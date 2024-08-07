import os
import argparse
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism

from torch.nn import L1Loss
from torch.cuda.amp import autocast, GradScaler
from generative.losses import PerceptualLoss, PatchAdversarialLoss

from gradacc import GradientAccumulation
from losses import KLDivergenceLoss
from monai_network_init import init_autoencoder, init_patch_discriminator

from dataloader_ import sdf_dataloader, aekl_dataloader, wav_dataloader
import wandb
from utils import get_recon_loss, num_of_params

from monai.data.image_reader import ITKReader
from monai.data import Dataset, PersistentDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.data import DataLoader

if __name__ == '__main__':
    json_path = r"E:\DTUTeams\bmsh\data\train_3.json" #r"C:\bjorn\train_3.json"
    #data = pd.read_json(json_path)
    # data = data.to_dict(orient='records')
    # c_dir = r"C:\bjorn\cache_dir"
    INPUT_SHAPE_AE = (128, 128, 128)
    # itk_reader = ITKReader()
    # transforms_fn = transforms.Compose([
    #     transforms.CopyItemsD(keys={'file_name'}, names=['image']),
    #     transforms.LoadImageD(image_only=True, keys=['image'], reader=itk_reader),
    #     transforms.EnsureChannelFirstD(keys=['image']),
    # ])
    # trainset = PersistentDataset(data=data, transform=transforms_fn, cache_dir=c_dir)
    #trainset = Dataset(data=data, transform=transforms_fn)
    # trainset = aekl_dataloader(data_dir=r"C:\bjorn\128_sdf",
    #                            json_data_dir=r"C:\bjorn\train_3.json")
    trainset = wav_dataloader(data_dir=r"E:\DTUTeams\bmsh\data\wavelet_sdf", json_data_dir=json_path)    
    dl_tr = DataLoader(dataset=trainset, 
                                num_workers=0, 
                                batch_size=2, 
                                shuffle=True, 
                                persistent_workers=False, 
                                pin_memory=False)
    
    out_dir = r"E:\DTUTeams\bmsh\experiments\BrLP_wav"#r"D:\DTUTeams\bjorn\experiments\BrLP_wav"
    os.makedirs(out_dir, exist_ok=True)

    autoencoder   = init_autoencoder().to(device)
    num_of_params(autoencoder)
    discriminator = init_patch_discriminator().to(device)
    num_of_params(discriminator)

    adv_weight          = 0.025
    perceptual_weight   = 0.001
    kl_weight           = 1e-5

    l1_loss_fn = get_recon_loss(mode="L1", reduction="mean", weight=None, device=device) #get_recon_loss(mode="L1", reduction="mean", weight=True, device=device)
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(device)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)


    gradacc_g = GradientAccumulation(actual_batch_size=2,
                                     expect_batch_size=16,
                                     loader_len=len(dl_tr),
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=2,
                                     expect_batch_size=16,
                                     loader_len=len(dl_tr),
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler())

    total_counter = 0

    wandb.init(project="PhD", entity="Bjonze")

    for epoch in range(10):
        
        autoencoder.train()
        progress_bar = tqdm(enumerate(dl_tr), total=len(dl_tr))
        progress_bar.set_description(f'Epoch {epoch}')

        for step, batch in progress_bar:

            with autocast(enabled=True):

                #images = batch["image"]#.to(device)
                #images = images.to(device)
                images = batch.to(device)
                #images = images.unsqueeze(1)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float()) #TODO: potentially back-transform images for this to work
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                
                loss_g = rec_loss + kld_loss + gen_loss #rec_loss + kld_loss + per_loss + gen_loss
                
            gradacc_g.step(loss_g, step)

            with autocast(enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss
        
            gradacc_d.step(loss_d, step)

            # log_dict = {"Generator/reconstruction_loss": rec_loss.item(), "Generator/perceptual_loss": per_loss.item(), "Generator/adverarial_loss" : gen_loss.item(),
            #             "Generator/kl_regularization": kld_loss.item(), "Discriminator/adverarial_loss": loss_d.item()}
            log_dict = {"Generator/reconstruction_loss": rec_loss.item(), "Generator/adverarial_loss" : gen_loss.item(),
                        "Generator/kl_regularization": kld_loss.item(), "Discriminator/adverarial_loss": loss_d.item()}
            wandb.log(log_dict)

        # Save the model after each epoch.
        torch.save(discriminator.state_dict(), os.path.join(out_dir, f'discriminator-ep-{epoch}.pth'))
        torch.save(autoencoder.state_dict(),   os.path.join(out_dir, f'autoencoder-ep-{epoch}.pth'))