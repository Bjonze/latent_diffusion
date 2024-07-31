import os
import argparse

import torch
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
from gradacc import GradientAccumulation

from torch.utils.data import DataLoader

from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm
from dataloader_ import wav_dataloader
from utils import num_of_params
import wandb
from monai.data import Dataset, PersistentDataset
from monai_network_init import init_wav_diffusion
from validate import validate_diffusion
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

set_determinism(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    

    #parser.add_argument('--cache_dir',  required=True, type=str)
    output_dir = r"C:\bjorn\wavelet_clipped_diffusion"
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, 'validation')
    os.makedirs(save_dir, exist_ok=True)

    normalization_dir = r"C:\bjorn\normalization_constants\constants.json"

    diff_ckpt = None
    num_workers = 8
    n_epochs = 15
    batch_size = 2
    lr = 2.5e-4


    data_dir = r"D:\DTUTeams\bjorn\thesis_data\wavelet_clipped_sdf"
    json_data_dir = r"C:\bjorn\train_3.json"
    trainset = wav_dataloader(data_dir, json_data_dir)


    train_loader = DataLoader(dataset=trainset, 
                              num_workers=num_workers, #num_workers
                              batch_size=batch_size, 
                              shuffle=True, 
                              persistent_workers=True,
                              pin_memory=False)
    
    diffusion = init_wav_diffusion(diff_ckpt).to(device)
    num_of_params(diffusion)
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr)
    scaler = GradScaler('cuda')
    gradacc_d = GradientAccumulation(actual_batch_size=2,
                                     expect_batch_size=16,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer, 
                                     grad_scaler=scaler)
    
    
    with torch.no_grad():
        with autocast('cuda', enabled=True):
            z,_, = trainset[0]
    if not torch.is_tensor(z):
        z = torch.tensor(z)

    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")
    wandb.init(project="PhD", entity="Bjonze")
    for epoch in range(n_epochs):
        diffusion.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
                        
            with autocast('cuda', enabled=True):
                wav_sdf, context = batch
                if context.dtype == torch.float64:
                    context = context.float()

                    
                optimizer.zero_grad(set_to_none=True)
                wav_sdf = wav_sdf.to(device) * scale_factor
                context = context.unsqueeze(1).to(device)
                n = wav_sdf.shape[0]
                                                        
                with torch.set_grad_enabled(True):#True if mode == 'train' else False
                    
                    noise = torch.randn_like(wav_sdf).to(device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()

                    noise_pred = inferer(
                        inputs=wav_sdf, 
                        diffusion_model=diffusion, 
                        noise=noise, 
                        timesteps=timesteps,
                        condition=context,
                        mode='crossattn'
                    )

                    loss = F.mse_loss( noise.float(), noise_pred.float() )
            log_dict = {"loss": loss.item() / (step + 1)}
            wandb.log(log_dict)
            gradacc_d.step(loss, step)
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
                
            #writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            #global_counter[mode] += 1
            
            
            # end of epoch
            #epoch_loss = epoch_loss / len(train_loader)
            
            #writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
        # save the model                
        savepath = os.path.join(output_dir, f'unet-ep-{epoch}.pth')
        torch.save(diffusion.state_dict(), savepath)