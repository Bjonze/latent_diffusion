import os
import argparse

import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm
from dataloader_ import diffusion_dataloader
from utils import num_of_params
import wandb

from monai_network_init import init_autoencoder, init_latent_diffusion



set_determinism(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    

    #parser.add_argument('--cache_dir',  required=True, type=str)
    output_dir = r"E:\DTUTeams\bmsh\experiments"
    aekl_ckpt = r"E:\DTUTeams\bmsh\experiments\autoencoder-ep-3.pth"
    diff_ckpt = None #r"E:\DTUTeams\bmsh\experiments"
    num_workers = 0
    n_epochs = 5
    batch_size = 16
    lr = 2.5e-5

    latent_code_dir = r"E:\DTUTeams\bmsh\data\latent_codes"
    trainset = diffusion_dataloader(latent_code_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=num_workers, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              persistent_workers=False,
                              pin_memory=True)
    
    autoencoder = init_autoencoder(aekl_ckpt).to(device)
    diffusion = init_latent_diffusion(diff_ckpt).to(device)
    num_of_params(diffusion)
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule='scaled_linear_beta', 
        beta_start=0.0015, 
        beta_end=0.0205
    )

    inferer = DiffusionInferer(scheduler=scheduler)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr)
    scaler = GradScaler()
    
    with torch.no_grad():
        with autocast(enabled=True):
            z,_,_ = trainset[0]
    
    if not torch.is_tensor(z):
        z = torch.tensor(z)
    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    for epoch in range(n_epochs):
        diffusion.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
                        
            with autocast(enabled=True):
                mu, sigma, context = batch
                    
                optimizer.zero_grad(set_to_none=True)
                latents = mu.to(device) * scale_factor
                context = context.unsqueeze(1).to(device)
                n = latents.shape[0]
                                                        
                with torch.set_grad_enabled(True):#True if mode == 'train' else False
                    
                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()

                    noise_pred = inferer(
                        inputs=latents, 
                        diffusion_model=diffusion, 
                        noise=noise, 
                        timesteps=timesteps,
                        condition=context,
                        mode='crossattn'
                    )

                    loss = F.mse_loss( noise.float(), noise_pred.float() )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            #writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            #global_counter[mode] += 1
        
            # end of epoch
            epoch_loss = epoch_loss / len(train_loader)
            #writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

            # visualize results
        # save the model                
        savepath = os.path.join(output_dir, f'unet-ep-{epoch}.pth')
        torch.save(diffusion.state_dict(), savepath)