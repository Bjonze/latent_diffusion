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
from monai.data import Dataset, PersistentDataset
from monai_network_init import init_latent_diffusion_2d_test
from validate import validate_diffusion
import socket
import PIL
import torchvision.transforms as transforms
from sampling import test_using_diffusion
import numpy as np 
from DWT_IDWT_layer import DWT_2D, IDWT_2D
from einops import rearrange
#import cv2
set_determinism(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    
    if socket.gethostname() == 'COMP-NB5713': # If running on HP Zbook
        output_dir = r"C:\Users\bmsha\Code\experiments\DDPI_test"
        os.makedirs(output_dir, exist_ok=True)
        save_dir = os.path.join(output_dir, 'validation')
        os.makedirs(save_dir, exist_ok=True)
    else:
        raise ValueError("Please specify the output directory for the experiment")
    
    dwt = DWT_2D("haar")
    iwt = IDWT_2D("haar")

    mean = (0.6883, 0.3497, 0.5180)
    std = (0.2456, 0.2026, 0.2277)
    transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda x: x/255.0),
    transforms.Normalize(mean, std)
    ])

    normalization_dir = None
    diff_ckpt = None
    n_epochs = 5000

    lr = 1e-4
    scale_factor = 1.0
    diffusion = init_latent_diffusion_2d_test(diff_ckpt).to(device)
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
    pil_img = PIL.Image.open(r"C:\Users\bmsha\data\test_cat\test_image.jpg")
    img = transform(pil_img)
    img = img.unsqueeze(0).to(device) * scale_factor
    dwts = dwt(img)
    dwts = torch.cat(dwts, dim=0)
    dwts = rearrange(dwts, 'b c h w -> 1 (b c) h w')
    img = dwts


    
    print(f"Scaling factor set to {scale_factor}")
    progress_bar = tqdm(range(n_epochs), desc='Epochs')
    for epoch in progress_bar:
        i = 0
        diffusion.train()
        epoch_loss = 0      
        with autocast(enabled=True):
            #load .jpg image from path
            context = None
                
            optimizer.zero_grad(set_to_none=True)
            n = img.shape[0]
                                                    
            with torch.set_grad_enabled(True):#True if mode == 'train' else False
                
                noise = torch.randn_like(img).to(device)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=device).long()

                noise_pred = inferer(
                    inputs=img, 
                    diffusion_model=diffusion, 
                    noise=noise, 
                    timesteps=timesteps,
                    condition=context,
                    mode='crossattn'
                )

                #loss = F.mse_loss( noise.float(), noise_pred.float() )
                loss = F.l1_loss( noise.float(), noise_pred.float() )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (i + 1)})
        i += 1
        
    
    nis = [50,100,150,200,250,500]
    mean = np.array(mean)
    mean = mean[:, None, None]
    std = np.array(std)
    std = std[:, None, None]
    for t in nis:
        savepath = os.path.join(output_dir, f"L1_wav_num_epochs_{n_epochs}_num_inf_{t}.png")
        x = test_using_diffusion(diffusion = diffusion, 
                                context = context,
                                device = device, 
                                scale_factor = scale_factor,
                                num_training_steps = 1000,
                                num_inference_steps = t,
                                schedule = 'scaled_linear_beta',
                                beta_start = 0.0015, 
                                beta_end = 0.0205,
                                verbose = True,
                                seed = None)
        x = rearrange(x, '1 (b c) h w -> b c h w', b=4, c=3)
        LL, LH, HL, HH = torch.split(x, 1, dim=0)
        x = iwt(LL, LH, HL, HH)
        x = x.squeeze().detach().cpu().numpy()
        #unormalize the image using the mean and std tuples

        x = x * std + mean
        x = x * 255.0
        x = np.clip(x, 0, 255)
        x = x.astype('uint8')
        x = x.transpose(1, 2, 0)
        y = PIL.Image.fromarray(x, 'RGB')
        y.save(savepath)
    #torch.save(diffusion.state_dict(), savepath)
