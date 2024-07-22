import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from sampling import sample_using_diffusion
import trimesh
from skimage.measure import marching_cubes
from utils import js_r, js_w
import numpy as np 
import json

#creates a validation for a model
normalized_to_unnormalized = lambda x, min_val, max_val: (x + 1) * (max_val - min_val) / 2 + min_val

@torch.no_grad()
def validate_diffusion(diffusion, autoencoder, val_loader, scheduler, scale_factor, inferer, epoch, normalization_dir, save_dir, device):
    normalization_constants = js_r(normalization_dir)
    len_context = len(val_loader.context_keys) 
    diffusion.eval()
    autoencoder.eval()
    total_loss = 0

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    progress_bar.set_description(f"Validation for epoch {epoch}") 
    for step, batch in progress_bar:
        with autocast(enabled=True):
            mu, sigma, context = batch
    
            latents = mu.to(device) * scale_factor
            context = context.unsqueeze(1).to(device)
            n = latents.shape[0]
                                         
            with torch.set_grad_enabled(False):#True if mode == 'train' else False
                
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

            total_loss += loss.item()
    mean_loss = sum(total_loss) / len(total_loss)  
    val_log_dict = {"Validation/Loss": mean_loss}

    random_context = torch.empty(val_loader.batch_size, len_context).uniform_(-1, 1)
    binary_elements = torch.randint(0, 2, (val_loader.batch_size, len_context)).float() * 2 - 1
    random_context[:, 1] = binary_elements[:, 1]
    random_context[:, 6] = binary_elements[:, 6]
    random_context = random_context.to(device)

    x = sample_using_diffusion(
        autoencoder, 
        diffusion, 
        random_context,
        device, 
        scale_factor,
        num_training_steps = 1000,
        num_inference_steps = 50,
        schedule = 'scaled_linear_beta',
        beta_start = 0.0015, 
        beta_end = 0.0205, 
        verbose = True
    )
    x = x.squeeze().detach().cpu().numpy()
    random_context_np = random_context.detach().cpu().numpy()
    random_context_unnormalized = np.empty_like(random_context_np)
    for c_key in normalization_constants.keys():
        min_val = normalization_constants[c_key]["min"]
        max_val = normalization_constants[c_key]["max"]
        random_context_unnormalized[:, val_loader.context_keys.index(c_key)] = normalized_to_unnormalized(random_context_np[:, val_loader.context_keys.index(c_key)], min_val, max_val)

    patient_dict = []
    for i, sdf in enumerate(x):
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(os.path.join(save_dir, f'synth_la_{i}.stl'))
        ht_bool = "True" if random_context_unnormalized[i, 1] > 0 else "False"
        sex = "male" if random_context_unnormalized[i,6] > 0 else "female"
        patient_dict.append({"file_name": f'synth_la_{i}.stl',
                    "age": round(random_context_unnormalized[i, 0], 2),
                    "med_hypertension": ht_bool, #True / false
                    "bp_sys": round(random_context_unnormalized[i, 2], 2),
                    "bp_dia": round(random_context_unnormalized[i, 3], 2),
                    "height": round(random_context_unnormalized[i, 4], 2),
                    "weight": round(random_context_unnormalized[i, 5], 2),
                    "sex": sex
                    })
    json_outdir = os.path.join(save_dir, "synth_la.json")
    js_w(json_outdir, patient_dict)
    return val_log_dict
