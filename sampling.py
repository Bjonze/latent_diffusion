import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm


@torch.no_grad()
def sample_using_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    context: torch.Tensor,
    device: str, 
    scale_factor: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True,
    seed: int = None
) -> torch.Tensor: 
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function.
    if context.dim() == 1:
        context = context[None, None, :].to(device)
    if context.dim() == 2:
        context = context[None, :].to(device)
    else:
        context = context.to(device)

    # drawing a random z_T ~ N(0,I)
    latent_shape_dm = (3, 8, 8, 8)
    if seed is not None:
        torch.manual_seed(seed)
    #torch.manual_seed(1234)
    z = torch.randn(latent_shape_dm).unsqueeze(0).to(device)
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                timestep = torch.tensor([t]).to(device)
                
                # predict the noise
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float(), 
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)
    
    # decode the latent
    z = z / scale_factor
    #z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    x = autoencoder.decode_stage_2_outputs(z)
    #x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x

if __name__ == "__main__":
    from monai_network_init import init_autoencoder, init_latent_diffusion
    import torch
    from dataloader_ import diffusion_dataloader
    import trimesh
    from skimage.measure import marching_cubes
    import os
    output_dir = r"E:\DTUTeams\bmsh\meshes"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aekl_ckpt = r"E:\DTUTeams\bmsh\experiments\autoencoder-ep-3.pth"
    diff_ckpt = r"E:\DTUTeams\bmsh\experiments\unet-ep-4.pth"
    autoencoder = init_autoencoder(aekl_ckpt).to(device)
    diffusion = init_latent_diffusion(diff_ckpt).to(device)
    latent_code_dir = r"E:\DTUTeams\bmsh\data\latent_codes"
    trainset = diffusion_dataloader(latent_code_dir)
    mu, sigma, context = trainset[1]
    context = torch.tensor(context).to(device)

    with torch.no_grad():
        with autocast(enabled=True):
            z,_,_ = trainset[0]
    
    if not torch.is_tensor(z):
        z = torch.tensor(z)
    scale_factor = 1 / torch.std(z)


    x = sample_using_diffusion(
        autoencoder, 
        diffusion, 
        context,
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
    verts, faces, _, _ = marching_cubes(x, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(os.path.join(output_dir, 'diff_2.stl'))
