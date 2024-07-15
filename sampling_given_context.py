from monai_network_init import init_autoencoder, init_latent_diffusion
import torch
from torch.cuda.amp.autocast_mode import autocast
from dataloader_ import diffusion_dataloader
import trimesh
from skimage.measure import marching_cubes
import os
from sampling import sample_using_diffusion
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = r"E:\DTUTeams\bmsh\meshes"
aekl_ckpt = r"E:\DTUTeams\bmsh\experiments\autoencoder-ep-3.pth"
diff_ckpt = r"E:\DTUTeams\bmsh\experiments\unet-ep-4.pth"
latent_code_dir = r"E:\DTUTeams\bmsh\data\latent_codes"

autoencoder = init_autoencoder(aekl_ckpt).to(device)
diffusion = init_latent_diffusion(diff_ckpt).to(device)

trainset = diffusion_dataloader(latent_code_dir)
age = np.empty(len(trainset), dtype=np.float32)
ht = np.empty(len(trainset), dtype=np.float32)
bp_sys = np.empty(len(trainset), dtype=np.float32)
bp_dia = np.empty(len(trainset), dtype=np.float32)
heigth = np.empty(len(trainset), dtype=np.float32)
weight = np.empty(len(trainset), dtype=np.float32)
vol = np.empty(len(trainset), dtype=np.float32)
sex = np.empty(len(trainset), dtype=np.float32)

["age", "med_hypertension", "bp_sys", "bp_dia:", "height", "weight", "vol", "sex"]
for i, (mu, sigma, context) in tqdm(enumerate(trainset), total=len(trainset)):
    age[i] = context[0]
    ht[i] = context[1]
    bp_sys[i] = context[2]
    bp_dia[i] = context[3]
    heigth[i] = context[4]
    weight[i] = context[5]
    vol[i] = context[6]
    sex[i] = context[7]

low_age = np.quantile(age, 0.1)
high_age = np.quantile(age, 0.9)
low_bp_sys = np.quantile(bp_sys, 0.1)
high_bp_sys = np.quantile(bp_sys, 0.9)
low_bp_dia = np.quantile(bp_dia, 0.1)
high_bp_dia = np.quantile(bp_dia, 0.9)
low_heigth = np.quantile(heigth, 0.1)
high_heigth = np.quantile(heigth, 0.9)
low_weight = np.quantile(weight, 0.1)
high_weight = np.quantile(weight, 0.9)
low_vol = np.quantile(vol, 0.1)
high_vol = np.quantile(vol, 0.9)

conditions = []
conditions.append([age.mean(), 0.0, bp_sys.mean(), bp_dia.mean(), heigth.mean(), weight.mean(), vol.mean(), 0.0])
conditions.append([age.mean(), 0.0, low_bp_sys, low_bp_dia, heigth.mean(), weight.mean(), vol.mean(), 0.0])
conditions.append([age.mean(), 1.0, high_bp_sys, high_bp_dia, heigth.mean(), weight.mean(), vol.mean(), 0.0])
# conditions.append([low_age, 0.0, low_bp_sys, low_bp_dia, low_heigth, low_weight, low_vol, 0.0])
# conditions.append([low_age, 1.0, low_bp_sys, low_bp_dia, low_heigth, low_weight, low_vol, 0.0])
# conditions.append([high_age, 0.0, high_bp_sys, high_bp_dia, high_heigth, high_weight, high_vol, 0.0])
# conditions.append([high_age, 1.0, high_bp_sys, high_bp_dia, high_heigth, high_weight, high_vol, 0.0])

# conditions.append([high_age, 0.0, bp_sys.mean(), bp_dia.mean(), heigth.mean(), weight.mean(), vol.mean(), 0.0])
# conditions.append([low_age, 1.0, bp_sys.mean(), bp_dia.mean(), heigth.mean(), weight.mean(), vol.mean(), 0.0])
# conditions.append([high_age, 1.0, bp_sys.mean(), bp_dia.mean(), heigth.mean(), weight.mean(), vol.mean(), 0.0])
names = ["baseline", "low_bp_noHT_random", "high_bp_HT_random"]
context = torch.tensor(np.array(conditions), device=device)

with torch.no_grad():
    with autocast(enabled=True):
        z,_,_ = trainset[0]

if not torch.is_tensor(z):
    z = torch.tensor(z)
scale_factor = 1 / torch.std(z)

for i in tqdm(range(len(context)), total=len(context)):
    name = names[i]
    c = context[i].to(device)

    x = sample_using_diffusion(
        autoencoder, 
        diffusion, 
        c,
        device, 
        scale_factor,
        num_training_steps = 1000,
        num_inference_steps = 50,
        schedule = 'scaled_linear_beta',
        beta_start = 0.0015, 
        beta_end = 0.0205, 
        verbose = True,
        deterministic=False
    )
    x = x.squeeze().detach().cpu().numpy()
    verts, faces, _, _ = marching_cubes(x, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(os.path.join(output_dir, f'{name}.stl'))