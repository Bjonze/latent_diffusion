from monai_network_init import init_autoencoder, init_patch_discriminator
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
import SimpleITK as sitk
from torch.cuda.amp import autocast
import os
import numpy as np
from tqdm import tqdm

model_path = r"D:\DTUTeams\bjorn\experiments\BrLP\autoencoder-ep-3.pth"
model = init_autoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
n = 3
sdf_list = os.listdir(r"D:\DTUTeams\bjorn\thesis_data\128_sdf")
sdf_outputs = np.empty((n, 128, 128, 128))
random.shuffle(sdf_list)
shuffle_list = sdf_list[:n]
mu = []
sigma = []
model.eval()
with torch.no_grad():
    for i, file in tqdm(enumerate(sdf_list), total=len(sdf_list)):
        if i >= n:
            break
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r"D:\DTUTeams\bjorn\thesis_data\128_sdf", file)))
        img = np.clip(img, a_min=-5, a_max=5) / 5.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
        #with autocast(enabled=True):
        reconstruction, z_mu, z_sigma = model(img)
        sdf_outputs[i] = reconstruction.squeeze().detach().cpu().numpy()
        mu.append(z_mu.squeeze().detach().cpu().numpy().flatten())
        sigma.append(z_sigma.squeeze().detach().cpu().numpy().flatten())
