import torch
import os 
from matplotlib.colors import Normalize
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import re 
from utils import np_to_tensor, get_recon_loss
from dataloader_ import sdf_dataloader
from torch.cuda.amp import autocast, GradScaler
from monai_network_init import init_autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_autoencoder()
model.load_state_dict(torch.load(r"D:\DTUTeams\bjorn\experiments\BrLP\autoencoder-ep-3.pth"))
model = model.to(device)


file_list = r"D:\DTUTeams\bjorn\thesis_data\train.json"
data_dir = r"D:\DTUTeams\bjorn\thesis_data\128_sdf"
dl_tr = sdf_dataloader(data_dir, file_list, eval=True)
ls_dir = r"D:\DTUTeams\bjorn\experiments\BrLP\latent_codes"
os.makedirs(ls_dir, exist_ok=True)
save_path = r"D:\DTUTeams\bjorn\experiments\BrLP\file_names"
os.makedirs(save_path, exist_ok=True)
names_list = []
mean_mu = None
mean_logvar = None
data_dict = []
full_names_list = []
if not len(os.listdir(ls_dir))>1:
    pattern = re.compile(r"Aorta-\d+_(\d{4})_SERIES\d+_segment_2_crop_label.nii.gz") #REGEX to clean the file name
    total_mu = total_logvar = total_attribute_latent = total_combined_latent = 0
    total_age = total_HT = total_bp_sys = total_bp_dia = total_height = total_weight = num_files = 0
    print("\nChecking for good files and cleaning names...")
    if file_list is not None: #json_data_dir_vl or json_data_dir_tr
        for i, data in tqdm(enumerate(dl_tr), total=len(dl_tr)):
            _, _, _, idx_dict = data #x = sdf, c = clinical data, p = values to predict, idx_dict = total patient info
            full_names_list.append(idx_dict["file_name"])
            match = pattern.search(idx_dict["file_name"])
            names_list.append(match.group(1))
            
        names_list_path = os.path.join(save_path, "names_list.npy")
        full_names_list_path = os.path.join(save_path, "full_names_list.npy")
        np.save(names_list_path, names_list)
        np.save(full_names_list_path, full_names_list)
model.eval()

for i, data, short_name in tqdm(zip(range(len(dl_tr)), dl_tr, names_list), total=len(dl_tr)):
    with autocast(enabled=True):
        x, c, p, idx_dict = data #x = sdf, c = clinical data, p = values to predict, idx_dict = total patient info
        out_dir = os.path.join(ls_dir, idx_dict["file_name"].split(".")[0])
        if not torch.is_tensor(x):
            x = np_to_tensor(x, device)
        if not torch.is_tensor(c):
            c = np_to_tensor(c, device)
        if not torch.is_tensor(p):
            p = np_to_tensor(p, device)
        if x.dim() <= 4:
            x = x[None, None, :, :, :]
        if c.dim() <= 1:
            c = c[None, :]
        if p.dim() <= 1:
            p = p[None, :]
        with torch.no_grad():
            z_mu, z_sigma = model.encode(x)
            c_np = c.squeeze().detach().cpu().numpy()
            p_np = p.squeeze().detach().cpu().numpy()

    mu_np = z_mu.squeeze().detach().cpu().numpy()
    sigma_np = z_sigma.squeeze().detach().cpu().numpy()
    total_mu += mu_np
    total_logvar += sigma_np
    total_age += c_np[0]
    total_HT += c_np[1]
    total_bp_sys += c_np[2]
    total_bp_dia += c_np[3]
    total_height += c_np[4]
    total_weight += c_np[5]
    num_files += 1
    np_savez_kwargs = {"mu": mu_np, "z_sigma": sigma_np, "id": short_name}
    np_savez_kwargs.update({"context":{
        "age": c_np[0],
        "med_hypertension": c_np[1],
        "bp_sys":c_np[2],
        "bp_dia:": c_np[3], 
        "height": c_np[4],
        "weight": c_np[5],
        "vol": p_np[0],
        "sex": p_np[1]
    }})
    np.savez(out_dir, **np_savez_kwargs)
