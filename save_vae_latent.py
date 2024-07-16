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
import pandas as pd 
from monai.data.image_reader import ITKReader
from monai.data import Dataset, PersistentDataset
from monai import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_autoencoder()
model.load_state_dict(torch.load(r"D:\DTUTeams\bjorn\experiments\BrLP_2\autoencoder-ep-2.pth"))
model = model.to(device)


file_list = r"C:\bjorn\train_3.json"
data_dir = r"C:\bjorn\128_sdf"
ls_dir = r"C:\bjorn\latent_codes"
os.makedirs(ls_dir, exist_ok=True)
save_path = r"C:\bjorn\file_names"
os.makedirs(save_path, exist_ok=True)
json_path = r"C:\bjorn\train_3.json"
data = pd.read_json(json_path)
data = data.to_dict(orient='records')
c_dir = r"C:\bjorn\cache_dir"
INPUT_SHAPE_AE = (128, 128, 128)
itk_reader = ITKReader()
transforms_fn = transforms.Compose([
    transforms.CopyItemsD(keys={'file_name'}, names=['image']),
    transforms.LoadImageD(image_only=True, keys=['image'], reader=itk_reader),
    transforms.EnsureChannelFirstD(keys=['image']),
])
trainset = PersistentDataset(data=data, transform=transforms_fn, cache_dir=c_dir)


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
        for i, data in tqdm(enumerate(trainset), total=len(trainset)):
            full_names_list.append(data["file_name"].split("/")[-1])
            match = pattern.search(data["file_name"])
            names_list.append(match.group(1))
            
        names_list_path = os.path.join(save_path, "names_list.npy")
        full_names_list_path = os.path.join(save_path, "full_names_list.npy")
        np.save(names_list_path, names_list)
        np.save(full_names_list_path, full_names_list)
model.eval()

for i, data, short_name in tqdm(zip(range(len(trainset)), trainset, names_list), total=len(trainset)):
    with autocast(enabled=True):

        with torch.no_grad():
            image = data["image"]
            image = image.unsqueeze(0).to(device)
            z_mu, z_sigma = model.encode(image)

    mu_np = z_mu.squeeze().detach().cpu().numpy()
    sigma_np = z_sigma.squeeze().detach().cpu().numpy()
    total_mu += mu_np
    total_logvar += sigma_np
    total_age += data["age"]
    total_HT += data["med_hypertension"]
    total_bp_sys += data["clin_BP_sys"]
    total_bp_dia += data["clin_BP_dia"]
    total_height += data["clin_height"]
    total_weight += data["clin_weight"]
    num_files += 1
    np_savez_kwargs = {"z_mu": mu_np, "z_sigma": sigma_np, "id": short_name}
    np_savez_kwargs.update({"context":{
        "age": data["age"],
        "med_hypertension": data["med_hypertension"],
        "bp_sys": data["clin_BP_sys"],
        "bp_dia": data["clin_BP_dia"], 
        "height": data["clin_height"],
        "weight": data["clin_weight"],
        "vol": data["vol"],
        "sex": data["clin_sex"]
    }})
    temp = data["file_name"].split("/")[-1].split(".")[0]
    out_name = os.path.join(ls_dir, f"{temp}.npz")
    np.savez(out_name, **np_savez_kwargs)
