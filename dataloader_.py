import torch 
import numpy as np
import os

import torch.utils 
from utils import load_results_from_json, normalize_and_transform, remove_nans_from_dicts, get_class_weights
import SimpleITK as sitk
from monai import transforms
from monai.data import Dataset, PersistentDataset
class sdf_dataloader(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_data_dir, min_max_values=None, eval=False):
        super().__init__
        self.dir_data_json = json_data_dir
        self.json = load_results_from_json(json_data_dir)
        self.eval = eval
        if min_max_values is None:
            self.json, self.min_max_values = normalize_and_transform(self.json)
        else:
            self.json, self.min_max_values = normalize_and_transform(self.json, min_max_values)
        self.json = remove_nans_from_dicts(self.json)
        self.data_dir = data_dir
        #self.class_weights = get_class_weights(self.json)
        self.keys_c = ["age", "med_hypertension", "clin_BP_sys", "clin_BP_dia", "clin_height", "clin_weight"]
        self.keys_p = ["vol", 'clin_sex']
    def __len__(self) -> int:
        self.len = len(self.json)
        return self.len 
    
    def __getitem__(self, idx):
        idx_dict = self.json[idx]
        file_name = idx_dict["file_name"]
        if self.data_dir is None:
            sdf = np.random.rand(128, 128, 128)
        else:
            sdf = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_dir, file_name)))
            sdf = np.clip(sdf, a_min=-5, a_max=5) / 5.0
        
        
        c_values = [idx_dict[key] for key in self.keys_c]
        c = np.array(c_values, dtype=np.float32)
        
        p_values = [idx_dict[key] for key in self.keys_p]
        p = np.array(p_values, dtype=np.float32)

        if self.eval:
            return sdf, c, p, idx_dict
        else:
            return sdf, c, p

class diffusion_dataloader(torch.utils.data.Dataset):
    def __init__(self, latent_code_dir):
        super().__init__
        self.latent_code_dir = latent_code_dir
        self.latent_code_files = os.listdir(latent_code_dir)
        self.context_keys = ["age", "med_hypertension", "bp_sys", "bp_dia:", "height", "weight", "vol", "sex"]
    def __len__(self) -> int:
        self.len = len(self.latent_code_files)
        return self.len
    
    def __getitem__(self, idx):
        file = self.latent_code_files[idx]
        latent_code = np.load(os.path.join(self.latent_code_dir, file), allow_pickle=True)
        context_dict = latent_code["context"].item()
        context = np.array([context_dict[c] for c in self.context_keys])
        mu = latent_code["mu"]
        sigma = latent_code["z_sigma"]
        id = latent_code["id"]
        #t1 = transforms.DivisiblePadD(k=4, mode='constant')(mu)
        #x = t1(mu)
        # transforms_fn = transforms.Compose([
        #     transforms.CopyItemsD(keys=['latent_path'], names=['latent']),
        #     transforms.LoadImageD(keys=['latent'], reader=npz_reader),
        #     transforms.EnsureChannelFirstD(keys=['latent'], channel_dim=0), 
        #     transforms.DivisiblePadD(keys=['latent'], k=4, mode='constant'),
        # ])
        
        return mu, sigma, context


if __name__ == '__main__':
    latent_code_dir = r"E:\DTUTeams\bmsh\data\latent_codes"
    ds_tr = diffusion_dataloader(latent_code_dir)
    mu, sigma, context = ds_tr[0]
