import torch 
import numpy as np
import os

import torch.utils 
from utils import load_results_from_json, normalize_and_transform, remove_nans_from_dicts, get_class_weights
import SimpleITK as sitk

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