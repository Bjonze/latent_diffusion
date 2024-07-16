import torch
import os 
import shutil
import numpy as np
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
import SimpleITK as sitk
import re
import json
import torch.nn as nn
import math

def get_class_weights(dict_list):
    """
    Compute class weights for a list of dictionaries.
    """
    # Initialize counters for each class
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for entry in dict_list:
        class_counts[entry['group']] += 1
    
    # Compute class weights
    class_weights = {key: sum(class_counts.values()) / (len(class_counts) * class_counts[key]) for key in class_counts}

    return class_weights

class ClassificationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(ClassificationLoss, self).__init__()
        if class_weights is not None:
            class_weights = torch.tensor([t for t in class_weights.values()], dtype=torch.float, device='cuda')

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, outputs, targets):
        """
        Compute the classification loss.

        Parameters:
        outputs (torch.Tensor): The output from the model of shape (batch_size, num_classes).
                                Each row corresponds to the output probabilities or logits for each class.
        targets (torch.Tensor): The ground truth labels of shape (batch_size). Each value is an integer 
                                representing the correct class for that sample.

        Returns:
        torch.Tensor: The computed cross-entropy loss.
        """
        loss = self.criterion(outputs, targets)
        return loss

class CompositeLoss(nn.Module):
    def __init__(self, weight_classification=1.0, weight_regression=1.0):
        super(CompositeLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.m = nn.Sigmoid()
        self.weight_classification = weight_classification
        self.weight_regression = weight_regression

    def forward(self, outputs, gt_hp_bool, gt_bp_sys, gt_bp_dia, gt_clin_bp_sys, gt_clin_bp_dia):
        # Split outputs and targets
        med_hypertension_pred = outputs[:, 0]
        bp_sys_pred = outputs[:,1]
        bp_dia_pred = outputs[:,2]
        clin_bp_sys_pred = outputs[:,3]
        clin_bp_dia_pred = outputs[:,4]
        # Compute the losses
        med_hypertension_loss = self.bce_loss(self.m(med_hypertension_pred), gt_hp_bool)
        bp_sys_loss = self.mse_loss(bp_sys_pred, gt_bp_sys)
        bp_dia_loss = self.mse_loss(bp_dia_pred, gt_bp_dia)
        clin_bp_sys_loss = self.mse_loss(clin_bp_sys_pred, gt_clin_bp_sys)
        clin_bp_dia_loss = self.mse_loss(clin_bp_dia_pred, gt_clin_bp_dia)

        total_loss = (self.weight_classification * med_hypertension_loss) + self.weight_regression * (bp_sys_loss+bp_dia_loss+clin_bp_sys_loss+clin_bp_dia_loss)
        
        return total_loss, med_hypertension_loss, bp_sys_loss, bp_dia_loss, clin_bp_sys_loss, clin_bp_dia_loss

def remove_nans_from_dicts(dict_list):
    """
    Removes dictionaries containing NaNs from a list of dictionaries.
    
    Parameters:
    dict_list (list): A list of dictionaries with keys 'file_name', 'med_hypertension', 
                      'scandata_BP_sys', 'scandata_BP_dia', 'clin_BP_sys', and 'clin_BP_dia'.
                      
    Returns:
    list: A list of dictionaries with no NaNs in the specified keys.
    """
    keys_to_check = ['file_name', 'med_hypertension', 'scandata_BP_sys', 
                     'scandata_BP_dia', 'clin_BP_sys', 'clin_BP_dia']
    
    def contains_nan(dictionary):
        for key in keys_to_check:
            if key not in dictionary or dictionary[key] is None or (isinstance(dictionary[key], float) and math.isnan(dictionary[key])):
                return True
        return False
    
    return [d for d in dict_list if not contains_nan(d)]
def normalize_and_transform_quantiles(data, min_max_values=None):
    # Initialize dictionaries to hold min and max values for normalization
    if min_max_values is None:
        min_max_values = {
            'scandata_BP_sys': {'min': float('inf'), 'max': float('-inf')},
            'scandata_BP_dia': {'min': float('inf'), 'max': float('-inf')},
            'clin_BP_sys': {'min': float('inf'), 'max': float('-inf')},
            'clin_BP_dia': {'min': float('inf'), 'max': float('-inf')},
            'age': {'min': float('inf'), 'max': float('-inf')},
            'clin_height': {'min': float('inf'), 'max': float('-inf')},
            'clin_weight': {'min': float('inf'), 'max': float('-inf')},
            'vol': {'min': float('inf'), 'max': float('-inf')},
            'med_hypertension': {'min': float(-1), 'max': float(1)},
            'clin_sex': {'min': float(0), 'max': float(1)}
            
        }

        # Find the min and max values for each key
        scandata_bp_sys = []
        scandata_bp_dia = []
        bp_sys = []
        bp_dia = []
        age = []
        height = []
        weight = []
        vol = []
        med_hypertension = []
        sex = []
        for entry in tqdm(data):
            scandata_bp_sys.append(entry['scandata_BP_sys'])
            scandata_bp_dia.append(entry['scandata_BP_dia'])
            bp_sys.append(entry['clin_BP_sys'])
            bp_dia.append(entry['clin_BP_dia'])
            age.append(entry['age'])
            height.append(entry['clin_height'])
            weight.append(entry['clin_weight'])
            vol.append(entry['vol'])

        scandata_bp_sys = np.array(scandata_bp_sys)
        scandata_bp_dia = np.array(scandata_bp_dia)
        bp_sys = np.array(bp_sys)
        bp_dia = np.array(bp_dia)
        age = np.array(age)
        height = np.array(height)
        weight = np.array(weight)
        vol = np.array(vol)

        min_max_values['scandata_BP_sys']['min'] = np.quantile(scandata_bp_sys, 0.01)
        min_max_values['scandata_BP_sys']['max'] = np.quantile(scandata_bp_sys, 0.99)
        min_max_values['scandata_BP_dia']['min'] = np.quantile(scandata_bp_dia, 0.01)
        min_max_values['scandata_BP_dia']['max'] = np.quantile(scandata_bp_dia, 0.99)
        min_max_values['clin_BP_sys']['min'] = np.quantile(bp_sys, 0.01)
        min_max_values['clin_BP_sys']['max'] = np.quantile(bp_sys, 0.99)
        min_max_values['clin_BP_dia']['min'] = np.quantile(bp_dia, 0.01)
        min_max_values['clin_BP_dia']['max'] = np.quantile(bp_dia, 0.99)
        min_max_values['age']['min'] = np.quantile(age, 0.01)
        min_max_values['age']['max'] = np.quantile(age, 0.99)
        min_max_values['clin_height']['min'] = np.quantile(height, 0.01)
        min_max_values['clin_height']['max'] = np.quantile(height, 0.99)
        min_max_values['clin_weight']['min'] = np.quantile(weight, 0.01)
        min_max_values['clin_weight']['max'] = np.quantile(weight, 0.99)
        min_max_values['vol']['min'] = np.quantile(vol, 0.01)
        min_max_values['vol']['max'] = np.quantile(vol, 0.99)
        min_max_values['med_hypertension']['min'] = -1.0
        min_max_values['med_hypertension']['max'] = 1.0
        min_max_values['clin_sex']['min'] = -1.0
        min_max_values['clin_sex']['max'] = 1.0
        for entry in data:
            for key in min_max_values:
                if entry[key] == None:
                    continue
                min_val = min_max_values[key]['min']
                max_val = min_max_values[key]['max']
                entry[key] = -1.0 + 2.0 * (entry[key] - min_val) / (max_val - min_val)

    return data, min_max_values

def normalize_and_transform(data, min_max_values=None):
    # Initialize dictionaries to hold min and max values for normalization
    if min_max_values is None:
        min_max_values = {
            'scandata_BP_sys': {'min': float('inf'), 'max': float('-inf')},
            'scandata_BP_dia': {'min': float('inf'), 'max': float('-inf')},
            'clin_BP_sys': {'min': float('inf'), 'max': float('-inf')},
            'clin_BP_dia': {'min': float('inf'), 'max': float('-inf')},
            'age': {'min': float('inf'), 'max': float('-inf')},
            'clin_height': {'min': float('inf'), 'max': float('-inf')},
            'clin_weight': {'min': float('inf'), 'max': float('-inf')},
            'vol': {'min': float('inf'), 'max': float('-inf')},
            'med_hypertension': {'min': float(-1), 'max': float(1)},
            'clin_sex': {'min': float(0), 'max': float(1)}
            
        }

        # Find the min and max values for each key
        for entry in data:
            for key in min_max_values:
                value = entry[key]
                if value == None:
                    continue
                if value < min_max_values[key]['min']:
                    min_max_values[key]['min'] = value
                if value > min_max_values[key]['max']:
                    min_max_values[key]['max'] = value

        # Normalize the values and transform med_hypertension
    for entry in data:
        for key in min_max_values:
            if entry[key] == None:
                continue
            min_val = min_max_values[key]['min']
            max_val = min_max_values[key]['max']
            entry[key] = -1.0 + 2.0 * (entry[key] - min_val) / (max_val - min_val)
        
        entry['med_hypertension'] = 1 if entry['med_hypertension'] else -1
        entry['clin_sex'] = 1 if entry['clin_sex']>0.5 else 0

    return data, min_max_values

def load_results_from_json(input_path):
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        print(f"Data successfully loaded from {input_path}")
        return data
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Please check the file path and try again.")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}. Please ensure the file contains valid JSON.")

def np_to_tensor(np_array, device):
    return torch.tensor(np_array, dtype=torch.float32, device=device)

def get_act(act_name="relu"):
    if act_name.lower()=="relu":
        act = torch.nn.ReLU(inplace=True)
    elif act_name.lower()=="silu":
        act = torch.nn.SiLU(inplace=True)
    else:
        raise ValueError("Did not recognize activation: " + act_name)
    return act

def num_of_params(net,full_print=False, no_print=False):
    n_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            n_param += param.data.numel()
            if full_print:
                print(name+", shape="+str(param.data.shape))
    if not no_print: print("Net has " + str(n_param) + " params.")
    return n_param

def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5, n_0=0):
    L = np.ones(n_iter) * stop
    # Set the first n_0 entries to 0
    for i in range(min(n_0, n_iter)):
        L[i] = 0

    period = (n_iter - n_0) / n_cycle  # Adjust period to exclude the first n_0 elements
    step = (stop - start) / (period * ratio)  # Linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period + n_0) < n_iter):
            index = int(i + c * period + n_0)
            L[index] = v
            v += step
            i += 1
    return L

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def load_model_by_time(directory):
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time)
    return sorted_items

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

class get_recon_loss(nn.Module):
    def __init__(self, mode="MSE", reduction="sum", weight=[45.0], device="cuda:0"):
        super(get_recon_loss, self).__init__()
        self.mode = mode
        self.reduction = reduction
        self.weight = weight
        assert isinstance(self.mode,str), f"Mode must be a string. Your input: {self.mode} which is of type {type(self.mode)}"
        if self.mode == "MSE":
            self.criterion = nn.MSELoss(reduction="none")
        elif self.mode == "smooth_l1":
            self.criterion = nn.SmoothL1Loss(reduction="none", beta=2.0)
        elif self.mode == "L1":
            self.criterion = nn.L1Loss(reduction="none")
        elif self.mode =="BCE":
            if weight is not None: 
                self.pos_weight = torch.tensor(weight).to(device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight, reduction="none")
            else:
                self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Invalid mode: {mode}")
    def forward(self, x_pred, x):
        loss = self.criterion(x_pred, x)

        if self.weight is not None and self.mode == "MSE" or self.mode == "smooth_l1" or self.mode == "L1":
            epsilon = 1e-6
            weight_tensor = 1.0/torch.tanh(abs(x)+epsilon)
            weighted_loss = loss * weight_tensor
            if self.reduction == "sum":
                return weighted_loss.sum()
            elif self.reduction == "mean":
                return weighted_loss.sum() / weight_tensor.sum()
            else:
                raise ValueError(f"Invalid reduction: {self.reduction}")
        else:
            if self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "mean":
                return loss.mean()
            else:
                raise ValueError(f"Invalid reduction: {self.reduction}. Please chose either 'sum' or 'mean'")

def adjust_learning_rate(optimizer, epoch, initial_lr, warm_up_epochs, total_epochs):
    if epoch < warm_up_epochs:  # Warm-up phase
        lr = initial_lr * (epoch + 1) / warm_up_epochs
    #else:  # After warm-up, you could keep it constant, or decrease it, e.g., using a decay factor
        #decay_factor = 0.5  # for example
        #lr = initial_lr * (decay_factor ** ((epoch - warm_up_epochs) / (total_epochs - warm_up_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def postprocess_mesh(mesh, num_faces=None):
    """Post processing mesh by removing small isolated pieces.

    Args:
        mesh (trimesh.Trimesh): input mesh to be processed
        num_faces (int, optional): min face num threshold. Defaults to 4096.
    """
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 10
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=3)
    mask = np.zeros(total_num_faces, dtype=np.bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)
    return mesh

def natural_sort_key(s):
    """
    A key function for natural (human) sorting.
    Extracts two parts: the number after 'CFA' and the number after 'SERIES'.
    """
    match = re.match(r'CFA(\d+)SERIES(\d+)', s)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0  # Default value if the pattern is not found

def load_sorted_data(directory):
    """
    Loads data from files in a directory with natural sorting of filenames.
    Assumes files are named in the format 'CFA<number>SERIES<number>'.
    """
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out files that do not match the 'CFA<number>SERIES<number>' pattern
    cfa_files = [f for f in files if re.match(r'CFA\d+SERIES\d+', f)]

    # Sort files using natural sort
    sorted_files = sorted(cfa_files, key=natural_sort_key)

    # Load data from each file (this part depends on your data format)
    data = []
    for file in sorted_files:
        file_path = os.path.join(directory, file)
        # Load your data here, e.g., using numpy or pandas if it's numeric data
        # For example:
        data.append(np.load(file_path))
        #print("Loaded file:", file_path)  # Placeholder for actual data loading

    return data