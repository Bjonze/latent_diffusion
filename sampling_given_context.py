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
from random import randint
from utils import js_r, js_w

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = randint(0, 1000)
normalized_to_unnormalized = lambda x, min_val, max_val: (x + 1) * (max_val - min_val) / 2 + min_val
normalization_dir = r"C:\bjorn\normalization_constants\constants.json"
output_dir = r"C:\bjorn\latent_diffusion_2.1\test_meshes"
os.makedirs(output_dir, exist_ok=True)
quantile_dir = r"C:\bjorn\latent_diffusion_2.1\quantiles"
os.makedirs(quantile_dir, exist_ok=True)
aekl_ckpt = r"D:\DTUTeams\bjorn\experiments\BrLP_2.1\autoencoder-ep-5.pth"
diff_ckpt = r"C:\bjorn\latent_diffusion_2.1\unet-ep-4.pth"
latent_code_dir = r"C:\bjorn\latent_codes_train_2.1"

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

#["age", "med_hypertension", "bp_sys", "bp_dia:", "height", "weight", "sex"]
if not os.path.exists(os.path.join(quantile_dir, "quantiles.npz")):
    for i, (mu, sigma, context) in tqdm(enumerate(trainset), total=len(trainset)):
        age[i] = context[0]
        ht[i] = context[1]
        bp_sys[i] = context[2]
        bp_dia[i] = context[3]
        heigth[i] = context[4]
        weight[i] = context[5]
        sex[i] = context[6]
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
    np.savez(os.path.join(quantile_dir, "quantiles.npz"), low_age=low_age, mean_age=age.mean(), high_age=high_age, low_bp_sys=low_bp_sys, mean_bp_sys=bp_sys.mean(),
             high_bp_sys=high_bp_sys, low_bp_dia=low_bp_dia, mean_bp_dia = bp_dia.mean(), high_bp_dia=high_bp_dia, low_heigth=low_heigth, mean_heigth=heigth.mean(),
             high_heigth=high_heigth, low_weight=low_weight, mean_weight = weight.mean(), high_weight=high_weight)
    
quantiles = np.load(os.path.join(quantile_dir, "quantiles.npz"))
low_age = quantiles["low_age"]
mean_age = quantiles["mean_age"]
high_age = quantiles["high_age"]
low_bp_sys = quantiles["low_bp_sys"]
mean_bp_sys = quantiles["mean_bp_sys"]
high_bp_sys = quantiles["high_bp_sys"]
low_bp_dia = quantiles["low_bp_dia"]
mean_bp_dia = quantiles["mean_bp_dia"]
high_bp_dia = quantiles["high_bp_dia"]
low_heigth = quantiles["low_heigth"]
mean_heigth = quantiles["mean_heigth"]
high_heigth = quantiles["high_heigth"]
low_weight = quantiles["low_weight"]
mean_weight = quantiles["mean_weight"]
high_weight = quantiles["high_weight"]
#male =1, female = 0
#["age", "med_hypertension", "bp_sys", "bp_dia", "height", "weight", "sex"]
normalization_constants = js_r(normalization_dir)

context_keys = ["age", "med_hypertension", "bp_sys", "bp_dia", "height", "weight", "sex"]
len_context = len(context_keys)


conditions = []
#baselines
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
conditions.append([np.random.uniform(low_age, high_age), np.random.uniform(0,1), np.random.uniform(low_bp_sys, high_bp_sys),
                   np.random.uniform(low_bp_dia, high_bp_dia), np.random.uniform(low_heigth, high_heigth), np.random.uniform(low_weight, high_weight), np.random.uniform(0, 1)])
#conditions.append([mean_age, 0.0, mean_bp_sys, mean_bp_dia, mean_heigth, mean_weight, 0.0])#female no ht
# conditions.append([mean_age, 0.0, mean_bp_sys, mean_bp_dia, mean_heigth, mean_weight, 1.0])#male no ht
# conditions.append([mean_age, 0.0, mean_bp_sys, mean_bp_dia, mean_heigth, mean_weight, 0.0])#female no ht
# #baseline with ht 
# conditions.append([mean_age, 1.0, mean_bp_sys, mean_bp_dia, mean_heigth, mean_weight, 1.0])#male ht
# conditions.append([mean_age, 1.0, mean_bp_sys, mean_bp_dia, mean_heigth, mean_weight, 0.0])#female ht
# #extremes high
# conditions.append([high_age, 1.0, high_bp_sys, high_bp_dia, high_heigth, high_weight, 1.0]) #male ht high
# conditions.append([high_age, 1.0, high_bp_sys, high_bp_dia, high_heigth, high_weight, 0.0]) #female ht high
# #extremes low
# conditions.append([low_age, 1.0, low_bp_sys, low_bp_dia, low_heigth, low_weight, 1.0]) #male ht low
# conditions.append([low_age, 1.0, low_bp_sys, low_bp_dia, low_heigth, low_weight, 0.0]) #female ht low
# #extremes old overweight man, high bp, low weight vs opposite:
# conditions.append([high_age, 1.0, high_bp_sys, high_bp_dia, low_heigth, high_weight, 1.0])#old, ht, high bp, low heigth, high weight, male
# conditions.append([high_age, 0.0, low_bp_sys, low_bp_dia, low_heigth, high_weight, 1.0])#old, no_ht, high bp, low heigth, high weight, male
# #extremes old overweight woman, high bp, low weight vs opposite:
# conditions.append([high_age, 1.0, high_bp_sys, high_bp_dia, low_heigth, high_weight, 0.0])#old, ht, high bp, low heigth, high weight, female
# conditions.append([high_age, 0.0, low_bp_sys, low_bp_dia, low_heigth, high_weight, 0.0])#old, no_ht, high bp, low heigth, high weight, female

# names = ["bl_male_no_ht", "bl_female_no_ht", "bl_male_ht", "bl_female_ht",
#          "male_ht_high", "female_ht_high", "male_ht_low", "female_ht_low", "old_ht_highBP_lowH_highW_male",
#          "old_noht_highBP_lowH_highW_male", "old_ht_highBP_lowH_highW_female", "old_noht_highBP_lowH_highW_female"]
names = ["bl1", "bl2", "bl3", "bl4", "bl5", "bl6", "bl7", "bl8", "bl9", "bl10"]




context_unormalized = np.empty((len(conditions), len_context))
for i, l in enumerate(conditions):
    for j, con, c_key in zip(range(len(context_keys)), l, context_keys):
        c_min = normalization_constants[c_key]["min"]
        c_max = normalization_constants[c_key]["max"]
        context_unormalized[i, j] = normalized_to_unnormalized(con, c_min, c_max)
context = torch.tensor(np.array(conditions), device=device)

with torch.no_grad():
    with autocast(enabled=True):
        z,_,_ = trainset[0]

if not torch.is_tensor(z):
    z = torch.tensor(z)
scale_factor = 1 / torch.std(z)
print(f"Scaling factor set to {scale_factor}")
patient_dict = []
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
        seed = None
    )
    x = x.squeeze().detach().cpu().numpy()
    verts, faces, _, _ = marching_cubes(x, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(os.path.join(output_dir, f'{name}.stl'))
    ht_bool = "True" if context_unormalized[i, 1] > 0 else "False"
    sex = "male" if context_unormalized[i, 6] > 0 else "female"
    patient_dict.append({"file_name": f'{name}.stl',
                "age":  str(round(context_unormalized[i, 0], 2)),
                "med_hypertension": ht_bool, #True / false
                "bp_sys": str(round(context_unormalized[i, 2], 2)),
                "bp_dia": str(round(context_unormalized[i, 3], 2)),
                "height": str(round(context_unormalized[i, 4], 2)),
                "weight": str(round(context_unormalized[i, 5], 2)),
                "sex": sex
                })
    json_outdir = os.path.join(output_dir, "synth_la_2.json")
    js_w(patient_dict, json_outdir)
