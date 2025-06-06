#%%
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


import torch
from models.networks import define_G
from collections import OrderedDict
from torchvision import transforms
import itk
import datetime
import os
from markdown_pdf import MarkdownPdf,Section

from train_temp_dirty import save_numpy_array_as_itk, save_iter, SSIM_inverted, PSNR_normalized
from tqdm import tqdm
#%%

if len(sys.argv) > 1:
    if "jupyter" not in sys.argv[1]:
        folder = Path(sys.argv[1])
    else :
        folder =  Path(r"checkpoints\experiment_name")
else :
    folder =  Path(r"checkpoints\experiment_name")
    
with open(folder/"command.txt","r") as f:
    command = f.readline()
    
opt = TrainOptions().parse(command.split(" ")[2:])
opt.phase="test" 
out_dir = Path(opt.checkpoints_dir)/opt.name

l1 = nn.L1Loss()
l2 = nn.MSELoss()
ssim = SSIM_inverted()
psnr = PSNR_normalized()
metrics = [l1,l2,ssim,psnr]
# %%

generator_name = r"checkpoints\experiment_name\epoch_124_L1Loss_0.0366_net_G_A.pth"


print(f"loading {generator_name}")
model_dict = torch.load(generator_name)
new_dict = OrderedDict()
for k, v in model_dict.items():
    # load_state_dict expects keys with prefix 'module.'
    new_dict["module." + k] = v

# make sure you pass the correct parameters to the define_G method
generator_model = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, norm=opt.norm,
                                use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                                n_downsampling=opt.n_downsampling, depth=opt.depth,heads=opt.heads, dropout=opt.dropout, ngf_cytran=opt.ngf_cytran)
generator_model.load_state_dict(new_dict)
generator_model.eval()
generator_model.to("cuda")


dataset_test_path = Path(opt.dataroot)
save_volumes_dir = out_dir/"volumes"
save_volumes_dir.mkdir(parents=True,exist_ok=True)

# %%
def volume_split(volume,split):
    for i in range(int(np.ceil(volume.shape[0] / split))):
        if i*split>volume.shape[0]:
            yield volume[i*split:]
        else :
            yield volume[i*split:(i+1)*split]

# std,mean = np.std(non_contrast),np.mean(non_contrast)
trf = transforms.Compose([
    lambda x : torch.Tensor(x),
    # transforms.Normalize(mean,std),
    transforms.Resize((256,256)),
    lambda x : x.unsqueeze(1).repeat(1,opt.input_nc,1,1),
])
            
with torch.no_grad():
    for i,patient in enumerate((dataset_test_path/"A"/"test").iterdir()):
        path = dataset_test_path/"A"/"test"
        non_contrast = itk.imread(patient, itk.SS)
        contrast = itk.imread(str(patient).replace("A","B"), itk.SS)

        non_contrast = itk.GetArrayFromImage(non_contrast)
        contrast = itk.GetArrayFromImage(contrast)
        print("non_contrast no norm")
        print(non_contrast.min())
        print(non_contrast.max())
        min = non_contrast.min()
        non_contrast-=min
        max = non_contrast.max()
        non_contrast = non_contrast/max
        
        non_contrast=trf(non_contrast)
        print("non_contrast norm")
        print(non_contrast.min())
        print(non_contrast.max())
        res = []
        for vol in tqdm(volume_split(non_contrast,10)):
            res.append(generator_model(vol.to("cuda")).cpu())
        res = torch.cat(res)
        print("res oob")
        print(res.min())
        print(res.max())
        res = res*max+min

        print("res unnorm")
        print(res.min())
        print(res.max())
        save_numpy_array_as_itk(contrast,save_volumes_dir/f"{i}_test_contrast.mha")
        save_numpy_array_as_itk((non_contrast[:,0,:,:]*max+min),save_volumes_dir/f"{i}_test_non_contrast.mha")
        save_numpy_array_as_itk(res,save_volumes_dir/f"{i}_test_reconstructed.mha")

        slice = 4*(contrast.shape[0]//9)
        save_iter(res[slice,0],non_contrast[slice,0],torch.Tensor(contrast[slice]),1000,save_volumes_dir/f"test_{i}.png",title=f"Test patient example slice nb {slice}")

# %%
dataset = list(range(800))
val_dataset = list(range(200))
text = []
text.append("# Training 2D\n")
text.append("\n")
text.append(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
text.append("\n")
text.append(f"This experiment trains a {opt.model} network on each CT slice of a non contrast scan and tries to recreate the matching contrast scan.\n")
text.append("\n")
text.append(f"New : Denormalize images to export them in the expected ranges.\n")
text.append("\n")
text.append("TODO Generate volume metadata properly.\n")
text.append("TODO Export volume in it's original shape.")
text.append("\n")
text.append("\n")
text.append("## Data setup\n")
text.append(f" got {len(dataset)} images in training, {len(val_dataset)} images in validation \n")
text.append("## Results\n")
text.append("\n")
text.append('Training and validation performances are a logged with L1, L2 and SSIM metrics.\n')
text.append("\n")
for i,metric in enumerate(metrics):
    text.append("\n")
    text.append(f"![image]({out_dir/'web'/'images'/f'{type(metric).__name__}_Final_losses.png'})")
    text.append("\n")

text.append("\n")
text.append(f"![image]({out_dir/'web'/'images'/f'avg_Final_losses.png'})")
text.append("\n")

text.append("\n")
text.append("\n")
text.append("Test results\n")
for i in range(len(os.listdir(dataset_test_path/"A"/"test"))):
    text.append("\n")
    text.append(f"![image]( { save_volumes_dir/f'test_{i}.png' } )")
    text.append("\n")
    text.append("\n")
text.append("## Experiment details\n")
for arg in vars(opt):
    text.append(f"{arg} : {getattr(opt, arg)}\n")
    text.append("\n")




pdf = MarkdownPdf(toc_level=2, optimize=True)
pdf.add_section(Section(''.join(text)))
pdf.save(out_dir/f"deck_2D_{datetime.datetime.now().strftime('%m_%d_%Y_%Hh_%M')}.pdf")

# %%
