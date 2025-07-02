#%%
%load_ext autoreload
%autoreload 2
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

from train_temp_dirty import save_numpy_array_as_itk, save_iter, SSIM_inverted, PSNR_normalized, MAE,MSE,SSIM,PSNR,volume_split
from tqdm import tqdm
import nibabel as nib
#%%
generator_name = Path(r"/home/ec2-user/cycle_contrast/checkpoints/test_invivo/epoch_184_periodic_net_G_A.pth")

if len(sys.argv) > 1:
    if "jupyter" not in sys.argv[1]:
        folder = Path(sys.argv[1])
    else :
        folder =  generator_name.parent
else :
    folder =  generator_name.parent
    
with open(folder/"command.txt","r") as f:
    command = f.readline()
    
opt = TrainOptions().parse(command.split(" ")[2:])
out_dir = Path(opt.checkpoints_dir)/opt.name

dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)    # get the number of images in the dataset.
opt.phase="val" 
val_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
val_dataset_size = len(val_dataset)
opt.phase="test" 

l1 = nn.L1Loss()
l2 = nn.MSELoss()
ssim = SSIM_inverted()
psnr = PSNR_normalized()
metrics = [l1,l2,ssim,psnr]


# %%


best_epoch = int(str(generator_name).split("epoch_")[1].split("_")[0])

print(f"loading {generator_name}")
model_dict = torch.load(generator_name)
new_dict = OrderedDict()
for k, v in model_dict.items():
    # load_state_dict expects keys with prefix 'module.'
    new_dict["module." + k] = v

# make sure you pass the correct parameters to the define_G method
generator_model = define_G(opt)#input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, norm=opt.norm,
                                #use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                                #n_downsampling=opt.n_downsampling, depth=opt.depth,heads=opt.heads, dropout=opt.dropout, ngf_cytran=opt.ngf_cytran)
generator_model.load_state_dict(new_dict)
generator_model.eval()
generator_model.to("cuda")


dataset_test_path = Path(opt.dataroot)
save_volumes_dir = out_dir/"volumes"
save_volumes_dir.mkdir(parents=True,exist_ok=True)

test_metrics = [MAE(), MSE(), SSIM(), PSNR()]
test_metrics_values = []

path_to_test = None
if opt.dataset_mode == "invivo":
    path_to_test = dataset_test_path/"test"/"ctcs"
    trf = transforms.Compose([
        lambda x : torch.Tensor(x),
        lambda x : x.permute(2,0,1),
        lambda x : transforms.Resize((opt.load_size,opt.load_size))(x),
        lambda x : x.unsqueeze(1).repeat(1,opt.input_nc,1,1),
    ])
else :
    path_to_test = dataset_test_path/"A"/"test"
    trf = transforms.Compose([
        lambda x : torch.Tensor(x),
        lambda x : x.unsqueeze(1).repeat(1,opt.input_nc,1,1),
    ])
            
with torch.no_grad():
    for i,patient in enumerate((path_to_test).iterdir()):
        print(patient)
        if opt.dataset_mode == "invivo":
            non_contrast = nib.load(patient).get_fdata()
            contrast = nib.load(str(patient).replace("ctcs","ccta")).get_fdata()
        else :
            non_contrast = itk.imread(patient, itk.SS)
            contrast = itk.imread(str(patient).replace("A","B"), itk.SS)                
            non_contrast = itk.GetArrayFromImage(non_contrast)
            contrast = itk.GetArrayFromImage(contrast)

        if opt.dataset_mode == "invivo":
            non_contrast,contrast = dataset.dataset.preprocess(non_contrast,contrast)
        else :
            non_contrast = dataset.dataset.normalize(non_contrast)
            contrast = dataset.dataset.normalize(contrast)
        print("non_contrast",non_contrast.shape)
        
        non_contrast=trf(non_contrast)
        print("non_contrast",non_contrast.shape)

        res = []
        
        for vol in tqdm(volume_split(non_contrast,10)):
            res.append(generator_model(vol.to("cuda")).cpu())
        res = torch.cat(res).clamp(0,1).to("cuda")
        contrast=trf(contrast).to('cuda')
        
        print("non_contrast",non_contrast.shape)
        print("res",res.shape)
        
        test_metrics_values.append([metric(res,contrast).item() for metric in test_metrics])
        
        
        slice = 4*(contrast.shape[0]//9)
        save_iter(res[slice,0].cpu(),non_contrast[slice,0],contrast[slice,0].cpu(),best_epoch,save_volumes_dir/f"test_{i}.png",title=f"Test patient example slice nb {slice}")

        non_contrast = dataset.dataset.unnormalize(contrast[:,0,:,:].squeeze().cpu().numpy())
        save_numpy_array_as_itk(non_contrast,save_volumes_dir/f"{i}_test_non_contrast.mha")
        res = dataset.dataset.unnormalize(res[:,0,:,:].squeeze().cpu().numpy())
        save_numpy_array_as_itk(res,save_volumes_dir/f"{i}_test_reconstructed.mha")
        contrast = dataset.dataset.unnormalize(contrast[:,0,:,:].squeeze().cpu().numpy())
        save_numpy_array_as_itk(contrast,save_volumes_dir/f"{i}_test_contrast.mha")

test_metrics_avg = np.mean(test_metrics_values,axis=0)

#%%

text = []
text.append("# Training 2D\n")
text.append("\n")
text.append(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
text.append("\n")
text.append(f"This experiment trains a {opt.model} network on each CT slice of a non contrast scan and tries to recreate the matching contrast scan.\n")
text.append(f"This experiment uses a {opt.netG} model as generator\n")
text.append(f"Scans Hounsfield units are clipped between {opt.min_clip} and {opt.max_clip} during preprocessing\n")
text.append(f"This experiment uses {opt.dataset_mode} data with a batch size of {opt.batch_size} images for {opt.n_epochs+opt.n_epochs_decay} epochs with an initial lr of {opt.lr} \n")
text.append("\n")
text.append("TODO : Generate volume metadata properly when exporting reconstructed volume.\n")
text.append("\n")
text.append("\n")
text.append("## Data setup\n")
text.append(f" got {len(dataset)} images in training, {len(val_dataset)} images in validation \n")
try :
    with open(os.path.join(opt.dataroot,"split.txt"),'r') as f:
        split = f.readlines()
    text.append(f"Patient split is as follow :\n")
    text.append("".join(split).replace(",", "as").replace("\n",", "))
    text.append("\n")
except:
    pass
text.append("## Training metrics\n")
text.append("\n")
text.append('Training and validation performances are a logged with L1, L2, SSIM and PSNR metrics.\n')
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
text.append("## Test results\n")
text.append("\n")

text.append("### Metrics\n")

text.append("\n")
text.append("Average metric over all test patients table")
text.append("\n")

text.append("| ")
text.append(" | ".join([type(metric).__name__ for metric in test_metrics]))
text.append(" |\n")

text.append("| ")
text.append(" | ".join([":---:" for _ in range(len(test_metrics))]))
text.append(" |\n")

text.append("| ")
text.append(" | ".join([str(np.around(metric,5)) for metric in test_metrics_avg]))
text.append(" |\n")
text.append("\n")
text.append("\n")
    
text.append("### Qualitative results\n")

for i in range(len(os.listdir(path_to_test))):
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
