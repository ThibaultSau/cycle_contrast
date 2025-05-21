#%%

# %load_ext autoreload
# %autoreload 2
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import shutil
import sys
from models.networks import define_G
from collections import OrderedDict
from torchvision import transforms
import itk
import datetime
import os
from pathlib import Path
import argparse

import SimpleITK as sitk

#%%

def save_numpy_array_as_itk(array,path):
    itk_image = itk.GetImageFromArray(array)
    itk.imwrite(itk_image,path)



file = "/home/ubuntu/pytorch-CycleGAN-and-pix2pix/checkpoints/cycleganexvivo_metrics/commmand.txt"
args = None
with open(file,"r") as f:
    args=f.readline()

opt = TrainOptions()
opt = opt.gather_options(args = args.split(' ')[2:])
out_dir = Path(opt.checkpoints_dir)/opt.name

#%%

def save_numpy_array_as_DICOM(array,path):
    # Convert the NumPy array to an ITK image
    itk_image = itk.GetImageFromArray(array)

    # Cast the image to the correct type if necessary
    cast_filter = itk.CastImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    cast_filter.SetInput(itk_image)
    cast_filter.Update()
    casted_image = cast_filter.GetOutput()

    # Set up the DICOM Image IO
    dicom_io = itk.GDCMImageIO.New()

    # Set up the Image Series Writer
    series_writer = itk.ImageSeriesWriter[itk.Image[itk.F, 3], itk.Image[itk.F, 2]].New()

    # Set the Image IO to the writer
    series_writer.SetImageIO(dicom_io)

    # Set the input image
    series_writer.SetInput(casted_image)
    # Set the file names to the writer
    # Generate file names for each slice using itk.StringArray
    file_names = []
    for i in range(casted_image.GetLargestPossibleRegion().GetSize()[2]):
        file_names.append(str(path/ f"slice_{i:03d}.dcm"))
    series_writer.SetFileNames(file_names)

    path.mkdir(parents=True,exist_ok=True)

    # Update the writer to write the DICOM series
    series_writer.Update()

def save_numpy_array_as_DICOM_sitk(array, output_directory):
    # Convert the NumPy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(array)

    # Cast the image to signed short (appropriate for CT images)
    casted_image = sitk.Cast(sitk_image, sitk.sitkInt16)

    # Create a directory to save the DICOM files
    os.makedirs(output_directory, exist_ok=True)

    # Use the ImageSeriesWriter to write the image as a DICOM series
    writer = sitk.ImageSeriesWriter()

    # Set the file names for each slice
    file_names = [os.path.join(output_directory, f"slice_{i:03d}.dcm") for i in range(casted_image.GetDepth())]

    # Set the file names to the writer
    writer.SetFileNames(file_names)

    # Execute the writer
    writer.Execute(casted_image)

    print(f"DICOM series saved to {output_directory}")


model = 'epoch_321_avg_metric_0.2175_net_G_A.pth'

print(f"loading {out_dir/model}")
model_dict = torch.load(out_dir/model)
new_dict = OrderedDict()
for k, v in model_dict.items():
    # load_state_dict expects keys with prefix 'module.'
    new_dict["module." + k] = v

# make sure you pass the correct parameters to the define_G method
generator_model = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, ['cuda'])
generator_model.load_state_dict(new_dict)
generator_model.eval()
generator_model.to("cuda")


dataset_test_path = Path(opt.dataroot)
save_volumes_dir = out_dir/"volumesTest"
save_volumes_dir.mkdir(parents=True,exist_ok=True)
save_volumes_dir_sitk= out_dir/"volumesTest_sitk"
save_volumes_dir_sitk.mkdir(parents=True,exist_ok=True)
with torch.no_grad():
    for i,patient in enumerate((dataset_test_path/"test_volumesA").iterdir()):
        non_contrast = itk.imread(patient, itk.F)
        contrast = itk.imread(str(patient).replace("test_volumesA","test_volumesB"), itk.F)
        
        non_contrast = itk.GetArrayFromImage(non_contrast)
        std,mean = np.std(non_contrast),np.mean(non_contrast)
        trf = transforms.Compose([
            lambda x : torch.Tensor(x),
            transforms.Normalize(mean,std),
            transforms.Resize((256,256)),
            lambda x : x.unsqueeze(1).repeat(1,3,1,1),
        ])

        contrast = itk.GetArrayFromImage(contrast)

        non_contrast=trf(non_contrast)
        contrast=trf(contrast)
        res = generator_model(non_contrast.to("cuda"))
        save_numpy_array_as_DICOM((non_contrast[:,1,:,:].squeeze().detach().cpu().numpy()*std)+mean,save_volumes_dir/f"{i}_test_non_contrast")
        save_numpy_array_as_DICOM((res[:,1,:,:].squeeze().detach().cpu()*std)+mean,save_volumes_dir/f"{i}_test_reconstructed")
        save_numpy_array_as_DICOM((contrast[:,1,:,:].squeeze()*std)+mean,save_volumes_dir/f"{i}_test_contrast")



        save_numpy_array_as_DICOM_sitk((non_contrast[:,1,:,:].squeeze().detach().cpu().numpy()*std)+mean,save_volumes_dir_sitk/f"{i}_test_non_contrast")
        save_numpy_array_as_DICOM_sitk((res[:,1,:,:].squeeze().detach().cpu()*std)+mean,save_volumes_dir_sitk/f"{i}_test_reconstructed")
        save_numpy_array_as_DICOM_sitk((contrast[:,1,:,:].squeeze()*std)+mean,save_volumes_dir_sitk/f"{i}_test_contrast")

        # slice = 4*(contrast.shape[0]//9)
        # save_iter(res[slice,0],non_contrast[slice,0],torch.Tensor(contrast[slice,0]),best_epoch,save_volumes_dir/f"test_{i}.png",title=f"Test patient example slice nb {slice}")



# %%
