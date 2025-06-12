import os
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from data.base_dataset import BaseDataset
from pathlib import Path
from skimage.transform import resize

class InvivoDataset(BaseDataset):
    def __init__(self, opt):
        self.ctcs_dir = Path(opt.dataroot)/opt.phase/"ctcs"
        self.ccta_dir = Path(opt.dataroot)/opt.phase/"ccta"
        self.ctcs_slices = []
        self.ccta_slices = []

        for file in os.listdir(self.ctcs_dir):
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                ctcs_img = nib.load(os.path.join(self.ctcs_dir, file)).get_fdata()
                ctcs_img = np.clip(ctcs_img, -100, 300)
                ctcs_img = (ctcs_img + 100) / 400  # Normalize to [0,1]
                if opt.load_size :
                    ctcs_img = resize(ctcs_img,(opt.load_size,opt.load_size))

                ccta_img = nib.load(os.path.join(self.ccta_dir, file)).get_fdata()
                ccta_img = np.clip(ccta_img, -100, 300)
                ccta_img = (ccta_img + 100) / 400  # Normalize to [0,1]
                if opt.load_size :
                    ccta_img = resize(ccta_img,(opt.load_size,opt.load_size))
                
                for i in range(ctcs_img.shape[2]-2):
                    self.ctcs_slices.append(ctcs_img[None, :, :, i])
                    self.ccta_slices.append(ccta_img[None, :, :, i])
    

    def __len__(self):
        return len(self.ctcs_slices)

    def __getitem__(self, idx):
        ctcs_slice_img = self.ctcs_slices[idx]
        ccta_slice_img = self.ccta_slices[idx]
        return {'A':ctcs_slice_img.astype(np.float32) , 'B':self.ccta_slices[idx].astype(np.float32)}
