# This code is released under the CC BY-SA 4.0 license.

import pickle

import numpy as np
import itk
from pathlib import Path

from data.base_dataset import BaseDataset
from tqdm import tqdm

from skimage.transform import resize

class ExVivoDataset(BaseDataset):
        
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.size = opt.load_size
        self.folder = Path(opt.dataroot)
        self.min_clip = opt.min_clip
        self.max_clip = opt.max_clip
        self.A = self.folder/"A"/opt.phase
        self.B = self.folder/"B"/opt.phase
        self.data={"A":[],"B":[]}
        for file in tqdm(self.A.iterdir(),desc="preloading dataset"):
            self.load_data(file)
    
    def clip(self,image):
        return np.clip(image, self.min_clip, self.max_clip)
    
    def normalize(self,image):
        return (self.clip(image) - self.min_clip) / (self.max_clip-self.min_clip)  # Normalize to [0,1]

    def unnormalize(self,image):
        return (image*(self.max_clip-self.min_clip) + self.min_clip)
        
    def load_data(self,file):
        a = itk.GetArrayFromImage(itk.imread(file,itk.SS))
        b = itk.GetArrayFromImage(itk.imread(self.B/file.name,itk.SS))
        
        a = self.normalize(a)
        b = self.normalize(b)

        if self.size is not None:
            a = resize(a,(a.shape[0],self.size,self.size))
            b = resize(b,(b.shape[0],self.size,self.size))
        
        for i in range(a.shape[0]):
            self.data["A"].append(a[None,i,:,:])
            self.data["B"].append(b[None,i,:,:])
        



    def __getitem__(self, index):
        return {'A': self.data["A"][index], 'B': self.data["B"][index]}

    def __len__(self):
        return len(self.data["A"])
