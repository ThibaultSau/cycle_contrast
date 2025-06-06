# This code is released under the CC BY-SA 4.0 license.

import pickle

import numpy as np
import itk
from pathlib import Path

from data.base_dataset import BaseDataset
from tqdm import tqdm

from skimage.transform import resize

class CTDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.size = opt.load_size
        self.folder = Path(opt.dataroot)
        self.A = self.folder/"A"/opt.phase
        self.B = self.folder/"B"/opt.phase
        self.data={"A":[],"B":[]}
        for file in tqdm(self.A.iterdir(),desc="preloading dataset"):
            self.load_data(file)
        
    def load_data(self,file):
        a = itk.GetArrayFromImage(itk.imread(file,itk.SS))
        b = itk.GetArrayFromImage(itk.imread(self.B/file.name,itk.SS))
        
        if self.size is not None:
            a = resize(a,(a.shape[0],self.size,self.size))
            b = resize(b,(b.shape[0],self.size,self.size))
        a-=a.min()
        a=np.expand_dims(a/a.max(), 1)
        
        b-=b.min()
        b=np.expand_dims(b/b.max(), 1)
        
        for i in range(a.shape[0]):
            self.data["A"].append(a[i])
            self.data["B"].append(b[i])
        



    def __getitem__(self, index):
        return {'A': self.data["A"][index], 'B': self.data["B"][index]}

    def __len__(self):
        return len(self.data["A"])
