import torch
import numpy as np
import cv2 

import OpenEXR as exr
import Imath

import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from natsort import natsorted

from abc import abstractmethod



class DriveSimDataset(Dataset):
    def __init__(self, dataset_root, dataset, split, input_dir, label_dir, transform=None, target_transform=None, max_file=-1):
        self.drivesim_dirs = os.path.join(dataset_root, dataset,split)
        self.dataset = dataset # "Pattern" or "HB"
        if self.dataset not in ["Pattern", "HB"]:
            raise ValueError("dataset must be 'Pattern' or 'HB'")

        self.dataset_root = dataset_root
        self.split = split
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.max_file = max_file

        self.inputs_file = natsorted(np.array(list(self.load_filesnames(self.input_dir, max_file=max_file))))
        self.labels_file = natsorted(np.array(list(self.load_filesnames(self.label_dir, max_file=max_file))))
            
    def load_filesnames(self,subdir,max_file=-1):
        files = []
        dirs = [os.path.join(self.drivesim_dirs, d) for d in os.listdir(self.drivesim_dirs) if os.path.isdir(os.path.join(self.drivesim_dirs, d))]
        for d in dirs:
            direct = os.path.join(d,subdir)
            files_to_add = self.get_files_list(direct)
            files_to_add = natsorted(files_to_add)
            if(max_file != -1 and len(files_to_add)> max_file):
                files_to_add = files_to_add[:max_file]
            files += files_to_add
        return files

    def get_files_list(self,dir):
        return [os.path.join(dir,file) for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]

    def __len__(self):
        return len(self.inputs_file)

    @abstractmethod
    def __getitem__(self, idx):
        return NotImplementedError # Implement in subclass

    

class DriveSimDepthDataset(DriveSimDataset):
    def __init__(self, dataset_root, dataset, split, transform=None, target_transform=None, max_depth=200, max_file=-1, max_depth_placeholder=0):
        self.max_depth = max_depth
        self.max_depth_placeholder = max_depth_placeholder
        
        input_dir = "ldr_color"
        label_dir = "distance_to_image_plane"
        super(DriveSimDepthDataset, self).__init__(dataset_root, dataset, split, input_dir, label_dir, transform, target_transform, max_file)


    def read_exr(self, path):
        exrfile = exr.InputFile(path)
        raw_bytes = exrfile.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
        depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
        height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
        width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
        depth_map = np.reshape(depth_vector, (height, width)).copy()
        depth_map[depth_map > self.max_depth] = self.max_depth_placeholder
        return depth_map

    
    def __getitem__(self, idx):
        img_path = self.inputs_file[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.read_exr(self.labels_file[idx])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label