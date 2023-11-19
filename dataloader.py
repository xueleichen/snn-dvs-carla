import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

class DVSDataset(Dataset):
    def __init__(self, data_txt):

        self.data_txt = data_txt

        with open(self.data_txt,"r") as f:
            self.data_list = f.readlines()

        self.transform = transforms.Compose([
                    transforms.RandomApply(
                    [transforms.RandomCrop(size=(270, 240)), transforms.Resize(size=(288, 256),antialias=True)],p=0.3),
                    ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the input and label

        input shape [T, C, H, W]
        """
        data = self.data_list[idx].split(" ")
        frame1 = np.array(Image.open(data[0]))
        frame2 = np.array(Image.open(data[1]))
        label = int(data[2])
        f1_polarized = -frame1[:,-256:,0]/255 + frame1[:,-256:,2]/255
        f2_polarized = -frame2[:,-256:,0]/255 + frame2[:,-256:,2]/255

        input = torch.tensor(np.vstack((f1_polarized[np.newaxis,np.newaxis,:,:],f2_polarized[np.newaxis,np.newaxis,:,:])),dtype=torch.float32)
        if self.data_txt == "./train.txt":
            input = self.transform(input)
        label = torch.tensor(label,dtype=torch.float32)
        return input, label


def get_DVSDataloader(data_txt,batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                DVSDataset(data_txt=data_txt),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )

class RGBDataset(Dataset):
    def __init__(self, data_txt):

        self.data_txt = data_txt

        with open(self.data_txt,"r") as f:
            self.data_list = f.readlines()

        self.transform = transforms.Compose([
                    transforms.RandomApply(
                    [transforms.RandomCrop(size=(270, 240)), transforms.Resize(size=(288, 256),antialias=True)],p=0.3),
                    ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the input and label

        input shape [T, C, H, W]
        """
        data = self.data_list[idx].split(" ")
        frame1 = np.array(Image.open(data[0]))
        frame2 = np.array(Image.open(data[1]))
        label = int(data[2])


        input = torch.tensor(np.concatenate((frame1[:,-256:,:],frame2[:,-256:,:]), axis=2),dtype=torch.float32).permute(2,0,1)

        if self.data_txt == "./train.txt":
            input = self.transform(input)
        label = torch.tensor(label,dtype=torch.float32)
        return input, label


def get_RGBDataloader(data_txt,batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                RGBDataset(data_txt=data_txt),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )

class ImageDataset(Dataset):
    def __init__(self, data_txt):

        self.data_txt = data_txt

        with open(self.data_txt,"r") as f:
            self.data_list = f.readlines()

        self.transform = transforms.Compose([
                    transforms.RandomApply(
                    [transforms.RandomCrop(size=(270, 240)), transforms.Resize(size=(288, 256),antialias=True)],p=0.3),
                    ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the input and label

        """
        data = self.data_list[idx].split(" ")
        frame1 = np.array(Image.open(data[0]))
        frame2 = np.array(Image.open(data[1]))
        label = int(data[2])


        input = torch.tensor(np.concatenate((frame1[:,-256:,:],frame2[:,-256:,:]), axis=2),dtype=torch.float32).permute(2,0,1)

        if self.data_txt == "./train.txt":
            input = self.transform(input)
        label = torch.tensor(label,dtype=torch.float32)
        return input, label


def get_ImageDataloader(data_txt,batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                ImageDataset(data_txt=data_txt),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )