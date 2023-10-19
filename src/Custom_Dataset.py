''' Custom_Dataset.py
This code file is used to define the custom dataset class that is used in our training.

  CS7180 Advanced Perception     10/13/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import os
import pandas as pd
import shutil
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda')

class Custom_Dataset(Dataset):
    def __init__(self, csv_file,data_percentage, transform=None):
        self.data = pd.read_csv(csv_file, nrows=int(len(pd.read_csv(csv_file)) * data_percentage))
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx, 0]).convert('RGB')
        target_image = Image.open(self.data.iloc[idx, 1]).convert('RGB')

        if self.transform:
            image = self.transform(image)
            target_image = self.transform(target_image)

        image = image.to(device = device)
        target_image = target_image.to(device = device)

        return image, target_image