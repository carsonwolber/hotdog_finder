import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import pandas as pd
import numpy as np


path = kagglehub.dataset_download("thedatasith/hotdog-nothotdog")

class HotDogDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data = ImageFolder(data_dir, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]
  
  @property
  def classes(self):
    return self.data.classes
  

dataset = HotDogDataset(path)
print(len(dataset))

