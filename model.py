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
  
transform = transforms.Compose([
  transforms.Resize((128,128)),
  transforms.ToTensor(),
])
  

dataset = HotDogDataset(path, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class HotDogClassifier(nn.Module):
  def __init__(self, num_classes=2):
    super(HotDogClassifier, self).__init__()
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    self.features= nn.Sequential(*list(self.base_model.children())[:-1])
    enet_out_size = 1280
    self.classifier = nn.Linear(enet_out_size, num_classes)
  
  def forward(self, x):
    x = self.features(x)
    output = self.classifier(x)

    return output
 