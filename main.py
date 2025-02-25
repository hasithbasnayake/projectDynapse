# .\env\Scripts\activate to activate virtual environment
# pip freeze > requirements.txt You can export a list of all installed packages
# pip install -r requirements.txt to install from a requirements file
# pip list to list all packages
 
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import snntorch.spikeplot as splt
import numpy as np
import matplotlib.pyplot as plt
from func import *


dim = np.array([6, 6])  # kernel dimensions (in pixels)
ppa = np.array([8, 8])  # pixels per arc (scaling factor)
ang = np.ceil(dim / ppa)  # angular size based on pixels per arc
ctr = (1/3) * dim[0]  # center size as a fraction of kernel size
sur = (2/3) * dim[0]  # surround size as a fraction of kernel size

split_params = (1000,200,2021)
kernel_params = (dim, ppa, ang, ctr, sur)

train_set, test_set = data_preprocessing('FashionMNIST','data', split_params, kernel_params)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# print(type(train_loader))
data = iter(train_loader) 
# print(type(data))
data_it, targets_it = next(data)  
# print(type(data_it))
# print(type(targets_it))


spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

