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

train_loader = DataLoader(train_set, batch_size=128, shuffle=True) #data loader loads train_loader with 128 image batch sets from train_set and shuffles it at every epoch
data = iter(train_loader)  #iter, makes train_loader a iterator
data_it, targets_it = next(data) #next(data) receives one batch from train_loader, returning two tensors, data_it and targets_it 

#data_it is a tensor in the shape, [128, C, H, W], so it should be [128,1,28,28]

test_img = data_it[0]
test_img = test_img.squeeze(0)
test_img_spike = spikegen.latency(test_img, num_steps = 255, normalize=True, linear=True)

test_img = test_img.numpy()
image_array_int = (test_img * 255).astype(int)

# Print row-by-row
for row in image_array_int:
    print(" ".join(f"{pixel:3}" for pixel in row))  # Align numbers neatly

plt.imshow(test_img, cmap="gray")  # Convert to NumPy and plot
plt.title(f"Label: {targets_it[0].item()}")  # Display the corresponding label
plt.axis("off")  # Hide axes
plt.show()



# Maybe use channels to see if you can incorporate both the DoG off and DoG on images, so [128,2,28,28].
# You now know the size and that the images are being convolved correctly 

# data_it is a tensor

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

# # The function in SNNTorch convert_to_time(data, tau, threshold) allows us to convert a feature of intensity X_ij [0,1] into a latency coded response 

# a = torch.Tensor([0.02, 0.5, 0.5])

# print(a)

# a_data = spikegen.latency(a, num_steps = 5, normalize=True, linear=True)

# print(a_data)

# print(data_it)
# print(data_it.dim())
# print(data_it.size())