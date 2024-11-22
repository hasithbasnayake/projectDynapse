# .\env\Scripts\activate to activate virtual environment
# pip freeze > requirements.txt You can export a list of all installed packages
# pip install -r requirements.txt to install from a requirements file
# pip list to list all packages

import torch 
from torchvision import datasets
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from func import *

dim = np.array([6, 6])  # kernel dimensions (in pixels)
ppa = np.array([8, 8])  # pixels per arc (scaling factor)
ang = np.ceil(dim / ppa)  # angular size based on pixels per arc
ctr = (1/3) * dim[0]  # center size as a fraction of kernel size
sur = (2/3) * dim[0]  # surround size as a fraction of kernel size

kernelON = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)
kernelOFF = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)

kernelON.setFilterCoefficients(ONOFF="ON")
kernelOFF.setFilterCoefficients(ONOFF="OFF")

FashionMNISTTrain = datasets.FashionMNIST(root='./data/raw', train=True, download=True, transform=None)
FashionMNISTTest = datasets.FashionMNIST(root='./data/raw', train=False, download=True, transform=None)

num_train_samples = 1000
num_test_samples = 200
random_seed = 40

training_set, testing_set = createTrainingTestingSets(training_images = FashionMNISTTrain, testing_images = FashionMNISTTest, num_train_samples = num_train_samples, num_test_samples = num_test_samples, random_seed = random_seed)

# For all images, min, mean, max, std dev, classification graph, pixel distributions [DONE]
# For training_set, min, mean, max, std dev, classification graph, pixel distributions [DONE]
# For testing_set, min, mean, max, std dev, classification graph, pixel distributions [DONE]
# For training_set, each label, min, mean, max, std dev, classification graph, pixel distributions
# For testing_set, each label, min, mean, max, std dev, classification graph, pixel distributions

full_set_analysis = dataAnalysis(training_set + testing_set)
training_set_analysis = dataAnalysis(training_set)
testing_set_analysis = dataAnalysis(testing_set)

plt.figure(figsize=(15,7))
plt.title("All Images")
plt.bar(full_set_analysis["label_plot"][0], full_set_analysis["label_plot"][1])
plt.show()

plt.figure(figsize=(15,7))
plt.title("Training Images")
plt.bar(training_set_analysis["label_plot"][0], training_set_analysis["label_plot"][1])
plt.show()
 
plt.figure(figsize=(15,7))
plt.title("Testing Images")
plt.bar(testing_set_analysis["label_plot"][0], testing_set_analysis["label_plot"][1])
plt.show()

 