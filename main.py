import torch
import numpy as np
from snntorch import spikegen
from model import *
from stdp_time import stdp_time
import matplotlib.pyplot as plt

model = Net(num_input=28*28, num_output=10, beta=0.9, threshold=20, reset_mechanism="zero")

img, label = torch.load("test_sample.pt", weights_only="True")
img_2, label_2 = torch.load("test_sample_two.pt", weights_only="True")

img = torch.flatten(img, start_dim=1)
img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)

img_2 = torch.flatten(img_2, start_dim=1)
img_2 = spikegen.latency(data=img_2, num_steps=255, normalize=True, linear=True)

A_plus = 5e-3
A_minus = 3.75e-3
tau = 200
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

data = []  

for iter in range(200):  
    if iter % 2 == 0:  
        data.append(img)  
    else:  
        data.append(img_2)  

# for img_num, val in enumerate(data):
    
data_list = torch.load("data\processed\convOFF_train.pt", weights_only=True)

for img_num, data in enumerate(data_list):
    val, label = data

    val = torch.flatten(val, start_dim=1)
    val = spikegen.latency(data=val, num_steps=255, normalize=True, linear=True)

    spk_rec, mem_rec = model(val)



    out_neuron, delta_w = stdp_time(weight_matrix=model.fc1.weight, in_spike=val.squeeze(1), out_spike=spk_rec.squeeze(1), params=params)

    with torch.no_grad():
        model.fc1.weight[out_neuron] += delta_w
        model.fc1.weight[out_neuron].clamp_(0.0, 1.0)

    print(f"Min Δw: {model.fc1.weight.min().item()}, Max Δw: {model.fc1.weight.max().item()}")

    print(f"Image: {img_num}")

    if img_num == 200:
        break 

weights = model.fc1.weight.data.cpu().numpy()

num_output_neurons = weights.shape[0]

rows = int(np.ceil(num_output_neurons / 10))  
cols = min(10, num_output_neurons)  

fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))

for i, ax in enumerate(axes.flat):
    if i < num_output_neurons:
        img = weights[i].reshape(28, 28)
        ax.imshow(img, cmap="gray") 
        ax.set_title(f"Neuron {i}")
        ax.axis("off") 
    else:
        ax.axis("off") 

# Title for the entire plot
plt.suptitle("Receptive Fields of Output Neurons")
plt.tight_layout()
plt.show()
