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

data_preprocessing('FashionMNIST','data', split_params, kernel_params)

# kernelON = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)
# kernelOFF = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)

# kernelON.setFilterCoefficients(ONOFF="ON")
# kernelOFF.setFilterCoefficients(ONOFF="OFF")

FashionMNISTTrain = datasets.FashionMNIST(root='./data/raw', train=True, download=True, transform=None)
FashionMNISTTest = datasets.FashionMNIST(root='./data/raw', train=False, download=True, transform=None)

# num_train_samples = 1000
# num_test_samples = 200
# rng = np.random.default_rng(2021)

# training_set, testing_set = createTrainingTestingSets(training_images = FashionMNISTTrain, testing_images = FashionMNISTTest, num_train_samples = num_train_samples, num_test_samples = num_test_samples, rng=rng)

# full_set_analysis = dataAnalysis(training_set + testing_set)
# training_set_analysis = dataAnalysis(training_set)
# testing_set_analysis = dataAnalysis(testing_set)

# # plt.title("Unaltered Sneaker")
# # plt.imshow(training_set[0][0], cmap="grey")
# # # plt.savefig("UnalteredSneaker.png")

# ON_training_set = genLGNActivityMaps(training_set, kernelON.kernel, False)
# OFF_training_set = genLGNActivityMaps(training_set, kernelON.kernel, False)

# # plt.title("Activity Map Sneaker")
# # plt.imshow(ON_training_set[0][0], cmap="grey")
# # # plt.savefig("ActivityMapSneaker.png")

# tensor_dataset = convertToTensor(ON_training_set)

# train_loader = DataLoader(tensor_dataset, batch_size=100)

# data = iter(train_loader) 
# data_it, targets_it = next(data)  

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)
# # print(f"Spike data shape: {spike_data.shape}")

# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spike_data[:, 0].view(100, -1), ax, s=25, c="black")

# plt.title(f"Spike Train of Sneaker")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# # plt.savefig("SpikeTrainOfSneaker.png")
# plt.show()

# spike_data_sample = spike_data[:, 0]
# fig, ax = plt.subplots()
# anim = splt.animator(spike_data_sample, fig, ax, 20, 500)

# # anim.save("SneakerSpikes.gif")

# # Min, max, and mean has been double checked with matlab values and are essentially the same. 
# # genLGNActivityMaps has been double checked to make sure it's convolving images, and keeping labels the same

# # Write latency encoding code using snntorch

# # So there's two options, we can either do the image augmentation through keeping everything a tensor, create a 
# # DoG kernel in the form of a tensor that we can use pytorch.functional.nn conv2d function on 
# # Then compare those final results to the tensor you produced before and checked against the matlab functions 

# # Need to create leakey neurons and set up my own STDP learning rules explained in the paper

# num_inputs = 28*28
# num_hidden = 1000
# num_outputs = 10

# num_steps = 100
# beta = 0.95

# batch_size = 100
# epochs = 5

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # Initialize layers
#         self.fc1 = nn.Linear(num_inputs, num_hidden)
#         self.lif1 = snn.Leaky(beta=beta)
#         self.fc2 = nn.Linear(num_hidden, num_outputs)
#         self.lif2 = snn.Leaky(beta=beta)

#     def forward(self, x):

#         # Initialize hidden states at t=0
#         mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()

#         # Record the final layer
#         spk2_rec = []
#         mem2_rec = []

#         for step in range(num_steps):
#             cur1 = self.fc1(x)
#             spk1, mem1 = self.lif1(cur1, mem1)
#             cur2 = self.fc2(spk1)
#             spk2, mem2 = self.lif2(cur2, mem2)
#             spk2_rec.append(spk2)
#             mem2_rec.append(mem2)

#         return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# loss = nn.CrossEntropyLoss()
# net = NeuralNetwork()
# optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# data, targets = next(iter(train_loader))

# spk_rec, mem_rec = net(data.view(batch_size, -1))

# print(mem_rec.size())

# num_epochs = 10
# loss_hist = []
# test_loss_hist = []
# counter = 0

# dtype = torch.float

# test_loader = DataLoader(convertToTensor(OFF_training_set), batch_size=batch_size, shuffle=True, drop_last=True)

# def print_batch_accuracy(data, targets, train=False):
#     output, _ = net(data.view(batch_size, -1))
#     _, idx = output.sum(dim=0).max(1)
#     acc = np.mean((targets == idx).detach().cpu().numpy())

#     if train:
#         print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
#     else:
#         print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

# def train_printer():
#     print(f"Epoch {epoch}, Iteration {iter_counter}")
#     print(f"Train Set Loss: {loss_hist[counter]:.2f}")
#     print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
#     print_batch_accuracy(data, targets, train=True)
#     print_batch_accuracy(test_data, test_targets, train=False)
#     print("\n")


# # Outer training loop
# for epoch in range(num_epochs):
#     iter_counter = 0
#     train_batch = iter(train_loader)

#     # Minibatch training loop
#     for data, targets in train_batch:

#         # forward pass
#         net.train()
#         spk_rec, mem_rec = net(data.view(batch_size, -1))

#         # initialize the loss & sum over time
#         loss_val = torch.zeros((1), dtype=dtype)
#         for step in range(num_steps):
#             loss_val += loss(mem_rec[step], targets)

#         # Gradient calculation + weight update
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()

#         # Store loss history for future plotting
#         loss_hist.append(loss_val.item())

#         # Test set
#         with torch.no_grad():
#             net.eval()
#             test_data, test_targets = next(iter(test_loader))

#             # Test set forward pass
#             test_spk, test_mem = net(test_data.view(batch_size, -1))

#             # Test set loss
#             test_loss = torch.zeros((1), dtype=dtype)
#             for step in range(num_steps):
#                 test_loss += loss(test_mem[step], test_targets)
#             test_loss_hist.append(test_loss.item())

#             # Print train/test loss/accuracy
#             if counter % 50 == 0:
#                 train_printer()
#             counter += 1
#             iter_counter +=1

# # Plot Loss
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# plt.plot(loss_hist)
# plt.plot(test_loss_hist)
# plt.title("Loss Curves")
# plt.legend(["Train Loss", "Test Loss"])
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()





# train_loader = DataLoader(convON_train, batch_size = 128, shuffle=True)
# data = iter(train_loader)
# data_it, targets_it = next(data) #[batch_size = 128, C = 1, H = 28, H = 28]
# # Neuron Model

# l1 = snn.Leaky(beta=0.8, threshold=1, reset_mechanism="zero")

# num_steps = 255
# l1_w = 0.31
# l1_cur_in = convr(data_it, num_steps) * l1_w
# l1_mem = torch.zeros(1)
# l1_spk = torch.zeros(1)
# l1_mem_rec = []
# l1_spk_rec = []

# for step in range(num_steps):
#     l1_spk, l1_mem = l1(l1_cur_in[step], l1_mem)
#     l1_mem_rec.append(l1_mem)
#     l1_spk_rec.append(l1_spk)

# l1_mem_rec = torch.stack(l1_mem_rec)
# l1_spk_rec = torch.stack(l1_spk_rec)

# # Network Model

# num_inputs = 784
# num_outputs = 1

# fc1 = nn.Linear(num_inputs, num_outputs, bias=False)
# lif1 = snn.Leaky(beta=0.8, threshold=1, reset_mechanism="zero")

# with torch.no_grad():
#     fc1.weight.fill_(.31)

# mem1 = lif1.init_leaky()

# mem1_rec = []
# spk1_rec = []

# img_spikes = spikegen.latency(data_it[0].squeeze(0), 255, normalize=True, linear=True)
# img_spikes = img_spikes.view(255, 1, -1)

# for step in range(num_steps):
#     cur1 = fc1(img_spikes[step])
#     spk1, mem1 = lif1(cur1, mem1)
    
#     mem1_rec.append(mem1)
#     spk1_rec.append(spk1)

# mem1_rec = torch.stack(mem1_rec)
# spk1_rec = torch.stack(spk1_rec)

# mem1_rec = mem1_rec.squeeze().detach()

# plot_cur_mem_spk(l1_cur_in, l1_mem_rec, l1_spk_rec, mem1_rec, thr_line = 1, ylim_max1 = 2, title="snn.Leaky Neuron Model")

# splt.traces(mem1_rec, spk=spk1_rec.squeeze(1))
# fig = plt.gcf()
# fig.set_size_inches(8, 6)
# plt.show()

# # Multi-Neuron Model





# # Now implement the full training loop with multiple neurons. STDP is an unsupervised learning rule so
# # you don't need to keep track of accuracy or anything like that, just keep running each iteration till a neuron spikes
# # then move on to the next image 
# # and continue for a certain amount of time till the network is "trained"