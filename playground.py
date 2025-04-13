import torch
import math
import matplotlib.pyplot as plt
from snntorch import spikegen
from model import *
from stdp_time import *

# img_ON = torch.rand((1,28,28))
# img_OFF = torch.rand((1,28,28))

data_ON = torch.load("data/processed/convON_train.pt", weights_only=True) 
data_OFF = torch.load("data/processed/convOFF_train.pt", weights_only=True) 

img_ON, label_ON = data_ON[0]
img_OFF, label_OFF = data_OFF[0]

# plt.imshow(img_ON.detach().numpy().squeeze(0), cmap="grey")
# plt.show()
# plt.imshow(img_OFF.detach().numpy().squeeze(0), cmap="grey")
# plt.show()

img = torch.stack([img_ON, img_OFF], dim=1)

A_plus = 5e-3
A_minus = 3.75e-3
tau = 200
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

test_net = Net(num_input=1568, num_output=140, beta=0.8, threshold=20, reset_mechanism="zero")
print(f"img shape: {img.shape}")
img = torch.flatten(img, start_dim=1)
print(f"img shape after flatten: {img.shape}")
img = spikegen.latency(data=img, num_steps=255)
print(f"img shape after spikegen: {img.shape}")
img = img.squeeze(1)
print(f"img shape after squeeze: {img.shape}")

iterations = 1000

# for iter in range(iterations):
#     print(f"Training Step {iter} out of {iterations}")
#     spk_rec, mem_rec = test_net(img)
#     out_neuron, delta_w = stdp_time(weight_matrix=test_net.fc1.weight, in_spike=img, out_spike=spk_rec, params=params)

#     print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

#     with torch.no_grad():
#         test_net.fc1.weight[out_neuron] += delta_w
#         test_net.fc1.weight[out_neuron].clamp_(0.0, 1.0)


for iter in range(iterations):
    print(f"Training Step {iter} out of {iterations}")
    img_ON, label_ON = data_ON[iter]
    img_OFF, label_OFF = data_OFF[iter]
    img = torch.stack([img_ON, img_OFF], dim=1)

    img = torch.flatten(img, start_dim=1)
    img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
    img = img.squeeze(1)

    spk_rec, mem_rec = test_net(img)
    out_neuron, delta_w = stdp_time(weight_matrix=test_net.fc1.weight, in_spike=img, out_spike=spk_rec, params=params)

    print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

    with torch.no_grad():
        test_net.fc1.weight[out_neuron] += delta_w
        test_net.fc1.weight[out_neuron].clamp_(0.0, 1.0)


# for iter in range(iterations):
#     print(f"Training Step {iter} out of {iterations}")

#     img = None

#     if iter % 2 == 0:
#         img_ON, label_ON = data_ON[0]
#         img_OFF, label_OFF = data_OFF[0]
#         img = torch.stack([img_ON, img_OFF], dim=1)
#     else:
#         img_ON, label_ON = data_ON[1]
#         img_OFF, label_OFF = data_OFF[1]
#         img = torch.stack([img_ON, img_OFF], dim=1)

#     img = torch.flatten(img, start_dim=1)
#     img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
#     img = img.squeeze(1)

#     spk_rec, mem_rec = test_net(img)
#     out_neuron, delta_w = stdp_time(weight_matrix=test_net.fc1.weight, in_spike=img, out_spike=spk_rec, params=params)

#     print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

#     with torch.no_grad():
#         test_net.fc1.weight[out_neuron] += delta_w
#         test_net.fc1.weight[out_neuron].clamp_(0.0, 1.0)


# five_images = []

# for step in range(10):
#     img_ON, label_ON = data_ON[step]
#     img_OFF, label_OFF = data_OFF[step]

#     show_img = img_ON.squeeze(0).detach().numpy()

#     print(show_img.shape)
#     plt.imshow(show_img, cmap="grey")
#     plt.show()

#     img = torch.stack([img_ON, img_OFF], dim=1)

#     five_images.append(img)


# for iter in range(iterations):
#     print(f"Training Step {iter} out of {iterations}")

#     img = five_images[iter % len(five_images)]

#     img = torch.flatten(img, start_dim=1)
#     img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
#     img = img.squeeze(1)

#     spk_rec, mem_rec = test_net(img)
#     out_neuron, delta_w = stdp_time(weight_matrix=test_net.fc1.weight, in_spike=img, out_spike=spk_rec, params=params)

#     print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

#     with torch.no_grad():
#         test_net.fc1.weight[out_neuron] += delta_w
#         test_net.fc1.weight[out_neuron].clamp_(0.0, 1.0)


torch.save(test_net.state_dict(), 'test_net.pt')




# spk_rec, mem_rec = test_net(img)


# print(f"spk_rec shape: {spk_rec.shape}")
# print(f"mem_rec shape: {mem_rec.shape}")

# spk_rec = spk_rec.detach().numpy()
# plt.plot(spk_rec)
# plt.show()

# mem_rec = mem_rec.detach().numpy()
# plt.plot(mem_rec)
# plt.show()

# Get the weights from the fully connected layer
# weights = test_net.fc1.weight.data  # Shape: [num_neurons, 1568]
# num_neurons = weights.shape[0]

# # Plot settings
# ncols = 3  # ON, OFF, Combined
# nrows = num_neurons

# fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

# if num_neurons == 1:
#     axs = axs[None, :]  # Handle single neuron case by adding batch dimension

# for i in range(num_neurons):
#     rf_weights = weights[i]  # Shape: [1568]

#     # Split into ON and OFF
#     rf_ON = rf_weights[:784].reshape(28, 28)
#     rf_OFF = rf_weights[784:].reshape(28, 28)
#     rf_BOTH = (rf_ON - rf_OFF).clamp_(0.0, 1.0)

#     axs[i, 0].imshow(rf_ON, cmap="gray")
#     axs[i, 0].set_title(f"Neuron {i} - ON")
#     axs[i, 0].axis("off")

#     axs[i, 1].imshow(rf_OFF, cmap="gray")
#     axs[i, 1].set_title(f"Neuron {i} - OFF")
#     axs[i, 1].axis("off")

#     axs[i, 2].imshow(rf_BOTH, cmap="gray")
#     axs[i, 2].set_title(f"Neuron {i} - Combined")
#     axs[i, 2].axis("off")

# plt.tight_layout()
# plt.show()




