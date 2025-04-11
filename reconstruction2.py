import torch
from sklearn.linear_model import LinearRegression
from snntorch import spikegen
import matplotlib.pyplot as plt
from model import *

torch.set_printoptions(precision=8)


test_net = Net(num_input=1568, num_output=140, beta=0.8, threshold=20, reset_mechanism="zero")

test_net.load_state_dict(torch.load('test_net.pt'))

ORG_data = torch.load("data/split/split_train.pt", weights_only=True)
ON_data = torch.load("data/processed/convON_train.pt", weights_only=True)
OFF_data = torch.load("data/processed/convOFF_train.pt", weights_only=True)

img_idx = 2

ORG_img = ORG_data[img_idx][0] # [C x H X W]
ON_img = ON_data[img_idx][0]
OFF_img = OFF_data[img_idx][0]

plt.imshow(ORG_img.squeeze(0).detach().numpy(), cmap="grey")
plt.title("Reconstructed Image")
plt.show()

IN_img = torch.stack([ON_img, OFF_img], dim=1)
IN_img = torch.flatten(IN_img, start_dim=1)
IN_img = spikegen.latency(data=IN_img, num_steps=255, normalize=True, linear=True)
IN_img = IN_img.squeeze(1)

spk_rec, mem_rec = test_net(IN_img) # spk_rec: [timestep x num_output]

# val_of_spk, idx_of_spk = spk_rec.max(dim=0)

# first_spk = torch.where(val_of_spk == 0, 999, idx_of_spk) 

first_spk = torch.argmax(spk_rec, dim=0)

# print(first_spk)

num_activated_neurons = 5
activation = 1. / first_spk

norm_activation = activation / torch.max(activation)

print(norm_activation)

top_neurons, top_neurons_idx = torch.topk(norm_activation, num_activated_neurons)

print(top_neurons)
print(top_neurons_idx)

# print(test_net.fc1.weight.shape)

neuron_weights = test_net.fc1.weight

receptive_field = torch.zeros((1,784))

for i in range(top_neurons_idx.size(0)):
    neuron_idx = top_neurons_idx[i].item()
    print(f"neuron_idx: {neuron_idx}")
    ON_RF = neuron_weights[neuron_idx, :784]
    OFF_RF = neuron_weights[neuron_idx, 784:]
    RF = ON_RF.reshape(28,28) - OFF_RF.reshape(28,28)

    plt.imshow(RF.detach().numpy(), cmap="grey")
    plt.show()
    receptive_field += (norm_activation[neuron_idx] * RF) 
    
receptive_field = receptive_field.reshape(28,28)

plt.imshow(receptive_field.detach().numpy(), cmap="grey")
plt.title("Last Image")
plt.show()

# ON_RF = receptive_field[0, 784:].reshape(28,28)
# OFF_RF = receptive_field[0, :784].reshape(28,28)

# REC = ON_RF - OFF_RF

# plt.figure(figsize=(10, 10))

# plt.subplot(1, 4, 1)
# plt.imshow(ORG_img.squeeze(0).detach().numpy(), cmap="grey")
# plt.title("Original Image")

# plt.subplot(1, 4, 2)
# plt.imshow(ON_RF.detach().numpy(), cmap="grey")
# plt.title("ON RF")

# plt.subplot(1, 4, 3)
# plt.imshow(OFF_RF.detach().numpy(), cmap="grey")
# plt.title("OFF RF")

# plt.subplot(1, 4, 4)
# plt.imshow(REC.detach().numpy(), cmap="grey")
# plt.title("ON RF - OFF RF")

# plt.tight_layout()
# plt.show()