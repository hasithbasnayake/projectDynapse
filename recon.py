import torch
import joblib
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from model import * 
from spikegen import *

decoder = joblib.load('decoder.pkl')

# Prediction Parameters 

org_image = None
spk_image = None

num_input = 784 # Model
num_output = 100
beta = 0.9
threshold = 20
reset_mechanism = "zero"

iterations = 10 # Training
num_steps = 255

transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

# Dataset

# training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)

training_subset = torch.load("training_subset.pt")

training_loader = torch.utils.data.DataLoader(training_subset, batch_size=1, shuffle=True) # print('Training set has {} images'.format(len(training_set)))

dataiter = iter(training_loader)

# Model Instantiation

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
model.load_state_dict(torch.load('models/model_99.pt'))

# Plot Parameters

fig, axes = plt.subplots(iterations, 2)

for step in range(iterations):

    print(f"Progress: {(step / iterations) * 100}%", end='\r', flush=True)

    org_image, label = next(dataiter)

    org_image = org_image.squeeze()

    image = torch.flatten(org_image, start_dim=0) 

    spike_image = spikegen(image=image, num_steps=num_steps) 

    spk_rec, mem_rec = model(spike_image) # Pass the [timestep x features] tensor into the model

    # Get First Spike Time for Output Neurons

    values, indices = spk_rec.max(dim=0)

    first_spk_rec = torch.where(values == 0, 255, indices)

    spk_image = first_spk_rec

    recon_img = decoder.predict(spk_image.unsqueeze(0).detach().numpy())

    axes[step, 0].imshow(org_image, cmap="grey")
    axes[step, 1].imshow(recon_img.reshape(28,28), cmap="grey")

plt.show()