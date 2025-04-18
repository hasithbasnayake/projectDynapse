import torch
import joblib
from model import * 
from spikegen import *
from torchvision import datasets
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import torchvision.transforms as transforms

# Hyperparameters

num_input = 784 # Model
num_output = 100
beta = 0.9
threshold = 20
reset_mechanism = "zero"

iterations = 1000 # Training
num_steps = 255

transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

# Dataloading 

# training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)

training_subset = torch.load("training_subset.pt")

training_loader = torch.utils.data.DataLoader(training_subset, batch_size=1, shuffle=True) # print('Training set has {} images'.format(len(training_set)))

dataiter = iter(training_loader)

# Model Instantiation

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
model.load_state_dict(torch.load('models/model_99.pt'))

image_to_reconstruct = None
spk_to_reconstruct = None

# Linear Regression Inputs 

X = [] # [num_images x outputs] tensor 
Y = [] # [num_images x features] tensor 

for step in range(iterations):

    # Progress Bar 
    
    print(f"Progress: {(step / iterations) * 100}%", end='\r', flush=True)

    # Process Features, Input Into Model

    image, label = next(dataiter)

    image = image.squeeze()

    image = torch.flatten(image, start_dim=0) 

    spike_image = spikegen(image=image, num_steps=num_steps) # visualize_spikegen(spike_image)

    spk_rec, mem_rec = model(spike_image) # Pass the [timestep x features] tensor into the model

    # Get First Spike Time for Output Neurons

    values, indices = spk_rec.max(dim=0)

    first_spk_rec = torch.where(values == 0, 255, indices)

    X.append(first_spk_rec)
    Y.append(image)

    if step == 0:

        image_to_reconstruct = image 
        spk_to_reconstruct = first_spk_rec

# Train Linear Regression Model

X = torch.stack(X).detach().numpy()
Y = torch.stack(Y).detach().numpy()

decoder = LinearRegression()
decoder.fit(X, Y)

print(f"Shape of image: {image_to_reconstruct.shape}")
print(f"Shape of spk: {spk_to_reconstruct.shape}")

recon_img = decoder.predict(spk_to_reconstruct.unsqueeze(0).detach().numpy())

print(f"Shape of recon_img: {recon_img.shape}")

image_to_reconstruct = image_to_reconstruct.unsqueeze(0).reshape(28, 28)
recon_img = recon_img.reshape(28,28)

fig, axes = plt.subplots(1, 2)

axes[0].imshow(image_to_reconstruct, cmap="grey")
axes[1].imshow(recon_img, cmap="grey")

plt.show()

joblib.dump(decoder, "decoder.pkl")