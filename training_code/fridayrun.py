# Code to train 5, 10, 25, 50, 200, 500, 750 neuron models till convergence on 1000 images
# At every 10 epochs, we train the decoder on 1000 images and print the test set, recording SSIM  

import torch 
import os
import time
import joblib
import torchvision
import numpy as np
import snntorch as snn
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt 
import pandas as pd
from IPython.display import HTML
from torchvision import datasets
from skimage.metrics import mean_squared_error, structural_similarity
from sklearn.linear_model import LinearRegression
import torchvision.transforms as transforms

from visualization_code.spikegen import *
from visualization_code.visualization import *
from model_def.model import * 
from learning_rules.stdp_time import *

# Hyperparameters

num_input = 784 # Model
models = [5, 10, 25, 50, 200, 500, 750]
beta = 0.9
threshold = 20
reset_mechanism = "zero"

training_set_size = 1000
testing_set_size = 200
num_steps = 255

A_plus = 5e-3 # STDP
A_minus = 3.75e-3
tau = 200 # Take note of the tau?
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

metrics = []

# Dataloading 

training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)
testing_set = datasets.MNIST('MNIST', train=False, transform=transform, download=True)

training_subset = torch.utils.data.Subset(training_set, range(training_set_size)) 
testing_subset = torch.utils.data.Subset(testing_set, range(testing_set_size))

# torch.save(training_subset, "fridayrun_training_set.pt")

training_loader = torch.utils.data.DataLoader(training_subset, batch_size=1, shuffle=True) 
testing_loader = torch.utils.data.DataLoader(testing_subset, batch_size=1, shuffle=True) 

# Checkpointing 5, 10, 25, 50

for model in [100, 200, 500]:

    print(f"---- Evaluating Model {model} ----")

    epoch = 0 

    while os.path.exists(f'newrun/model_{model}/model_{epoch}.pt'):

        print(f"---- Epoch {epoch} ----")
            
        net = Net(num_input=num_input, num_output=model, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        net.load_state_dict(torch.load(f'newrun/model_{model}/model_{epoch}.pt', weights_only=True)) 
        net.eval()

        dataitertrain = iter(training_loader) 
        dataitertest = iter(testing_loader)

        X = [] # [num_images x outputs] tensor 
        Y = [] # [num_images x features] tensor     

        with torch.no_grad():

            for step in range(1000):
        
                # Process Features, Input Into Model

                print(f"Progress: {(step / 1000) * 100}%", end='\r', flush=True)

                image, label = next(dataitertrain)

                image = image.squeeze()

                image = torch.flatten(image, start_dim=0) 

                spike_image = spikegen(image=image, num_steps=num_steps) # visualize_spikegen(spike_image)

                spk_rec, mem_rec = net(spike_image) # Pass the [timestep x features] tensor into the model

                # Get First Spike Time for Output Neurons

                values, indices = spk_rec.max(dim=0)

                first_spk_rec = torch.where(values == 0, 255, indices)

                X.append(first_spk_rec)
                Y.append(image)

        # Train Linear Regression Model

        X = torch.stack(X).detach().numpy()
        Y = torch.stack(Y).detach().numpy()

        decoder = LinearRegression()
        decoder.fit(X, Y)

        print("Trained linear regression model")

        orig_images = []
        recon_images = []

        with torch.no_grad():

            for step in range(200):

                print(f"Progress: {(step / 200) * 100}%", end='\r', flush=True)

                org_image, label = next(dataitertest)

                org_image = org_image.squeeze()

                image = torch.flatten(org_image, start_dim=0)

                spike_image = spikegen(image=image, num_steps=num_steps)

                spk_rec, mem_rec = net(spike_image)

                values, indices = spk_rec.max(dim=0)

                first_spk_rec = torch.where(values == 0, 255, indices)

                spk_image = first_spk_rec

                recon_img = decoder.predict(spk_image.unsqueeze(0).detach().numpy())
                recon_img = recon_img.reshape(28,28)

                orig_images.append(org_image.numpy())
                recon_images.append(recon_img)
        
        mses = []
        ssims = []

        for orig, recon in zip(orig_images, recon_images):
            mses.append(mean_squared_error(orig, recon))
            ssims.append(structural_similarity(orig, recon, data_range=1.0))

        avg_ssim = np.mean(ssims)
        std_ssim = np.std(ssims)
        max_ssim = np.max(ssims)
        min_ssim = np.min(ssims)

        avg_mse  = np.mean(mses)
        std_mse = np.std(mses)
        max_mse = np.max(mses)
        min_mse = np.min(mses)

        print("Evaluated decoder")

        metrics.append({
            'neurons': model,
            'epoch': epoch,
            'avg_ssim': avg_ssim,
            'max_ssim': max_ssim,
            'min_ssim': min_ssim,

            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'max_mse': max_mse,
            'min_mse': min_mse
        })

        print("Saved Metrics")

        path = f'decoder/decoder_{model}_epoch_{epoch}.pkl'
        joblib.dump(decoder, path)

        print("Saved Decoder")

        if epoch == 90 or epoch == 190 or epoch == 490:
            epoch += 9
        else:
            epoch += 10

df = pd.DataFrame(metrics)
df.to_csv('decoder_metrics.csv', index=False)
print("Saved metrics to decoder_metrics.csv")

