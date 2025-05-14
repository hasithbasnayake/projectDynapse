import torch
import joblib
import math
import numpy as np
from torchvision import datasets
from skimage import io
from skimage.metrics import mean_squared_error, structural_similarity
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from model.model import * 
from visualization_code.spikegen import *

decoder = joblib.load('decoder500.pkl')

# Prediction Parameters 

org_image = None
spk_image = None

num_input = 784 # Model
num_output = 50
beta = 0.9
threshold = 20
reset_mechanism = "zero"

i_ndices = list(range(1000, 1200)) # Training
iterations = 200
num_steps = 255

transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

# Dataset

# training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)

testing_set = datasets.MNIST('MNIST', train=False, transform=transform, download=True)

testing_set = torch.utils.data.Subset(testing_set, range(200))

print(len(testing_set))

training_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, shuffle=False) # print('Training set has {} images'.format(len(training_set)))

dataiter = iter(training_loader)

# Model Instantiation

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
model.load_state_dict(torch.load('newrun/model_50/model_99.pt'))

# Lists 

orig_images = []
recon_images = []

# Reconstruction 

for step in range(iterations):

    print(f"Progress: {(step / iterations) * 100}%", end='\r', flush=True)

    org_image, label = next(dataiter)

    org_image = org_image.squeeze()

    image = torch.flatten(org_image, start_dim=0)

    spike_image = spikegen(image=image, num_steps=num_steps)

    spk_rec, mem_rec = model(spike_image)

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


print(f"Average SSIM over {len(ssims)} images: {avg_ssim:.4f}")
print(f"STD SSIM over {len(ssims)} images: {std_ssim:.4f}")
print(f"Max SSIM over {len(ssims)} images: {max_ssim:.4f}")
print(f"Min SSIM over {len(ssims)} images: {min_ssim:.4f}")

print(f"Average MSE over {len(mses)} images:  {avg_mse:.4f}")
print(f"STD MSE over {len(mses)} images: {std_mse:.4f}")
print(f"Max MSE over {len(mses)} images:  {max_mse:.4f}")
print(f"Min MSE over {len(mses)} images:  {min_mse:.4f}")

all_pixels = np.concatenate([orig.flatten() for orig in orig_images])
var_Y = np.var(all_pixels)

R2 = 1.0 - (avg_mse / var_Y)
print(f"Global R² over {len(orig_images)} images: {R2:.4f}")

total = len(orig_images)                       
pairs_per_row = 10
rows = math.ceil(total / pairs_per_row)        
cols = pairs_per_row * 2                       

fig, axes = plt.subplots(rows, cols,
                         figsize=(cols * 1.2, rows * 1.2))
axes = axes.flatten()

for i in range(total):
    orig_ax = axes[2 * i]
    recon_ax = axes[2 * i + 1]

    orig_ax.imshow(orig_images[i], cmap='gray')
    orig_ax.axis('off')

    recon_ax.imshow(recon_images[i], cmap='gray', vmin=0, vmax=1)
    recon_ax.axis('off')


plt.tight_layout()
plt.savefig("fig_100", dpi=300)
plt.show()
