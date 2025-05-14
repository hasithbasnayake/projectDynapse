import torch 
import time
import torchvision
import numpy as np
import snntorch as snn
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt 
from IPython.display import HTML
from torchvision import datasets
import torchvision.transforms as transforms

from visualization_code.spikegen import *
from visualization_code.visualization import *
from model.model import * 
from training.stdp_time import *

num_input = 784 # Model
num_output = 500
beta = 0.9
threshold = 20
reset_mechanism = "zero"


for img in range(82,183):

    model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
    model.load_state_dict(torch.load(f'newrun/model_{num_output}/model_{img}.pt'))

    model_weights = model.fc1.weight
    num_neurons = model_weights.shape[0]
    cols = int(np.ceil(np.sqrt(num_neurons)))
    rows = int(np.ceil(num_neurons / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes)  

    for i in range(num_neurons):
        neuron_weights = model_weights[i].reshape(28, 28).detach().numpy()
        ax = axes[i // cols, i % cols]
        im = ax.imshow(neuron_weights, cmap="gray")
        ax.set_title(f"Neuron {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"newrun/model_{num_output}/modelfig_{img}.png", dpi=50)
    plt.close()