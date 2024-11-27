# .\env\Scripts\activate to activate virtual environment
# pip freeze > requirements.txt You can export a list of all installed packages
# pip install -r requirements.txt to install from a requirements file
# pip list to list all packages
 
from torchvision import datasets
from torch.utils.data import DataLoader
import snntorch as snn
from IPython.display import HTML
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

kernelON = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)
kernelOFF = dogKernel(dim = dim, ang = ang, ppa = ppa, ctr = ctr, sur = sur)

kernelON.setFilterCoefficients(ONOFF="ON")
kernelOFF.setFilterCoefficients(ONOFF="OFF")

FashionMNISTTrain = datasets.FashionMNIST(root='./data/raw', train=True, download=True, transform=None)
FashionMNISTTest = datasets.FashionMNIST(root='./data/raw', train=False, download=True, transform=None)

num_train_samples = 1000
num_test_samples = 200
rng = np.random.default_rng(2021)

training_set, testing_set = createTrainingTestingSets(training_images = FashionMNISTTrain, testing_images = FashionMNISTTest, num_train_samples = num_train_samples, num_test_samples = num_test_samples, rng=rng)

full_set_analysis = dataAnalysis(training_set + testing_set)
training_set_analysis = dataAnalysis(training_set)
testing_set_analysis = dataAnalysis(testing_set)

plt.title("Unaltered Sneaker")
plt.imshow(training_set[0][0], cmap="grey")
# plt.savefig("UnalteredSneaker.png")

ON_training_set = genLGNActivityMaps(training_set, kernelON.kernel, True)
OFF_training_set = genLGNActivityMaps(training_set, kernelOFF.kernel, False)

plt.title("Activity Map Sneaker")
plt.imshow(ON_training_set[0][0], cmap="grey")
# plt.savefig("ActivityMapSneaker.png")

tensor_dataset = convertToTensor(ON_training_set)

train_loader = DataLoader(tensor_dataset, batch_size=100)

data = iter(train_loader) 
data_it, targets_it = next(data)  

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)
print(f"Spike data shape: {spike_data.shape}")

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(100, -1), ax, s=25, c="black")

plt.title(f"Spike Train of Sneaker")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
# plt.savefig("SpikeTrainOfSneaker.png")
plt.show()

spike_data_sample = spike_data[:, 0]
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax, 20, 500)

anim.save("SneakerSpikes.gif")

# Min, max, and mean has been double checked with matlab values and are essentially the same. 
# genLGNActivityMaps has been double checked to make sure it's convolving images, and keeping labels the same

# Write latency encoding code using snntorch

# So there's two options, we can either do the image augmentation through keeping everything a tensor, create a 
# DoG kernel in the form of a tensor that we can use pytorch.functional.nn conv2d function on 
# Then compare those final results to the tensor you produced before and checked against the matlab functions 

num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

num_steps = 25
beta = 0.95

batch_size = 100
epochs = 5
