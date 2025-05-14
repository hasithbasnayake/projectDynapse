import torch
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

def spikegen(image, num_steps):

    spike_rec = torch.zeros([num_steps, image.shape[0]])

    norm_latency = latency(image, num_steps)

    for index, pixel_latency in enumerate(norm_latency):
        spike_rec[pixel_latency, index] = 1

    return spike_rec

def visualize_spikegen(spike_image):

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(spike_image, ax, s=25, c="black")

    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

def visualize_latency(image, num_steps, cmap):

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    spike_image = latency(image, num_steps)
    
    org_image = torch.reshape(image, (28,28))
    reshape = torch.reshape(spike_image, (28,28))

    axes[0].imshow(org_image.cpu().numpy(), cmap=cmap)
    axes[0].set_title("Original Image")
    axes[1].imshow(reshape.cpu().numpy(), cmap=cmap)
    axes[1].set_title("Spiking Image")

    plt.tight_layout()

    plt.show()


def latency(image, num_steps):

    latency = 1.0 / image
    
    finite_vals = latency[torch.isfinite(latency)]
    min_val = torch.min(finite_vals)
    max_val = torch.max(finite_vals)
    
    norm_latency = (latency - min_val) / (max_val - min_val) * (num_steps - 1)
    
    norm_latency[~torch.isfinite(norm_latency)] = num_steps - 1

    norm_latency = norm_latency.int()
    
    return norm_latency


# def old_latency(image, num_steps):

#     latency = 1.0 / image # Use y = 1/x to convert pixel intensities to latencies where x is the pixel value 

#     latency = torch.clamp(latency, min=0.1)

#     min_val = torch.min(latency)
#     max_val = torch.max(latency)

#     norm_latency = (latency - min_val) / (max_val - min_val) * (num_steps - 1)
    
#     return norm_latency
