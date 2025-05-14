import matplotlib.pyplot as plt 
import numpy as np
import torch

def sample_images(num_images, training_loader, show_img=False):

    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 6))

    for index in range(num_images):
        image, label = training_loader.dataset[index]
        
        image_mean = image.mean().item()
        image_median = image.median().item()
        image_max = image.max().item()
        image_min = image.min().item()

        flatten_image = torch.flatten(image)
        hist = torch.histc(flatten_image, bins=10, min=0.0, max=1.0)

        print(f"{'-'*50}")
        print(f"Image #{index+1}")
        print(f"Label: {label}, Mean: {image_mean:.4f}, Median: {image_median} Min: {image_min:.4f}, Max: {image_max:.4f}")
        print(f"Histogram: {hist}")

        axes[0, index].imshow(image.squeeze().numpy(), cmap='gray')
        axes[0, index].set_title(f'Label: {label}')
        axes[0, index].axis('on')

        bin_edges = torch.linspace(0.0, 1.0, steps=11)
        axes[1, index].bar(bin_edges[:-1].numpy(), hist.numpy(), width=0.1, align='edge', color='gray')
        axes[1, index].set_xlim(0, 1)
        axes[1, index].set_ylim(0, hist.max().item() + 1)
        axes[1, index].set_title('Pixel Histogram')
        axes[1, index].set_xlabel('Pixel Value')
        axes[1, index].set_ylabel('Count')

    if show_img: 
        plt.tight_layout()
        plt.show()

    print(f"{'-'*50}")