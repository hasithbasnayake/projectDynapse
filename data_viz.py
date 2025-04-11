import matplotlib.pyplot as plt 
import numpy as np

def sample_images(num_images, training_loader, show_img=False):

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for index in range(num_images):
        image, label = training_loader.dataset[index]
        
        image_mean = image.mean().item()
        image_max = image.max().item()
        image_min = image.min().item()

        print(f"{'-'*50}")
        print(f"Image #{index+1}")
        print(f"Label: {label}, Mean: {image_mean:.4f}, Min: {image_min:.4f}, Max: {image_max:.4f}")

        axes[index].imshow(image.squeeze().numpy(), cmap='gray')
        axes[index].set_title(f'Label: {label}')
        axes[index].axis('on')

    if show_img: 
        plt.tight_layout()
        plt.show()

    print(f"{'-'*50}")