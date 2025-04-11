import torch

def spikegen(image, num_steps):

    image = 1 / image 

    # Add a noise? 

    return image 

a = torch.tensor([0, 1, 2, 255])

# Handle the edge case in which the pixel value is 0.

print(spikegen(a, 2))

