import torch
import torch.nn as nn
import snntorch as snn

def stdp(fc, in_spike, out_spike):
    '''
    fc: nn.Linear (The model's matrix of weights)
    in_spike: torch.Tensor (A tensor containing the input spikes with the dimensions [# Timestep, # Input Neurons])
    out_spike: torch.Tensor (A tensor containing the output spikes with the dimension [# Timestep. # Output Neurons])
    
    Need to finish writing the rest of the parameters here, as well as an example 

    '''
    # Identify the neuron that spiked first 
    post_spike_time = None 
    neuron = None

    for time_step, spk in enumerate(out_spike):
        if spk.any(): # For every timestep's spike record, check if any neurons spiked 
            spiking_neurons = torch.where(spk == 1)[0] # If there's a spike(s), create an index list of the neuron(s) that spiked
            neuron = spiking_neurons[torch.randint(0, len(spiking_neurons), (1,)).item()] # Choose a random index, that's the neuron chosen for a weight update
            post_spike_time = time_step # Save the timestep

            break
    
    out_spike = out_spike[:, neuron] # Keep only the spike train of the neuron that spiked, disregard the rest (crucial for WTA)

    # Compute weight update 

    # Create a list of all delta T's, the input spike train is in the form 255 x 784

    # We need to find the spike time that every pixel fires at, wherever the column is 1
    # Save that time in a list 
    # Then compute for every post-synaptic fire, get that timestep (row) and compute a delta t 

    input_spike_times = []

    # So remember that the weight update is by synaptic connection, so order matters
    # This is because your delta_w is a 1x784 matrix, for that single neuron that you are updating 
    # So you have to go column wise, for each column, where the column corresponds to the first, second, third, etc pixel
    # You find the timestep (row) in which it spiked.
    # You then take the timestep of when your post-synaptic neuron spiked, then compute the weight update and append it to the list of delta_W

    for col in range(input_spike.shape[1]): # Iterate over every column
        row = torch.argmax(input_spike[:, col]).item() # Get the timestep in which they fire
    

    return neuron, post_spike_time, out_spike


# Need to check the zero-based indexing of your snntorch functions 
# Since you're going off zero-based indexing for stdp weight update
# Testing

input_spike = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])

out_spike = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0]
])


neuron, post_spike_time, out_spike = stdp(None, input_spike, out_spike)
print(f"neuron: {neuron}")
print(f"post_spike_time: {post_spike_time}")
print(f"out_spike: {out_spike}")

