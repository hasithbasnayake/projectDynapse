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
    first_spike_time = None 
    neuron = None

    for time_step, spk in enumerate(out_spike):
        if spk.any(): # For every timestep's spike record, check if any neurons spiked 
            spiking_neurons = torch.where(spk == 1)[0] # If there's a spike(s), create an index list of the neuron(s) that spiked
            neuron = spiking_neurons[torch.randint(0, len(spiking_neurons), (1,)).item()] # Choose a random index, that's the neuron chosen for a weight update
            first_spike_time = time_step # Save the timestep

            break
    
    out_spike = out_spike[:, neuron] # Keep only the spike train of the neuron that spiked, disregard the rest (crucial for WTA)

    # Compute weight update 
    

    return neuron, first_spike_time, out_spike


# Testing

out_spike = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0]
])


neuron, first_spike_time, out_spike = stdp(None, None, out_spike)
print(f"neuron: {neuron}")
print(f"first_spike_time: {first_spike_time}")
print(f"out_spike: {out_spike}")

