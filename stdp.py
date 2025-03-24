import torch
import math
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

    input_spike_times = torch.zeros(input_spike.shape[1])
    

    # So remember that the weight update is by synaptic connection, so order matters
    # This is because your delta_w is a 1x784 matrix, for that single neuron that you are updating 
    # So you have to go column wise, for each column, where the column corresponds to the first, second, third, etc pixel
    # You find the timestep (row) in which it spiked.
    # You then take the timestep of when your post-synaptic neuron spiked, then compute the weight update and append it to the list of delta_W

    for col in range(input_spike.shape[1]): # Iterate over every column
        step = torch.argmax(input_spike[:, col]).item() # Get the timestep in which they fire
        input_spike_times[col] = step # Append to a list that's tracking their spike times, should be the # of Input Synapses Long
    
    delta_w = torch.zeros_like(input_spike_times)
    delta_w_org = fc[neuron, :].clone()

    A_plus = 5e-3
    A_minus = 3.75e-3
    tau = 20
    mu_plus = 0.65
    mu_minus = 0.05

    for idx, post_spike in enumerate(out_spike):
        print(f"idx: {idx}")
        if post_spike == 1:
            for pre_spike_time in input_spike_times:
                delta_t = idx - pre_spike_time
                w = delta_w_org[idx]

                if delta_t > 0:
                    delta_w[idx] += A_plus * (1 - w)**mu_plus * math.exp(-abs(delta_t) / tau)

                if delta_t < 0:
                    delta_w[idx] += -A_minus * w**mu_minus * math.exp(-abs(delta_t) / tau)


        print(f"post_spike: {post_spike}")

        

    print(f"DEBUG: Length of input_spike_times: {len(input_spike_times)}")
    print(f"DEBUG: Num col in input_spike (should match above val): {input_spike.shape[1]}")
    

    return neuron, post_spike_time, out_spike, input_spike_times, delta_w


# Need to check the zero-based indexing of your snntorch functions 
# Since you're going off zero-based indexing for stdp weight update
# Testing

weight_matrix = torch.tensor([
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [.2, .1, .3, .5, 0.8],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

input_spike = torch.tensor([
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
])

out_spike = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0]
])


neuron, post_spike_time, out_spike, input_spike_times, delta_w = stdp(weight_matrix, input_spike, out_spike)
print(f"neuron: {neuron}")
print(f"post_spike_time: {post_spike_time}")
print(f"out_spike: {out_spike}")
print(f"input_spike_times: {input_spike_times}")
print(f"delta_w: {delta_w}")
