import torch
import math
import torch.nn as nn
import snntorch as snn

def stdp(fc, in_spike, out_spike, params):
    '''
    fc: nn.Linear (The model's matrix of weights with the dimensions [# Output Neurons, # Input Neurons]
    in_spike: torch.Tensor (A tensor containing the input spikes with the dimensions [# Timestep, # Input Neurons])
    out_spike: torch.Tensor (A tensor containing the output spikes with the dimension [# Timestep. # Output Neurons])
    params: Python List (A list containing A_plus, A_minus, tau, mu_plus, mu_minus, the hyperparameters of the STDP rule)
    
    Need to finish writing the rest of the parameters here, as well as an example 

    '''

    post_synaptic_neuron_idx = None
    A_plus, A_minus, tau, mu_plus, mu_minus = params

    for time_step in range(out_spike.shape[0]):  # Each row of out_spike represents a timestep, where each element is 0 (no spike) or 1 (spike) for a neuron
        if out_spike[time_step].any():  # If there's a spike for any neuron
            spiking_neuron_indices = torch.where(out_spike[time_step] == 1)[0]  # Get the indices of neurons that spiked
            post_synaptic_neuron_idx = spiking_neuron_indices[torch.randint(0, len(spiking_neuron_indices), (1,)).item()]  # Randomly choose one neuron

            break  # Exit the loop after finding the first spike

    post_synaptic_neuron_spike_train = out_spike[:, post_synaptic_neuron_idx]  # Keep only the spike train of the chosen neuron, discard the rest
    spike_times_of_input_neurons = torch.argmax(in_spike, dim=0)
    print(f"spike_times_of_input_neurons: {spike_times_of_input_neurons}")
    # DEBUG: Note that if there's no input spike from an input neuron, the column contains just zeros, it will mark it as 0, this might lead to undefined behavior

    # Weight Update

    neuron_weights = fc[post_synaptic_neuron_idx, :] # Get the existing weight matrix for the chosen neuron
    print(f"neuron_weights: {neuron_weights}")
 
    delta_w = torch.zeros_like(neuron_weights) # Create a tensor to store the weight changes
    print(f"delta_w: {delta_w}")

    spike_times_of_output_neuron = torch.where(post_synaptic_neuron_spike_train == 1)[0]

    print(f"spk_indices: {spike_times_of_output_neuron}")

    for synapse, t_pre in enumerate(spike_times_of_input_neurons):
        # synapse is the index, first index is first pixel and so on
        # t_pre is the t of that index (pixel)

        for t_post in spike_times_of_output_neuron:
            delta_t = t_post - t_pre
            w = neuron_weights[synapse]

            if delta_t > 0:
                # delta_w[synapse] += A_plus * (1 - w)**mu_plus * math.exp(-abs(delta_t) / tau)
                delta_w[synapse] += 1

            if delta_t < 0: 
                # delta_w[synapse] += -A_minus * w**mu_minus * math.exp(-abs(delta_t) / tau)
                delta_w[synapse] += -1

    print(f"delta_w: {delta_w}")

    return post_synaptic_neuron_idx, post_synaptic_neuron_spike_train

























    # Compute weight update 

    # input_spike_times = torch.zeros(in_spike.shape[1])
    
    # # So remember that the weight update is by synaptic connection, so order matters
    # # This is because your delta_w is a 1x784 matrix, for that single neuron that you are updating 
    # # So you have to go column wise, for each column, where the column corresponds to the first, second, third, etc pixel
    # # You find the timestep (row) in which it spiked.
    # # You then take the timestep of when your post-synaptic neuron spiked, then compute the weight update and append it to the list of delta_W

    # for col in range(in_spike.shape[1]): # Iterate over every column
    #     step = torch.argmax(in_spike[:, col]).item() # Get the timestep in which they fire
    #     input_spike_times[col] = step # Append to a list that's tracking their spike times, should be the # of Input Synapses Long
    
    # delta_w = torch.zeros_like(input_spike_times)
    # delta_w_org = fc[neuron, :].clone()



    # for idx, post_spike in enumerate(out_spike):
    #     print(f"idx: {idx}")
    #     if post_spike == 1:
    #         for pre_spike_time in input_spike_times:
    #             delta_t = idx - pre_spike_time
    #             w = delta_w_org[idx]

    #             if delta_t > 0:
    #                 delta_w[idx] += A_plus * (1 - w)**mu_plus * math.exp(-abs(delta_t) / tau)

    #             if delta_t < 0:
    #                 delta_w[idx] += -A_minus * w**mu_minus * math.exp(-abs(delta_t) / tau)


    #     print(f"post_spike: {post_spike}")

        

    # print(f"DEBUG: Length of input_spike_times: {len(input_spike_times)}")
    # print(f"DEBUG: Num col in input_spike (should match above val): {input_spike.shape[1]}")
    

    # return neuron, post_spike_time, out_spike, input_spike_times, delta_w


# Need to check the zero-based indexing of your snntorch functions 
# Since you're going off zero-based indexing for stdp weight update
# Testing

# fc: nn.Linear (The model's matrix of weights with the dimensions [# Output Neurons, # Input Neurons]
# in_spike: torch.Tensor (A tensor containing the input spikes with the dimensions [# Timestep, # Input Neurons])
# out_spike: torch.Tensor (A tensor containing the output spikes with the dimension [# Timestep. # Output Neurons])



weight_matrix = torch.tensor([
    [.2, .5, .1, .3],
    [.3, .2, .6, .9]
])

in_spike = torch.tensor([
    [0, 1, 0, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
])

out_spike = torch.tensor([
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 1]
])

A_plus = 5e-3
A_minus = 3.75e-3
tau = 20
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

for x in range(1):
    r_post_synaptic_neuron_idx, r_post_synaptic_neuron_spike_train = stdp(weight_matrix, in_spike, out_spike, params)
    print(f"post_synaptic_neuron_idx: {r_post_synaptic_neuron_idx}")
    print(f"out_spike: {r_post_synaptic_neuron_spike_train}")


