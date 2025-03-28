import torch
import math 
from old.oldstdp import *

def stdp_time(weight_matrix: torch.Tensor, 
              in_spike: torch.Tensor, 
              out_spike: torch.Tensor, 
              params: list):
    '''
    Generate a pre-synaptic weight update for a single neuron
    using a time-based STDP learning rule that explicitly utilizes
    the difference in time between pre-synaptic and post-synaptic spikes.

    Parameters:
        weight_matrix: A matrix of model weights with the shape [output_nuerons x input_features]
        in_spike: A tensor of input spikes with the shape [timesteps x input_features]
        out_spike: A tensor of output spikes with the shape [timesteps x output_neurons]
        params: A list of STDP parameters in the form [A_plus, A_minus, tau, mu_plus, mu_minus]
            A_plus = The maximum weight update for LTP
            A_minus = The minimum weight update for LTD
            tau = Temporal decay constant of learning window
            mu_plus = Learning rate parameter for LTP
            mu_minus = Learning rate parameter for LTD
    '''

    # WTA Inhibition

    A_plus, A_minus, tau, mu_plus, mu_minus = params
    out_neuron = None

    for timestep in range(out_spike.shape[0]):
        if out_spike[timestep].any():
            spike_idx = torch.where(out_spike[timestep] == 1)[0]
            out_neuron = spike_idx[torch.randint(0, len(spike_idx), (1,)).item()]

            break
    
    print(f"Firing neuron: {out_neuron}")

    if out_neuron == None:
        print(f"No weight update, no output spikes")
        return None 
    
    # Save spike train and weight matrix of chosen output neuron
    
    out_spike = out_spike[:, out_neuron]
    weight_matrix = weight_matrix[out_neuron, :]

    print(f"out_spike: {out_spike}")
    print(f"weight_matrix: {weight_matrix}")
    
    # Retrieve spike timesteps of input neurons 

    has_spikes = in_spike.any(dim=0)
    in_spike_times = torch.where(has_spikes, torch.argmax(in_spike, dim=0), -1)

    print(f"in_spike_times: {in_spike_times}")

    # Retrieve spike timesteps of chosen output neuron

    out_spike_times = torch.where(out_spike == 1)[0]

    print(f"out_spike_times: {out_spike_times}")

    # Weight Update

    in_spike_times = in_spike_times.float()
    out_spike_times = out_spike_times.float()

    delta_t = out_spike_times[:, None] - in_spike_times[None, :]
    
    exp_decay = torch.exp(-torch.abs(delta_t) / tau)
    
    pos_mask = delta_t > 0
    ltp_updates = (pos_mask * A_plus * (1 - weight_matrix[None, :])**mu_plus * exp_decay).sum(dim=0)
    
    neg_mask = delta_t < 0
    ltd_updates = (neg_mask * -A_minus * weight_matrix[None, :]**mu_minus * exp_decay).sum(dim=0)
    
    delta_w = ltp_updates + ltd_updates

    return delta_w
    

weight_matrix = torch.tensor([
    [.2, .5, .1, .3],
    [.3, .2, .6, .9]
])

in_spike = torch.tensor([
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
])

out_spike = torch.tensor([
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])

A_plus = 5e-3
A_minus = 3.75e-3
tau = 20
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

A_delta_w = stdp_time(weight_matrix, in_spike, out_spike, params)
trash, trash2, B_delta_w = stdp(weight_matrix, in_spike, out_spike, params)

print(f"A_delta_w: {A_delta_w}")
print(f"B_delta_w: {B_delta_w}")