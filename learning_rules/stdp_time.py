import torch
import math 

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
    
    # print(f"Firing neuron: {out_neuron}")

    # TURNED OFF FOR NOW TURN BACK ON LATER

    if out_neuron == None:
        # print(f"No weight update, no output spikes")
        return 0, torch.zeros_like(weight_matrix[0, :])
    
    # Save spike train and weight matrix of chosen output neuron
    
    out_spike = out_spike[:, out_neuron]
    spike_indices = torch.nonzero(out_spike, as_tuple=True)[0]
    # if spike_indices.numel() > 0:
    #     first_spike_idx = spike_indices[0]
    #     out_spike[first_spike_idx+1:] = 0
    weight_matrix = weight_matrix[out_neuron, :]

    # print(f"out_spike: {out_spike}")
    # print(f"weight_matrix: {weight_matrix}")
    
    # Retrieve spike timesteps of input neurons 

    has_spikes = in_spike.any(dim=0)
    in_spike_times = torch.where(has_spikes, torch.argmax(in_spike, dim=0), -1)

    # print(f"in_spike_times: {in_spike_times}")

    # Retrieve spike timesteps of chosen output neuron

    out_spike_times = torch.where(out_spike == 1)[0]

    # print(f"out_spike_times: {out_spike_times}")

    # Weight Update

    in_spike_times = in_spike_times
    out_spike_times = out_spike_times

    delta_t = out_spike_times[:, None] - in_spike_times[None, :]
    
    exp_decay = torch.exp(-torch.abs(delta_t) / tau)
    
    pos_mask = delta_t > 0
    ltp_updates = (pos_mask * A_plus * (1 - weight_matrix[None, :])**mu_plus * exp_decay).sum(dim=0)
    
    neg_mask = delta_t < 0
    ltd_updates = (neg_mask * -A_minus * weight_matrix[None, :]**mu_minus * exp_decay).sum(dim=0)
    
    delta_w = ltp_updates + ltd_updates

    return out_neuron, delta_w
    
