import torch
from model import * 

# Hyperparameters

num_input = 784 # Model
num_output = 10
beta = 0.9
threshold = 20
reset_mechanism = "zero"

iterations = 10000 # Training
num_steps = 255

A_plus = 5e-3 # STDP
A_minus = 3.75e-3
tau = 200 # Take note of the tau?
mu_plus = 0.65
mu_minus = 0.05

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
model.load_state_dict(torch.load('test_net.pt'))