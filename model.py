import torch
import torch.nn as nn
import snntorch as snn

class Net(nn.Module):
    def __init__(self,
                 num_input,
                 num_output,
                 beta,
                 threshold,
                 reset_mechanism):
        super().__init__()

        self.fc1 = nn.Linear(in_features= num_input, out_features = num_output, bias = False)
        self.lif = snn.Leaky(beta = beta, threshold = threshold, reset_mechanism = reset_mechanism)

    def forward(self, x):

        mem = self.lif.init_leaky()
        spk_rec = []
        mem_rec = []

        for step in range(x.shape[0]):  
            cur = self.fc1(x[step]) 
            spk, mem = self.lif(cur, mem) 
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec), torch.stack(mem_rec)
