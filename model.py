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

        torch.nn.init.uniform_(self.fc1.weight, a=0.0, b=1.0)

    def forward(self, x):

        # print(f"Shape of x: {x.shape}")
        mem = self.lif.init_leaky()
        spk_rec = []
        mem_rec = []

        for step in range(x.shape[0]):  
            # print(f"Shape of x: {x[step].shape}")
            cur = self.fc1(x[step]) 
            spk, mem = self.lif(cur, mem)
            # print(spk)
            # print(mem)
            spk_rec.append(spk)
            mem_rec.append(mem)

            d = []

            for l in range(mem.shape[0]):
                d.append((mem, spk))

            print(d)


            # if torch.any(spk): # WTA Inhibition
            #     steps_left = x.shape[0] - step - 1
            #     # print(f"x.shape[0] ({x.shape[0]}) - step {step} = steps_left {steps_left}")

            #     spk_rec.extend([torch.zeros_like(spk) for _ in range(steps_left)])
            #     mem_rec.extend([torch.zeros_like(mem) for _ in range(steps_left)])
            #     break  


        return torch.stack(spk_rec), torch.stack(mem_rec)
