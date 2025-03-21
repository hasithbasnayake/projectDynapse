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

            spk_rec.append(spk)
            mem_rec.append(mem)
            
            # print(spk)
            # print(mem)

            # d = [(f"{mem[0, l].item():.1f}", spk[0, l].item()) for l in range(mem.shape[1])]
            # print(f"{d}{step}")


            # if torch.any(spk):
            #     print(f"SPIKE DETECTED AT {step}")
                
            # if torch.any(spk):
            #     neurons_that_spiked = [i for i, val in enumerate(spk[0].tolist()) if val == 1]
            #     # print(neurons_that_spiked)
            #     random_neuron = neurons_that_spiked[torch.randint(0, len(neurons_that_spiked), (1,)).item()]
            #     # print(random_neuron)

            #     spk = torch.zeros_like(spk)
            #     mem = torch.zeros_like(mem)

            #     spk[0, random_neuron] = 1

            #     spk_rec.append(spk)
            #     mem_rec.append(mem)

            #     # print(f"Shape of spk_rec: {len(spk_rec)}")

            #     while len(spk_rec) < 255:
            #         spk_rec.append(torch.zeros((1, spk.shape[1]
            #                                     )))
            #         mem_rec.append(torch.zeros((1, spk.shape[1])))

            #     break
                
            # spk_rec.append(spk)
            # mem_rec.append(mem)

        return torch.stack(spk_rec), torch.stack(mem_rec)
