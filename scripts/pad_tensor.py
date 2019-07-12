import torch

t_path="xlm-embeddings-810000-820000"

ind = 4500

t = torch.load(t_path+".pt")

first_half = t[0:ind]
second_half = t[ind:]
pad = torch.zeros(500, 1024)

t_new = torch.cat((first_half, pad, second_half))

torch.save(t_new, t_path+"-padded.pt")