import torch

def build_MLP(sizes, hidden_activation=torch.nn.Tanh, output_activation=None):
    layers = []
    for s in zip(sizes[:-2], sizes[1:-1]):
        layers.append(torch.nn.Linear(*s))
        #layers.append(torch.nn.BatchNorm1d(s[1]))
        layers.append(hidden_activation())
    layers.append(torch.nn.Linear(*sizes[-2:]))
    if output_activation is not None:
        layers.append(output_activation())
    return torch.nn.Sequential(*layers)
