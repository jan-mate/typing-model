import torch.nn as nn


class FixedMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DynamicMLP(nn.Module):
    def __init__(self, in_dim, n_layers, hidden_dim, dropout, activation_name):
        super().__init__()
        activations = {"ReLU": nn.ReLU(), "GELU": nn.GELU(), "SiLU": nn.SiLU()}
        act = activations[activation_name]
        layers = []
        curr_dim = in_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(curr_dim, hidden_dim), act, nn.Dropout(dropout)])
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)