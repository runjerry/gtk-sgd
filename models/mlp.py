'''MLP in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

use_bias = False

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(3*32*32, 2634, bias=use_bias)
        self.fc2   = nn.Linear(2634, 2196, bias=use_bias)
        self.fc3   = nn.Linear(2196, 1758, bias=use_bias)
        self.fc4   = nn.Linear(1758, 1320, bias=use_bias)
        self.fc5   = nn.Linear(1320, 882, bias=use_bias)
        self.fc6   = nn.Linear(882, 444, bias=use_bias)
        self.fc7   = nn.Linear(444, 10, bias=use_bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x
