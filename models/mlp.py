'''MLP in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, use_bias=False, initializer='kaiming'):
        super().__init__()
        self.fc1   = nn.Linear(3*32*32, 2634, bias=use_bias)
        self.fc2   = nn.Linear(2634, 2196, bias=use_bias)
        self.fc3   = nn.Linear(2196, 1758, bias=use_bias)
        self.fc4   = nn.Linear(1758, 1320, bias=use_bias)
        self.fc5   = nn.Linear(1320, 882, bias=use_bias)
        self.fc6   = nn.Linear(882, 444, bias=use_bias)
        self.fc7   = nn.Linear(444, 10, bias=use_bias)

        if initializer == 'xavier':
            init_fn = nn.init.xavier_normal_
        if initializer == 'xavier_u':
            init_fn = nn.init.xavier_uniform_
        elif initializer == 'kaiming':
            init_fn = nn.init.kaiming_normal_
        elif initializer == 'kaiming_u':
            init_fn = nn.init.kaiming_uniform_
        elif initializer == 'normal':
            init_fn = nn.init.normal_
        elif initializer == 'uniform':
            init_fn = nn.init.uniform_
        elif initializer == 'trunc':
            init_fn = nn.init.trunc_normal_
        elif initializer == 'dirac':
            init_fn = nn.init.dirac_
        elif initializer == 'ones':
            init_fn = nn.init.ones_
        elif initializer == 'eye':
            init_fn = nn.init.eye_
        elif initializer == 'orthogonal':
            init_fn = nn.init.orthogonal_
        else:
            init_fn = None

        if init_fn is not None:
            init_fn(self.fc1.weight.data)
            init_fn(self.fc2.weight.data)
            init_fn(self.fc3.weight.data)
            init_fn(self.fc4.weight.data)
            init_fn(self.fc5.weight.data)
            init_fn(self.fc6.weight.data)
            init_fn(self.fc7.weight.data)

            if use_bias:
                init_fn(self.fc1.bias.data)
                init_fn(self.fc2.bias.data)
                init_fn(self.fc3.bias.data)
                init_fn(self.fc4.bias.data)
                init_fn(self.fc5.bias.data)
                init_fn(self.fc6.bias.data)
                init_fn(self.fc7.bias.data)

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


class MLP2(nn.Module):
    def __init__(self, use_bias=False, initializer='kaiming'):
        super().__init__()
        self.fc1   = nn.Linear(3*32*32, 1024, bias=use_bias)
        self.fc2   = nn.Linear(1024, 1024, bias=use_bias)
        self.fc3   = nn.Linear(1024, 512, bias=use_bias)
        self.fc4   = nn.Linear(512, 256, bias=use_bias)
        self.fc5   = nn.Linear(256, 128, bias=use_bias)
        self.fc6   = nn.Linear(128, 10, bias=use_bias)

        if initializer == 'xavier':
            init_fn = nn.init.xavier_normal_
        if initializer == 'xavier_u':
            init_fn = nn.init.xavier_uniform_
        elif initializer == 'kaiming':
            init_fn = nn.init.kaiming_normal_
        elif initializer == 'kaiming_u':
            init_fn = nn.init.kaiming_uniform_
        elif initializer == 'normal':
            init_fn = nn.init.normal_
        elif initializer == 'uniform':
            init_fn = nn.init.uniform_
        elif initializer == 'trunc':
            init_fn = nn.init.trunc_normal_
        elif initializer == 'dirac':
            init_fn = nn.init.dirac_
        elif initializer == 'ones':
            init_fn = nn.init.ones_
        elif initializer == 'eye':
            init_fn = nn.init.eye_
        elif initializer == 'orthogonal':
            init_fn = nn.init.orthogonal_
        else:
            init_fn = None

        if init_fn is not None:
            init_fn(self.fc1.weight.data)
            init_fn(self.fc2.weight.data)
            init_fn(self.fc3.weight.data)
            init_fn(self.fc4.weight.data)
            init_fn(self.fc5.weight.data)
            init_fn(self.fc6.weight.data)

            if use_bias:
                init_fn(self.fc1.bias.data)
                init_fn(self.fc2.bias.data)
                init_fn(self.fc3.bias.data)
                init_fn(self.fc4.bias.data)
                init_fn(self.fc5.bias.data)
                init_fn(self.fc6.bias.data)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x


class MLP3(nn.Module):
    def __init__(self, use_bias=False, initializer='kaiming'):
        super().__init__()
        self.fc1   = nn.Linear(3*32*32, 512, bias=use_bias)
        self.fc2   = nn.Linear(512, 512, bias=use_bias)
        self.fc3   = nn.Linear(512, 256, bias=use_bias)
        self.fc4   = nn.Linear(256, 128, bias=use_bias)
        self.fc5   = nn.Linear(128, 10, bias=use_bias)

        if initializer == 'xavier':
            init_fn = nn.init.xavier_normal_
        if initializer == 'xavier_u':
            init_fn = nn.init.xavier_uniform_
        elif initializer == 'kaiming':
            init_fn = nn.init.kaiming_normal_
        elif initializer == 'kaiming_u':
            init_fn = nn.init.kaiming_uniform_
        elif initializer == 'normal':
            init_fn = nn.init.normal_
        elif initializer == 'uniform':
            init_fn = nn.init.uniform_
        elif initializer == 'trunc':
            init_fn = nn.init.trunc_normal_
        elif initializer == 'ones':
            init_fn = nn.init.ones_
        elif initializer == 'eye':
            init_fn = nn.init.eye_
        elif initializer == 'orthogonal':
            init_fn = nn.init.orthogonal_
        else:
            init_fn = None

        if init_fn is not None:
            init_fn(self.fc1.weight.data)
            init_fn(self.fc2.weight.data)
            init_fn(self.fc3.weight.data)
            init_fn(self.fc4.weight.data)
            init_fn(self.fc5.weight.data)

            if use_bias:
                init_fn(self.fc1.bias.data)
                init_fn(self.fc2.bias.data)
                init_fn(self.fc3.bias.data)
                init_fn(self.fc4.bias.data)
                init_fn(self.fc5.bias.data)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x
