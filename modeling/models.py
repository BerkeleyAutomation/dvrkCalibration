import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class CalibrationModel(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_dim = [128]):
        super(CalibrationModel, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[0])
        self.output = nn.Linear(hidden_dim[0], num_outputs)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return torch.tanh(self.output(x))
