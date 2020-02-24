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

class CalibrationLSTM(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(num_inputs, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, num_outputs)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
