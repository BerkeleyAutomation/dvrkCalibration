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
        # self.linear1 = nn.Linear(num_inputs, num_outputs)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.output(x)
        # return self.linear1(state)

class LinearModel(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_dim = [128]):
        super(CalibrationModel, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.apply(weights_init_)

    def forward(self, state):
        return self.linear(state)

class CalibrationLSTM(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer_size=120):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(num_inputs, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, num_outputs)

    def forward(self, input_seq):
        input_seq = input_seq.permute([1, 0, 2])
        hidden = self.init_hidden(input_seq)

        lstm_out, self.hidden_cell = self.lstm(input_seq, hidden)
        # print(lstm_out.shape)
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out[-1])
        return predictions

    def init_hidden(self, x):
        if next(self.parameters()).is_cuda:
            return (torch.zeros(1, x.size(1), self.hidden_layer_size).float().cuda(),
                    torch.zeros(1, x.size(1), self.hidden_layer_size).float().cuda())
        return (torch.zeros(1, x.size(1), self.hidden_layer_size).float(),
                torch.zeros(1, x.size(1), self.hidden_layer_size).float())
