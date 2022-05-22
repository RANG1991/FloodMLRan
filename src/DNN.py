import torch.nn.functional as F
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size: int, num_hidden_layers: int, num_hidden_units: int, dropout_rate: float = 0.0):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.dropout_rate = dropout_rate
        self.input_layer = nn.Linear(self.input_size, self.num_hidden_units)
        self.hidden_layer = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.output_layer = nn.Linear(self.num_hidden_units, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        batch_size, timesteps, ts_size = x.size()
        x = x.view(batch_size, timesteps * ts_size)
        x = self.input_layer(x)
        for i in range(0, self.num_hidden_layers):
            x = self.hidden_layer(F.relu(self.hidden_layer(x)))
        pred = self.dropout(self.output_layer(x))
        return pred
