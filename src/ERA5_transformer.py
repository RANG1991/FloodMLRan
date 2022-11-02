import torch
from torch import nn


class ERA5_Transformer(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, sequence_length=270):
        super(ERA5_Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.transformer = nn.Transformer(d_model=input_dim, nhead=16, num_encoder_layers=12, batch_first=True)

    def forward(self, x):
        output = self.transformer(x, x)
        output = torch.nn.Dropout(0.4)(output)
        return output
