import torch
from torch import nn


class ERA5_Transformer(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, sequence_length=270):
        super(ERA5_Transformer, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.transformer = nn.Transformer(d_model=input_dim, nhead=input_dim, num_encoder_layers=12, batch_first=True)
        self.fc = torch.nn.Linear(in_features=self.input_dim, out_features=1)

    def forward(self, x):
        output = self.transformer(x, x)
        output = torch.nn.Dropout(0.4)(output)
        return self.fc(output[:, -1, :])
