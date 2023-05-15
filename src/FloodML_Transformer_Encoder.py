from torch import nn
import torch
from torch import Tensor
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


class Transformer_Encoder(nn.Module):

    def __init__(self, num_features, sequence_length, intermediate_dim, dropout, num_heads, num_layers):
        super(Transformer_Encoder, self).__init__()
        self.fc_1 = nn.Linear(num_features, intermediate_dim)
        self.positional_encoding = PositionalEncoding(intermediate_dim, sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermediate_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_2 = nn.Linear(intermediate_dim, 1)
        exponential_decay = torch.exp(torch.tensor([-1 * (sequence_length - i) / 25 for i in range(sequence_length)]))
        exponential_decay = exponential_decay.unsqueeze(0).unsqueeze(-1).repeat(1, 1, intermediate_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.intermediate_dim = intermediate_dim
        self.register_buffer('exponential_decay', exponential_decay)

    def forward(self, x):
        out_fc_1 = self.fc_1(x) * math.sqrt(self.intermediate_dim)
        out_pe = self.positional_encoding(out_fc_1)
        out_transformer = self.transformer_encoder(out_pe)[:, 0, :]
        # out_decay = torch.sum(out_transformer * self.exponential_decay, dim=1)
        out_fc_2 = self.dropout(self.fc_2(out_transformer))
        return out_fc_2
