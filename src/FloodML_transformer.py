import torch
from torch import nn
import math


class ERA5_Transformer(nn.Module):
    def __init__(self, input_dim, sequence_length, dim_model, num_heads, num_encoder_layers, dropout_p):
        super().__init__()

        self.dim_model = dim_model
        self.sequence_length = sequence_length
        self.positional_encoder = PositionalEncoding(max_seq_length=sequence_length, encodings_dim=dim_model,
                                                     dropout=dropout_p)
        self.linear_1 = nn.Linear(input_dim, dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.linear_2 = nn.Linear(dim_model * sequence_length, 1)

    @staticmethod
    def get_tgt_mask(size) -> torch.tensor:
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    @staticmethod
    def create_pad_mask(matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token

    def forward(self, src):
        # embedding + positional encoding - out size = (batch_size, sequence_length, dim_model)
        src = self.linear_1(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        # we permute to obtain size (sequence length, batch_size, dim_model)
        src = src.permute((1, 0, 2))
        # Transformer blocks - out size = (sequence length, batch_size, sequence_length)
        transformer_encoder_out = self.transformer_encoder(src)
        transformer_encoder_out = transformer_encoder_out.permute((1, 0, 2)).reshape(-1,
                                                                                     self.dim_model * self.sequence_length)
        out = self.linear_2(transformer_encoder_out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, encodings_dim, max_seq_length, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_encodings_matrix = torch.zeros(max_seq_length, encodings_dim)
        positions_vector = torch.arange(1, max_seq_length + 1).reshape(-1, 1)
        positions_matrix = torch.tile(positions_vector, (1, encodings_dim))
        denominator = (10000 ** (torch.arange(0, encodings_dim) / encodings_dim)).reshape(1, -1)
        denominator_matrix = torch.tile(denominator, (max_seq_length, 1))
        pos_encodings_matrix[:, :] = positions_matrix / denominator_matrix
        pos_encodings_matrix[:, 0::2] = torch.sin(pos_encodings_matrix[:, 0::2])
        pos_encodings_matrix[:, 1::2] = torch.cos(pos_encodings_matrix[:, 1::2])
        self.register_buffer("pos_encodings_matrix", pos_encodings_matrix)

    def forward(self, input_sequence: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(input_sequence + self.pos_encodings_matrix[:input_sequence.size(0), :].unsqueeze(0))
