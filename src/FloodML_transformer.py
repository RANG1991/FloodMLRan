import torch
from torch import nn
import math


class ERA5_Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(max_seq_length=num_tokens, encodings_dim=dim_model,
                                                     dropout=dropout_p)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

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

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
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
        self.register_buffer("pos_encoding_matrix", pos_encodings_matrix)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encodings_matrix[:token_embedding.size(0), :])
