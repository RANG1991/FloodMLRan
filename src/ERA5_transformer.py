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

    def forward(self, src, tgt):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
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
