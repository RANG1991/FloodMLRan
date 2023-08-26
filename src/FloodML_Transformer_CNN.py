import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from FloodML_Transformer_Encoder import PositionalEncoding
from FloodML_CNN_LSTM import CNN


class Transformer_CNN(nn.Module):

    def __init__(self,
                 embedding_size,
                 num_dynamic_attributes,
                 sequence_length,
                 intermediate_dim,
                 num_heads,
                 num_layers,
                 num_channels,
                 image_input_size,
                 dropout=0.0):
        """
        Initialize model
       :param hidden_size: Number of hidden units/LSTM cells
       :param dropout_rate: Dropout rate of the last fully connected layer. Default 0.0
        """
        super(Transformer_CNN, self).__init__()
        self.dropout = dropout
        self.num_channels = num_channels
        input_size = 4
        self.input_image_size = image_input_size
        self.embedding_size = embedding_size
        self.num_dynamic_attr = num_dynamic_attributes
        self.cnn = CNN(num_channels=num_channels, output_size_cnn=input_size,
                       image_input_size=image_input_size)
        self.fc_1 = nn.Linear(num_dynamic_attributes + self.embedding_size + input_size, intermediate_dim)
        self.positional_encoding = PositionalEncoding(intermediate_dim, sequence_length + 1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=intermediate_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_2 = nn.Linear(intermediate_dim, 1)
        # exponential_decay = torch.exp(torch.tensor([-1 * (sequence_length - i) / 25 for i in range(sequence_length)]))
        # exponential_decay = exponential_decay.unsqueeze(0).unsqueeze(-1).repeat(1, 1, intermediate_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.intermediate_dim = intermediate_dim
        # self.register_buffer('exponential_decay', exponential_decay)
        # self.fc_3 = nn.Linear(16, 1)

    def forward(self, x_non_spatial, x_spatial, memory) -> torch.Tensor:
        """
        Forward pass through the Network.
        param x: Tensor of shape [batch size, seq length, num features]
        containing the input data for the LSTM network
        return: Tensor containing the network predictions
        """
        batch_size, time_steps, _ = x_non_spatial.size()
        c_in = x_spatial.reshape(batch_size * time_steps, self.num_channels, self.input_image_size[0],
                                 self.input_image_size[1])
        # CNN.plot_as_image(c_in)
        # self.number_of_images_counter += (batch_size * time_steps)
        c_out = self.cnn(c_in)
        cnn_out = c_out.reshape(batch_size, time_steps, -1)
        r_in = torch.cat([cnn_out, x_non_spatial], dim=2)
        out_fc_1 = self.fc_1(r_in)
        out_pe = self.positional_encoding(out_fc_1)
        out_transformer = self.transformer_decoder(out_pe, memory)[:, 0, :]
        # out_decay = torch.sum(out_transformer * self.exponential_decay, dim=1)
        out_fc_2 = self.fc_2(self.dropout(out_transformer))
        return out_fc_2
