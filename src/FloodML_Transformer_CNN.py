import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from FloodML_Transformer_Encoder import PositionalEncoding
from FloodML_CNN_LSTM import CNN


class Transformer_CNN(nn.Module):

    def __init__(self,
                 num_static_attributes,
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
        self.embedding_size = 10
        self.input_image_size = image_input_size
        self.num_static_attr = num_static_attributes
        self.num_dynamic_attr = num_dynamic_attributes
        self.cnn = CNN(num_channels=num_channels, output_size_cnn=input_size,
                       image_input_size=image_input_size)
        self.fc_1 = nn.Linear(num_dynamic_attributes + self.embedding_size + input_size, intermediate_dim)
        self.positional_encoding = PositionalEncoding(intermediate_dim, sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermediate_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_2 = nn.Linear(intermediate_dim, 1)
        exponential_decay = torch.exp(torch.tensor([-1 * (sequence_length - i) / 25 for i in range(sequence_length)]))
        exponential_decay = exponential_decay.unsqueeze(0).unsqueeze(-1).repeat(1, 1, intermediate_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.intermediate_dim = intermediate_dim
        self.register_buffer('exponential_decay', exponential_decay)
        # self.fc_3 = nn.Linear(16, 1)
        self.embedding = torch.nn.Linear(in_features=self.num_static_attr, out_features=self.embedding_size)

    def forward(self, x_non_spatial, x_spatial) -> torch.Tensor:
        """
        Forward pass through the Network.
        param x: Tensor of shape [batch size, seq length, num features]
        containing the input data for the LSTM network
        return: Tensor containing the network predictions
        """
        x_d = x_non_spatial[:, :, :self.num_dynamic_attr]
        x_s = x_non_spatial[:, :, -self.num_static_attr:]
        x_s = self.embedding(x_s)
        x_non_spatial = torch.cat([x_d, x_s], axis=2)
        batch_size, time_steps, _ = x_non_spatial.size()
        c_in = x_spatial.reshape(batch_size * time_steps, self.num_channels, self.input_image_size[0],
                                 self.input_image_size[1])
        # CNN.plot_as_image(c_in)
        # self.number_of_images_counter += (batch_size * time_steps)
        # CNN part
        c_out = self.cnn(c_in)
        # CNN output should be in the size of (input size - attributes_size)
        cnn_out = c_out.reshape(batch_size, time_steps, -1)
        # getting the "non-image" part of the input (last 4 attributes)
        # (removing the "image" part)
        r_in = torch.cat([cnn_out, x_non_spatial], dim=2)
        out_fc_1 = self.fc_1(r_in)
        out_pe = self.positional_encoding(out_fc_1)
        out_transformer = self.transformer_encoder(out_pe)[:, 0, :]
        # out_decay = torch.sum(out_transformer * self.exponential_decay, dim=1)
        # out_fc_2 = self.fc_2(self.dropout(out_transformer))
        out_fc_2 = self.fc_2(self.dropout(out_transformer))
        return out_fc_2
