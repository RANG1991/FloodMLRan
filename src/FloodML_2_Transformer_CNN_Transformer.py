import torch
from FloodML_Transformer_CNN import Transformer_CNN
from torch import nn


class TWO_Transformer_CNN_Transformer(torch.nn.Module):
    def __init__(self,
                 dropout,
                 in_cnn_channels,
                 sequence_length_cnn_transformer,
                 image_width,
                 image_height,
                 num_dynamic_attributes,
                 num_static_attributes,
                 intermediate_dim_transformer,
                 num_heads_transformer,
                 num_layers_transformer):
        super(TWO_Transformer_CNN_Transformer, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.in_cnn_channels = in_cnn_channels
        self.num_static_attr = num_static_attributes
        self.num_dynamic_attr = num_dynamic_attributes
        self.sequence_length_cnn_transformer = sequence_length_cnn_transformer
        self.intermediate_dim_transformer = intermediate_dim_transformer
        self.num_heads_transformer = num_heads_transformer
        self.num_layers_transformer = num_layers_transformer
        self.embedding_size = 10
        self.dropout = dropout
        self.lstm = torch.nn.LSTM(
            input_size=self.num_dynamic_attr + self.embedding_size,
            hidden_size=self.hidden_dim, bias=True,
            batch_first=True
        )
        self.transformer_cnn = Transformer_CNN(image_input_size=(self.image_width, self.image_height),
                                               embedding_size=self.embedding_size,
                                               num_dynamic_attributes=self.num_dynamic_attr,
                                               sequence_length=self.sequence_length_cnn_transformer,
                                               intermediate_dim=self.intermediate_dim_transformer,
                                               dropout=self.dropout,
                                               num_heads=self.num_heads_transformer,
                                               num_layers=self.num_layers_transformer,
                                               num_channels=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.intermediate_dim_transformer,
                                                   nhead=self.num_heads_transformer,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers_transformer)
        self.embedding = torch.nn.Linear(in_features=self.num_static_attr, out_features=self.embedding_size)

    def forward(self, x_non_spatial, x_spatial):
        x_d = x_non_spatial[:, :, :self.num_dynamic_attr]
        x_s = x_non_spatial[:, :, -self.num_static_attr:]
        x_s = self.embedding(x_s)
        x_non_spatial = torch.cat([x_d, x_s], dim=-1)
        memory = self.transformer_encoder(x_non_spatial[:, :-self.sequence_length_cnn_transformer, :])
        if self.use_only_precip_feature:
            output = self.transformer_cnn(x_non_spatial[:, -self.sequence_length_cnn_transformer:, 1:], x_spatial,
                                          memory)
        else:
            output = self.transformer_cnn(x_non_spatial[:, -self.sequence_length_cnn_transformer:, :], x_spatial,
                                          memory)
        return output
