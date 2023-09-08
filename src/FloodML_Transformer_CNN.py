import torch.nn as nn
import torch
from FloodML_Transformer_Encoder import PositionalEncoding
from FloodML_CNN_LSTM import CNN
from Transformer.layers import multi_head_attention, layer_norm, position_wise_feed_forward


class EncoderCrossAttentionLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderCrossAttentionLayer, self).__init__()
        self.self_attention = multi_head_attention.MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = layer_norm.LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.enc_dec_attention = multi_head_attention.MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = layer_norm.LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = position_wise_feed_forward.PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden,
                                                                      drop_prob=drop_prob)
        self.norm3 = layer_norm.LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, enc_1, enc_2):
        _x = enc_1
        x = self.self_attention(q=enc_1, k=enc_1, v=enc_1)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.enc_dec_attention(q=x, k=enc_2, v=enc_2)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class EncoderCrossAttention(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([EncoderCrossAttentionLayer(d_model=d_model,
                                                                ffn_hidden=ffn_hidden,
                                                                n_head=n_head,
                                                                drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, trg, enc_src):
        for layer in self.layers:
            trg = layer(trg, enc_src)
        return trg


class CNN_Transformer(nn.Module):

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
        super(CNN_Transformer, self).__init__()
        self.dropout = dropout
        self.num_channels = num_channels
        input_size = 4
        self.input_image_size = image_input_size
        self.embedding_size = embedding_size
        self.num_dynamic_attr = num_dynamic_attributes
        self.cnn = CNN(num_channels=num_channels, output_size_cnn=input_size,
                       image_input_size=image_input_size)
        self.fc_1 = nn.Linear(intermediate_dim + input_size, intermediate_dim)
        self.positional_encoding = PositionalEncoding(intermediate_dim, sequence_length)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=intermediate_dim, nhead=num_heads, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.cross_attention_encoder = EncoderCrossAttention(n_head=num_heads,
                                                             n_layers=num_layers,
                                                             d_model=intermediate_dim,
                                                             ffn_hidden=2048,
                                                             drop_prob=0.1)
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
        out_transformer = self.cross_attention_encoder(out_pe, memory)
        # out_decay = torch.sum(out_transformer * self.exponential_decay, dim=1)
        out_fc_2 = self.fc_2(self.dropout(out_transformer[:, 0, :]))
        return out_fc_2
