import torch
from FloodML_Conv_LSTM import FloodML_Conv_LSTM


class TWO_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, in_channels_cnn,
                 sequence_length_conv_lstm, image_width, image_height, num_channels):
        super(TWO_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.image_width = image_width
        self.image_height = image_height
        self.num_channels = num_channels
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.head = torch.nn.Linear(in_features=self.num_channels * self.image_width * self.image_height,
                                    out_features=1)
        self.conv_lstm = FloodML_Conv_LSTM(in_channels_cnn, sequence_length_conv_lstm, self.image_width,
                                           self.image_height)
        self.sequence_length_conv_lstm = sequence_length_conv_lstm
        self.linear_states = torch.nn.Linear(self.hidden_dim, self.num_channels * self.image_width * self.image_height)

    def forward(self, x):
        batch_size, time_steps, _ = x.size()
        x_non_spatial = x[:, :-self.sequence_length_conv_lstm, :self.input_dim]
        _, (h_n, c_n) = self.lstm(x_non_spatial)
        x_spatial = x[:, -self.sequence_length_conv_lstm:, :]
        x_spatial = x_spatial.view(batch_size, self.sequence_length_conv_lstm, self.num_channels,
                                   self.image_width * self.image_height)
        x_spatial = x_spatial.view(batch_size, self.sequence_length_conv_lstm, self.num_channels, self.image_width,
                                   self.image_height)
        h_n = self.linear_states(h_n).reshape(batch_size, 1, self.num_channels, self.image_width, self.image_height)
        c_n = self.linear_states(c_n).reshape(batch_size, 1, self.num_channels, self.image_width, self.image_height)
        output = self.conv_lstm(x_spatial, h_n, c_n)
        output = self.dropout(output)
        pred = self.head(output)
        return pred
