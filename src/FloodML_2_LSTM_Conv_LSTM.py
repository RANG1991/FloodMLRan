import torch
from FloodML_Conv_LSTM import FloodML_Conv_LSTM


class TWO_LSTM_CONV_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, in_channels_cnn,
                 sequence_length_conv_lstm, image_width, image_height):
        super(TWO_LSTM_CONV_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.image_width = image_width
        self.image_height = image_height
        self.in_cnn_channels = in_channels_cnn
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.conv_lstm = FloodML_Conv_LSTM(self.in_cnn_channels, sequence_length_conv_lstm, self.image_width,
                                           self.image_height)
        self.sequence_length_conv_lstm = sequence_length_conv_lstm
        self.linear_states = torch.nn.Linear(self.hidden_dim,
                                             self.in_cnn_channels * self.image_width * self.image_height)
        self.head = torch.nn.Linear(
            in_features=self.image_width * self.image_height * self.in_cnn_channels,
            out_features=1)

    def forward(self, x_non_spatial, x_spatial):
        batch_size = x_non_spatial.shape[0]
        _, (h_n, c_n) = self.lstm(x_non_spatial)
        x_spatial = x_spatial.view(batch_size, self.sequence_length_conv_lstm, self.in_cnn_channels,
                                   self.image_width * self.image_height)
        x_spatial = x_spatial.view(batch_size, self.sequence_length_conv_lstm, self.in_cnn_channels, self.image_width,
                                   self.image_height)
        h_n = self.linear_states(h_n).reshape(batch_size, self.in_cnn_channels, self.image_width, self.image_height)
        c_n = self.linear_states(c_n).reshape(batch_size, self.in_cnn_channels, self.image_width, self.image_height)
        output = self.conv_lstm(x_spatial, c_n, h_n)
        output = output.reshape(batch_size,
                                self.sequence_length_conv_lstm,
                                self.in_cnn_channels *
                                self.image_width *
                                self.image_height)
        output = self.dropout(output)
        pred = self.head(output)
        return pred[:, -1, :]