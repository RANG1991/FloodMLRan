import torch
from FloodML_Conv_LSTM import FloodML_Conv_LSTM


class TWO_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, in_channels_cnn,
                 sequence_length_conv_lstm, image_width, image_height):
        super(TWO_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.head = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.conv_lstm = FloodML_Conv_LSTM(in_channels_cnn, sequence_length_conv_lstm, image_width, image_height)
        self.image_width = image_width
        self.image_height = image_height
        self.sequence_length_conv_lstm = sequence_length_conv_lstm
        self.linear = torch.nn.Linear(self.image_width * self.image_height, self.hidden_dim)

    def forward(self, x):
        batch_size, time_steps, _ = x.size()
        x_spatial = x[:, :, -self.num_channels * self.image_width * self.image_height:]
        x_spatial = x_spatial.view(batch_size, time_steps, self.num_channels, self.image_width * self.image_height)
        x_spatial = x_spatial.view(batch_size, time_steps, self.num_channels, self.image_width, self.image_height)
        x_spatial = x_spatial[:, :self.sequence_length_conv_lstm, :, :]
        c, h = self.conv_lstm(x_spatial)
        x_non_spatial = x[:, :, :-self.num_channels * self.image_width * self.image_height]
        output, (h_n, c_n) = self.lstm(x_non_spatial, self.linear(h), self.linear(c))
        output = self.dropout(output)
        pred = self.head(output)
        return pred[:, -1, :]
