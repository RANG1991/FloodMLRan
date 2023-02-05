import torch
from FloodML_CNN_LSTM import CNN_LSTM


class TWO_LSTM_CNN_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, in_channels_cnn,
                 sequence_length_conv_lstm, image_width, image_height):
        super(TWO_LSTM_CNN_LSTM, self).__init__()
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
        self.cnn_lstm = CNN_LSTM(lat=image_width,
                                 lon=image_height,
                                 hidden_size=hidden_dim,
                                 num_channels=in_channels_cnn,
                                 dropout_rate=dropout,
                                 image_input_size=(image_width, image_height))
        self.sequence_length_cnn_lstm = sequence_length_conv_lstm
        self.linear_states = torch.nn.Linear(self.hidden_dim,
                                             self.in_cnn_channels * self.image_width * self.image_height)
        self.head = torch.nn.Linear(
            in_features=self.image_width * self.image_height * self.in_cnn_channels,
            out_features=1)

    def forward(self, x_non_spatial, x_spatial):
        batch_size = x_non_spatial.shape[0]
        _, (h_n, c_n) = self.lstm(x_non_spatial[:, :-self.sequence_length_cnn_lstm, :])
        output = self.cnn_lstm(x_non_spatial[:, -self.sequence_length_cnn_lstm:, :], x_spatial, h_n, c_n)
        output = output.reshape(batch_size,
                                self.sequence_length_cnn_lstm,
                                self.in_cnn_channels *
                                self.image_width *
                                self.image_height)
        output = self.dropout(output)
        pred = self.head(output)
        return pred[:, -1, :]
