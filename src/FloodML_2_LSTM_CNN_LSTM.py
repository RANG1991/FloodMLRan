import torch
from FloodML_CNN_LSTM import CNN_LSTM


class TWO_LSTM_CNN_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout,
                 in_cnn_channels, sequence_length_conv_lstm,
                 image_width, image_height,
                 num_dynamic_attributes, num_static_attributes):
        super(TWO_LSTM_CNN_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.image_width = image_width
        self.image_height = image_height
        self.in_cnn_channels = in_cnn_channels
        self.num_static_attr = num_static_attributes
        self.num_dynamic_attr = num_dynamic_attributes
        self.sequence_length_cnn_lstm = sequence_length_conv_lstm
        self.embedding_size = 10
        self.lstm = torch.nn.LSTM(
            input_size=self.num_dynamic_attr + self.embedding_size,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.cnn_lstm = CNN_LSTM(lat=self.image_width,
                                 lon=self.image_height,
                                 hidden_size=self.hidden_dim,
                                 num_channels=self.in_cnn_channels,
                                 dropout_rate=dropout,
                                 image_input_size=(self.image_width, self.image_height),
                                 num_attributes=(self.num_dynamic_attr + self.embedding_size))
        self.embedding = torch.nn.Linear(in_features=self.num_static_attr, out_features=self.embedding_size)

    def forward(self, x_non_spatial, x_spatial):
        x_d = x_non_spatial[:, :, :self.num_dynamic_attr]
        x_s = x_non_spatial[:, :, -self.num_static_attr:]
        x_s = self.embedding(x_s)
        x_non_spatial = torch.cat([x_d, x_s], axis=-1)
        _, (h_n, c_n) = self.lstm(x_non_spatial[:, :-self.sequence_length_cnn_lstm, :])
        output = self.cnn_lstm(x_non_spatial[:, -self.sequence_length_cnn_lstm:, :], x_spatial, h_n, c_n)
        return output
