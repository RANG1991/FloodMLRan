import torch
import torch.nn as nn
from CNN import CNN


class CNNLSTM(nn.Module):

    def __init__(self, lat, lon, input_size: int, hidden_size: int, num_channels: int,
                 dropout_rate: float = 0.0,
                 num_layers: int = 1, num_attributes: int = 0, image_input_size=(int,)):
        """
        Initialize model
       :param hidden_size: Number of hidden units/LSTM cells
       :param dropout_rate: Dropout rate of the last fully connected layer. Default 0.0
        """
        super(CNNLSTM, self).__init__()
        self.lat = lat
        self.lon = lon
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_channels = num_channels
        self.cnn = CNN(num_channels=num_channels, output_size_cnn=(input_size - num_attributes),
                       image_input_size=image_input_size)
        # create required layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=num_layers, bias=True,
                            batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Network.
        param x: Tensor of shape [batch size, seq length, num features] containing the input data for the LSTM network
        return: Tensor containing the network predictions
        """
        # x is of size:
        # 1. batch_size (some sample of all the training set)
        # 2. times_steps - the length of the sequence (for example 30,
        # if we are talking about one month)
        # 3. (num_channels*H_LAT*W_LON + 4)
        # the 4 is for the 4 static features
        # for example, currently, x.size() is - (64, 30, 840)
        batch_size, time_steps, _ = x.size()
        # getting the "image" part of the input
        # (removing the last 4 static features)
        image = x[:, :, :self.num_channels * self.lat * self.lon]
        # reshaping the image to 4 dimensional tensor of (batch_size, time_steps, num_channels, H_LAT*W_LON)
        image = image.view(batch_size, time_steps, self.num_channels, self.lat * self.lon)
        # reshaping the image to 5 dimensional tensor of (batch_size, time_steps, num_channels, H_LAT, W_LON)
        image = image.view(batch_size, time_steps, self.num_channels, self.lat, self.lon)
        # reshaping the image to 4 dimensional tensor of (batch_size * time_steps, num_channels, H_LAT, W_LON)
        c_in = image.view(batch_size * time_steps, self.num_channels, self.lat, self.lon)
        # CNN part
        c_out = self.cnn(c_in)
        # CNN output should be in the size of (input size - attributes_size)
        cnn_out = c_out.view(batch_size, time_steps, -1)
        # getting the "non-image" part of the input (last 4 attributes)
        # (removing the "image" part)
        a_in = x[:, :, self.num_channels * self.lat * self.lon:]
        r_in = torch.cat((cnn_out, a_in), 2)
        output, (h_n, c_n) = self.lstm(r_in)
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1, :, :]))
        return pred
