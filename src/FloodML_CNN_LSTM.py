import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# convolution has 3 parameters:
# 1. filter size - width X height (all filters are squared, so it's actually a single number)
# 2. stride size (how many pixels the filter is "jumping" at each iteration through the input image)
# 3. number of filters (also called channels as this is the number of output channels - each
# filter is producing *one* channel)
class CNN(nn.Module):
    """
    It's important to note that one convolution operation on a single part of the image
    only generates a single number!!! (single scalar)
    """

    def __init__(self, num_channels: int, output_size_cnn, image_input_size):
        super(CNN, self).__init__()
        self.initial_num_channels = num_channels
        self.initial_input_size = image_input_size
        self.filter_size_conv = 3
        self.filter_size_pool = 2
        self.stride_size_conv = 1
        self.stride_size_pool = 2
        # The operation list - the operation type to how many times we
        # are doing this operation (how much filter applications)
        self.op_list = [("conv", 16), ("pool", 1), ("conv", 32), ("pool", 1)]
        dims_fc = self.calc_dims_after_all_conv_op(self.initial_input_size, self.op_list)
        # declaring convolution with 3 by 3 filter matrix
        # the input is: 1 or 2 or 3 channels (depending on the number of channels) -
        # this is the number of channel of the input image
        # the output is: 16 channels (the number of filters we apply)
        self.conv1 = nn.Conv2d(self.initial_num_channels, 16, 3)
        # declaring max pooling to the output matrix of the previous stage
        # (getting the max of the pool of 2 by 2 matrix going
        # over the large matrix from previous stage)
        self.pool = nn.MaxPool2d(2, 2)
        # declaring convolution with 3 by 3 filter matrix
        # the input is: 16 channels
        # the output is: 32 channels (the number of filters we apply)
        self.conv2 = nn.Conv2d(16, 32, 3)
        size_for_fc = dims_fc[0] * dims_fc[1] * 32
        self.size_for_fc = int(size_for_fc)
        self.fc1 = nn.Linear(self.size_for_fc, 120)
        self.fc2 = nn.Linear(120, output_size_cnn)
        self.dropout1 = nn.Dropout()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.size_for_fc)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    @staticmethod
    def calc_dims_after_filter(input_image_shape, filter_size, stride):
        if len(input_image_shape) != 2:
            raise Exception("The dimensions of the image are not 2, their "
                            "Should be exactly 2 - (width, height)")
        width = input_image_shape[0]
        height = input_image_shape[1]
        new_dims = np.zeros(2)
        new_dims[0] = ((width - filter_size) / stride) + 1
        new_dims[1] = ((height - filter_size) / stride) + 1
        return new_dims

    def calc_dims_after_all_conv_op(self, input_image_shape: [int], ops_list: [str]):
        image_dims = (input_image_shape[1], input_image_shape[2])
        for op in ops_list:
            if op[0] == "conv":
                image_dims = CNN.calc_dims_after_filter(image_dims,
                                                        self.filter_size_conv,
                                                        self.stride_size_conv)
            elif op[0] == "pool":
                image_dims = CNN.calc_dims_after_filter(image_dims,
                                                        self.filter_size_pool,
                                                        self.stride_size_pool)
        return image_dims


class CNN_LSTM(nn.Module):

    def __init__(self, lat, lon,
                 input_size: int,
                 hidden_size: int,
                 num_channels: int,
                 dropout_rate: float = 0.0,
                 num_layers: int = 1,
                 num_attributes: int = 0,
                 image_input_size=(int,)):
        """
        Initialize model
       :param hidden_size: Number of hidden units/LSTM cells
       :param dropout_rate: Dropout rate of the last fully connected layer. Default 0.0
        """
        super(CNN_LSTM, self).__init__()
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
        param x: Tensor of shape [batch size, seq length, num features]
        containing the input data for the LSTM network
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
