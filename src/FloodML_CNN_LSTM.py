import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


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

    def __init__(self, num_channels, output_size_cnn, image_input_size):
        super(CNN, self).__init__()
        self.initial_num_channels = num_channels
        self.initial_input_size = image_input_size
        self.channels_out_conv_1 = 16
        self.channels_out_conv_2 = 32
        self.channels_out_conv_3 = 64
        self.channels_out_conv_4 = 128
        self.filter_size_conv = 2
        self.stride_size_conv = 1
        self.filter_size_pool = 2
        self.stride_size_pool = self.filter_size_pool
        self.cnn_layers = nn.ModuleList([
            torch.nn.Conv2d(in_channels=self.initial_num_channels, out_channels=self.channels_out_conv_1,
                            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="valid"),
            torch.nn.BatchNorm2d(self.channels_out_conv_1),
            nn.ReLU(),
            torch.nn.MaxPool2d(self.filter_size_pool, stride=self.stride_size_pool),
            torch.nn.Conv2d(in_channels=self.channels_out_conv_1, out_channels=self.channels_out_conv_2,
                            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="valid"),
            torch.nn.BatchNorm2d(self.channels_out_conv_2),
            nn.ReLU(),
            torch.nn.MaxPool2d(self.filter_size_pool, stride=self.stride_size_pool),
            # torch.nn.Conv2d(in_channels=self.channels_out_conv_2, out_channels=self.channels_out_conv_3,
            #                 kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="valid"),
            # torch.nn.BatchNorm2d(self.channels_out_conv_3),
            # nn.ReLU(),
            # torch.nn.AvgPool2d(self.filter_size_pool, stride=self.stride_size_pool),
            # torch.nn.Conv2d(in_channels=self.channels_out_conv_3, out_channels=self.channels_out_conv_4,
            #                 kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="valid"),
            # torch.nn.BatchNorm2d(self.channels_out_conv_4),
            # nn.ReLU()
        ])
        self.size_for_fc = int(self.calc_dims_after_all_conv_op(self.initial_input_size, self.initial_num_channels))
        self.fc = nn.Linear(self.size_for_fc, output_size_cnn)
        self.relu_for_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        x = self.dropout(x)
        # self.plot_as_image(x)
        x_after_cnn = x.view(-1, self.size_for_fc)
        x_after_fc = self.fc(x_after_cnn)
        return x_after_fc

    @staticmethod
    def plot_as_image(x):
        import matplotlib.pyplot as plt
        x_to_plot = x.permute((0, 2, 3, 1)).sum(axis=0).sum(axis=-1).squeeze().cpu().detach().numpy()
        plt.imsave("check_x_after_cnn.png", x_to_plot)

    @staticmethod
    def calc_dims_after_filter(input_image_shape, filter_size, stride):
        if len(input_image_shape) != 2:
            raise Exception("The dimensions of the image are not 2, their "
                            "Should be exactly 2 - (width, height)")
        width = input_image_shape[0]
        height = input_image_shape[1]
        new_dims = np.zeros(2)
        if type(stride) is tuple:
            new_dims[0] = ((width - filter_size) // stride[0]) + 1
            new_dims[1] = ((height - filter_size) // stride[1]) + 1
        else:
            new_dims[0] = ((width - filter_size) // stride) + 1
            new_dims[1] = ((height - filter_size) // stride) + 1
        return new_dims

    def calc_dims_after_all_conv_op(self, input_image_shape: [int], num_in_channels):
        image_dims = (input_image_shape[-2], input_image_shape[-1])
        num_out_channels = num_in_channels
        for op in self.cnn_layers:
            if "conv" in str(op).lower():
                if op.padding == "valid":
                    image_dims = CNN.calc_dims_after_filter(image_dims,
                                                            self.filter_size_conv,
                                                            self.stride_size_conv)
                num_out_channels = op.out_channels
            elif "pool" in str(op).lower():
                image_dims = CNN.calc_dims_after_filter(image_dims,
                                                        self.filter_size_pool,
                                                        self.stride_size_pool)
        return image_dims[0] * image_dims[1] * num_out_channels


class CNN_LSTM(nn.Module):

    def __init__(self,
                 image_width,
                 image_height,
                 hidden_size: int,
                 num_channels: int,
                 dropout_rate: float = 0.0,
                 num_layers: int = 1,
                 num_attributes=0,
                 image_input_size=(int,)):
        """
        Initialize model
       :param hidden_size: Number of hidden units/LSTM cells
       :param dropout_rate: Dropout rate of the last fully connected layer. Default 0.0
        """
        super(CNN_LSTM, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_channels = num_channels
        input_size = 4
        self.cnn = CNN(num_channels=num_channels, output_size_cnn=input_size,
                       image_input_size=image_input_size)
        self.lstm = nn.LSTM(input_size=input_size + num_attributes, hidden_size=self.hidden_size,
                            num_layers=num_layers, bias=True,
                            batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.number_of_images_counter = 0

    def forward(self, x_non_spatial, x_spatial, h_n, c_n) -> torch.Tensor:
        """
        Forward pass through the Network.
        param x: Tensor of shape [batch size, seq length, num features]
        containing the input data for the LSTM network
        return: Tensor containing the network predictions
        """
        batch_size, time_steps, _ = x_non_spatial.size()
        c_in = x_spatial.reshape(batch_size * time_steps, self.num_channels, self.image_width, self.image_height)
        # c_in = torch.zeros_like(c_in)
        # CNN.plot_as_image(c_in)
        # self.number_of_images_counter += (batch_size * time_steps)
        # CNN part
        c_out = self.cnn(c_in)
        # CNN output should be in the size of (input size - attributes_size)
        cnn_out = c_out.reshape(batch_size, time_steps, -1)
        # getting the "non-image" part of the input (last 4 attributes)
        # (removing the "image" part)
        r_in = torch.cat([cnn_out, x_non_spatial], dim=2)
        output, (h_n, c_n) = self.lstm(r_in, (h_n, c_n))
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(output))
        return pred[:, -1, :]
