import torch
from torch import nn
from torch.nn import functional as F
import math


class FloodML_Conv_LSTM(nn.module):

    def __init__(self, in_channels_cnn, sequence_length, image_width, image_height):
        self.filter_size_convolutions = 3
        self.conv_x_input_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_h_input_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_x_forget_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_h_forget_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_x_output_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_h_output_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_x_cell_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.conv_h_cell_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=self.in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv))
        self.sequence_length = sequence_length
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, x):
        c_curr = torch.zeros_like(x[:, 0, :, :].squeeze())
        h_curr = torch.zeros_like(x[:, 0, :, :].squeeze())
        for i in range(self.sequence_length):
            c_prev = c_curr
            h_prev = h_curr
            curr_x = x[:, i, :, :].squeeze()
            input_gate = nn.Sigmoid(self.conv_x_input_gate(curr_x) + self.conv_h_input_gate(h_prev))
            forget_gate = nn.Sigmoid(self.conv_x_forget_gate(curr_x) + self.conv_h_forget_gate(h_prev))
            output_gate = nn.Sigmoid(self.conv_x_output_gate(curr_x) + self.conv_h_output_gate(h_prev))
            c_curr = forget_gate * c_prev + input_gate * nn.Tanh(self.conv_x_cell_gate(curr_x) +
                                                                 self.conv_h_cell_gate(h_prev))
            h_curr = output_gate * torch.nn.Tanh(c_curr)
            max_width_right = int(self.image_width / 2)
            max_width_left = math.ceil(self.image_width / 2)
            max_height_right = int(self.image_height / 2)
            max_height_left = math.ceil(self.image_height / 2)
            pad = (max_width_left - int(c_curr.shape[1] / 2), max_width_right - math.ceil(c_curr.shape[1] / 2),
                   max_height_left - int(c_curr.shape[2] / 2), max_height_right - math.ceil(c_curr.shape[2] / 2))
            c_curr = F.pad(c_curr, pad, "constant", 0)
            h_curr = F.pad(h_curr, pad, "constant", 0)
        return c_curr, h_curr
