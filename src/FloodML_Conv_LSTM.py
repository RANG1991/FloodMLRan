import torch
from torch import nn
from torch.nn import functional as F
import math


class FloodML_Conv_LSTM(nn.Module):

    def __init__(self, in_channels_cnn, sequence_length, image_width, image_height):
        super(FloodML_Conv_LSTM, self).__init__()
        self.filter_size_conv = 3
        self.conv_x_input_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_h_input_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_x_forget_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_h_forget_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_x_output_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_h_output_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_x_cell_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.conv_h_cell_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
        self.sequence_length = sequence_length
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, x, c_curr, h_curr):
        for i in range(self.sequence_length):
            c_prev = c_curr
            h_prev = h_curr
            curr_x = x[:, i, :, :, :]
            input_gate = nn.Sigmoid()(self.conv_x_input_gate(curr_x) + self.conv_h_input_gate(h_prev))
            forget_gate = nn.Sigmoid()(self.conv_x_forget_gate(curr_x) + self.conv_h_forget_gate(h_prev))
            output_gate = nn.Sigmoid()(self.conv_x_output_gate(curr_x) + self.conv_h_output_gate(h_prev))
            c_curr = forget_gate * c_prev + input_gate * nn.Tanh()(self.conv_x_cell_gate(curr_x) +
                                                                   self.conv_h_cell_gate(h_prev))
            h_curr = output_gate * torch.nn.Tanh()(c_curr)
        return h_curr
