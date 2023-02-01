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
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_h_input_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_x_forget_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_h_forget_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_x_output_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_h_output_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_x_cell_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        self.conv_h_cell_gate = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=in_channels_cnn,
            kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same", bias=False)
        # self.bn = nn.BatchNorm2d(in_channels_cnn)
        self.sequence_length = sequence_length
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, x, c_prev, h_prev):
        outputs = []
        for i in range(self.sequence_length):
            curr_x = x[:, i, :, :, :]
            input_gate = torch.sigmoid(self.conv_x_input_gate(curr_x) + self.conv_h_input_gate(h_prev))
            # input_gate = self.bn(input_gate)
            forget_gate = torch.sigmoid(self.conv_x_forget_gate(curr_x) + self.conv_h_forget_gate(h_prev))
            # forget_gate = self.bn(forget_gate)
            output_gate = torch.sigmoid(self.conv_x_output_gate(curr_x) + self.conv_h_output_gate(h_prev))
            # output_gate = self.bn(output_gate)
            c_curr = forget_gate * c_prev + input_gate * (self.conv_x_cell_gate(curr_x) +
                                                          self.conv_h_cell_gate(h_prev))
            h_curr = output_gate * torch.tanh(c_curr)
            outputs.append(h_curr)
            c_prev = c_curr
            h_prev = h_curr
        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
