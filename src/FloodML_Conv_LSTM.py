import torch
from torch import nn
from torch.nn import functional as F
import math


class FloodML_Conv_LSTM(nn.Module):

    def __init__(self, in_channels_cnn, sequence_length, image_width, image_height):
        super(FloodML_Conv_LSTM, self).__init__()
        self.in_channels_cnn = in_channels_cnn
        self.filter_size_conv = 3
        self.layers_list = []
        for i in range(sequence_length):
            conv = torch.nn.Conv2d(
                in_channels=self.in_channels_cnn * 2,
                out_channels=self.in_channels_cnn * 4,
                kernel_size=(self.filter_size_conv, self.filter_size_conv), padding="same")
            self.layers_list.append(conv)
        self.layers_list = nn.ModuleList(self.layers_list)
        # self.bn = nn.BatchNorm2d(in_channels_cnn)
        self.sequence_length = sequence_length
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, x, c_prev, h_prev):
        outputs = []
        for i in range(self.sequence_length):
            curr_x = x[:, i, :, :, :]
            combined = torch.cat([curr_x, h_prev], dim=1)
            gates = self.layers_list[i](combined)
            input_gate, forget_gate, cell_gate, output_gate = torch.split(gates, self.in_channels_cnn, dim=1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            output_gate = torch.sigmoid(output_gate)
            c_curr = forget_gate * c_prev + input_gate * cell_gate
            h_curr = output_gate * torch.tanh(c_curr)
            outputs.append(h_curr)
            c_prev = c_curr
            h_prev = h_curr
        return torch.cat(outputs, dim=1).contiguous()
