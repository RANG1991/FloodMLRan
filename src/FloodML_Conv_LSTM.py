import torch
import torch.nn.functional as F
import numpy as np
from FloodML_CNN_LSTM import CNN


class Conv_LSTM(torch.nn.Module):
    def __init__(self, hidden_dim_lstm, image_input_size, in_channels_cnn, sequence_length=270):
        super(Conv_LSTM, self).__init__()
        self.hidden_dim_lstm = hidden_dim_lstm
        self.sequence_length = sequence_length
        self.in_channels_cnn = in_channels_cnn
        self.initial_input_size = image_input_size
        self.filter_size_conv = 3
        self.filter_size_pool = 2
        self.stride_size_conv = 1
        self.stride_size_pool = 2
        # The operation list - the operation type to how many times we
        # are doing this operation (how much filter applications)
        self.op_list = [("conv", 16), ("pool", 1), ("conv", 32), ("pool", 1)]
        dims_fc = self.calc_dims_after_all_conv_op(self.initial_input_size, self.op_list)
        size_for_fc = dims_fc[0] * dims_fc[1] * 32
        self.size_for_fc = int(size_for_fc)
        self.lstm_cell = torch.nn.LSTMCell(
            input_size=self.size_for_fc, hidden_size=self.hidden_dim_lstm
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=16, kernel_size=(3, 3)
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3)
        )
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.4)
        self.head = torch.nn.Linear(in_features=self.hidden_dim_lstm, out_features=1)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_lstm = []
        batch_size = x.shape[0]
        c_i = torch.randn((batch_size, self.hidden_dim_lstm), device=device)
        h_i = torch.randn((batch_size, self.hidden_dim_lstm), device=device)
        for i in range(self.sequence_length):
            output_cnn = self.pool(
                F.relu(
                    self.conv1(
                        x[:, i, :, :].squeeze(1)
                    )
                )
            )
            output_cnn = self.pool(F.relu(self.conv2(output_cnn)))
            output_cnn = output_cnn.reshape(output_cnn.shape[0], -1)
            h_i, c_i = self.lstm_cell(output_cnn, (h_i, c_i))
            output_lstm.append(h_i)
        output_lstm = torch.stack(output_lstm, dim=0)
        output_final = self.dropout(output_lstm)
        return self.head(output_final.permute((1, 0, 2)))

    @staticmethod
    def calc_dims_after_filter(input_image_shape, filter_size, stride):
        if len(input_image_shape) != 2:
            raise Exception("The dimensions of the image are not 2, their "
                            "Should be exactly 2 - (width, height)")
        width = input_image_shape[0]
        height = input_image_shape[1]
        new_dims = np.zeros(2)
        new_dims[0] = ((width - filter_size) // stride) + 1
        new_dims[1] = ((height - filter_size) // stride) + 1
        return new_dims

    def calc_dims_after_all_conv_op(self, input_image_shape: [int], ops_list: [str]):
        image_dims = (input_image_shape[-2], input_image_shape[-1])
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
