import torch
import torch.nn.functional as F
import numpy as np


class Conv_LSTM(torch.nn.Module):
    def __init__(
            self, input_dim_lstm, hidden_dim_lstm, sequence_length=270, in_channels_cnn=3
    ):
        super(Conv_LSTM, self).__init__()
        self.input_dim_lstm = input_dim_lstm
        self.hidden_dim_lstm = hidden_dim_lstm
        self.sequence_length = sequence_length
        self.in_channels_cnn = in_channels_cnn
        self.lstm_cell = torch.nn.LSTMCell(
            input_size=32, hidden_size=self.hidden_dim_lstm
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels_cnn, out_channels=16, kernel_size=(3, 3)
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3)
        )
        self.pool = torch.nn.MaxPool2d(2, 2)
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
                        x[:, i, :, :].squeeze(1).unsqueeze(3).permute((0, 3, 1, 2))
                    )
                )
            )
            output_cnn = self.pool(F.relu(self.conv2(output_cnn)))
            output_cnn = output_cnn.reshape(output_cnn.shape[0], -1)
            h_i, c_i = self.lstm_cell(output_cnn, (h_i, c_i))
            output_lstm.append(h_i)
        output_lstm = torch.stack(output_lstm, dim=0)
        output_final = torch.nn.Dropout(0.4)(output_lstm)
        return self.head(output_final.permute((1, 0, 2)))

    @staticmethod
    def calc_dims_after_filter(input_image_shape, filter_size, stride):
        if len(input_image_shape) != 2:
            raise Exception(
                "The dimensions of the image are not 2, their "
                "Should be exactly 2 - (width, height)"
            )
        width = input_image_shape[0]
        height = input_image_shape[1]
        new_dims = np.zeros(2)
        new_dims[0] = ((width - filter_size) / stride) + 1
        new_dims[1] = ((height - filter_size) / stride) + 1
        return new_dims
