import torch
import torch.nn.functional as F


class ERA5_CNN_LSTM(torch.nn.Module):

    def __init__(self, input_dim_lstm, hidden_dim_lstm, sequence_length=270, in_channels_cnn=3):
        super(ERA5_CNN_LSTM, self).__init__()
        self.input_dim_lstm = input_dim_lstm
        self.hidden_dim_lstm = hidden_dim_lstm
        self.sequence_length = sequence_length
        self.in_channels_cnn = in_channels_cnn
        self.lstm_cell = torch.nn.LSTM(input_size=self.input_dim_lstm, hidden_size=self.hidden_dim_lstm)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels_cnn, out_channels=16, ketnel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, ketnel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.head = torch.nn.Linear(in_features=self.hidden_dim_lstm, out_features=1)

    def forward(self, x):
        for i in range(self.sequence_length):
            x = self.pool(F.relu(self.conv1(x[:, i, :])))
            x = self.pool(F.relu(self.conv2(x)))
            output, h_i, c_i = self.lstm_cell()
        output = torch.nn.Dropout(0.4)(output)
        return self.head(output[:, -1, :])

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
