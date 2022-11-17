import torch


class ERA5_LSTM(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(ERA5_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.head = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = torch.nn.Dropout(0.4)(output)
        return self.head(output[:, -1, :])
