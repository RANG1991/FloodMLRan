import torch


class LSTM_ERA5(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, sequence_length=270):
        super(LSTM_ERA5, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                  bias=True, batch_first=True)
        self.fc = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n)
