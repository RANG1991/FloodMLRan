import torch


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.dropout = dropout
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bias=True,
        )
        self.head = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = torch.nn.Dropout(self.dropout)(output)
        return self.head(output[:, -1, :])
