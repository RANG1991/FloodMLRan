import torch


class TWO_LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(TWO_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.head = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.dropout(output)
        return self.head(output)