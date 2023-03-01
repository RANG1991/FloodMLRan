import torch


class LSTM(torch.nn.Module):
    def __init__(self, hidden_dim, dropout, num_dynamic_attr, num_static_attr):
        super(LSTM, self).__init__()
        self.input_dim = (num_dynamic_attr + num_static_attr)
        self.embedding_size = 30
        self.num_dynamic_attr = num_dynamic_attr
        self.num_static_attr = num_static_attr
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size=(self.num_dynamic_attr + self.embedding_size),
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.head = torch.nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.embedding = torch.nn.Linear(in_features=self.num_static_attr, out_features=self.embedding_size)

    def forward(self, x):
        x_d = x[:, :, :self.num_dynamic_attr]
        x_s = x[:, :, -self.num_static_attr:]
        x_s = self.embedding(x_s)
        x = torch.cat([x_d, x_s], axis=-1)
        output, (h_n, c_n) = self.lstm(x)
        output = self.dropout(output)
        pred = self.head(output)
        return pred[:, -1, :]
