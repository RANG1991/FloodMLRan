import torch
from torch import nn


class Transformer_Seq2Seq(nn.Module):

    def __init__(self, in_features):
        super(Transformer_Seq2Seq, self).__init__()
        self.d_model = 8
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=1, batch_first=True)
        self.linear_x = nn.Linear(in_features, self.d_model)
        self.linear_y = nn.Linear(1, self.d_model)
        self.linear_output = nn.Linear(self.d_model, 1)

    def forward(self, x, y):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[1], device="cuda")
        x = self.linear_x(x)
        y = self.linear_y(y.unsqueeze(-1))
        output = self.transformer(x, y, tgt_mask=tgt_mask)
        output = self.linear_output(output)
        return output
