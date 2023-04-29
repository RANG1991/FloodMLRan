from torch import nn
import torch


class Transformer_Encoder(nn.Module):

    def __init__(self, num_features, sequence_length, intermediate_dim, dropout):
        super(Transformer_Encoder, self).__init__()
        self.fc_1 = nn.Linear(num_features, intermediate_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermediate_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc_2 = nn.Linear(intermediate_dim, 1)
        exponential_decay = torch.exp(torch.tensor([-1 * (sequence_length - i) / 25 for i in range(sequence_length)]))
        exponential_decay = exponential_decay.unsqueeze(0).unsqueeze(-1).repeat(1, 1, intermediate_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('exponential_decay', exponential_decay)

    def forward(self, x):
        out_fc_1 = self.fc_1(x)
        out_transformer = self.transformer_encoder(out_fc_1)
        out_decay = torch.sum(out_transformer * self.exponential_decay, dim=1)
        out_fc_2 = self.dropout(self.fc_2(out_decay))
        return out_fc_2
