from torch import nn


class Transformer_Encoder(nn.Module):

    def __init__(self, num_features):
        super(Transformer_Encoder, self).__init__()
        self.fc_1 = nn.Linear(num_features, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc_2 = nn.Linear(512, 1)

    def forward(self, x):
        out_fc_1 = self.fc_1(x)
        out_transformer = self.transformer_encoder(out_fc_1)
        out_fc_2 = self.fc_2(out_transformer[:, -1, :])
        return out_fc_2
