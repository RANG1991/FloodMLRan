import torch
from torch import nn
import math
from Transformer.Layers import EncoderLayer
from Transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from FloodML_CNN_LSTM import CNN


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class ERA5_Transformer(nn.Module):

    def __init__(self, out_features, in_features, sequence_length=270):
        super(ERA5_Transformer, self).__init__()
        assert out_features % 8 == 0
        self.encoder = Encoder(in_features_dim=in_features,
                               out_features_dim=out_features,
                               n_layers=6,
                               d_k=out_features // 8,
                               d_v=out_features // 8,
                               n_head=8,
                               d_model=out_features,
                               d_inner=out_features,
                               sequence_length=sequence_length)

        self.fc = nn.Linear(out_features, 1)

    def forward(self, x):
        enc_output, *_ = self.encoder_1(x)
        dec_output, *_ = self.decoder(enc_output)
        return self.fc(dec_output)


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, sequence_length=270):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(sequence_length, d_hid))

    def _get_sinusoid_encoding_table(self, sequence_length, d_hid):
        """ Sinusoid position encoding table """

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return torch.FloatTensor([position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)])

        sinusoid_table = torch.vstack([get_position_angle_vec(pos_i) for pos_i in range(sequence_length)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, in_features_dim, out_features_dim, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, sequence_length=270, scale_emb=False):

        super().__init__()

        self.src_embeddings = nn.Linear(in_features_dim, out_features_dim)
        self.position_enc = PositionalEncoding(out_features_dim, sequence_length=sequence_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask=None, return_attns=False):

        enc_slf_attn_list = []

        enc_output = self.src_embeddings(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
