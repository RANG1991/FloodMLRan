import torch
from torch import nn
import math
from Transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from transformers import ViTConfig, ViTModel


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, sequence_length=270):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(sequence_length, d_hid))

    def _get_sinusoid_encoding_table(self, sequence_length, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return torch.FloatTensor([position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)])

        sinusoid_table = torch.vstack([get_position_angle_vec(pos_i) for pos_i in range(sequence_length)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Transformer_CNN(nn.Module):
    def __init__(self, sequence_length_spatial, num_dynamic_attr, num_static_attr, embedding_size, image_size, d_model):
        super(Transformer_CNN, self).__init__()
        self.vit = VIT(d_model=d_model, image_size=image_size)
        self.fc_1 = nn.Linear((num_dynamic_attr + embedding_size), d_model)
        self.encoder = Encoder(n_layers=6, n_head=8, d_model=d_model, d_inner=d_model)
        self.fc_2 = nn.Linear(d_model, 1)
        self.num_static_attr = num_static_attr
        self.num_dynamic_attr = num_dynamic_attr
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.embedding = torch.nn.Linear(in_features=self.num_static_attr, out_features=self.embedding_size)

    def forward(self, non_spatial_input, spatial_input):
        x_d = non_spatial_input[:, :, :self.num_dynamic_attr]
        x_s = non_spatial_input[:, :, -self.num_static_attr:]
        x_s = self.embedding(x_s)
        non_spatial_input = torch.cat([x_d, x_s], axis=-1)
        batch_size, seq_length, _ = spatial_input.shape
        spatial_input = spatial_input.reshape(batch_size * seq_length, 1, self.image_size, self.image_size)
        spatial_input = self.vit(spatial_input).last_hidden_state[:, 0, :]
        spatial_input = spatial_input.reshape(batch_size, seq_length, -1)
        non_spatial_input = self.fc_1(non_spatial_input)
        enc_out = self.encoder(non_spatial_input, spatial_input)
        return self.fc_2(enc_out)[:, -1, :]


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input_non_spatial, enc_input_spatial, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input_spatial, enc_input_non_spatial, enc_input_non_spatial,
                                                 mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, n_layers, n_head, d_model, d_inner, dropout=0.1, sequence_length=270, scale_emb=False):
        super(Encoder, self).__init__()
        # the nn.Embedding creates a lookup table that can retrieve a word embedding using an index in the table
        # d_word_vec should be equal to d_model (?) - for example 512
        self.position_enc = PositionalEncoding(d_model, sequence_length=sequence_length)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_model // n_head, d_model // n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, enc_input_non_spatial, enc_input_spatial):
        if self.scale_emb:
            enc_input_non_spatial *= self.d_model ** 0.5
        enc_input_non_spatial = self.dropout(self.position_enc(enc_input_non_spatial))
        enc_input_non_spatial = self.layer_norm(enc_input_non_spatial)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_input_non_spatial, enc_input_spatial)
        return enc_output


class VIT(nn.Module):
    def __init__(self, d_model, image_size):
        super(VIT, self).__init__()
        configuration = ViTConfig(num_hidden_layers=6, num_attention_heads=8, image_size=image_size,
                                  hidden_size=d_model,
                                  num_channels=1)
        self.vit_model = ViTModel(configuration)

    def forward(self, spatial_input):
        return self.vit_model(spatial_input)
