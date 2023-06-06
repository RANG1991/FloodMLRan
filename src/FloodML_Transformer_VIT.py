import torch
from torch import nn
import math
from Transformer.layers.multi_head_attention import MultiHeadAttention
from Transformer.layers.position_wise_feed_forward import PositionwiseFeedForward
from transformers import ViTConfig, ViTModel
from FloodML_Transformer_Encoder import PositionalEncoding


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Transformer_VIT(nn.Module):
    def __init__(self, sequence_length_spatial, num_dynamic_attr, num_static_attr,
                 embedding_size, image_size, d_model, sequence_length,
                 dropout, num_heads, num_layers):
        super(Transformer_VIT, self).__init__()
        self.vit = VIT(d_model=d_model, image_size=image_size)
        self.fc_1 = nn.Linear((num_dynamic_attr + embedding_size), d_model)
        self.encoder = Encoder(n_layers=num_layers, n_head=num_heads, d_model=d_model, d_inner=d_model,
                               sequence_length=sequence_length)
        self.fc_2 = nn.Linear(d_model, 1)
        self.sequence_length_spatial = sequence_length_spatial
        self.num_static_attr = num_static_attr
        self.num_dynamic_attr = num_dynamic_attr
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Linear(in_features=self.num_static_attr, out_features=self.embedding_size)

    def forward(self, non_spatial_input, spatial_input):
        x_d = non_spatial_input[:, :, :self.num_dynamic_attr]
        x_s = non_spatial_input[:, :, -self.num_static_attr:]
        x_s = self.embedding(x_s)
        non_spatial_input = torch.cat([x_d, x_s], axis=2)
        batch_size, seq_length, _ = spatial_input.shape
        spatial_input = spatial_input.reshape(batch_size * seq_length, 1, self.image_size, self.image_size)
        spatial_input = self.vit(spatial_input).pooler_output
        spatial_input = spatial_input.reshape(batch_size, seq_length, -1)
        non_spatial_input = self.fc_1(non_spatial_input)
        enc_out = self.encoder(non_spatial_input, spatial_input)
        return self.fc_2(self.dropout(enc_out))[:, 0, :]


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, drop_prob=dropout)

    def forward(self, enc_input_non_spatial, enc_input_spatial, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input_non_spatial, enc_input_spatial, enc_input_spatial,
                                   mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, n_layers, n_head, d_model, d_inner, dropout=0.1, sequence_length=270, scale_emb=False):
        super(Encoder, self).__init__()
        # the nn.Embedding creates a lookup table that can retrieve a word embedding using an index in the table
        # d_word_vec should be equal to d_model (?) - for example 512
        self.position_enc = PositionalEncoding(d_model, max_len=sequence_length + 1)
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
        enc_input = self.layer_norm(enc_input_non_spatial)
        for ind, enc_layer in enumerate(self.layer_stack):
            if ind > 0:
                enc_input = enc_layer(enc_input, enc_input)
            else:
                enc_input = enc_layer(enc_input, enc_input_spatial)
        return enc_input


class VIT(nn.Module):
    def __init__(self, d_model, image_size):
        super(VIT, self).__init__()
        configuration = ViTConfig(num_hidden_layers=6, num_attention_heads=8, image_size=image_size,
                                  hidden_size=d_model, patch_size=image_size // 2, num_channels=1)
        self.vit_model = ViTModel(configuration)

    def forward(self, spatial_input):
        return self.vit_model(spatial_input)
