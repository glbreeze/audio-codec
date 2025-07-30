import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from dac.nn import utils
from dac.nn.multihead_attention import MultiheadAttention
from dac.nn.custom_layers import LayerNorm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class AsymmetricPad1d(nn.Module):
    def __init__(self, pad_left: int, pad_right: int):
        super().__init__()
        self.padding = (pad_left, pad_right)

    def forward(self, x):
        return F.pad(x, self.padding)


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class TransformerSentenceEncoderLayer(nn.Module):
    """ Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained models. """

    def __init__(self, embedding_dim: float = 768, ffn_embedding_dim: float = 3072, num_attention_heads: int = 8,
        dropout: float = 0.1, attention_dropout: float = 0.1, activation_dropout: float = 0.0,
        activation_fn: str = "gelu", layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_first = layer_norm_first


        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim, 
            num_attention_heads,
            dropout=attention_dropout, self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self, x, self_attn_mask = None, self_attn_padding_mask = None,
        need_weights = False, att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)
