import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn
from torch.nn import functional as F

from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d, WNConvTranspose1d
from dac.nn.layers import TransformerSentenceEncoderLayer
from dac.nn.quantize import ResidualVectorQuantize, VectorQuantize
from dac.nn.custom_layers import Fp32LayerNorm, TransposeLast

from dac.model.dac import ResidualUnit, EncoderBlock, init_weights, DecoderBlock


class SharedEncoder(nn.Module):
    def __init__(self, d_model=64, strides=[2,4]):
        super().__init__()
        self.blocks = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        for stride in strides:
            d_model *= 2
            self.blocks += [EncoderBlock(d_model, stride=stride)]
        
        self.blocks = nn.Sequential(*self.blocks)
        self.d_model = d_model

    def forward(self, x):
        return self.blocks(x)


class AcousticHead(nn.Module):
    def __init__(self, d_model=256, strides=[8, 8], d_latent=64):
        super().__init__()
        self.blocks = []
        for stride in strides:
            d_model *= 2
            self.blocks += [EncoderBlock(d_model, stride=stride)]
        
        self.blocks += [Snake1d(d_model)]
        self.blocks += [WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.blocks = nn.Sequential(*self.blocks)

        self.enc_dim = d_model
       
    def forward(self, x):
        return self.blocks(x)


class SemanticHead(nn.Module):
    def __init__(self,
        d_model=256,           # From shared encoder
        d_latent=64,          # For VQ
        strides = [2, 2, 2, 2, 2, 2],
        n_transformer=2,
        n_heads=8,
        dropout=0.1,
        activation_fn="gelu",
        layer_norm_first=True,
    ):
        super().__init__()

        self.pre_conv = []
        self.strides = strides
        for i, stride in enumerate(strides):
            in_dim = d_model
            if i == (len(strides)-1)//2:
                d_model *= 2
            self.pre_conv += [
                nn.Conv1d(in_dim, d_model, 3, stride=stride, padding=1, bias=False),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(), # (B, C, T) → (B, T, C)
                    Fp32LayerNorm(d_model, elementwise_affine=True),
                    TransposeLast(), # (B, T, C) → (B, C, T)
                ),
                nn.GELU(),
            ]
        self.pre_conv = nn.Sequential(*self.pre_conv)

        self.transformer_layers = nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim=d_model,
                ffn_embedding_dim=d_model*4,
                num_attention_heads=n_heads,
                dropout=dropout,
                attention_dropout=dropout,
                activation_dropout=0.0,
                activation_fn=activation_fn,
                layer_norm_first=layer_norm_first,
            )
            for _ in range(n_transformer)
        ])

        self.project = nn.Sequential(
            Fp32LayerNorm(d_model, elementwise_affine=True), # [B, T, C]
            TransposeLast(),  # [B, C, T]
            nn.Conv1d(d_model, d_latent, kernel_size=1),
        )

    def forward(self, x):
        """
        x: [B, C, T] from shared encoder
        returns: [B, proj_dim, T'] for semantic VQ
        """
        x = self.pre_conv(x)        # [B, 256, T/8]->[B, 512, T/320]
        x = x.transpose(1, 2)       # [B, T/320, C]

        for layer in self.transformer_layers:
            x, _ = layer(x)         # [B, T/320, C]

        x = self.project(x)         # [B, proj_dim, T]
        return x


class FiLMGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, kernel_size=3, depth=2, strides = []):
        super().__init__()
        layers = []
        for i in range(depth):
            dim_in = in_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(dim_in, hidden_dim, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
        self.shared_net = nn.Sequential(*layers)
        
        if len(strides) > 0:
            upsample_layers = []
            for stride in strides:
                upsample_layers.append(nn.ConvTranspose1d(
                    hidden_dim, hidden_dim, kernel_size=stride*2, stride=stride, padding=stride//2
                ))
                upsample_layers.append(nn.ReLU())
            self.upsample = nn.Sequential(*upsample_layers)
        else:
            self.upsample = nn.Identity()

        self.to_gamma = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        self.to_beta = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, sem_embedding):  # [B, D_sem, T]
        h = self.shared_net(sem_embedding)  # [B, H, T]
        h = self.upsample(h)
        gamma = self.to_gamma(h)  # [B, D, T]
        beta = self.to_beta(h)    # [B, D, T]
        return gamma, beta


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out = 1, 
                 film_layer_idx=1,):
        super().__init__()

        # Add first conv layer
        self.pre_conv = WNConv1d(input_channel, channels, kernel_size=7, padding=3)

        # Add upsampling + MRF blocks
        self.layers = nn.ModuleList()
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            self.layers.append(DecoderBlock(input_dim, output_dim, stride))

        # Add final conv layer
        self.post_conv = nn.Sequential(
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        # FiLM layer
        self.film_layer_idx = film_layer_idx
        self.film_channels = channels // 2 ** film_layer_idx
        self.film = FiLMGenerator(in_dim = input_channel, out_dim=self.film_channels, strides=rates[0:film_layer_idx])

    def forward(self, z_acs, z_sem):
        gamma, beta = self.film(z_sem) # [B, D, T]

        z = self.pre_conv(z_acs)

        if self.film_layer_idx == 0:
            if gamma.shape[-1] != z.shape[-1]:
                gamma = F.interpolate(gamma, size=z.shape[-1], mode='nearest')
                beta = F.interpolate(beta, size=z.shape[-1], mode='nearest')
            z = gamma * z + beta

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i + 1 == self.film_layer_idx:
                if gamma.shape[-1] != z.shape[-1]:
                    gamma = F.interpolate(gamma, size=z.shape[-1], mode='nearest')
                    beta = F.interpolate(beta, size=z.shape[-1], mode='nearest')
                z = gamma * z + beta
        
        out = self.post_conv(z)
        return out


class DiscoDAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim = 64,
        encoder_rates = [[2, 4], [8, 8], [2, 2, 2, 2, 2, 2]],
        latent_dim = None,
        decoder_dim = 1536,
        decoder_rates = [8, 8, 4, 2],
        n_codebooks = 9,
        codebook_size = 1024,
        codebook_dim: Union[int, list] = 8,
        sem_codebook_size=512,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        film_layer_idx: int = 1,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates[0] + encoder_rates[1])
        self.enc = SharedEncoder(d_model=encoder_dim, strides=encoder_rates[0])
        self.acs_enc = AcousticHead(d_model=self.enc.d_model, strides=encoder_rates[1], d_latent=self.latent_dim)
        self.sem_enc = SemanticHead(d_model=self.enc.d_model, strides=encoder_rates[2], d_latent=self.latent_dim)
        
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.sem_codebook_size = sem_codebook_size
        self.acs_quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.sem_quantizer = VectorQuantize(
            input_dim=latent_dim, 
            codebook_size=sem_codebook_size, 
            codebook_dim=codebook_dim)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            film_layer_idx= film_layer_idx,
        )
        self.proj_sem = nn.Conv1d(latent_dim, 768, kernel_size=1)
        
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(self, audio_data: torch.Tensor, n_quantizers: int = None,):
        enc = self.enc(audio_data)  # [B, 256, T/8]

        sem_enc = self.sem_enc(enc)  # [B, 512, T/320]
        acs_enc = self.acs_enc(enc)  # [B, 512, T/320]
        z_sem, commitment_loss_sem, codebook_loss_sem, codes_sem, latents_sem = self.sem_quantizer(sem_enc)
        commitment_loss_sem = commitment_loss_sem.mean()
        codebook_loss_sem = codebook_loss_sem.mean()
        z_acs, codes_acs, latents_acs, commitment_loss_acs, codebook_loss_acs = self.acs_quantizer(acs_enc, n_quantizers)
        return {
            "z_sem": z_sem, "codes_sem": codes_sem, "latents_sem": latents_sem,
            "z_acs": z_acs, "codes_acs": codes_acs, "latents_acs": latents_acs,
            "vq/commit_loss_sem": commitment_loss_sem, "vq/codebook_loss_sem": codebook_loss_sem,
            "vq/commit_loss_acs": commitment_loss_acs, "vq/codebook_loss_acs": codebook_loss_acs,
        }

    def decode(self, z_acs, z_sem):
        """z : Tensor[B x D x T]"""
        return self.decoder(z_acs, z_sem)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """RETURNS
        dict{"z" : Tensor[B x D x T],Quantized continuous representation of input
            "codes" : Tensor[B x N x T], Codebook indices for each codebook
            "latents" : Tensor[B x N*D x T], Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1], Commitment loss to train encoder to predict vectors closer to codebook
            "vq/codebook_loss" : Tensor[1], Codebook loss to update the codebook
            "length" : int, Number of samples in input audio
            "audio" : Tensor[B x 1 x length], Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        out = self.encode(audio_data, n_quantizers=n_quantizers)

        e_sem = self.proj_sem(out['z_sem'])           # [B, 512, T/320]
        x = self.decode(out['z_acs'], out['z_sem'])   # [B, 1, T]

        out.update({
            "audio": x[..., :length], "e_sem": e_sem
        })

        return out