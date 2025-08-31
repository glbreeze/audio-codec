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
from dac.nn.layers import WNConv1d, WNConvTranspose1d, AsymmetricPad1d
from dac.nn.quantize import VectorQuantize, ResidualVectorQuantize

from dac.model.dac import ResidualUnit, EncoderBlock, init_weights, DecoderBlock, Encoder


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
                
                if stride % 2 == 0:
                    pad_left = pad_right = stride//2
                else:
                    pad_left = stride//2 + 1
                    pad_right = stride//2
                
                upsample_layers.extend([
                    AsymmetricPad1d(pad_left, pad_right),
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=stride*2, stride=stride, padding=0),
                    nn.ReLU()
                    ])
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
                 film_layers_idx=[1],):
        super().__init__()
        print(f"--check model structure, film layers are {film_layers_idx}")

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
        self.film_layers_idx = film_layers_idx
        self.films = nn.ModuleDict()
        for film_idx in self.film_layers_idx:
            film_channels = channels // 2 ** film_idx
            self.films[str(film_idx)] = FiLMGenerator(in_dim = input_channel, out_dim=film_channels, strides=rates[0:film_idx])

    def forward(self, z_acs, z_sem):
        z = self.pre_conv(z_acs)

        if 0 in self.film_layers_idx:
            gamma, beta = self.films["0"](z_sem) # [B, D, T]
            if gamma.shape[-1] != z.shape[-1]:
                gamma = F.interpolate(gamma, size=z.shape[-1], mode='nearest')
                beta = F.interpolate(beta, size=z.shape[-1], mode='nearest')
            z = gamma * z + beta

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i + 1 in self.film_layers_idx:
                gamma, beta = self.films[str(i+1)](z_sem)
                if gamma.shape[-1] != z.shape[-1]:
                    gamma = F.interpolate(gamma, size=z.shape[-1], mode='nearest')
                    beta = F.interpolate(beta, size=z.shape[-1], mode='nearest')
                z = gamma * z + beta
        
        out = self.post_conv(z)
        return out


class SemDAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim = 64,
        encoder_rates = [2, 4, 8, 8],
        latent_dim = None,
        decoder_dim = 1536,
        decoder_rates = [8, 8, 4, 2],
        n_codebooks = 9,
        codebook_size = [512, 1024], # 512/1024: codebook size for semantic/acoustic code 
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        film_layer_idx: list = '0',
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

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            film_layers_idx=[int(i) for i in list(film_layer_idx)],
        )
        
        self.proj_sem = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),  # or Linear if shape is [B, T, D]
            nn.GELU(),
            nn.Conv1d(latent_dim, 768, kernel_size=1)
        )
        
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
        z = self.encoder(audio_data)
        
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return {
            "z": z, "codes": codes, "latents": latents,
            "vq/commit_loss": commitment_loss, "vq/codebook_loss": codebook_loss,
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
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        
        out = self.encode(audio_data, n_quantizers=n_quantizers)

        z_sem = out['latents'][0]
        z_acs = out['z']
        e_sem = self.proj_sem(z_sem)           # [B, 512, T/320]
        x = self.decode(z_acs, z_sem)   # [B, 1, T]

        out.update({
            "audio": x[..., :length], "e_sem": e_sem
        })

        return out