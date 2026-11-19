"""Conditional Gaussian policy over density and integration endpoints."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 128
BINS = 64


@dataclass(frozen=True)
class PolicySpec:
    condition_dim: int
    token_dim: int
    noise_channels: int


def projection(inputs):
    return nn.Sequential(nn.Linear(inputs, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))


class BinBlock(nn.Module):
    """Four-head bin attention with cached rotary positions."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.attention_output = nn.Linear(WIDTH, WIDTH)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(
            nn.Linear(WIDTH, 256), nn.SiLU(), nn.Dropout(0.0), nn.Linear(256, WIDTH)
        )
        centers = (torch.arange(BINS - 1) + 0.5) / BINS
        angles = centers[:, None] * torch.arange(1, 17)[None] * math.pi
        self.register_buffer(
            "cosine", angles.cos().repeat_interleave(2, -1), persistent=False
        )
        self.register_buffer(
            "sine", angles.sin().repeat_interleave(2, -1), persistent=False
        )

    def rotate(self, x):
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
        return x * self.cosine + rotated * self.sine

    def forward(self, tokens):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, 4, 32)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v)
        tokens = tokens + self.attention_output(
            attention.transpose(1, 2).reshape(batch, length, width)
        )
        return tokens + self.ff(self.norm2(tokens))


class NoiseSummary(nn.Module):
    def forward(self, features):
        batch, channels, height, width = features.shape
        if (height, width) != (4, 4):
            if height % 4 or width % 4:
                raise ValueError(
                    "Noise feature dimensions must divide into a 4×4 summary."
                )
            features = features.reshape(
                batch, channels, 4, height // 4, 4, width // 4
            ).mean((3, 5))
        return features.flatten(1)


class _DensityStart(nn.Module):
    def __init__(self, spec: PolicySpec):
        super().__init__()
        self.spec = spec
        self.condition = projection(spec.condition_dim)
        self.blocks = nn.ModuleList([BinBlock(), BinBlock()])
        self.norm = nn.LayerNorm(WIDTH)
        geometry = torch.stack(
            ((torch.arange(BINS) + 0.5) / BINS, torch.full((BINS,), 1 / BINS)), dim=-1
        )
        self.register_buffer("geometry", geometry[:-1], persistent=False)
        self.input = projection(2)
        self.output = nn.Linear(WIDTH, 2)
        nn.init.zeros_(self.output.weight)
        p = (0.5 - 0.05) / 1.95
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([0.0, math.log(p / (1 - p))]))
        self.text_mlp = nn.Sequential(
            nn.LayerNorm(spec.token_dim, elementwise_affine=False),
            nn.Linear(spec.token_dim, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.register_buffer("prior_logratios", torch.zeros(BINS - 1))
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(spec.noise_channels, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            NoiseSummary(),
            nn.Linear(1024, WIDTH),
        )
        nn.init.zeros_(self.noise_encoder[-1].weight)
        nn.init.zeros_(self.noise_encoder[-1].bias)
        self.start_head = nn.Linear(WIDTH, 2)
        nn.init.zeros_(self.start_head.weight)
        with torch.no_grad():
            self.start_head.bias.copy_(self.output.bias)

    def forward(self, inputs):
        condition, text, padding, noise = (
            inputs[k] for k in ("condition", "text", "padding_mask", "noise")
        )
        batch = len(condition)
        if (
            condition.shape != (batch, self.spec.condition_dim)
            or text.ndim != 3
            or (text.shape[0], text.shape[-1]) != (batch, self.spec.token_dim)
            or padding.shape != text.shape[:2]
            or padding.dtype != torch.bool
            or padding.all(-1).any()
            or noise.ndim != 4
            or noise.shape[:2] != (batch, self.spec.noise_channels)
        ):
            raise ValueError(
                "Invalid pooled context, native token/mask, or auxiliary noise shape."
            )
        if not all(torch.isfinite(x).all() for x in (condition, text, noise)):
            raise ValueError("Policy inputs must be finite.")
        features = self.text_mlp(text.float()).masked_fill(padding[..., None], 0.0)
        summary = features.sum(1) / (~padding).sum(
            1, keepdim=True
        ) + self.noise_encoder(noise.float())
        queries = self.input(self.geometry)[None] + summary[:, None]
        encoded = queries + self.condition(condition)[:, None]
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.norm(encoded)
        density = self.output(encoded)
        start = self.start_head(encoded.mean(1))
        mean = torch.cat((density[..., 0] + self.prior_logratios, start[:, :1]), dim=-1)
        scale = torch.cat((density[..., 1], start[:, 1:]), dim=-1)
        return mean, 0.05 + 1.95 * scale.sigmoid()


class DensityPolicy(nn.Module):
    """Predict the mean and scale of 63 density ratios, start, and end."""

    def __init__(self, spec: PolicySpec):
        super().__init__()
        self.start_policy = _DensityStart(spec)
        self.end_head = nn.Linear(64, 2)
        nn.init.zeros_(self.end_head.weight)
        with torch.no_grad():
            self.end_head.bias.copy_(self.start_policy.output.bias)

    def forward(self, inputs):
        mean, std = self.start_policy(inputs)
        end_mean, raw_scale = self.end_head(mean).split(1, dim=-1)
        return torch.cat((mean, end_mean), -1), torch.cat(
            (std, 0.05 + 1.95 * raw_scale.sigmoid()), -1
        )


@contextmanager
def policy_determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous)
