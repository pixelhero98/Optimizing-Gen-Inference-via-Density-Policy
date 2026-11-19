"""Replayable Gaussian actions and bounded inverse-CDF schedules."""

import hashlib
import math

import numpy as np
import torch

START_MAX = 0.95
START_LOGIT = math.log(0.10 / (0.95 - 0.10))


def keyed_seed(*parts):
    data = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**63)


def innovations(ids, recipe, seed=3109):
    """Draw independently keyed density, start and endpoint noise for each request."""

    def draw(size, *key):
        generator = torch.Generator().manual_seed(keyed_seed(*key))
        return torch.randn(size, generator=generator)

    return torch.stack(
        [
            torch.cat(
                (
                    draw(63, seed, key),
                    draw(1, "gen-inf-learned-start-v1", seed, key),
                    draw(1, recipe.endpoint_key, seed, key),
                )
            )
            for key in ids
        ]
    )


def antithetic_innovations(ids, recipe, seed=3109):
    if len(ids) != 8 * recipe.clocks:
        raise ValueError("Supply one request ID for each rollout in the recipe.")
    base = innovations(ids[::2], recipe, seed).reshape(8, recipe.clocks // 2, 65)
    return torch.stack((base, -base), dim=2).flatten(1, 2)


def start_time(raw_start):
    logit = START_LOGIT + 2.0 * raw_start
    if not math.isfinite(logit):
        raise ValueError("Start coordinate must be finite.")
    if logit >= 0.0:
        return START_MAX / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return START_MAX * exp_logit / (1.0 + exp_logit)


def _quantiles(mass, start, nfe):
    values = np.asarray(mass, dtype=np.float64)
    if values.shape != (64,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected 64 finite, nonnegative bin masses.")
    if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=1e-7):
        raise ValueError("Bin masses must sum to one.")
    if not 0.0 <= start <= START_MAX or nfe < 1:
        raise ValueError("Invalid starting time or model-call budget.")
    values = (1.0 - 1e-8) * values + 1e-8 / 64
    cdf = np.concatenate(([0.0], np.cumsum(values / values.sum())))
    cdf[-1] = 1.0
    base = np.interp(np.linspace(0.0, 1.0, nfe + 1), cdf, np.linspace(0.0, 1.0, 65))
    base[0], base[-1] = 0.0, 1.0
    for i in range(1, len(base)):
        if base[i] <= base[i - 1]:
            base[i] = min(1.0, base[i - 1] + 1e-8)
    base[-1] = 1.0
    if not (
        np.all(np.diff(base.astype(np.float32)) > 0)
        and np.all(np.diff((1.0 - base).astype(np.float32)) < 0)
    ):
        raise ValueError("Canonical clock collapses in float32.")
    grid = (
        float(start),
        *(float(start + (1.0 - start) * node) for node in base[1:-1]),
        1.0,
    )
    if not np.all(np.diff(np.array(grid, dtype=np.float64)) > 0):
        raise ValueError("Density clock is not strictly increasing.")
    return grid


def decode(actions, nfe=4, *, fixed_end=False):
    """Decode a batch of joint actions into NFE+1 ordered Euler nodes."""
    if actions.ndim != 2 or actions.shape[1] != 65 or not torch.isfinite(actions).all():
        raise ValueError("Expected finite actions of shape [batch, 65].")
    logits = actions[:, :63].detach().double()
    mass = torch.cat((logits, torch.zeros_like(logits[:, :1])), -1).softmax(-1)
    clocks = []
    for weights, (raw_start, raw_end) in zip(
        mass.cpu().numpy(), actions[:, 63:].detach().tolist(), strict=True
    ):
        grid = _quantiles(weights, start_time(raw_start), nfe)
        logit = math.log(0.1 / 0.9) + 2 * raw_end
        tail = math.exp(-abs(logit))
        fraction = (1.0 if logit >= 0 else tail) / (1.0 + tail)
        end = 1.0 if fixed_end else 1.0 - min(0.05, (1.0 - grid[-2]) / 2) * fraction
        nodes = (*grid[:-1], end)
        native = np.asarray(nodes, dtype=np.float32)
        sigma = (1.0 - np.asarray(nodes)).astype(np.float32)
        if not (
            np.all(np.diff(native) > 0)
            and np.all(np.diff(sigma) < 0)
            and np.all(np.diff(sigma[:-1] * 1000) < 0)
        ):
            raise ValueError("A schedule interval collapses at solver precision.")
        clocks.append(nodes)
    return clocks
