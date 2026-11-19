"""Validation-selected policies and frozen feature conventions."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .clocks import keyed_seed
from .policy import DensityPolicy, PolicySpec


@dataclass(frozen=True)
class Recipe:
    spec: PolicySpec
    clocks: int
    endpoint_key: str
    update: int
    checksum: str


_CIFAR = PolicySpec(4, 1, 3)
_CIFAR_KEY = "gen-inf-cifar-endpoint-v1"
RECIPES = {
    "sana": Recipe(
        PolicySpec(2307, 2304, 32),
        8,
        "sana-endpoint-v1",
        180,
        "8198079ca28305ffbef6fa921374df3bc9b3c362913b517fb23958ab4633145f",
    ),
    "1rf": Recipe(
        _CIFAR,
        16,
        _CIFAR_KEY,
        190,
        "617e7694ccea62359062cd2b5123c66363021d8b534610b105140f14b3bc73ae",
    ),
    "otcfm": Recipe(
        _CIFAR,
        16,
        _CIFAR_KEY,
        180,
        "46a2e724468d390084ac3754dc71b0e2cc07c201014040db08dbbbb963a18d7d",
    ),
}


def _load_weights(model, path, checksum):
    if sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Unexpected checksum for {path.name}.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def load_policy(name, assets, *, device="cpu"):
    """Load the validation-selected policy for inference."""
    recipe = RECIPES[name]
    model = _load_weights(
        DensityPolicy(recipe.spec), Path(assets) / f"{name}.pt", recipe.checksum
    )
    return model.to(device).eval()


def initialize_policy(name, assets=None, *, device="cpu"):
    """Return the campaign's update-zero policy for online training."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(3109)
        model = DensityPolicy(RECIPES[name].spec)
    if name == "sana":
        if assets is None:
            raise ValueError("Supply the directory containing sana-init.pt.")
        _load_weights(
            model,
            Path(assets) / "sana-init.pt",
            "7d093fb01b6e16287e0248c21201cb00bec3e5c454bdf4eecb798ea37ca16691",
        )
    return model.to(device)


def cifar_inputs(seeds, nfe=4, *, device="cpu"):
    """Build the unconditional features and independently seeded auxiliary noise."""
    if nfe not in (4, 6, 8):
        raise ValueError("The CIFAR recipes cover Euler budgets 4, 6 and 8.")
    noise = []
    for seed in seeds:
        key = keyed_seed("cifar-lpips-independent-policy-noise", 3109, int(seed))
        rng = np.random.Generator(np.random.PCG64(key))
        noise.append(rng.standard_normal((3, 32, 32), dtype=np.float32))
    noise = torch.from_numpy(np.stack(noise)).to(device)
    batch = len(noise)
    budget = np.log1p(nfe) - np.log1p(4)
    return {
        "condition": noise.new_tensor([0.0, 1.0, budget, budget]).expand(batch, -1),
        "text": noise.new_zeros((batch, 1, 1)),
        "padding_mask": torch.zeros((batch, 1), device=device, dtype=torch.bool),
        "noise": noise,
    }


def sana_inputs(pooled, tokens, padding_mask, seeds, assets):
    """Use native SANA embeddings, positive tokens and True-for-padding masks."""
    data = (Path(assets) / "sana-conditioning.json").read_bytes()
    if (
        sha256(data).hexdigest()
        != "5e1998736c949411775068360cb70a2c51ea77e16956c7ab66ffac66f06e7786"
    ):
        raise ValueError("Unexpected SANA feature normalization checksum.")
    payload = json.loads(data)
    context = payload["context"]
    norm = (np.asarray(pooled, dtype=np.float64) - context["mean"]) / context["std"]
    if norm.shape != (len(tokens), 2304) or len(seeds) != len(tokens):
        raise ValueError("Supply a pooled embedding and a generation seed per prompt.")
    settings = payload["settings"]
    budget = (np.log1p([4, 4]) - settings["mean"]) / settings["std"]
    extra = np.broadcast_to([1.0, *budget], (len(norm), 3))
    condition = torch.from_numpy(
        np.concatenate((norm, extra), axis=1).astype(np.float32)
    )
    noise = []
    for seed in seeds:
        key = keyed_seed("four-factor-independent-latent-v1", 3109, int(seed))
        generator = torch.Generator(device=tokens.device).manual_seed(key)
        noise.append(
            torch.randn((1, 32, 16, 16), device=tokens.device, generator=generator)
        )
    return {
        "condition": condition.to(tokens.device),
        "text": tokens,
        "padding_mask": padding_mask,
        "noise": torch.cat(noise),
    }
