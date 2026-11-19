"""Batched native noise-to-data Euler integration."""

import torch


@torch.no_grad()
def euler(field, initial, clocks):
    """Call field(state, time) once per interval from the paired initial latent."""
    times = torch.as_tensor(clocks, dtype=initial.dtype, device=initial.device)
    if (
        times.ndim != 2
        or times.shape[0] != len(initial)
        or times.shape[1] < 2
        or not torch.isfinite(times).all()
        or (times[:, 0] < 0).any()
        or (times[:, -1] > 1).any()
        or not (times.diff(dim=1) > 0).all()
    ):
        raise ValueError(
            "Each latent needs a strictly increasing schedule inside [0,1]."
        )
    state = initial
    for left, right in zip(times.T[:-1], times.T[1:], strict=True):
        dt = (right - left).reshape(-1, *([1] * (state.ndim - 1)))
        state = state + dt * field(state, left)
    return state
