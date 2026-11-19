"""One score per entire clock with leave-one-pair-out advantages."""

import math

import torch
from torch.nn.utils import clip_grad_norm_

CLIP_LOW, CLIP_HIGH = 0.20, 0.30
TARGET_KL = 0.20
PASSES = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4


def advantages(rewards):
    if rewards.shape not in ((8, 8), (8, 16)) or not torch.isfinite(rewards).all():
        raise ValueError(
            "Expected finite utilities for eight groups of eight or sixteen clocks."
        )
    values = rewards.detach().double()
    pair_sums = values.reshape(8, -1, 2).sum(-1)
    baseline = (
        (values.sum(-1, keepdim=True) - pair_sums) / (values.shape[1] - 2)
    ).repeat_interleave(2, -1)
    return values - baseline


def joint_log_prob(mean, std, action):
    return (
        torch.distributions.Normal(mean.double(), std.double())
        .log_prob(action.double())
        .sum(-1)
    )


def joint_kl(old_mean, old_std, mean, std):
    old_mean, old_std, mean, std = [v.double() for v in (old_mean, old_std, mean, std)]
    return (
        torch.log(std / old_std)
        + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square())
        - 0.5
    ).sum(-1)


def update(model, optimizer, inputs, actions, rewards, old_mean, old_std):
    if (
        actions.shape != (*rewards.shape, 65)
        or old_mean.shape != (len(rewards), 65)
        or old_std.shape != old_mean.shape
    ):
        raise ValueError(
            "Expected one 65-coordinate behavior distribution per group and one action per reward."
        )
    actions, old_mean, old_std = [t.detach() for t in (actions, old_mean, old_std)]
    old_logp = joint_log_prob(old_mean[:, None], old_std[:, None], actions)
    advantage = advantages(rewards)
    stats = []
    for _ in range(PASSES):
        mean, std = model(inputs)
        before = joint_kl(old_mean, old_std, mean, std).mean().item()
        if not math.isfinite(before):
            raise FloatingPointError("Nonfinite rollout KL.")
        if before > TARGET_KL:
            break
        ratio = (joint_log_prob(mean[:, None], std[:, None], actions) - old_logp).exp()
        if not torch.isfinite(ratio).all():
            raise FloatingPointError("Nonfinite joint likelihood ratio.")
        loss = -torch.minimum(
            ratio * advantage, ratio.clamp(1.0 - CLIP_LOW, 1.0 + CLIP_HIGH) * advantage
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item()
        optimizer.step()
        with torch.no_grad():
            new_mean, new_std = model(inputs)
            after = joint_kl(old_mean, old_std, new_mean, new_std).mean().item()
            if not math.isfinite(after):
                raise FloatingPointError(
                    "Nonfinite joint KL after the optimizer update."
                )
        stats.append(
            {
                "kl_before": before,
                "kl_after": after,
                "grad_norm": norm,
                "loss": loss.item(),
                "ratio_min": ratio.min().item(),
                "ratio_max": ratio.max().item(),
            }
        )
    return stats
