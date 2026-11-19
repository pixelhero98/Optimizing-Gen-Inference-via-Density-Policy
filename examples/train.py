"""Train online against a frozen linear ODE with an analytic full-interval teacher."""

import argparse
import math

import torch

from density_grpo import (
    RECIPES,
    antithetic_innovations,
    cifar_inputs,
    decode,
    initialize_policy,
    policy_determinism,
    update,
)
from density_grpo.euler import euler
from density_grpo.optimizer import LR, WEIGHT_DECAY


def main(steps, device):
    recipe = RECIPES["1rf"]
    torch.manual_seed(3109)
    policy = initialize_policy("1rf", device=device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for step in range(steps):
        seeds = range(step * 8, (step + 1) * 8)
        inputs = cifar_inputs(seeds, device=device)
        ids = [
            f"train:{seed}:{draw}" for seed in seeds for draw in range(recipe.clocks)
        ]
        with torch.no_grad(), policy_determinism():
            mean, std = policy(inputs)
            eps = antithetic_innovations(ids, recipe).to(mean)
            actions = mean[:, None] + std[:, None] * eps
            clocks = decode(actions.flatten(0, 1))
            initial = torch.randn(8, 1, device=device).repeat_interleave(
                recipe.clocks, 0
            )
            prediction = euler(lambda state, time: -state, initial, clocks)
            reward = -(prediction - initial / math.e).abs().reshape(8, recipe.clocks)
        with policy_determinism():
            passes = update(policy, optimizer, inputs, actions, reward, mean, std)
        print(
            f"update {step + 1}: mean reward {reward.mean().item():.6f}, passes {len(passes)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.steps, args.device)
