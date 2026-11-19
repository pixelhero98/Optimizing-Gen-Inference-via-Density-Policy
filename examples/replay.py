"""Sample a selected policy's clock from the release assets."""

import argparse

import torch

from density_grpo import (
    RECIPES,
    cifar_inputs,
    decode,
    innovations,
    load_policy,
    policy_determinism,
    sana_inputs,
)


def main(args):
    model = load_policy(args.recipe, args.assets, device=args.device)
    if args.recipe == "sana":
        if args.nfe != 4 or args.context is None:
            raise ValueError(
                "SANA needs --nfe 4 and --context with pooled, tokens and padding_mask tensors."
            )
        context = torch.load(args.context, map_location="cpu", weights_only=True)
        inputs = sana_inputs(
            context["pooled"].numpy(),
            context["tokens"].to(args.device),
            context["padding_mask"].to(args.device),
            [args.seed],
            args.assets,
        )
    else:
        inputs = cifar_inputs([args.seed], args.nfe, device=args.device)
    with torch.no_grad(), policy_determinism():
        mean, std = model(inputs)
        noise = innovations(
            [args.request_id or f"test:{args.seed}"], RECIPES[args.recipe]
        ).to(mean)
        print(decode(mean + std * noise, args.nfe, fixed_end=args.fixed_end)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=RECIPES)
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--seed", type=int, default=3109)
    parser.add_argument("--request-id")
    parser.add_argument("--context", help="SANA native prompt features for one prompt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nfe", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument(
        "--fixed-end", action="store_true", help="Paired terminal-step control"
    )
    main(parser.parse_args())
