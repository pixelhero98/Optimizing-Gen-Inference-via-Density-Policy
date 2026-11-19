# Optimizing generative inference with a density policy

Density-GRPO learns where a frozen generator spends its model calls. A single conditional policy samples 63 log density ratios for 64 bins, a starting time, and an endpoint. An inverse CDF places the Euler queries. The endpoint head adjusts the final step while keeping those queries fixed.

## Train and replay

Install from the checkout with Python 3.11 or newer. The training example runs one online update on a frozen linear ODE. It generates fresh rollouts, scores their terminal states against the analytic solution, and updates the policy.

```bash
python -m pip install '.[test]'
python -m pytest -q
python examples/train.py --steps 1
```

Download the **v1.0.0** assets into `assets/`. The replay example loads the validation-selected checkpoint and prints one sampled schedule, including both endpoints. The same seed and request ID reproduce the same draw on the same backend. Use the native CUDA backend for SANA campaign replay because its auxiliary Gaussian latent is device-generated. [Weights and checksums](https://github.com/pixelhero98/Optimizing-Gen-Inference-via-Density-Policy/releases/tag/v1.0.0)

```bash
gh release download v1.0.0 -D assets
python examples/replay.py 1rf --seed 3109 --nfe 4
python examples/replay.py otcfm --seed 3109 --nfe 8
python examples/replay.py sana --context assets/sana-example.pt --seed 3109
```

For a paired control, repeat a command with `--fixed-end`. For CIFAR, pass the resulting nodes and the generation seed’s original Gaussian latent to `density_grpo.euler.euler(field, initial, clocks)`. The field receives native noise-to-data time. For SANA, use successive `1 - node` values as the flow scheduler’s sigmas, including the learned terminal sigma. The model queries use the first four sigmas scaled by 1000, CFG 4.5, and the original Gaussian latent.

## Online policy training

Use `initialize_policy(name, assets)` for the recorded training initialization and `load_policy(name, assets)` for the selected inference weights. Both return a joint density, start, and end policy. The recipe supplies the input dimensions, clocks per group, endpoint random stream, selected update, and weight checksum.

| Recipe | Conditions and reward | Batch | Selected update |
|---|---|---:|---:|
| `sana` | SANA 600M, 512 px. Native text tokens, normalized pooled text, settings and independently seeded noise. Balanced ImageReward and VQAScore utility. | 8 prompts × 8 clocks | 180 |
| `1rf` | CIFAR-10 original 1-RF. Unconditional Euler features and independently seeded noise. Negative VGG LPIPS to a full-interval RK45 teacher. | 8 cases × 16 clocks | 190 |
| `otcfm` | CIFAR-10 OT-CFM. Same inputs, teacher construction and reward as 1-RF, using its own frozen backbone. | 8 cases × 16 clocks | 180 |

Each campaign used seed 3109, 200 updates, and Euler NFE 4. Validation ran every ten updates, including update zero. The selected checkpoint maximized sampled utility for SANA or minimized sampled LPIPS for CIFAR, with earlier updates winning ties. SANA utility was `(0.5 * ImageReward / 1.2403747548508492 + 0.5 * VQAScore / 0.3385359817450678) / 0.9470129752862206`.

To use your own generator, follow the loop in `examples/train.py`. For each case, sample adjacent antithetic action pairs, decode each action, generate with the same initial latent for the entire group, and return one scalar reward per clock. GRPO subtracts the mean reward outside the action’s pair. The likelihood ratio and KL sum over all 65 coordinates. AdamW uses learning rate and weight decay of 0.0001, gradient norm limit 1, and ratio clipping to [0.8, 1.3]. Before each of up to four passes, a mean joint KL above 0.20 stops additional updates.

For SANA, supply the positive native tokens with shape `[B, T, 2304]`, their Boolean padding mask, and the mask-weighted mean embedding. The feature helper applies the released training-set normalization. The example context encodes “A blue ceramic cup on a wooden table.” The initial start is 0.10 for a zero action. With last query `q` and endpoint action `u`, the terminal time is `1 - min(0.05, (1-q)/2) * sigmoid(2*u + log(0.1/0.9))`. The last step retains at least half its original length.

## Results

**SANA and Stable Diffusion 1.5.** Mean balanced utility on paired development panels of 512 prompts with two generation seeds each. Higher is better. Intervals are paired 95% prompt-bootstrap intervals.

| Backbone and solver | NFE | Learned schedule | Uniform | Gain over uniform [95% CI] |
|---|---:|---:|---:|---|
| SANA 600M, Euler | 4 | 1.6197 | 0.6627 | +0.9570 [+0.8894, +1.0276] |
| SD 1.5, iPNDM | 4 | 0.9241 | 0.8726 | +0.0515 [+0.0109, +0.0912] |
| SD 1.5, DPM++(2M) | 4 | 0.9179 | 0.8625 | +0.0555 [+0.0185, +0.0915] |
| SD 1.5, iPNDM | 6 | 1.1061 | 1.0899 | +0.0162 [-0.0177, +0.0504] |
| SD 1.5, iPNDM | 8 | 1.1593 | — | — |

SANA learned both endpoints. Its earlier learned-start policy scored 1.5918, and the joint checkpoint scored 1.6199 when forced to end at one. SD 1.5 used a learned start and a fixed end, with a fresh policy trained for each solver and budget. The 8-call comparison used the 6-call learned policy as its control.

**SiT-XL/2 on ImageNet-256, four Euler calls.** FID30k uses the full ADM real reference and one matched generation panel per schedule (30 seeds per class). LPIPS uses 1,024 held-out class/seed pairs. Lower is better for both metrics.

| Schedule | FID30k | VGG-LPIPS |
|---|---:|---:|
| Uniform Euler | 95.316 | 0.57306 |
| Joint learned start/end | 81.906 | 0.42480 |
| Same joint policy, end forced to 1 | 81.074 | 0.42466 |

Each LPIPS target came from a higher-accuracy DOPRI5 solve of the same frozen class-conditional SiT with the same noise seed, CFG 1.0 and EMA VAE. Validation LPIPS selected update 200 of 200. The paired held-out LPIPS difference for joint minus uniform was -0.14825 (95% case-bootstrap interval [-0.15294, -0.14346]). One policy training seed.

**CIFAR-10 zero-shot budget transfer.** FID50k averaged over five paired panels per row. Lower is better. Both policies trained at four calls. The six- and eight-call rows use the same validation-selected weights without additional training or checkpoint selection.

| Backbone | NFE | Uniform | Learned start only | Joint start/end | Joint, end forced to 1 |
|---|---:|---:|---:|---:|---:|
| 1-RF | 4 | 52.9384 | 16.4015 | **16.3550** | 16.8139 |
| 1-RF | 6 | 26.0171 | 8.0246 | **7.8099** | 8.1322 |
| 1-RF | 8 | 17.1326 | 6.1143 | **5.8157** | 6.0887 |
| OT-CFM | 4 | 30.3672 | 17.2441 | 16.3167 | **16.1478** |
| OT-CFM | 6 | 18.4901 | 9.5278 | 8.9566 | **8.8815** |
| OT-CFM | 8 | 14.1358 | 6.9658 | 6.5506 | **6.5349** |

The joint policies improved FID over the earlier learned-start policy on all 20 zero-shot panels (2.67%–6.00% lower mean FID). Shortening the final step improved 1-RF FID at both transfer budgets. OT-CFM achieved its lowest FID when forced to end at one. On 1,024 paired teacher cases, transfer LPIPS improved for 1-RF and worsened for OT-CFM relative to learned start alone. Each policy used one training seed.
