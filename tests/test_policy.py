import numpy as np
import pytest
import torch

from density_grpo import (
    RECIPES,
    DensityPolicy,
    antithetic_innovations,
    cifar_inputs,
    decode,
)
from density_grpo.optimizer import advantages, joint_kl, joint_log_prob


@pytest.mark.parametrize("nfe", [4, 6, 8])
def test_terminal_step_control(nfe):
    action = torch.zeros(3, 65)
    action[:, 64] = torch.tensor([-5.0, 0.0, 5.0])
    nodes = np.asarray(decode(action, nfe))
    control = np.asarray(decode(action, nfe, fixed_end=True))
    np.testing.assert_equal(nodes[:, :-1], control[:, :-1])
    np.testing.assert_allclose(nodes[:, 0], 0.1)
    assert np.all(control[:, -1] == 1)
    assert np.all(nodes[:, -1] >= np.maximum(0.95, (nodes[:, -2] + 1) / 2))
    assert np.all(np.diff(nodes[:, -1]) < 0)
    np.testing.assert_allclose(nodes[1, -1], 0.995)


@pytest.mark.parametrize("name", ["sana", "1rf"])
def test_pairs_and_joint_likelihood(name):
    recipe = RECIPES[name]
    ids = list(range(8 * recipe.clocks))
    draws = antithetic_innovations(ids, recipe)
    torch.testing.assert_close(draws[:, ::2], -draws[:, 1::2])
    reward = torch.arange(len(ids)).reshape(8, recipe.clocks)
    for i in range(recipe.clocks):
        keep = torch.arange(recipe.clocks) // 2 != i // 2
        expected = reward[:, i] - reward[:, keep].double().mean(1)
        torch.testing.assert_close(advantages(reward)[:, i], expected)
    mean, std = torch.zeros(65), torch.ones(65)
    for i in range(65):
        shift = mean.clone()
        shift[i] = 1.0
        assert joint_kl(mean, std, shift, std).item() == pytest.approx(0.5)
        assert (
            joint_log_prob(mean, std, shift) - joint_log_prob(mean, std, mean)
        ).item() == pytest.approx(-0.5)


def test_joint_heads_and_input_budget():
    features = cifar_inputs([1, 2], 6)
    np.testing.assert_allclose(
        features["condition"][0], [0, 1, np.log(7 / 5), np.log(7 / 5)]
    )
    policy = DensityPolicy(RECIPES["1rf"].spec)
    mean, scale = policy(features)
    assert mean.shape == scale.shape == (2, 65)
    torch.testing.assert_close(mean, torch.zeros_like(mean))
    torch.testing.assert_close(scale, torch.full_like(mean, 0.5))
