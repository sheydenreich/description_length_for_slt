"""The declared randomized-response channel is part of every predictor's label law."""

import math

import numpy as np
import torch

from bitsev.alignment import controls as C
from bitsev.alignment import maps as M
from bitsev.alignment.score import score_configuration
from bitsev.alignment.targets import NoiseChannel, make_targets
from bitsev.codes import schedule as S
from bitsev.testbeds.oracle import make_oracle


def test_channel_matches_probability_mixture():
    ch = NoiseChannel(0.05, 3)
    p = np.array([[0.9, 0.08, 0.02], [1e-9, 1 - 2e-9, 1e-9]])
    observed = ch.probs(p)
    assert np.allclose(observed.sum(1), 1.0)
    lp = torch.log(torch.as_tensor(p))
    assert torch.allclose(ch.log_prob(lp), torch.log(torch.as_tensor(observed)))


def test_flipped_label_costs_at_most_the_channel_floor():
    ch = NoiseChannel(0.05, 3)
    sharp_wrong = torch.tensor([-40.0], dtype=torch.float64)  # clean model puts e^-40 on it
    assert float(-ch.log_prob(sharp_wrong)) <= -math.log(0.05 / 3) + 1e-6
    # and a confident correct label is barely dearer than without the channel
    sharp_right = torch.tensor([-1e-3], dtype=torch.float64)
    assert float(-ch.log_prob(sharp_right)) < 0.06


def test_zero_rate_is_the_identity():
    ch = NoiseChannel(0.0, 4)
    lp = torch.tensor([-0.5, -3.0])
    assert torch.equal(ch.log_prob(lp), lp)


def test_scoring_declares_the_channel_and_bounds_noisy_cost():
    oracle = make_oracle(d=5, n_pool=400, noise_rate=0.1, seed=0)
    targets = make_targets(oracle.algorithm, oracle.pool, 96, "V", 0.1, seed=1)
    cfg = C.Configuration(
        "oracle/h/subspace", oracle.network, "h", "subspace", lambda d: M.SubspaceMap(d, 1)
    )
    sched = S.geometric(len(targets), 2.0, start=16)
    res = score_configuration(cfg, targets, oracle.pool, sched, seed=0)
    assert "rr=0.1" in res.learner
    floor = -math.log(0.1 / oracle.algorithm.n_outputs)
    assert float(res.code.item_losses.max()) <= floor + 1e-6
