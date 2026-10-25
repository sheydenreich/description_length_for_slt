"""End-to-end Stage 1 pipeline on a tiny hierarchical-equality MLP (CPU, a few seconds)."""

import numpy as np
import pytest
import torch

from bitsev.alignment import controls as C
from bitsev.alignment import maps as M
from bitsev.alignment.algorithms import HierarchicalEquality
from bitsev.alignment.learners import TrainingBudget
from bitsev.alignment.network import mlp, reinitialised_twin
from bitsev.alignment.score import gains, label_only_code, score_configuration
from bitsev.alignment.targets import make_targets
from bitsev.codes import schedule as S
from bitsev.testbeds import hierarchical_equality as he
from bitsev.testbeds.training import train_classifier


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    vocab = he.make_vocabulary(6, 2, rng)
    data = he.sample(2000, vocab, rng)
    net = mlp(8, [16, 16], n_outputs=2)
    run = train_classifier(net, data.x, data.y, steps=600, lr=1e-2, seed=0)
    alg = HierarchicalEquality(vocab.dim)
    pool = he.sample(300, vocab, rng).x
    targets = make_targets(alg, pool, 64, "V1", noise_rate=0.05, seed=1)
    eval_targets = make_targets(alg, pool, 200, "V1", noise_rate=0.05, seed=2)
    return net, run, pool, targets, eval_targets


def test_training_checkpoints_and_learning(world):
    net, run, *_ = world
    assert run.steps()[0] == 0 and run.steps()[-1] == 600
    assert run.checkpoints[-1].train_accuracy > run.checkpoints[0].train_accuracy


def test_label_only_code_is_about_n_log_2(world):
    *_, targets, _ = world
    code = label_only_code(targets, S.per_sample(len(targets), start=0)).code
    n = len(targets)
    assert abs(code.total - n * np.log(2)) < 0.5 * np.log(n) + 2.0


def test_pipeline_runs_and_records(world):
    net, run, pool, targets, eval_targets = world
    hyp = C.Configuration("trained/h2/rotation", net, "h2", "rotation", lambda d: M.Rotation(d, 1))
    untrained = C.untrained_control(hyp, reinitialised_twin(net, seed=5))
    wrong = C.wrong_locus_control(hyp, "input")
    other = C.other_map_control(
        hyp, "coupling", lambda d: M.AdditiveCoupling(d, 1, n_blocks=1, width=8)
    )
    sched = S.geometric(len(targets), 2.0, start=4)
    budget = TrainingBudget(steps=40, lr=5e-2)
    results = [
        score_configuration(
            cfg, targets, pool, sched, budget, seed=0, eval_targets={"iid": eval_targets}
        )
        for cfg in (hyp, untrained, wrong, other)
    ]
    g = gains(results[0], results[1:])
    assert set(g) == {untrained.name, wrong.name, other.name}
    for r in results:
        assert r.code.schedule.n == len(targets) and np.all(np.isfinite(r.code.item_losses))
        assert 0 <= r.faithfulness["iid"].iia <= 1
        assert r.targets_hash == targets.content_hash()
        rec = r.record()
        assert len(rec["item_losses"]) == sched.n_coded and rec["sees"] == [
            "labels",
            "locus",
            "network",
        ]
    assert results[0].distortion_bound == 1.0 and results[3].distortion_bound >= 1.0
    with pytest.raises(ValueError):
        C.Configuration(
            "bad", net, "h2", "rotation", lambda d: M.Rotation(d, 1), sees=frozenset({C.ALGORITHM})
        )


def test_mixture_learner_hedges_relative_to_single_map(world):
    """The mixture learner runs on the same schedule and reports the same map size."""
    net, run, pool, targets, eval_targets = world
    hyp = C.Configuration("trained/h1/rotation", net, "h1", "rotation", lambda d: M.Rotation(d, 1))
    sched = S.geometric(len(targets), 2.0, start=4)
    budget = TrainingBudget(steps=30, lr=5e-2)
    single = score_configuration(hyp, targets, pool, sched, budget, seed=0)
    mix = score_configuration(hyp, targets, pool, sched, budget, seed=0, n_members=3)
    assert mix.learner.startswith("mixture3:")
    assert np.all(np.isfinite(mix.code.item_losses)) and len(mix.code.item_losses) == len(
        single.code.item_losses
    )
    assert mix.map_params == single.map_params


def test_least_decodable_locus_prefers_where_v_is_absent(world):
    net, run, pool, targets, _ = world
    from bitsev.alignment.algorithms import HierarchicalEquality
    from bitsev.alignment.controls import least_decodable_locus

    values = HierarchicalEquality(2).variable_values(pool)["V1"].astype(int)
    locus, acc = least_decodable_locus(net, pool, values)
    assert set(acc) == set(net.loci) and all(0 <= a <= 1 for a in acc.values())
    assert locus == min(acc, key=acc.get)
