"""D1 checks from the plan's Appendix D: kernel identity, Poisson law for blind geometry."""

import numpy as np
import pytest

from bitsev.diagnostics import separation as D


def test_first_kernel_birth_equals_separation(rng):
    """Exact identity in 20 random three-colour clouds (plan: 20 of 20)."""
    for _ in range(20):
        n = int(rng.integers(30, 120))
        states = rng.normal(size=(n, 3))
        labels = rng.integers(0, 3, n)
        kr = D.kernel_rank(states, labels)
        assert np.isclose(kr.first_birth, D.separation(states, labels), rtol=0, atol=1e-12)
        assert np.all(kr.rank >= 0) and kr.rank[-1] == len(np.unique(labels)) - 1


def test_nearest_bichromatic_distances_match_brute_force(rng):
    states = rng.normal(size=(60, 2))
    labels = rng.integers(0, 2, 60)
    d = D.bichromatic_nn_distances(states, labels)
    full = np.linalg.norm(states[:, None] - states[None], axis=-1)
    full[labels[:, None] == labels[None]] = np.inf
    np.testing.assert_allclose(d, full.min(axis=1))


@pytest.mark.parametrize("k", [2, 3])
def test_poisson_scaling_on_the_torus(k, rng):
    """Fitted slope of the median sep_V against n within 0.05 of -2/k over n = 2^7 ... 2^12,
    and medians within 25% of the Poisson prediction with 40 replicates (the plan's full
    check uses 200 replicates up to 2^15 and finds 10%)."""
    sizes = 2 ** np.arange(7, 13)
    medians = []
    for n in sizes:
        seps = [
            D.separation(rng.random((n, k)), rng.integers(0, 2, n), boxsize=1.0) for _ in range(40)
        ]
        medians.append(np.median(seps))
    slope = np.polyfit(np.log(sizes), np.log(medians), 1)[0]
    assert abs(slope + 2 / k) < 0.05, slope
    for n, med in zip(sizes, medians, strict=True):
        pred = D.poisson_median_separation(int(n), k, np.array([0.5, 0.5]))
        assert 0.75 < med / pred < 1.25, (n, med, pred)


def test_computed_variable_keeps_separation_bounded(rng):
    """States that encode V with a margin have sep_V >= margin for every n."""
    for n in (100, 1000, 5000):
        labels = rng.integers(0, 2, n)
        states = np.column_stack([labels * 2.0 + 0.1 * rng.normal(size=n), rng.normal(size=n)])
        assert D.separation(states, labels) > 1.0


def test_shuffle_null_detects_structure(rng):
    n = 300
    labels = rng.integers(0, 2, n)
    structured = np.column_stack([labels * 3.0, rng.normal(size=n)])
    blind = rng.normal(size=(n, 2))
    assert D.shuffle_null(structured, labels, n_shuffles=99).p_value < 0.02
    assert D.shuffle_null(blind, labels, n_shuffles=99).p_value > 0.05
    assert D.stretch_bits_required(0.25) == 2.0 and D.stretch_bits_required(2.0) == 0.0


def test_margin_ratio_separates_a_computed_variable_from_a_blind_one(rng):
    n = 400
    labels = rng.integers(0, 2, n)
    computed = rng.normal(size=(n, 3)) * 0.1 + 3.0 * labels[:, None]
    blind = rng.normal(size=(n, 3))
    assert D.margin_statistics(computed, labels).sep_ratio > 1.0
    assert D.margin_statistics(computed, labels).nn_ratio > 5.0
    stats = D.margin_statistics(blind, labels)
    assert stats.sep_ratio < 1.0 and 0.5 < stats.nn_ratio < 2.0
