import numpy as np
import pytest

from bitsev.diagnostics import realizability as R

gudhi = pytest.importorskip("gudhi")


def test_statistics_vanish_when_intervened_states_are_natural(rng):
    natural = rng.normal(size=(200, 3))
    intervened = natural[:50].copy()
    spacing = float(np.median(R.nearest_neighbour_spacing(natural)))
    assert spacing > 0
    np.testing.assert_allclose(R.counterfactual_errors(intervened, natural[:50], spacing), 0.0)
    assert R.one_sided_hausdorff(intervened, natural) == 0.0
    ip = R.inclusion_persistence(natural, intervened, max_points=200)
    assert ip.hausdorff == 0.0 and all(v < 1e-12 for v in ip.bottleneck.values())


def test_off_manifold_states_are_detected_and_stability_bound_holds(rng):
    natural = rng.normal(size=(150, 2))
    far = rng.normal(size=(30, 2)) + np.array([12.0, 0.0])
    ip = R.inclusion_persistence(natural, far, max_points=200)
    assert ip.hausdorff > 8.0
    assert ip.bottleneck[0] > 0.0
    assert ip.within_bound()
    assert R.realizability_bound(distortion=1.0, eps=0.1) == pytest.approx((np.sqrt(2) + 1) * 0.1)


def test_loop_filled_by_intervened_states_changes_degree_one(rng):
    angles = rng.uniform(0, 2 * np.pi, 120)
    circle = np.column_stack([np.cos(angles), np.sin(angles)]) + 0.02 * rng.normal(size=(120, 2))
    filling = 0.6 * rng.uniform(-1, 1, size=(80, 2))
    ip = R.inclusion_persistence(circle, filling, max_points=200, max_edge=2.5)
    assert ip.bottleneck[1] > 0.3  # the long degree-one bar of the circle is destroyed
