import numpy as np
import pytest

from bitsev.codes import schedule as S


def test_batching_factor_values():
    assert S.batching_factor(1) == 1.0
    assert np.isclose(S.batching_factor(2), 1 / np.log(2))
    assert np.isclose(S.batching_factor(4), 3 / np.log(4))
    with pytest.raises(ValueError):
        S.batching_factor(0.5)


def test_per_sample_schedule():
    s = S.per_sample(5, start=1)
    assert s.boundaries.tolist() == [1, 2, 3, 4, 5]
    assert s.n == 5 and s.start == 1 and s.n_coded == 4 and s.n_blocks == 4
    assert s.fit_sizes().tolist() == [1, 2, 3, 4]
    assert list(s.blocks()) == [(1, 2), (2, 3), (3, 4), (4, 5)]


def test_geometric_matches_plan_definition():
    # n_j = ceil(r^j), deduplicated, clipped to n
    s = S.geometric(20, growth=2.0)
    assert s.boundaries.tolist() == [1, 2, 4, 8, 16, 20]
    s = S.geometric(20, growth=1.5)
    expected = sorted(set(int(np.ceil(1.5**j)) for j in range(12)) | {20})
    assert s.boundaries.tolist() == [b for b in expected if b <= 20]


def test_geometric_fit_sizes_repeat_block_starts():
    s = S.geometric(10, growth=2.0)  # boundaries 1,2,4,8,10
    assert s.fit_sizes().tolist() == [1, 2, 2, 4, 4, 4, 4, 8, 8]


def test_invalid_schedules():
    with pytest.raises(ValueError):
        S.Schedule(np.array([3, 2]))
    with pytest.raises(ValueError):
        S.Schedule(np.array([2]))
    with pytest.raises(ValueError):
        S.geometric(10, growth=2.0, start=0)
