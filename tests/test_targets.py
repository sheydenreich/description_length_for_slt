import numpy as np

from bitsev.alignment.algorithms import HierarchicalEquality
from bitsev.alignment.targets import make_targets
from bitsev.testbeds import hierarchical_equality as he


def test_task_labels_are_consistent(rng):
    vocab = he.make_vocabulary(8, 3, rng)
    data = he.sample(500, vocab, rng)
    assert np.array_equal(data.y, (data.v1 == data.v2).astype(int))
    v1, v2 = he.variables_from_input(data.x, vocab.dim)
    assert np.array_equal(v1, data.v1) and np.array_equal(v2, data.v2)
    assert 0.35 < data.v1.mean() < 0.65 and 0.35 < data.y.mean() < 0.65
    # unequal pairs really are different vectors
    assert np.all(data.slots[~data.v1, 0] != data.slots[~data.v1, 1])


def test_disjoint_vocabularies_do_not_share_vectors(rng):
    a, b = he.make_disjoint_vocabularies(5, 2, rng)
    assert not np.any(np.all(a.vectors[:, None] == b.vectors[None], axis=-1))


def test_counterfactual_targets(rng):
    vocab = he.make_vocabulary(6, 2, rng)
    data = he.sample(300, vocab, rng)
    alg = HierarchicalEquality(vocab.dim)
    t = make_targets(alg, data.x, 200, "V1", noise_rate=0.0, seed=1)
    expected = (data.v1[t.source] == data.v2[t.base]).astype(int)
    assert np.array_equal(t.y_clean, expected) and np.array_equal(t.y, expected)
    assert np.all(t.base != t.source)


def test_randomized_response_rate_and_hash(rng):
    vocab = he.make_vocabulary(6, 2, rng)
    data = he.sample(400, vocab, rng)
    alg = HierarchicalEquality(vocab.dim)
    t = make_targets(alg, data.x, 4000, "V2", noise_rate=0.2, seed=3)
    flipped = np.mean(t.y != t.y_clean)
    assert abs(flipped - 0.1) < 0.02  # a uniform redraw agrees half the time
    assert t.content_hash() == make_targets(alg, data.x, 4000, "V2", 0.2, 3).content_hash()
    assert t.content_hash() != make_targets(alg, data.x, 4000, "V2", 0.2, 4).content_hash()
    np.testing.assert_allclose(t.target_law(), 0.9)
