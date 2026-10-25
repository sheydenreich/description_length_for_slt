import numpy as np

from bitsev.baselines.probing_mdl import online_probe_code, uniform_code_nats
from bitsev.codes import schedule as S


def test_probe_compresses_a_linearly_readable_variable(rng):
    n = 400
    v = rng.integers(0, 2, n)
    states = np.column_stack([v * 3.0 + rng.normal(size=n), rng.normal(size=(n, 4))])
    sched = S.geometric(n, 2.0, start=8)
    code = online_probe_code(states, v, sched)
    assert code.total < 0.5 * uniform_code_nats(sched.n_coded, 2)
    assert np.all(np.isfinite(code.item_losses))


def test_probe_cannot_compress_noise(rng):
    n = 400
    v = rng.integers(0, 3, n)
    states = rng.normal(size=(n, 6))
    sched = S.geometric(n, 2.0, start=8)
    code = online_probe_code(states, v, sched)
    assert code.total > 0.9 * uniform_code_nats(sched.n_coded, 3)
