"""System 04: predator-prey agent-based models (Lotka-Volterra abstraction).

Original: ``systems/04_predator_prey.py`` of MelouxM/CAE (``AgentBasedModel``, numba kernels
``_run_nonspatial_step`` and ``_run_spatial_simulation``, ``solve_lotka_volterra``,
``calibrate_high_level_model``).

Variables (numpy order): ``prey_t`` (batch, 1), ``predator_t`` (batch, 1),
``final_populations`` (batch, 2).  Inputs: ``prey_t``, ``predator_t`` (defaults 100, 40).

Semantics reproduced exactly:

* Interventions on ``prey_t`` / ``predator_t`` replace the inputs (they are the only
  variables the original reads; an intervention on ``final_populations`` is ignored, as in
  the original).  A size-1 batch of one input is broadcast against the other (the original
  uses ``np.resize``, which for size 1 is the same thing).
* Non-spatial deterministic variant: ``num_steps`` Euler-like float steps with the
  ``prey < 1 or pred < 1 -> (0, 0)`` gate and ``max(0, .)`` clamps.  Deterministic and
  differentiable almost everywhere (the gate and the clamps are piecewise constant).
* Non-spatial stochastic variant: binomial draws in the exact order of the original
  (births, eaten, predator births, starvation, then the two aging deaths), with
  ``int()`` truncation of the float populations.  Not differentiable.
* Spatial variant: the sequential grid algorithm of ``_run_spatial_simulation`` (random
  placement by rejection, shuffled agent order, death / reproduction / move / predation,
  then births spawned into shuffled empty cells), run per batch item in Python on an
  ``int8`` grid tensor, with every random draw in the original order.  Not differentiable,
  not vectorised (the algorithm is inherently sequential).  Unlike the original, which
  loops forever if the initial agents do not fit on the grid, this raises ``ValueError``.

Randomness: the original uses numba's internal ``np.random`` stream.  Here all draws go
through an :class:`AgentRNG`; the default :class:`TorchAgentRNG` uses ``torch.binomial``,
``torch.randint``, ``torch.rand`` and ``torch.randperm`` on an optional
``torch.Generator``.  Samples therefore differ from the original and only distributions
can be compared.  Because numba's stream is bit-identical to numpy's legacy
``RandomState``, the tests inject a replaying RNG (one ``RandomState`` per batch item,
same seed as the numba stream of the corresponding original run) and check exact
equivalence of the stochastic variants as well.

High-level abstraction (as in the original and ``test/04_predator_prey.py``):

* Lotka-Volterra Euler integration :func:`solve_lotka_volterra` with parameters
  ``alpha, beta, delta, gamma`` and ``dt = num_steps``.
* Calibration: grid search :data:`LV_SEARCH_SPACE` against the ideal ABM
  (non-spatial, deterministic, no aging) started at (100, 40) from :data:`BASE_CONFIG`.
* Valid condition: calibrated LV versus the ideal ABM.  Invalid conditions:
  ``fail_alpha`` (``alpha * 1.5``), ``spatial``, ``stochastic``, ``aging``, ``complex``
  (all three toggles) and ``noise`` (Gaussian noise of std 5.0 on the ABM output).
* Coarse-graining: identity on ``prey_t``, ``predator_t``, ``final_populations``; value
  map continuous with ``prey_t`` in [50, 250], ``predator_t`` in [20, 100].
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from cae_torch.base import TorchSystem

F64 = torch.float64

# High-level abstraction data (plain Python) ------------------------------------------

BASE_CONFIG: dict[str, Any] = {
    "prey_reproduce_prob": 0.1,
    "predation_prob": 0.001,
    "predator_reproduce_prob": 0.5,
    "predator_starvation_prob": 0.05,
    "num_steps": 50,
    "grid_size": 20,
    "prey_age_death_prob": 0.02,
    "predator_age_death_prob": 0.04,
}

IDEAL_CONFIG: dict[str, Any] = dict(
    BASE_CONFIG, spatial=False, stochastic_reproduction=False, agent_aging=False
)

LV_SEARCH_SPACE: dict[str, tuple[float, ...]] = {
    "alpha": (0.08, 0.10, 0.12),
    "beta": (0.0008, 0.0010, 0.0012),
    "delta": (0.0004, 0.0005, 0.0006),
    "gamma": (0.04, 0.05, 0.06),
}

PREY_DOMAIN = (50.0, 250.0)
PREDATOR_DOMAIN = (20.0, 100.0)
COARSE_GRAINING = {"prey_t": ["prey_t"], "predator_t": ["predator_t"],
                   "final_populations": ["final_populations"]}
CONDITIONS = {
    "valid": dict(spatial=False, stochastic=False, aging=False, alpha_factor=1.0, noise_std=0.0),
    "fail_alpha": dict(spatial=False, stochastic=False, aging=False, alpha_factor=1.5, noise_std=0.0),
    "spatial": dict(spatial=True, stochastic=False, aging=False, alpha_factor=1.0, noise_std=0.0),
    "stochastic": dict(spatial=False, stochastic=True, aging=False, alpha_factor=1.0, noise_std=0.0),
    "aging": dict(spatial=False, stochastic=False, aging=True, alpha_factor=1.0, noise_std=0.0),
    "noise": dict(spatial=False, stochastic=False, aging=False, alpha_factor=1.0, noise_std=5.0),
    "complex": dict(spatial=True, stochastic=True, aging=True, alpha_factor=1.0, noise_std=0.0),
}


def solve_lotka_volterra(prey_t: float, predator_t: float, params: Mapping[str, float]) -> tuple[float, float]:
    """Euler integration of the LV equations for ``params['dt']`` steps (original, plain floats)."""
    prey, pred = float(prey_t), float(predator_t)
    for _ in range(int(params["dt"])):
        if prey < 1e-3 or pred < 1e-3:
            break
        d_prey = (params["alpha"] * prey) - (params["beta"] * prey * pred)
        d_pred = (params["delta"] * prey * pred) - (params["gamma"] * pred)
        prey += d_prey
        pred += d_pred
    return max(0.0, prey), max(0.0, pred)


def calibrate_high_level_model(ideal_model: "AgentBasedModel",
                               search_space: Mapping[str, Sequence[float]] = LV_SEARCH_SPACE) -> dict[str, float]:
    """Grid-search the LV parameters against the ideal ABM run from (100, 40), as the original."""
    out = ideal_model.forward({"prey_t": torch.tensor([[100.0]], dtype=F64),
                               "predator_t": torch.tensor([[40.0]], dtype=F64)})
    target = out["final_populations"][0].tolist()
    names = list(search_space.keys())
    best, min_err = None, float("inf")
    for combo in itertools.product(*search_space.values()):
        params = dict(zip(names, combo))
        params["dt"] = ideal_model.steps
        pr, pd = solve_lotka_volterra(100.0, 40.0, params)
        err = (pr - target[0]) ** 2 + (pd - target[1]) ** 2
        if err < min_err:
            min_err, best = err, params
    return best


# Random number source ------------------------------------------------------------------

class AgentRNG:
    """Protocol for the random draws of the ABM. ``b`` is the batch index of the caller.

    ``binomial`` is batched (used by the non-spatial stochastic variant): ``n`` is an int64
    tensor of counts, ``mask`` tells which items actually draw (the original does not draw
    for items whose populations collapsed); masked-out entries must be 0.  The scalar
    methods are used by the sequential spatial variant.
    """

    def binomial(self, n: Tensor, p: float, mask: Tensor) -> Tensor:
        raise NotImplementedError

    def randint(self, b: int, low: int, high: int) -> int:
        raise NotImplementedError

    def rand(self, b: int) -> float:
        raise NotImplementedError

    def permutation(self, b: int, n: int) -> Sequence[int]:
        raise NotImplementedError


class TorchAgentRNG(AgentRNG):
    """Default RNG on a ``torch.Generator`` (samples differ from the numpy original)."""

    def __init__(self, generator: torch.Generator | None = None):
        self.generator = generator

    def binomial(self, n: Tensor, p: float, mask: Tensor) -> Tensor:
        counts = n.to(F64)
        probs = torch.full_like(counts, float(p))
        draws = torch.binomial(counts, probs, generator=self.generator)
        return torch.where(mask, draws, torch.zeros_like(draws))

    def randint(self, b: int, low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=self.generator).item())

    def rand(self, b: int) -> float:
        return float(torch.rand((1,), generator=self.generator, dtype=F64).item())

    def permutation(self, b: int, n: int) -> Sequence[int]:
        return torch.randperm(n, generator=self.generator).tolist()


# Kernels ---------------------------------------------------------------------------------

def nonspatial_step(prey: Tensor, pred: Tensor, params: Sequence[float], use_stochastic: bool,
                    use_aging: bool, rng: AgentRNG | None = None) -> tuple[Tensor, Tensor]:
    """One batched tick of ``_run_nonspatial_step``. ``prey``, ``pred``: (batch,) float64."""
    r_prey, p_pred, r_pred, s_pred, d_prey_age, d_pred_age = (float(x) for x in params)
    alive = (prey >= 1) & (pred >= 1)
    zero = torch.zeros_like(prey)

    if use_stochastic:
        if rng is None:
            rng = TorchAgentRNG()
        n_prey = prey.floor().to(torch.int64)      # int(prey), prey >= 0
        n_pred = pred.floor().to(torch.int64)
        births = rng.binomial(n_prey, r_prey, alive)
        eaten = rng.binomial(n_prey * n_pred, p_pred, alive)
        eaten = torch.where(eaten > prey, n_prey.to(F64), eaten)
        pred_births = rng.binomial(eaten.to(torch.int64), r_pred, alive)
        pred_starve = rng.binomial(n_pred, s_pred, alive)
        age_death_prey = zero
        age_death_pred = zero
        if use_aging:
            age_death_prey = rng.binomial((prey - eaten).floor().to(torch.int64), d_prey_age, alive)
            age_death_pred = rng.binomial(n_pred, d_pred_age, alive)
    else:
        births = prey * r_prey
        eaten = prey * pred * p_pred
        pred_births = eaten * r_pred
        pred_starve = pred * s_pred
        age_death_prey = zero
        age_death_pred = zero
        if use_aging:
            age_death_prey = (prey - eaten) * d_prey_age
            age_death_pred = pred * d_pred_age

    new_prey = prey + births - eaten - age_death_prey
    new_pred = pred + pred_births - pred_starve - age_death_pred
    new_prey = torch.clamp(new_prey, min=0.0)
    new_pred = torch.clamp(new_pred, min=0.0)
    return torch.where(alive, new_prey, zero), torch.where(alive, new_pred, zero)


def spatial_simulation(b: int, steps: int, init_prey: int, init_pred: int, grid_size: int,
                       params: Sequence[float], use_aging: bool, rng: AgentRNG) -> tuple[float, float]:
    """Sequential translation of ``_run_spatial_simulation`` for batch item ``b``.

    Grid codes: 0 empty, 1 prey, 2 predator. Every random draw happens in the original
    order so that a replaying RNG reproduces the original exactly.
    """
    r_prey = float(params[0])
    r_pred = float(params[2])
    s_pred = float(params[3])
    d_prey_age = float(params[4])
    d_pred_age = float(params[5])

    if init_prey + init_pred > grid_size * grid_size:
        raise ValueError("initial agents do not fit on the grid (the original would loop forever)")
    grid = torch.zeros((grid_size, grid_size), dtype=torch.int8)

    placed = 0
    while placed < init_prey:
        r, c = rng.randint(b, 0, grid_size), rng.randint(b, 0, grid_size)
        if grid[r, c] == 0:
            grid[r, c] = 1
            placed += 1
    placed = 0
    while placed < init_pred:
        r, c = rng.randint(b, 0, grid_size), rng.randint(b, 0, grid_size)
        if grid[r, c] == 0:
            grid[r, c] = 2
            placed += 1

    for _ in range(steps):
        occupied = grid.nonzero().tolist()          # row-major, as the original double loop
        n_pop = len(occupied)
        if n_pop == 0:
            break
        order = rng.permutation(b, n_pop)
        prey_birth_queue = 0
        pred_birth_queue = 0

        for idx in order:
            r, c = occupied[int(idx)]
            val = int(grid[r, c])
            if val == 0:
                continue
            is_prey = val == 1

            died = False
            if is_prey:
                if use_aging and rng.rand(b) < d_prey_age:
                    grid[r, c] = 0
                    died = True
            else:
                prob_death = s_pred + (d_pred_age if use_aging else 0.0)
                if rng.rand(b) < prob_death:
                    grid[r, c] = 0
                    died = True
            if died:
                continue

            if is_prey:
                if rng.rand(b) < r_prey:
                    prey_birth_queue += 1

            dr = rng.randint(b, -1, 2)
            dc = rng.randint(b, -1, 2)
            if dr == 0 and dc == 0:
                continue
            nr, nc = (r + dr) % grid_size, (c + dc) % grid_size
            target = int(grid[nr, nc])

            if target == 0:
                grid[nr, nc] = val
                grid[r, c] = 0
            elif is_prey and target == 2:
                grid[r, c] = 0
                if rng.rand(b) < r_pred:
                    pred_birth_queue += 1
            elif (not is_prey) and target == 1:
                grid[nr, nc] = 2
                grid[r, c] = 0
                if rng.rand(b) < r_pred:
                    pred_birth_queue += 1

        empties = (grid == 0).nonzero().tolist()
        n_empty = len(empties)
        if n_empty > 0:
            spawn_order = rng.permutation(b, n_empty)
            ptr = 0
            for _ in range(min(prey_birth_queue, n_empty)):
                rr, cc = empties[int(spawn_order[ptr])]
                grid[rr, cc] = 1
                ptr += 1
            remaining = n_empty - ptr
            for _ in range(min(pred_birth_queue, remaining)):
                rr, cc = empties[int(spawn_order[ptr])]
                grid[rr, cc] = 2
                ptr += 1

    return float((grid == 1).sum().item()), float((grid == 2).sum().item())


# Low-level model ----------------------------------------------------------------------

class AgentBasedModel(TorchSystem):
    """Torch translation of ``AgentBasedModel(config_dict)``.

    Extra keyword arguments (not in the original): ``generator`` for the default torch
    RNG and ``rng`` to supply any :class:`AgentRNG` (used by the tests to replay the
    numpy stream exactly).  Only the stochastic / spatial variants draw random numbers.
    """

    variables = ("prey_t", "predator_t", "final_populations")
    input_variables = ("prey_t", "predator_t")

    def __init__(self, config_dict: Mapping[str, Any], generator: torch.Generator | None = None,
                 rng: AgentRNG | None = None):
        self.cfg = dict(config_dict)
        self.params = (
            float(self.cfg["prey_reproduce_prob"]),
            float(self.cfg["predation_prob"]),
            float(self.cfg["predator_reproduce_prob"]),
            float(self.cfg["predator_starvation_prob"]),
            float(self.cfg.get("prey_age_death_prob", 0)),
            float(self.cfg.get("predator_age_death_prob", 0)),
        )
        self.steps = int(self.cfg["num_steps"])
        self.grid_size = int(self.cfg["grid_size"])
        self.spatial = bool(self.cfg.get("spatial", False))
        self.stochastic = bool(self.cfg.get("stochastic_reproduction", False))
        self.aging = bool(self.cfg.get("agent_aging", False))
        self.generator = generator
        self.rng = rng if rng is not None else TorchAgentRNG(generator)

    @staticmethod
    def _as_batch(value, default: float) -> Tensor:
        if value is None:
            return torch.tensor([[default]], dtype=F64)
        t = torch.as_tensor(value, dtype=F64)
        if t.ndim == 0:
            t = t.reshape(1, 1)
        return t

    def forward_with_interventions(self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]) -> dict[str, Tensor]:
        prey_in = self._as_batch(interventions.get("prey_t", inputs.get("prey_t")), 100.0)
        pred_in = self._as_batch(interventions.get("predator_t", inputs.get("predator_t")), 40.0)
        batch = max(prey_in.shape[0], pred_in.shape[0])
        if prey_in.shape[0] < batch:          # np.resize: cyclic repetition
            prey_in = prey_in.repeat((batch + prey_in.shape[0] - 1) // prey_in.shape[0], *([1] * (prey_in.ndim - 1)))[:batch]
        if pred_in.shape[0] < batch:
            pred_in = pred_in.repeat((batch + pred_in.shape[0] - 1) // pred_in.shape[0], *([1] * (pred_in.ndim - 1)))[:batch]

        prey = prey_in.reshape(batch, -1)[:, 0]
        pred = pred_in.reshape(batch, -1)[:, 0]

        if self.spatial:
            rows = []
            for b in range(batch):
                f_prey, f_pred = spatial_simulation(
                    b, self.steps, int(prey[b].item()), int(pred[b].item()),
                    self.grid_size, self.params, self.aging, self.rng)
                rows.append([f_prey, f_pred])
            final = torch.tensor(rows, dtype=F64).reshape(batch, 2)
        else:
            cur_p, cur_pr = prey, pred
            for _ in range(self.steps):
                cur_p, cur_pr = nonspatial_step(cur_p, cur_pr, self.params, self.stochastic, self.aging, self.rng)
            final = torch.stack([cur_p, cur_pr], dim=1)

        return {"prey_t": prey_in, "predator_t": pred_in, "final_populations": final}
