# cae-torch: the CAE benchmark systems in PyTorch

Faithful PyTorch translations of the simulated complex systems of
*Validating causal abstraction metrics on simulated complex systems* (Méloux, Pimentel,
Portet, Peyrard 2026; code `MelouxM/CAE`, MIT). The originals are numpy structural models
driven through `forward_with_interventions(SystemState, interventions)`. These versions are
batched torch modules with the same semantics, so that interventions on micro-variables can
be optimised by gradient descent where the dynamics are differentiable, and so that the
systems can be used from any torch code base.

This directory is self-contained (its own `pyproject.toml`) so it can be shared on its own.

## Contract (every system follows it)

```python
from cae_torch.base import TorchSystem

sys = SomeSystem(**params)                     # same constructor parameters as the numpy class
sys.variables         # tuple of all micro-variable names, in the numpy model's order
sys.input_variables   # names the numpy model expects in the input SystemState
out = sys.forward_with_interventions(inputs, interventions)
```

* `inputs` and `interventions` map variable names to tensors with a leading batch
  dimension; per-variable trailing shape equals the numpy model's per-sample shape
  (a scalar variable is `(batch, 1)`, matching `SystemState` conventions).
* `out` maps **every** variable name to a `(batch, ...)` tensor, including inputs,
  internals and outputs; it equals the numpy `forward_with_interventions(...).values`
  stacked over the batch.
* Intervened variables are **clamped** to the given values for the whole computation,
  exactly as in the numpy model (check how each original treats "forced" variables).
* Default dtype is `torch.float64` so that equivalence with numpy is exact to 1e-9;
  callers may convert.
* No randomness inside `forward_with_interventions` unless the numpy model has it; then the
  constructor takes `generator: torch.Generator | None` and the tests compare
  distributions, not samples (document this in the system's docstring).
* Discrete steps (thresholds, `argmax`, boolean gates) are kept **exact** in the forward
  pass. If a smooth relaxation is natural (for example a sigmoid gate), expose it as an
  optional constructor flag `soft: bool = False`; it must default to the exact behaviour.
* Each module records in its docstring: the original file, the variables, which steps are
  non-differentiable, and the parameters of the valid / invalid high-level abstractions
  as defined in the original file (as plain data, no `causal_abstraction` dependency).

## Equivalence tests

`tests/test_<nn>_<name>.py` loads the original numpy system by path from the CAE clone
(`cae_torch.reference.load_numpy_system`, directory from `CAE_DIR`, default
`~/scratch/third_party/CAE`), builds both systems with identical parameters, draws seeded
random inputs and interventions (including interventions on internal variables), and
asserts `torch.allclose(out[name], numpy_out[name], atol=1e-9, rtol=1e-9)` for every
variable, for batch sizes 1 and 8. A second test checks gradients flow through the
differentiable path (`torch.autograd.gradcheck` on a small case, or at least that
`.grad` is finite and nonzero) and states explicitly if the system has no differentiable
path. Tests skip, with a message, if the CAE clone or its dependencies are absent.

## Status

| # | System | Module | Equivalence | Differentiable |
|---|---|---|---|---|
| 01 | Logic circuit | `s01_logic_circuit.NetlistSimulator` | exact (1e-9), inputs and interventions on internal wires | no (0.5 thresholds, boolean gates); `soft=True` product relaxation is differentiable in all wires |
| 02 | Transistor circuit | `s02_transistor_circuit.GenericSpiceModel` | 1e-2 V (observed ≤ 2e-3 V on floating nodes, < 1e-7 V elsewhere; limited by ngspice reltol 1e-3) | yes: every node voltage w.r.t. every driven voltage (implicit gradient through a final autograd Newton step) |
| 03 | Gas simulation | `s03_gas_simulation.UniversalGasModel` | exact (1e-9) on the full MD trajectory via noise replay, all four modes; energy minimisation of overlapping configurations 1e-6 | yes a.e.: positions, velocities, volume, temperature, pressure |
| 04 | Predator-prey | `s04_predator_prey.AgentBasedModel` | exact (1e-9) for all eight variants via RNG replay, plus distributional check | partial: deterministic non-spatial variants; stochastic draws detached; spatial none |
| 05 | Heat equation 1D | `s05_heat_equation.BrownianParticleSystem` | exact for deterministic parts; distributional for the final temperature | yes: final temperature w.r.t. initial temperature for fixed noise |
| 06 | Heat equation 2D | `s06_heat_equation_2d.PhononLatticeModel` | exact (0.0) without scattering using replayed kick angles; distributional with scattering | partial: temperature map w.r.t. source energy; source position not (integer truncation) |
| 07 | Ising model | `s07_ising_model.MolecularDynamicsIsingModel` | inputs, seed and common random numbers exact; magnetisation distributional (mean within 0.08, observed ≤ 0.021) | no (Metropolis accept/reject on discrete spins) |
| 09 | Gene regulatory network | `s09_grn.SegmentPolarityLowLevelModel` | exact; embedded network equals the parsed `.ginml` | no (integer thresholds) |
| 11 | Controlled experiments | `s11_controlled_experiments.*` (six models) | exact | yes for 1, 3, 4, 6; models 2 and 5 are gated (exact gate has no gradient; `soft=True` sigmoid option) |

Run the suite with `cd cae_torch && pytest` (about a minute). The transistor tests need
`LD_LIBRARY_PATH=$HOME/scratch/ngspice_env/lib` (libngspice from conda-forge) and the
reference tests need the clone at `~/scratch/third_party/CAE` (or `CAE_DIR`); they skip
otherwise. Stochastic systems take an injectable RNG so that trajectories can be replayed
from the numpy originals; numba's `np.random` stream is bit-identical to numpy's legacy
`RandomState`, which is what makes the exact replay possible.

Systems 08 (tracr, needs JAX) and 10 (MOS 6502, needs native emulator libraries) are out
of scope.
