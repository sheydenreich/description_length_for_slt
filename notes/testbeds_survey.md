# Ground-truth testbeds and manufactured-alignment tooling: web survey

Date: 2026-10-15. Scope: weeks 3-4 of `research_plan_synthesis_v7_2.tex`
("Ground truth and manufactured alignments": compiled transformers, semi-synthetic
models with known circuits, MIB causal-variable tasks, Sutter-style expressive
maps on untrained networks). Environment: PyTorch 2.14, Linux cluster, internet.
Everything below is read from the cited pages; items marked *inferred* were not
verified against code or docs.

## Summary table

| Item | Available | Main dependency | Blocker / caveat |
|---|---|---|---|
| 1. InterpBench (86 SIIT transformers + ground-truth circuits) | yes (weights, CC-BY-4.0; code MIT) | `transformer_lens` (pinned 1.19.0 in repo), `iit` (git, no licence), `tracr` fork (JAX) only for the *high-level* tracr model | Low-level `.pth` + pickled `HookedTransformerConfig` load with TransformerLens alone; the tracr HL model needs JAX+haiku; unpickling a TL-1.19 config under TL 4.x is a compatibility risk (*inferred*) |
| 2. MIB causal-variable track | yes (datasets MIT on HF; code Apache-2.0) | `pyvene`, `transformers`, `sae_lens` (py3.12), `CausalAbstraction` submodule (now `goodfire-ai/causalab`) | Smallest model GPT-2 small (IOI only); other tasks need Gemma-2-2B / Llama-3.1-8B (gated HF). Submodule renamed; pin the commit |
| 3. Meloux et al. 2026, CAE benchmark | yes (`causal-abstraction-eval` 1.0.0 on PyPI, MIT) | numpy/scipy only; extras torch, numba, PySpice | Python >= 3.11; system 08 needs `tracr` (JAX); system 10 needs native 6502 emulator libs built locally |
| 4. Sutter et al. 2025, non-linear dilemma | yes (code), no licence file | plain `torch`+`transformers` (own DAS/RevNet impl, no pyvene) | No LICENSE (reuse = read, re-implement); Pythia via HF; default script targets Llama-3.2-1B (gated) |
| 5. pyvene 0.1.8 | yes (Apache-2.0) | `transformers>=4.45.1`, `torch>=2.0` | No `HookedTransformer` in the model card; HF classes or custom `nn.Module` via `type_to_module_mapping` |
| tracr (DeepMind) | yes (Apache-2.0) | `jax`, `dm-haiku`, `chex` | JAX required to compile; InterpBench ships PyTorch weights so tracr is unneeded for the low-level model |

## 1. InterpBench

**Artefacts.** Paper arXiv:2407.14494 (NeurIPS 2024 D&B, CC-BY-4.0)
[https://arxiv.org/abs/2407.14494]. Code (MIT): `FlyingPumba/circuits-benchmark`
(the paper's `FlyingPumba/InterpBench` link resolves there)
[https://github.com/FlyingPumba/circuits-benchmark]. Weights (CC-BY-4.0):
`cybershiptrooper/InterpBench` on HF, 84 numeric case dirs + `ioi` + `ioi_next_token`,
each with `ll_model.pth`, `ll_model_cfg.pkl`, `meta.json`, `edges.pkl`
[https://huggingface.co/cybershiptrooper/InterpBench],
[https://huggingface.co/api/models/cybershiptrooper/InterpBench/tree/main].
`benchmark_metadata.json` (339 kB) lists per case: `case_id`, `task_description`, `vocab`,
`max_seq_len`, `transformer_cfg`, `training_args`, file URLs
[https://huggingface.co/cybershiptrooper/InterpBench/resolve/main/benchmark_metadata.json].
The paper counts 86 models: 85 tracr-derived + IOI
[https://arxiv.org/html/2407.14494v3].

**Sizes.** Tracr cases are tiny: e.g. case 52 ("square root of each element") 2 layers,
4 heads, d_model 4; case 7 (token counts) 2L/4H/d_model 17; case 82 is 4L/4H/d_model 24;
all with `max_seq_len` 10. IOI case: 6 layers, 4 heads, d_model 64, d_head 16,
d_vocab 50257 (GPT-2 tokenizer), seq_len 16 [metadata JSON above].

**Dependencies.** `circuits-benchmark` is Poetry-managed, python ^3.10, pins
`transformer-lens = "1.19.0"`, `torch>=2.2`, `transformers ^4.37`, `chex`, `dm-haiku`,
and git deps `tracr` (FlyingPumba fork), `iit` (cybershiptrooper), `auto-circuit`, `acdc`
[https://raw.githubusercontent.com/FlyingPumba/circuits-benchmark/main/pyproject.toml].
The tracr fork's `install_requires` includes `jax`
[https://raw.githubusercontent.com/FlyingPumba/tracr/main/setup.py]. `iit` needs only
`torch ^2.3.1`, `transformer-lens ^1.19.0`, `networkx`
[https://raw.githubusercontent.com/cybershiptrooper/iit/main/pyproject.toml]; its GitHub
licence field is null (no LICENSE file) [https://api.github.com/repos/cybershiptrooper/iit].
Current TransformerLens is 4.2.0 (`torch>=2.6`, `transformers>=5.9`)
[https://pypi.org/pypi/transformer-lens/json], so the 1.19 pin conflicts with a modern env.

**Loading the low-level model (PyTorch only).** The repo loader downloads the two files,
unpickles the config (`HookedTransformerConfig.from_dict` if dict, else the object),
and calls `HookedTransformer(cfg).load_state_dict(torch.load(...))`; for IOI it also sets
`use_hook_mlp_in`, `use_attn_result`, `use_split_qkv_input`
[https://raw.githubusercontent.com/FlyingPumba/circuits-benchmark/main/circuits_benchmark/utils/ll_model_loader/interp_bench_model_loader.py].
Reconstructed from that description (*inferred* snippet, not copied verbatim):

```python
import pickle, torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig
case = "3"
w = hf_hub_download("cybershiptrooper/InterpBench", subfolder=case, filename="ll_model.pth")
c = hf_hub_download("cybershiptrooper/InterpBench", subfolder=case, filename="ll_model_cfg.pkl")
cfg = pickle.load(open(c, "rb"))
cfg = HookedTransformerConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
model = HookedTransformer(cfg); model.load_state_dict(torch.load(w, map_location="cpu"))
_, cache = model.run_with_cache(tokens)          # cache["blocks.1.attn.hook_result"][:, pos, head]
```

**Ground truth.** The HL model is the tracr-compiled program, converted to a TransformerLens
`HookedTracrTransformer` (`from_tracr_model(tracr_model)`, stores `residual_stream_labels`
and `tracr_input_encoder`; imports `jax.numpy` only for type hints)
[https://raw.githubusercontent.com/FlyingPumba/circuits-benchmark/main/circuits_benchmark/transformers/hooked_tracr_transformer.py].
Compiling it (`case.get_hl_model()`) calls `compiling.compile_rasp_to_model` each time, so
JAX is needed for the HL side [tracr_benchmark_case.py in the repo]. Variables are tracr
basis directions (e.g. `is_x_3`, `frac_prevs_1`); the correspondence maps each HL
`(layer, "attn"|"mlp", head)` to LL hook `blocks.{l}.attn.hook_result` with
`Ix[:, :, head, :]` or `blocks.{l}.mlp.hook_post` with `Ix[[None]]`
[https://raw.githubusercontent.com/FlyingPumba/circuits-benchmark/main/circuits_benchmark/utils/iit/correspondence.py].
`iit` types: `HLNode(name, num_classes, index)`, `LLNode(name, index, subspace)`,
`Correspondence(dict[HLNode, set[LLNode]])` with `make_corr_from_dict`
[https://raw.githubusercontent.com/cybershiptrooper/iit/main/iit/utils/nodes.py],
[.../iit/utils/correspondence.py]. Interchange interventions are `run_with_hooks` with a hook
that overwrites `out[index]` from a cached source run, on both HL and LL models
(`BaseModelPair.do_intervention`)
[https://raw.githubusercontent.com/cybershiptrooper/iit/main/iit/model_pairs/base_model_pair.py].
`edges.pkl` holds the head/MLP-level circuit; the card warns QKV nodes should be promoted to
whole heads. *Inferred*: for a JAX-free setup, re-implement the RASP program for a handful of
cases in Python (the `task_description` and vocab are in the metadata) and use the HL
variables as Python functions of the input; the LL alignment loci come from `edges.pkl`.

## 2. MIB causal-variable localization track

**Artefacts.** Paper arXiv:2504.13151 (ICML 2025) [https://arxiv.org/abs/2504.13151];
site [https://mib-bench.github.io]; code Apache-2.0 [https://github.com/aaronmueller/MIB],
track folder `MIB-causal-variable-track/` with `example_submission.ipynb`,
`ioi_example_submission.ipynb`, `baselines/`, `tasks/`, `evaluate_submission.py` and a
`CausalAbstraction` submodule [https://github.com/aaronmueller/MIB/tree/main/MIB-causal-variable-track].
The submodule URL `atticusg/CausalAbstraction` now resolves to `goodfire-ai/causalab`
(pyproject: `transformers>=5.16`, `torch`, optional `nnsight`; Rust toolchain needed for
`uv sync`; no licence field seen)
[https://api.github.com/repos/atticusg/CausalAbstraction],
[https://raw.githubusercontent.com/goodfire-ai/causalab/main/pyproject.toml]. Pin the
submodule commit from MIB's `.gitmodules` rather than tracking `main`.
Track requirements: `torch, pyvene, transformers, sae_lens, scikit-learn, networkx, ...`
[https://raw.githubusercontent.com/aaronmueller/MIB/main/MIB-causal-variable-track/requirements.txt];
`sae_lens` requires Python 3.12 per the README.

**Tasks, variables, models** [https://arxiv.org/html/2504.13151v2]:
IOI (S_Tok, S_Pos; GPT-2 small, metric = MSE on logit differences); two-digit addition
(X_Carry; Gemma-2-2B, Llama-3.1-8B); MCQA (X_Order, O_Answer; Qwen-2.5-0.5B, Gemma-2-2B,
Llama-3.1-8B); ARC-easy (same variables; Gemma-2-2B, Llama-3.1-8B); RAVEL (A_Country, A_Cont,
A_Lang; Gemma-2-2B, Llama-3.1-8B). Featurizers compared: full vector, PCA, SAE
(GemmaScope/LlamaScope), DBM masks, DAS. Metric: interchange intervention accuracy.

**Data.** HF collection `mib-bench`: `ioi`, `copycolors_mcqa`, `arithmetic_addition`,
`arithmetic_subtraction`, `arc_easy`, `arc_challenge`, `ravel`
[https://huggingface.co/collections/mib-bench/mib-datasets-67f55273612ec3067a42a56b].
`mib-bench/ioi` (MIT; train ~10k, val ~10k, test ~1k) rows carry `template`, `metadata`
(subject, indirect_object, place, object, random names), `prompt`, `choices`, `answerKey`,
and counterfactual dicts (`abc_counterfactual`, `s1_io_flip_counterfactual`,
`s2_io_flip_counterfactual`, `random_names_*`, `s1_ioi_flip_s2_ioi_flip_counterfactual`, ...),
each with its own `prompt/choices/answerKey` [https://huggingface.co/datasets/mib-bench/ioi].

```python
from datasets import load_dataset
ds = load_dataset("mib-bench/ioi", split="train")
ex = ds[0]; base, cf = ex["prompt"], ex["s1_io_flip_counterfactual"]["prompt"]
```

Submission format: per `{TASK}_{MODEL}_{VARIABLE}` folder, `{ModelUnit}_featurizer`,
`_inverse_featurizer`, `_indices` files plus optional `token_position.py`
[track README]. The reference DAS lives in the `CausalAbstraction`/`causalab` library
(JSON protocols; DAS and DBM supported) [https://github.com/goodfire-ai/causalab], not in
pyvene, but pyvene's `LowRankRotatedSpaceIntervention` is the same operation (see 5).

## 3. Meloux, Pimentel, Portet, Peyrard 2026 (CAE benchmark)

**Availability: yes.** arXiv:2607.00267 (ICML 2026 MechInterp workshop, spotlight)
[https://arxiv.org/abs/2607.00267]; code MIT [https://github.com/MelouxM/CAE]; PyPI
`causal-abstraction-eval` 1.0.0 (2026-06-30), import name `causal_abstraction`, Python >= 3.11,
deps `numpy>=2.2, scipy>=1.16, tqdm, joblib`; extras `neural` (torch>=2.7), `physics`
(numba), `circuit` (PySpice), `figures` [https://pypi.org/pypi/causal-abstraction-eval/json].

**Structure** [README]: library modules `primitives.py, schema.py, valuemap.py, models/,
sampling.py, metrics.py, analytical_metrics.py, engine.py, experiment.py`; one
`systems/NN_<name>.py` per system plus `test/NN_*.py` suites writing
`test/results/NN_*_results.json`. Building blocks: `MicroVariableSchema`,
`CoarseGrainingMap` (defines the variable map `a` and the unmapped sentinel Phi),
`ValueMap` (tau_X), `LowLevelModel`, `CausalGraph`, `EvaluationEngine(EvaluationConfig)`,
`StandardTasks`. Samplers `BottomUpSampler` (CAE-up) and `TopDownSampler` (CAE-down), each
with faithful and `_nf` variants.

**Ten systems** [https://arxiv.org/html/2607.00267 and README]: logic circuit (2-bit adder,
wire -> gate), transistor circuit (SPICE -> Boolean), Lennard-Jones gas -> ideal/vdW gas,
agent-based predator-prey -> Lotka-Volterra, Brownian particles -> 1D heat equation, lattice ->
2D heat equation, MD/MC -> Ising, tracr compiled transformer -> symbolic sort-rank program,
multi-valued GRN -> binary Wg-Fz rule, MOS 6502 transistor -> gate -> ISA. Invalid conditions
perturb the model, the map, or the validity regime; six controlled experiments: hidden
confounder, XOR backup path, wrong intermediate variable, spurious mediator, unreachable
states, wrong causal direction. Faithfulness: Phi is intervened with probability p and
overwritten with noise (Gaussian / {-1,0,1} / resample); power ~100% within ~30 interventions
(Mann-Whitney, 100 runs). Cluster blockers: system 08 needs `tracr` (JAX, not on PyPI);
system 10 needs native libs via `systems/10_cpu_6502_libs/build_libs.sh`; the rest are pure
Python/numpy. Pip: `pip install "causal-abstraction-eval[neural,physics]"`.

## 4. Sutter, Minder, Hofmann, Pimentel 2025 (non-linear representation dilemma)

arXiv:2507.08802 (NeurIPS 2025 spotlight, CC-BY-4.0) [https://arxiv.org/abs/2507.08802];
code [https://github.com/densutter/non-linear-representation-dilemma], licence field null
[https://api.github.com/repos/densutter/non-linear-representation-dilemma];
`requirements.txt`: `torch, torchvision, torchaudio, numpy, wandb, tqdm, pandas, transformers,
matplotlib, pathlib` (no pyvene, no transformer_lens)
[https://raw.githubusercontent.com/densutter/non-linear-representation-dilemma/main/requirements.txt].

**Maps.** `das/Rotation_Model.py` (orthogonal rotation + inverse) and `das/RevNet.py`:
`RevNet(number_blocks, in_features, hidden_size, depth=1)` of `RevNet_Block`s whose coupling
functions are `F = G = MLP(half, half, [hidden_size]*depth)`; `forward` applies blocks,
`inverse` applies their inverses in reverse
[https://raw.githubusercontent.com/densutter/non-linear-representation-dilemma/main/das/RevNet.py].
Complexity is varied by `L_rn` (blocks) and `d_rn` (hidden): Fig. 2 uses L=10, d=16;
Fig. 3 L=1 with d in {1,2,4,8,16}; IOI uses d=64, L in {1,2,4,8}
[https://www.alphaxiv.org/abs/2507.08802.md].

**Hierarchical equality (their version).** Input [-0.5,0.5]^16 = four 4-dim blocks; 3-layer MLP
width 16; trained with Adam 1e-3, batch 1024, 1,048,576 samples; alignment map trained with
batch 6400, Adam 1e-3, up to 50 epochs on 1,280,000 interventions (prob 1/3 each on
eta(x1==x2), eta(x3==x4), both) [alphaxiv md, App. I.1]. Dataset code:
`make_dataset_sample_model(embedding_size, ...)`, pairs sampled "same"/"different" by coin
flip, label 1.0 iff both pairs share type [`das/Dataset_Generation.py`].

**IOI.** Pythia 31M-410M (HF `transformers`, `model.gpt_neox.layers[layer_index]`), random
init vs trained vs checkpoints (steps 0, 1k-4k, full); middle layer (L3/L3/L6/L12); intervention
size half of d_model; map trained 2 epochs on 10^6 interventions, batch 256, lr 1e-4
[alphaxiv md; `scripts/das_llm.py` argparse: `--transformation_type {RevNet,Rotation}`,
`--revnet_blocks 10`, `--revnet_hidden_size 16`, `--intervention_dim_proportion 0.5`,
`--layer_index 9`, `--name-split`]. Algorithm ABAB-ABBA: one inner node = pattern, output B for
ABAB else A. Name split: 100 names split by index 60/20/20 (train/eval/test), 15 BABA + 15 ABBA
templates, 20 places, 20 objects, 200k sentences (80/10/10)
[https://raw.githubusercontent.com/densutter/non-linear-representation-dilemma/main/das/Dataset_Generation.py].
Result: with disjoint names, random-init models yield no alignment even with complex maps;
trained Pythia-410M keeps perfect IIA with phi_nonlin and "reasonably high" with phi_lin
(no exact numbers in text) [alphaxiv md, App. I.2.3]. 8-layer RevNet gives near-perfect IIA at
every checkpoint; 1-2 layer maps reach it only as training advances [arXiv HTML Fig. 4].

## 5. pyvene and tracr

pyvene 0.1.8 (2025-05-26, Apache-2.0; `torch>=2.0`, `transformers>=4.45.1`, `datasets`,
`accelerate`) [https://pypi.org/pypi/pyvene/json]. Supported classes are HF modules
(GPT2, Llama, GPTNeoX = Pythia, Gemma/Gemma2, Qwen2/3, Mistral, OLMo, ...) plus pyvene's own
`MLPModel`, `GRUModel`; **no `HookedTransformer`** entry
[https://raw.githubusercontent.com/stanfordnlp/pyvene/main/pyvene/models/intervenable_modelcard.py].
Custom modules are registered via `pv.type_to_module_mapping[type(m)] = {"block_output": "encoder.block[%s]..."}`
and `pv.type_to_dimension_mapping[type(m)] = {"block_output": ("d_model",)}` [pyvene_101.ipynb].
Interventions: `VanillaIntervention`, `RotatedSpaceIntervention` (full orthogonal rotation,
swap `interchange_dim` dims, rotate back), `LowRankRotatedSpaceIntervention`
(`low_rank_dimension`; `base + (R(src)-R(base)) R^T`), `BoundlessRotatedSpaceIntervention`
(learned boundary, `temperature` parameter init 50, sigmoid mask), `SigmoidMask*`, `PCA*`,
`Autoencoder*` [https://raw.githubusercontent.com/stanfordnlp/pyvene/main/pyvene/models/interventions.py].

```python
import pyvene as pv
cfg = pv.IntervenableConfig({"layer": 0, "component": "mlp_output"},
                            intervention_types=pv.VanillaIntervention)
pv_gpt2 = pv.IntervenableModel(cfg, model=gpt2)
out = pv_gpt2(base=base_inputs, sources=[src_inputs],
              unit_locations={"sources->base": 3}, subspaces=[10, 11, 12])
```
[https://raw.githubusercontent.com/stanfordnlp/pyvene/main/pyvene_101.ipynb]. DAS on the
equality MLP (tutorial): `RepresentationConfig(0, "block_output", "pos", 1,
subspace_partition=None, intervention_link_key=0)` with `intervention_types=RotatedSpaceIntervention`;
optimizer over `v.rotate_layer.parameters()`; call
`intervenable({"inputs_embeds": x}, [{"inputs_embeds": src}, None], {"sources->base": ([[[0]]*B, None], [[[0]]*B, None])}, subspaces=[[list(range(0, 2*d))]*B, None])`;
loss = CE on counterfactual labels; `intervenable.set_zero_grad()`
[https://raw.githubusercontent.com/stanfordnlp/pyvene/main/tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb].
Boundless DAS: optimizer over `rotate_layer.parameters()` + `intervention_boundaries`,
`intervenable.set_temperature(...)` on a `linspace(50, 0.1)` schedule
[.../Boundless_DAS.ipynb]. Save/load: `pv_model.save(dir)`, `pv.IntervenableModel.load(dir, model=m)`.

tracr: `pip3 install .` from the repo; `install_requires = chex, einops, dm-haiku, jax,
networkx, numpy, typing_extensions`; `compile_rasp_to_model(program, vocab, max_seq_len,
compiler_bos)` returns a Haiku model exposing `out.residuals`, `out.layer_outputs`
[https://github.com/google-deepmind/tracr], [https://raw.githubusercontent.com/google-deepmind/tracr/main/setup.py].
So JAX is unavoidable for compiling; InterpBench's `ll_model.pth` removes that need for the
low-level side only.

## Geiger et al. 2023 hierarchical equality (original DAS spec)

arXiv:2303.02536 [https://arxiv.org/html/2303.02536v4]: inputs are four objects [w,x,y,z]
as random vectors in R^n (concatenated; n not fixed in the main text, 16-dim used in the
random-network check); label True iff (w=x) == (y=z); causal model V1=(w=x), V2=(y=z),
V3=(V1=V2); alternatives: left-equality only, and identity of the first object. Network:
3 ReLU layers of width 4n, softmax head; trained to 100% train/test accuracy, batch 6.4K,
lr 1e-3, 1.92M examples. DAS: 640K training pairs, 19.2K test, rotation learned with the
network frozen, best of 3 seeds; intervention at layer 1-3, k in {1,2,8} (|N|=16) or {2,4,16}
(|N|=32) neurons per variable; IIA at layer 1 ranges 0.88 (k=1) to 1.00 (k=8); brute force
0.60; random-network control 0.50-0.64. The pyvene tutorial uses `embedding_dim=2`,
20 entities, `h_dim=embedding_dim*4`, `n_layer=3`, and splits layer-0 output into dims
[0,2d) for WX and [2d,4d) for YZ [DAS_Main_Introduction.ipynb]. Our implementation should
report n, width, k and the layer so these numbers are comparable.

## Recommendation for week 4

Wire **InterpBench low-level models first**: pure PyTorch + TransformerLens, tiny
(d_model 4-36, 2-4 layers; IOI 6L/d64), known head/MLP loci in `edges.pkl`, and `meta.json`
training args. Avoid the repo's Poetry environment and JAX: load with the inferred snippet
above and encode HL variables as Python functions of the input for 3-5 cases (sort-rank,
counts, first-occurrence) chosen from `benchmark_metadata.json`. If the pickled config does
not unpickle under TransformerLens 4.x, fall back to `HookedTransformerConfig(**transformer_cfg)`
from the metadata JSON (*inferred*). Second, **MIB IOI on GPT-2 small** via `datasets` +
pyvene (`LowRankRotatedSpaceIntervention` on `block_output`) gives a natural-model DAS
baseline with provided counterfactual pairs. Third, build the manufactured-alignment
adversary from Sutter's recipe (own RevNet, `[hidden]*depth` coupling MLPs, 60/20/20 name
split) on randomly initialised `EleutherAI/pythia-70m` through HF; no pyvene needed.
Defer the CAE benchmark (needs Python >= 3.11; fine for H-A4 in weeks 5-6) and the tracr HL
models (JAX) unless a separate env is created.

```bash
pip install transformer_lens huggingface_hub pyvene datasets   # torch 2.14 already present
pip install "causal-abstraction-eval[neural,physics]"           # py>=3.11; weeks 5-6
# optional, separate env: pip install "jax[cpu]" dm-haiku chex git+https://github.com/google-deepmind/tracr.git
# do NOT pip-install iit/circuits-benchmark into the main env (pins transformer-lens 1.19.0)
```

## Addendum (2026-10-15, after wiring): what was verified in code

* **InterpBench.** `src/bitsev/testbeds/tl_transformer.py` reproduces TransformerLens 1.19
  logits to 1e-3 (scratch env `~/venvs/tl1`, pins: transformer-lens 1.19.0, torch 2.5.1 CPU,
  transformers 4.40.2, numpy<2, pyarrow<16, datasets<2.20, typeguard<5). Token conventions
  found by brute force: integer vocabularies use token = value and BOS = len(vocab); string
  vocabularies use BOS = 0, PAD = 1, letters from index 2. Cases 52, 7 and 18 reproduce their
  programs at 100% on random full-length sequences; case 13's low-level model does not
  (about 80%, it rarely outputs "constant"), so it is not used as ground truth.
* **Méloux CAE benchmark.** The pip package `causal-abstraction-eval` (import name
  `causal_abstraction`) ships the framework only; the ten systems are in the repository
  `systems/` directory (cloned to `~/scratch/third_party/CAE`). Systems are numpy structural
  models with `forward_with_interventions(SystemState, interventions)`; the logic circuit is
  discrete (thresholded gates) and the physical systems are numpy simulations, so the
  gradient-trained alignment learner of Stage 1 cannot be applied as is. Options: a
  gradient-free learner (random search or CMA-ES over low-dimensional rotations), a torch
  re-implementation of one or two systems, or restricting H-A4's pilot to the framework's
  own metrics on our transformer testbeds. This is a design decision for the preregistration.
