"""InterpBench (Gupta et al. 2024): semi-synthetic transformers with known circuits.

The low-level models are downloaded from the Hugging Face repository
``cybershiptrooper/InterpBench`` (CC-BY-4.0) and loaded into
:class:`~bitsev.testbeds.tl_transformer.TLTransformer`. The high-level programs are
tracr RASP programs; for the cases registered here the algorithm and its intermediate
variables are re-implemented as plain Python so that counterfactual targets can be built
without JAX. Each registered case records the token convention (BOS is the index after the
sorted vocabulary, tokens are their sorted index), the output position and the class order,
all verified by the accuracy test in ``tests/test_interpbench.py``.

Registered cases (case id: task; variables):

* 52: square root of each element (elementwise, no intermediate variable; sanity check)
* 7: count of each token in the sequence (output is the count; no intermediate variable)
* 13: trend of numeric tokens, ``trend(token_p, token_{p+1})``; variable ``next``
* 18: frequency class of each token from its count and the length; variables ``freq``,
  ``length`` (constant for full-length sequences)

Only the cases in ``VERIFIED`` reproduce their program (100% agreement on random
full-length sequences). The forward pass matches TransformerLens 1.19 to 1e-3 on all four
cases. Case 13's low-level model itself disagrees with its program on about 20% of
positions (it rarely outputs "constant" for equal neighbours), so it is loaded but not
verified; the token conventions were found by brute force over BOS id and offset.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from bitsev.testbeds.tl_transformer import TLTransformer, from_state_dict

REPO = "cybershiptrooper/InterpBench"


def cache_dir() -> Path:
    return Path(os.environ.get("BITSEV_HF_CACHE", Path.home() / "scratch" / "hf_cache"))


def _download(case: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(REPO, subfolder=case, filename=filename, cache_dir=str(cache_dir()))
    )


def load_metadata() -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "benchmark_metadata.json", cache_dir=str(cache_dir()))
    return {str(c["case_id"]): c for c in json.load(open(path))["cases"]}


def load_model(case: str, output_position: int = -1) -> TLTransformer:
    cfg = pickle.load(open(_download(case, "ll_model_cfg.pkl"), "rb"))
    state = torch.load(_download(case, "ll_model.pth"), map_location="cpu")
    return from_state_dict(cfg, state, output_position)


def load_edges(case: str) -> list[tuple[str, str]]:
    """Ground-truth circuit edges (TransformerLens hook names)."""
    return pickle.load(open(_download(case, "edges.pkl"), "rb"))


# ----------------------------------------------------------------------------------------
# Token convention


@dataclass(frozen=True)
class Tokens:
    """Token convention of an InterpBench model, found empirically and checked by tests.

    Integer vocabularies: token id = value, BOS = ``len(vocab)``, PAD = ``len(vocab) + 1``.
    String vocabularies: BOS = 0, PAD = 1, token id = sorted index + 2 (tracr sorts the
    basis directions as strings, so ``'BOS'`` and ``'PAD'`` come first).
    """

    vocab: tuple  # sorted vocabulary values
    n_ctx: int

    @property
    def _strings(self) -> bool:
        return isinstance(self.vocab[0], str)

    @property
    def bos(self) -> int:
        return 0 if self._strings else len(self.vocab)

    @property
    def pad(self) -> int:
        return 1 if self._strings else len(self.vocab) + 1

    @property
    def offset(self) -> int:
        return 2 if self._strings else 0

    @property
    def n_values(self) -> int:
        return len(self.vocab)

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Value indices ``(n, n_ctx - 1)`` -> token ids with BOS prepended, ``(n, n_ctx)``."""
        values = np.asarray(values, dtype=np.int64) + self.offset
        bos = np.full((len(values), 1), self.bos, dtype=np.int64)
        return np.concatenate([bos, values], axis=1)

    def decode_values(self, tokens: np.ndarray) -> np.ndarray:
        """Token ids with BOS ``(n, n_ctx)`` -> value indices ``(n, n_ctx - 1)``."""
        return np.asarray(tokens)[:, 1:] - self.offset

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Random full-length sequences as value indices ``(n, n_ctx - 1)``."""
        return rng.integers(0, self.n_values, size=(n, self.n_ctx - 1))


# ----------------------------------------------------------------------------------------
# Algorithms on value-index sequences


@dataclass
class SequenceAlgorithm:
    """An algorithm on token sequences with named per-position intermediate variables.

    ``variable_fns`` map a value-index array ``(n, T)`` (without BOS) to a value array
    ``(n,)`` for the output position; ``output_fn`` maps the dict of variables plus the
    sequences to the output class ``(n,)``.
    """

    name: str
    variables: tuple[str, ...]
    n_outputs: int
    tokens: Tokens
    output_position: int  # position in the BOS-prefixed sequence
    variable_fns: dict[str, Callable[[np.ndarray], np.ndarray]]
    output_fn: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]
    classes: tuple = field(default_factory=tuple)

    def _values(self, x: np.ndarray) -> np.ndarray:
        """Token arrays (with BOS) -> value indices."""
        return self.tokens.decode_values(x)

    def variable_values(self, x):
        v = self._values(x)
        return {name: fn(v) for name, fn in self.variable_fns.items()}

    def output(self, x):
        return self.output_fn(self.variable_values(x), self._values(x))

    def counterfactual(self, base, source, variable):
        if variable not in self.variables:
            raise KeyError(variable)
        vb = self.variable_values(base)
        vb[variable] = self.variable_values(source)[variable]
        return self.output_fn(vb, self._values(base))


def _case_tokens(meta: dict) -> Tokens:
    return Tokens(tuple(sorted(meta["vocab"])), int(meta["max_seq_len"]))


def _build_52(meta: dict) -> SequenceAlgorithm:
    tok = _case_tokens(meta)
    p = tok.n_ctx - 1  # last position; value index == token value
    return SequenceAlgorithm(
        "ib52_sqrt",
        ("value",),
        tok.n_values,
        tok,
        p,
        {"value": lambda v: v[:, p - 1]},
        lambda var, v: var["value"],
    )


def _build_7(meta: dict) -> SequenceAlgorithm:
    tok = _case_tokens(meta)
    p = tok.n_ctx - 1

    def count(v):
        return (v == v[:, p - 1 : p]).sum(axis=1)

    return SequenceAlgorithm(
        "ib7_hist", ("count",), 10, tok, p, {"count": count}, lambda var, v: var["count"]
    )


def _build_13(meta: dict) -> SequenceAlgorithm:
    tok = _case_tokens(meta)
    p = tok.n_ctx - 2  # the last position with a defined next token
    classes = ("constant", "increasing", "decreasing")  # observed class order

    def output(var, v):
        cur, nxt = v[:, p - 1], var["next"]
        return np.where(cur < nxt, 1, np.where(cur > nxt, 2, 0))

    return SequenceAlgorithm(
        "ib13_trend", ("next",), 3, tok, p, {"next": lambda v: v[:, p]}, output, classes
    )


def _build_18(meta: dict) -> SequenceAlgorithm:
    tok = _case_tokens(meta)
    p = tok.n_ctx - 1
    classes = ("common", "frequent", "rare")

    def freq(v):
        return (v == v[:, p - 1 : p]).sum(axis=1)

    def output(var, v):
        f, total = var["freq"], var["length"]
        return np.where(f > total / 2, 1, np.where(f > total / 4, 0, 2))

    return SequenceAlgorithm(
        "ib18_frequency",
        ("freq", "length"),
        3,
        tok,
        p,
        {"freq": freq, "length": lambda v: np.full(len(v), v.shape[1])},
        output,
        classes,
    )


BUILDERS: dict[str, Callable[[dict], SequenceAlgorithm]] = {
    "52": _build_52,
    "7": _build_7,
    "13": _build_13,
    "18": _build_18,
}
VERIFIED: frozenset[str] = frozenset({"52", "7", "18"})


@dataclass
class InterpBenchCase:
    case: str
    model: TLTransformer
    algorithm: SequenceAlgorithm
    edges: list[tuple[str, str]]
    description: str


def load_case(case: str, metadata: dict[str, dict] | None = None) -> InterpBenchCase:
    meta = (metadata or load_metadata())[case]
    algorithm = BUILDERS[case](meta)
    model = load_model(case, output_position=algorithm.output_position)
    return InterpBenchCase(case, model, algorithm, load_edges(case), meta["task_description"])


def task_accuracy(ib: InterpBenchCase, n: int = 500, seed: int = 0) -> float:
    """Agreement of the model's argmax at the output position with the algorithm."""
    rng = np.random.default_rng(seed)
    x = ib.algorithm.tokens.encode(ib.algorithm.tokens.sample(n, rng))
    with torch.no_grad():
        pred = ib.model.logits(ib.model.prepare_input(x)).argmax(1).numpy()
    return float(np.mean(pred == ib.algorithm.output(x)))
