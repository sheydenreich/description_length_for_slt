"""MIB causal-variable tasks (Mueller et al. 2025) as Stage 1 testbeds.

First task: two-digit addition (``mib-bench/arithmetic_addition``), prompts of the form
``"Q: How much is 27 plus 64? A: "`` with the answer as the next token(s). The high-level
model has the operands' digit sums as variables; the track's variable is the carry from the
ones digits. The output alphabet is the set of distinct *first tokens* of the possible
answer strings ``"0" .. "198"``: a single token per answer for tokenizers that keep 2-3
digit numbers whole (Llama 3), the first digit for digit-level tokenizers (Gemma, Qwen).
Prompts have a fixed token length under all three, so the adapter's fixed-position readout
applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ANSWER = 198


@dataclass
class ArithmeticPrompts:
    text: list[str]
    tokens: np.ndarray  # (n, T) token ids, fixed length, read position -1
    op1: np.ndarray
    op2: np.ndarray
    output_token_ids: np.ndarray  # alphabet: distinct first tokens of "0" .. "198"
    answer_index: np.ndarray  # answer value k -> index into the alphabet

    def __len__(self) -> int:
        return len(self.op1)

    @staticmethod
    def concat(a: ArithmeticPrompts, b: ArithmeticPrompts) -> ArithmeticPrompts:
        if not np.array_equal(a.output_token_ids, b.output_token_ids):
            raise ValueError("prompt sets have different output alphabets")
        return ArithmeticPrompts(
            a.text + b.text,
            np.concatenate([a.tokens, b.tokens]),
            np.concatenate([a.op1, b.op1]),
            np.concatenate([a.op2, b.op2]),
            a.output_token_ids,
            a.answer_index,
        )


def _bos_prefix(tokenizer) -> list[int]:
    bos = tokenizer.bos_token_id
    probe = tokenizer.encode("a", add_special_tokens=True)
    return [bos] if (bos is not None and len(probe) > 0 and probe[0] == bos) else []


def answer_alphabet(tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Distinct first tokens of the answer strings and the map answer value -> alphabet index."""
    first = [tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(MAX_ANSWER + 1)]
    alphabet: list[int] = []
    index = np.empty(MAX_ANSWER + 1, dtype=np.int64)
    for k, t in enumerate(first):
        if t not in alphabet:
            alphabet.append(t)
        index[k] = alphabet.index(t)
    return np.array(alphabet, dtype=np.int64), index


DEFAULT_TEMPLATE = "Q: How much is "


def build_arithmetic(
    tokenizer,
    split: str,
    n: int,
    rng: np.random.Generator,
    two_digit_only: bool = True,
    template: str = DEFAULT_TEMPLATE,
) -> ArithmeticPrompts:
    """Sample ``n`` prompts of the MIB addition task from ``split`` (train / validation / test).

    The dataset mixes six prompt templates of different token lengths ("Q: How much is N plus
    N? A: ", "N+N=", ...); ``template`` keeps the prompts starting with that string so that every
    prompt has the same token length and the read position is the last token.
    """
    from datasets import load_dataset

    rows = load_dataset("mib-bench/arithmetic_addition")[split]
    rows = rows.filter(
        lambda r: (
            (not two_digit_only or r["num_digit"] == 2)
            and (not template or r["prompt"].startswith(template))
        )
    )
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    rows = rows.select(sorted(idx.tolist()))
    prefix = _bos_prefix(tokenizer)
    encoded = [prefix + tokenizer.encode(r["prompt"], add_special_tokens=False) for r in rows]
    lengths = np.array([len(e) for e in encoded])
    modal = np.bincount(lengths).argmax()
    keep = [i for i, L in enumerate(lengths) if L == modal]
    alphabet, index = answer_alphabet(tokenizer)
    return ArithmeticPrompts(
        [rows[i]["prompt"] for i in keep],
        np.array([encoded[i] for i in keep], dtype=np.int64),
        np.array([rows[i]["operand1"] for i in keep]),
        np.array([rows[i]["operand2"] for i in keep]),
        alphabet,
        index,
    )


class ArithmeticAlgorithm:
    """Variables of two-digit addition: ``carry`` (ones digits sum to 10 or more), ``ones``
    (sum of the ones digits, 0-18) and ``tens`` (sum of the tens digits, 0-18); the output is
    the alphabet index of the answer ``10 * (tens + carry) + ones mod 10``."""

    name = "mib-arithmetic"
    variables = ("carry", "ones", "tens")

    def __init__(self, prompts: ArithmeticPrompts):
        self.prompts = prompts
        self.n_outputs = len(prompts.output_token_ids)
        self._lookup = {tuple(row): i for i, row in enumerate(prompts.tokens)}

    def _rows(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._lookup[tuple(r)] for r in np.asarray(x)])

    def variable_values(self, x):
        i = self._rows(x)
        o1, o2 = self.prompts.op1[i], self.prompts.op2[i]
        ones = o1 % 10 + o2 % 10
        return {"carry": (ones >= 10).astype(int), "ones": ones, "tens": o1 // 10 + o2 // 10}

    def _answer(self, tens, ones_sum, carry):
        value = 10 * (tens + carry) + ones_sum % 10
        return self.prompts.answer_index[np.clip(value, 0, MAX_ANSWER)]

    def output(self, x):
        v = self.variable_values(x)
        return self._answer(v["tens"], v["ones"], v["carry"])

    def counterfactual(self, base, source, variable):
        vb, vs = self.variable_values(base), self.variable_values(source)
        if variable == "carry":
            return self._answer(vb["tens"], vb["ones"], vs["carry"])
        if variable == "ones":
            # the ones sum carries its own carry: replacing it replaces both
            return self._answer(vb["tens"], vs["ones"], (vs["ones"] >= 10).astype(int))
        if variable == "tens":
            return self._answer(vs["tens"], vb["ones"], vb["carry"])
        raise KeyError(variable)
