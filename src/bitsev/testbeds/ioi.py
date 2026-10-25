"""Indirect-object identification (IOI) prompts with disjoint-name splits.

Templates follow the ABBA / BABA patterns of the IOI task: two names appear, one is
repeated (the subject S), and the correct continuation is the other name (the indirect
object IO). The high-level variables are the identity of IO, the identity of S, and the
order (whether IO is the first name). The output alphabet is the name list, so the network's
logits are restricted to the names' token ids.

Distribution shift (Sutter et al.): the alignment is learned on prompts built from one set
of names and tested on prompts built from a disjoint set. All prompts in a split are built
to have the same token length so that positions are comparable without padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# {X} and {Y} are the first and second mention; {S} is the repeated name (the subject).
TEMPLATES = (
    "When {X} and {Y} went to the {place}, {S} gave a {object} to",
    "When {X} and {Y} went to the {place}, {S} gave the {object} to",
    "Then, {X} and {Y} were working at the {place}. {S} decided to give a {object} to",
    "Afterwards, {X} and {Y} went to the {place}. {S} gave a {object} to",
    "Then, {X} and {Y} had a long argument, and afterwards {S} said to",
)
PLACES = ("store", "garden", "school", "office", "park", "station", "hospital", "restaurant")
OBJECTS = ("drink", "ring", "book", "kiss", "bone", "necklace", "snack", "basketball")
NAMES = (
    "Mary",
    "John",
    "Tom",
    "James",
    "Dan",
    "Sid",
    "Martin",
    "Kevin",
    "Alex",
    "Anna",
    "Sarah",
    "Lisa",
    "Mike",
    "Sam",
    "Max",
    "Emma",
    "Paul",
    "Peter",
    "David",
    "Robert",
    "Laura",
    "Jane",
    "Andrew",
    "Rachel",
    "Mark",
    "Steve",
    "Brian",
    "Eric",
    "Jack",
    "Adam",
    "Charlie",
    "Kate",
)


@dataclass
class IOIPrompts:
    text: list[str]
    tokens: np.ndarray  # (n, T) token ids, fixed length
    io: np.ndarray  # index into `names`
    s: np.ndarray
    order: np.ndarray  # 1 if IO is the first name (ABB), 0 if second (BAB)
    names: tuple[str, ...]
    name_token_ids: np.ndarray  # token id of " Name" for each name

    def __len__(self) -> int:
        return len(self.io)


def single_token_names(tokenizer, candidates=NAMES) -> list[str]:
    """Names whose ``" Name"`` form is a single token."""
    out = []
    for name in candidates:
        if len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1:
            out.append(name)
    return out


def build_prompts(
    tokenizer, names: tuple[str, ...], n: int, rng: np.random.Generator, templates=TEMPLATES
) -> IOIPrompts:
    """Sample prompts from ``names`` and keep the modal token length only."""
    rows = []
    for _ in range(n):
        a, b = rng.choice(len(names), 2, replace=False)
        t = templates[rng.integers(len(templates))]
        order = int(rng.integers(2))  # 1: IO mentioned first (ABB), 0: S first (BAB)
        io, s = a, b
        first, second = (io, s) if order else (s, io)
        text = t.format(
            X=names[first],
            Y=names[second],
            S=names[s],
            place=rng.choice(PLACES),
            object=rng.choice(OBJECTS),
        )
        rows.append((text, io, s, order))
    # models whose tokenizer adds a beginning-of-sequence token by default (Gemma, Llama 3)
    # need it: without it Gemma-3-1b scores 1% among names, with it 100%
    bos = tokenizer.bos_token_id
    probe = tokenizer.encode("a", add_special_tokens=True)
    prefix = [bos] if (bos is not None and len(probe) > 0 and probe[0] == bos) else []
    encoded = [prefix + tokenizer.encode(r[0], add_special_tokens=False) for r in rows]
    lengths = np.array([len(e) for e in encoded])
    modal = np.bincount(lengths).argmax()
    keep = [i for i, L in enumerate(lengths) if L == modal]
    tokens = np.array([encoded[i] for i in keep], dtype=np.int64)
    name_ids = np.array([tokenizer.encode(" " + nm, add_special_tokens=False)[0] for nm in names])
    return IOIPrompts(
        [rows[i][0] for i in keep],
        tokens,
        np.array([rows[i][1] for i in keep]),
        np.array([rows[i][2] for i in keep]),
        np.array([rows[i][3] for i in keep]),
        tuple(names),
        name_ids,
    )


class IOIAlgorithm:
    """Variables ``io``, ``s``, ``order``; the output is the IO name index."""

    name = "ioi"
    variables = ("io", "s", "order")

    def __init__(self, prompts: IOIPrompts):
        self.prompts = prompts
        self.n_outputs = len(prompts.names)
        # x is the token array; recover variables by row lookup (token rows are unique enough
        # to index, but we key on the row index via a dict for robustness)
        self._lookup = {tuple(row): i for i, row in enumerate(prompts.tokens)}

    def _rows(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._lookup[tuple(r)] for r in np.asarray(x)])

    def variable_values(self, x):
        i = self._rows(x)
        return {"io": self.prompts.io[i], "s": self.prompts.s[i], "order": self.prompts.order[i]}

    def output(self, x):
        return self.variable_values(x)["io"]

    def counterfactual(self, base, source, variable):
        vb, vs = self.variable_values(base), self.variable_values(source)
        if variable == "io":
            return vs["io"]
        if variable == "s":
            return vb["io"]  # changing the subject does not change the answer
        if variable == "order":
            return vb["io"]  # nor does the order
        raise KeyError(variable)
