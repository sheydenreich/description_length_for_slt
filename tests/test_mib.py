"""MIB two-digit addition: alphabet construction and the high-level counterfactuals."""

import numpy as np

from bitsev.testbeds import mib


class _WholeNumberTokenizer:
    """Every number is one token (Llama-3 style)."""

    def encode(self, s, add_special_tokens=False):
        return [1000 + int(s)]


class _DigitTokenizer:
    """One token per digit (Gemma / Qwen style)."""

    def encode(self, s, add_special_tokens=False):
        return [100 + int(c) for c in s]


def _prompts(tokenizer, pairs):
    alphabet, index = mib.answer_alphabet(tokenizer)
    tokens = np.array([[7, 7, i] for i in range(len(pairs))])  # distinct rows stand in for prompts
    return mib.ArithmeticPrompts(
        [f"{a} plus {b}" for a, b in pairs],
        tokens,
        np.array([a for a, _ in pairs]),
        np.array([b for _, b in pairs]),
        alphabet,
        index,
    )


def test_alphabet_sizes():
    a, idx = mib.answer_alphabet(_WholeNumberTokenizer())
    assert len(a) == 199 and idx[141] == 141
    a, idx = mib.answer_alphabet(_DigitTokenizer())
    assert len(a) == 10 and idx[141] == idx[1] and idx[91] == idx[9]


def test_counterfactual_carry_moves_the_tens_digit():
    p = _prompts(_WholeNumberTokenizer(), [(27, 64), (20, 61), (77, 64)])
    alg = mib.ArithmeticAlgorithm(p)
    x = p.tokens
    v = alg.variable_values(x)
    np.testing.assert_array_equal(v["carry"], [1, 0, 1])
    np.testing.assert_array_equal(v["ones"], [11, 1, 11])
    np.testing.assert_array_equal(v["tens"], [8, 8, 13])
    np.testing.assert_array_equal(alg.output(x), [91, 81, 141])
    # base 27+64 (carry 1) with the carry of 20+61 (0): 10*8 + 1 = 81; the reverse: 10*(8+1)+1 = 91
    assert alg.counterfactual(x[:1], x[1:2], "carry")[0] == 81
    assert alg.counterfactual(x[1:2], x[:1], "carry")[0] == 91
    # tens of 77+64 (13) into 27+64: 10*(13+1) + 1 = 141
    assert alg.counterfactual(x[:1], x[2:3], "tens")[0] == 141
    # ones sum of 20+61 (1, no carry) into 27+64: 10*8 + 1 = 81
    assert alg.counterfactual(x[:1], x[1:2], "ones")[0] == 81


def test_digit_tokenizer_output_is_the_first_digit():
    p = _prompts(_DigitTokenizer(), [(27, 64), (77, 64)])
    alg = mib.ArithmeticAlgorithm(p)
    out = alg.output(p.tokens)
    assert p.output_token_ids[out[0]] == 100 + 9  # "91" -> "9"
    assert p.output_token_ids[out[1]] == 100 + 1  # "141" -> "1"
