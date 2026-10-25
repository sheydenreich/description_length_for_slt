"""Tie handling of the causal wrong-locus rule."""

from bitsev.alignment.controls import _earliest_within


def test_ties_within_tolerance_go_to_the_earliest_locus():
    scores = {"layer0": 0.072, "layer6": 0.069, "layer9": 0.093, "layer10": 0.97}
    order = ["layer0", "layer6", "layer9", "layer10"]
    assert _earliest_within(scores, order, 0.02) == "layer0"
    assert _earliest_within(scores, order, 0.0) == "layer6"
    assert _earliest_within(scores, order, 0.03) == "layer0"


def test_a_clear_minimum_wins_regardless_of_order():
    scores = {"resid_0": 0.985, "resid_1": 0.106, "resid_2": 0.642}
    assert _earliest_within(scores, ["resid_0", "resid_1", "resid_2"], 0.02) == "resid_1"
