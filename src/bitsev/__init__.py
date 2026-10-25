"""Bits of Evidence: code lengths, exact singular models and alignment sample efficiency.

Layers (no upward imports):
    bitsev.codes       prequential codes, learners, gains, e-values      (numpy, scipy)
    bitsev.exact       exactly solvable models used as regression targets (layer 1)
    bitsev.alignment   Stage 1 alignment score                          (torch)
    bitsev.baselines   rival metrics                                    (torch)
    bitsev.diagnostics D1-D3 metric certificates                         (torch, gudhi)
    bitsev.testbeds    ground-truth and manufactured models              (transformer_lens, ...)
    bitsev.experiments / bitsev.report
"""

__version__ = "0.0.1"
