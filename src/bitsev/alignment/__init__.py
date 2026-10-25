"""Layer 2: the Stage 1 score, alignment sample efficiency.

A *configuration* is a network, a locus and an alignment learner (a map family with a
training procedure). Its prequential code length on a fixed set of counterfactual
targets, and its gain over controls that differ in one respect, are computed here.
Requires torch.
"""
