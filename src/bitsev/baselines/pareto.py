"""Accuracy against map size (Pimentel et al. 2020, Pareto probing).

The rival reads IIA together with the number of free parameters of the alignment map (or
its certified distortion bound) and asks for the Pareto frontier: alignments that reach
a given accuracy with the smallest map. It is computed from Stage 1 records, which
already store ``map_params``, ``distortion_bound`` and the held-out IIA.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParetoPoint:
    configuration: str
    size: float
    accuracy: float

    def dominates(self, other: ParetoPoint) -> bool:
        return (self.size <= other.size and self.accuracy >= other.accuracy) and (
            self.size < other.size or self.accuracy > other.accuracy
        )


def pareto_points(
    records: list[dict], split: str = "iid", size_key: str = "map_params"
) -> list[ParetoPoint]:
    """One point per configuration: mean IIA on ``split`` against the map size."""
    by_name: dict[str, list[dict]] = {}
    for r in records:
        if r.get("faithfulness") and split in r["faithfulness"] and r.get(size_key) is not None:
            by_name.setdefault(r["configuration"], []).append(r)
    points = []
    for name, recs in sorted(by_name.items()):
        acc = sum(r["faithfulness"][split]["iia"] for r in recs) / len(recs)
        points.append(ParetoPoint(name, float(recs[0][size_key]), acc))
    return points


def frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Points not dominated by any other (smaller or equal size with higher or equal accuracy)."""
    return [p for p in points if not any(q.dominates(p) for q in points)]
