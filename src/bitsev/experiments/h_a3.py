"""H-A3 on a recorded Stage 1 run: does the Bayes (Laplace) code's regret grow like λ ln n?

For every configuration in a run directory with the final predictor's per-item losses
recorded (``extra.final_item_nats``), the regret at each block end is the cumulative
prequential loss minus the cumulative loss of the final fit on the same targets (the
best-in-family proxy; exact when the family is realizable). Its slope against ln n, after
the batching correction of the schedule, is compared with λ = d_V (d − d_V) / 2 for the
subspace family. Example:
  bitsev-h-a3 results/stage1_ib18_laplace --n-min 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bitsev.codes.gain import fit_log_slope


def regret_curve(record: dict, reference: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Regret against the final fit of ``reference`` (same configuration and targets; default:
    the record itself)."""
    losses = np.asarray(record["item_losses"], float)
    final = np.asarray((reference or record)["extra"]["final_item_nats"], float)
    bounds = np.asarray(record["boundaries"], int)
    start = bounds[0]
    final = final[start : start + len(losses)]
    cum = np.cumsum(losses - final)
    ends = bounds[1:]
    return ends, cum[ends - start - 1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--n-min", type=int, default=64)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--d-v", type=int, default=None, help="for λ of the subspace family")
    p.add_argument("--width", type=int, default=None, help="locus width d, for λ")
    p.add_argument(
        "--reference-run",
        type=Path,
        default=None,
        help="take the final fit's losses from this run's records (same configuration, seed and "
        "targets), e.g. the MAP run as the best-in-family proxy for a Bayes code",
    )
    args = p.parse_args(argv)
    reference: dict[tuple[str, int], dict] = {}
    if args.reference_run is not None:
        for path in sorted((args.reference_run / "records").glob("*.json")):
            r = json.loads(path.read_text())
            if "final_item_nats" in r.get("extra", {}):
                reference[(r["configuration"], r["seed"])] = r
    by_name: dict[str, list[dict]] = {}
    for path in sorted((args.run_dir / "records").glob("*.json")):
        r = json.loads(path.read_text())
        if "final_item_nats" in r.get("extra", {}):
            by_name.setdefault(r["configuration"], []).append(r)
    if not by_name:
        print("no records with final_item_nats (run with --learner laplace)", file=sys.stderr)
        return 1
    lam = None
    if args.d_v and args.width:
        lam = args.d_v * (args.width - args.d_v) / 2
        print(f"λ for the subspace family: d_V (d − d_V) / 2 = {lam:g}")
    rows = []
    for name, recs in sorted(by_name.items()):
        growth = recs[0]["growth"]
        curves = []
        for r in recs:
            ref = reference.get((r["configuration"], r["seed"])) if reference else None
            if reference and ref is None:
                continue
            if ref is not None and ref["targets_hash"] != r["targets_hash"]:
                raise ValueError(f"reference targets differ for {r['configuration']}")
            curves.append(regret_curve(r, ref))
        if not curves:
            continue
        ends = curves[0][0]
        mean = np.mean([c[1] for c in curves], axis=0)
        fit = fit_log_slope(ends, mean, n_min=args.n_min, n_max=args.n_max, growth=growth)
        per_seed = [
            fit_log_slope(e, c, n_min=args.n_min, n_max=args.n_max, growth=growth).corrected_slope
            for e, c in curves
        ]
        se = np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)) if len(per_seed) > 1 else np.nan
        rows.append((name, len(recs), fit.corrected_slope, se, fit.slope_stderr, mean[-1]))
        print(
            f"{name:<44} seeds={len(recs)} corrected slope {fit.corrected_slope:6.2f} "
            f"± {se:4.2f} (fit s.e. {fit.slope_stderr:4.2f}); regret at n={ends[-1]}: "
            f"{mean[-1]:.1f} nats; raw slope {fit.slope:.2f}"
        )
    out = args.run_dir / "h_a3.json"
    out.write_text(
        json.dumps(
            {
                "lambda": lam,
                "n_min": args.n_min,
                "rows": [
                    dict(
                        zip(
                            (
                                "configuration",
                                "seeds",
                                "corrected_slope",
                                "seed_se",
                                "fit_se",
                                "final_regret_nats",
                            ),
                            r,
                            strict=True,
                        )
                    )
                    for r in rows
                ],
            },
            indent=1,
        )
    )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
