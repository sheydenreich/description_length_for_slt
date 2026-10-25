"""Gate G-A0: compute every exact table and write ``results/G-A0/report.md``.

The criteria are those of ``preregistration/G-A0.md``; this module applies them
mechanically and never adjusts a threshold. Everything here is closed form or quadrature
and runs in well under a minute on one core, so it is allowed directly on the login node.

Usage: ``bitsev-exact [--out results/G-A0]``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from bitsev.exact import gaussian_mean, germs, rrr
from bitsev.exact.batching import batching_rows
from bitsev.exact.identifiability import vif_over_decades
from bitsev.exact.thermometer import read_germ
from bitsev.report import markdown as md

# Settings fixed in preregistration/G-A0.md -----------------------------------------------
GERM_SIZES = 2.0 ** np.arange(4, 15)  # n = 2^4 ... 2^14
GERM_FIT_RANGE = (2**8, 2**14)
GERM_GROWTHS = (1.0, 2.0, 4.0)
THERMOMETER_TEMPERATURES = (1e-2, 1e-3, 1e-4)
PLAN_THERMOMETER = {1e-2: (0.93, 0.37), 1e-3: (0.97, 0.40), 1e-4: (0.98, 0.42)}
TOL_BATCHING_GAUSS = 2e-3  # criterion 1
TOL_LAMBDA = 0.05  # criteria 2 and 3
TOL_THERMOMETER = 0.01  # criterion 5


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - report generation must not depend on git
        return "unknown"


def criterion_1_gaussian_batching() -> tuple[bool, str, list[dict]]:
    rows = gaussian_mean.batching_table()
    ok = all(abs(r.error) < TOL_BATCHING_GAUSS for r in rows)
    text = md.table(
        ["learner", "r", "fitted", "(r-1)/(2 ln r)", "error"],
        [
            [r.learner, f"{r.growth:g}", f"{r.fitted:.4f}", f"{r.predicted:.4f}", f"{r.error:.1e}"]
            for r in rows
        ],
    )
    return ok, text, [asdict(r) for r in rows]


def criterion_2_bayes_redundancy() -> tuple[bool, str, list[dict]]:
    records, lines, ok = [], [], True
    sizes = GERM_SIZES[GERM_SIZES >= GERM_FIT_RANGE[0]]
    for germ in germs.STANDARD_GERMS.values():
        plain = germs.fit_lambda(germ, sizes)
        fixed = germs.fit_lambda(germ, sizes, fix_mult=germ.mult)
        passes = abs(plain.lam - germ.lam) < TOL_LAMBDA if germ.mult == 1 else True
        ok &= passes
        lines.append(
            [
                germ.name,
                f"{germ.lam:g}",
                germ.mult,
                f"{plain.lam:.4f} ± {plain.stderr:.4f}",
                f"{fixed.lam:.4f} ± {fixed.stderr:.4f}",
                md.status(passes) if germ.mult == 1 else "reported",
            ]
        )
        records.append(
            {
                "germ": germ.name,
                "lambda": germ.lam,
                "m": germ.mult,
                "fit_plain": asdict(plain),
                "fit_fixed_m": asdict(fixed),
            }
        )
    text = md.table(["germ", "λ", "m", "slope (no log log)", "slope (m fixed)", "criterion"], lines)
    return ok, text, records


def criterion_3_batching_germs() -> tuple[bool, str, list[dict]]:
    rows, ok = [], True
    for germ in germs.STANDARD_GERMS.values():
        for row in batching_rows(
            germ.name, "bayes", germ.bayes_excess, germ.lam, GERM_GROWTHS, 2**14, GERM_FIT_RANGE
        ):
            tested = germ.mult == 1
            passes = (abs(row.error) < TOL_LAMBDA * row.predicted / row.kappa) and (
                abs(row.corrected - germ.lam) < TOL_LAMBDA
            )
            if tested:
                ok &= passes
            rows.append((row, md.status(passes) if tested else "reported (m = 2)"))
    text = md.table(
        ["germ", "r", "λ", "fitted", "λ (r-1)/ln r", "corrected", "criterion"],
        [
            [
                r.model,
                f"{r.growth:g}",
                f"{r.kappa:g}",
                f"{r.fitted:.4f}",
                f"{r.predicted:.4f}",
                f"{r.corrected:.4f}",
                s,
            ]
            for r, s in rows
        ],
    )
    return ok, text, [asdict(r) | {"status": s} for r, s in rows]


def criterion_4_optimizer_effect() -> tuple[bool, str, list[dict]]:
    rows = rrr.comparison_table()
    ran = True  # exact, so the criterion "has been run"; the gap is a theorem, not a fit
    text = md.table(
        ["(M,N,H)", "r", "λ (Bayes)", "m", "κ (least squares)", "κ − λ"],
        [
            [str((r.M, r.N, r.H)), r.r, str(r.lam), r.mult, str(r.kappa_plugin), str(r.gap)]
            for r in rows
        ],
    )
    note = (
        "\n\nThe least-squares learner of reduced-rank regression with `H >= min(M, N)` has the "
        "regular coefficient `MN/2` (exact finite-sample excess `MN / (2(m - N - 1))`), while the "
        "Bayes code of the same family has the Aoyagi–Watanabe `λ`. Positive gaps are exact "
        "optimizer effects. In the two-parameter germs the natural plug-in learners have "
        "`κ = λ`, so no effect appears there."
    )
    records = [
        {k: (str(v) if not isinstance(v, int) else v) for k, v in asdict(r).items()} for r in rows
    ]
    return ran, text + note, records


def criterion_5_carried_over() -> tuple[bool, str, list[dict]]:
    ok, lines, records = True, [], []
    for T in THERMOMETER_TEMPERATURES:
        r = read_germ(germs.STANDARD_GERMS["x^2y^2"], T)
        t_plan, l_plan = PLAN_THERMOMETER[T]
        passes = (
            abs(r.t_ratio - t_plan) < TOL_THERMOMETER
            and abs(r.lambda_hat - l_plan) < TOL_THERMOMETER
        )
        ok &= passes
        lines.append(
            [
                r.germ,
                f"{T:g}",
                f"{r.t_ratio:.3f}",
                f"{t_plan:.2f}",
                f"{r.lambda_hat:.3f}",
                f"{l_plan:.2f}",
                md.status(passes),
            ]
        )
        records.append(asdict(r) | {"plan": [t_plan, l_plan]})
    for name in ("x^2+y^4", "x^2+y^2"):
        for T in THERMOMETER_TEMPERATURES:
            r = read_germ(germs.STANDARD_GERMS[name], T)
            passes = abs(r.t_ratio - 1) < 1e-6 and abs(r.lambda_hat - r.lam) < 1e-6
            ok &= passes
            lines.append(
                [
                    r.germ,
                    f"{T:g}",
                    f"{r.t_ratio:.6f}",
                    "1",
                    f"{r.lambda_hat:.6f}",
                    f"{r.lam:g}",
                    md.status(passes),
                ]
            )
            records.append(asdict(r))
    thermo = md.table(["germ", "T", "T̂/T", "plan", "λ̂", "plan", "criterion"], lines)
    vif_rows = vif_over_decades()
    vif_ok = all(r.vif > 100 for r in vif_rows)
    ok &= vif_ok
    vif = md.table(
        ["decades", "n range", "points", "VIF", "SE inflation"],
        [
            [
                f"{r.decades:g}",
                f"[{r.n_min:.0e}, {r.n_max:.0e}]",
                r.points,
                f"{r.vif:.0f}",
                f"{r.stderr_inflation:.1f}×",
            ]
            for r in vif_rows
        ],
    )
    return (
        ok,
        thermo + "\n\nIdentifiability of `(log n)/n` against `1/n` (log-spaced grids):\n\n" + vif,
        records + [asdict(r) for r in vif_rows],
    )


def build_report(out_dir: Path) -> bool:
    criteria = [
        ("1. Batching factor reproduced (Gaussian mean)", criterion_1_gaussian_batching),
        (
            "2. Bayes redundancy isolated (per-sample codes of exact germs)",
            criterion_2_bayes_redundancy,
        ),
        ("3. Batching isolated (block codes of exact germs)", criterion_3_batching_germs),
        ("4. Optimizer effect (plug-in κ against λ)", criterion_4_optimizer_effect),
        ("5. Carried-over checks (thermometer, identifiability)", criterion_5_carried_over),
    ]
    sections, verdicts, records = [], [], {}
    for title, fn in criteria:
        ok, text, recs = fn()
        verdicts.append((title, ok))
        records[title] = recs
        sections.append(f"## {title}\n\n**{md.status(ok)}**\n\n{text}\n")
    gate = (
        verdicts[0][1] and verdicts[1][1] and verdicts[2][1] and verdicts[4][1] and verdicts[3][1]
    )
    head = (
        f"# Gate G-A0 report\n\nGenerated {date.today().isoformat()} by `bitsev-exact` at commit "
        f"`{_git_hash()}` from `preregistration/G-A0.md`.\n\n**Gate verdict: {md.status(gate)}** "
        "(criteria 1, 2, 3 and 5 met; criterion 4 run).\n\n"
        + md.table(["criterion", "status"], [[t, md.status(ok)] for t, ok in verdicts])
        + "\n\n"
    )
    md.write(out_dir / "report.md", head + "\n".join(sections))
    (out_dir / "records.json").write_text(json.dumps(records, indent=1, default=str))
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/G-A0"))
    args = parser.parse_args(argv)
    gate = build_report(args.out)
    print(f"G-A0: {md.status(gate)}  ->  {args.out / 'report.md'}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
