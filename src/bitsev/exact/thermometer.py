"""The loss-fluctuation thermometer (Appendix C.1 of the plan).

Under a Gibbs law proportional to ``exp(-K / T)`` near a minimum of multiplicity one, the
excess loss is asymptotically Gamma-distributed with shape ``lambda`` and scale ``T``, so

    T_hat = Var(K) / E(K),        lambda_hat = E(K)^2 / Var(K).

The estimators are exact for regular germs and biased by logarithms when ``m > 1``.
They apply only to samplers whose target is Gibbs by construction; SGD's stationary law
is not (see the caveat in the plan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bitsev.exact.germs import Germ


@dataclass
class ThermometerReading:
    germ: str
    temperature: float
    t_hat: float
    lambda_hat: float
    lam: float

    @property
    def t_ratio(self) -> float:
        return self.t_hat / self.temperature


def thermometer(mean: float, var: float) -> tuple[float, float]:
    """``(T_hat, lambda_hat)`` from the first two moments of the excess loss."""
    return var / mean, mean**2 / var


def read_germ(germ: Germ, temperature: float) -> ThermometerReading:
    mean, var = germ.gibbs_moments(1.0 / temperature)
    t_hat, lam_hat = thermometer(mean, var)
    return ThermometerReading(germ.name, temperature, t_hat, lam_hat, germ.lam)


def thermometer_from_samples(losses: np.ndarray) -> tuple[float, float]:
    """Plug the sample mean and variance of excess losses into the thermometer."""
    losses = np.asarray(losses, dtype=float)
    return thermometer(float(losses.mean()), float(losses.var()))


def format_readings(readings: list[ThermometerReading]) -> str:
    lines = [f"{'germ':<12}{'T':>8}{'T_hat/T':>10}{'lambda_hat':>12}{'lambda':>8}"]
    for r in readings:
        lines.append(
            f"{r.germ:<12}{r.temperature:>8.0e}{r.t_ratio:>10.3f}{r.lambda_hat:>12.3f}{r.lam:>8.3g}"
        )
    return "\n".join(lines)
