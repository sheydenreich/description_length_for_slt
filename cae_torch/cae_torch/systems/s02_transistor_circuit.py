"""System 02: CMOS half-adder / NAND operating point (``systems/02_transistor_circuit.py``).

The original builds a PySpice netlist (ngspice MOSFET ``level=1`` with ``Vto=±1``, ``Kp=20e-6``
and all other parameters at ngspice defaults) and solves the DC operating point with ngspice.
This module re-implements that netlist and the ngspice level-1 DC equations in torch
(``mos1load.c``: Shichman-Hodges channel current with ``lambda = gamma = 0``,
``beta = Kp * W / L = Kp`` for the default ``W = L = 100 um``, bulk-source / bulk-drain
junction diodes with ``Is = 1e-14 A`` and the ``gmin = 1e-12 S`` shunt, ``vt = kT/q`` at
25 degC) and solves KCL by a batched damped Newton iteration. The last Newton step is taken with
autograd enabled, which gives the exact implicit-function gradient of every node voltage with
respect to every driven voltage. The capacitors of the original are open in DC and dropped.

``GenericSpiceModel(build_fn, input_names)`` keeps the constructor of the original;
``build_fn`` is one of :func:`nand_topology` / :func:`half_adder_topology` (or any function with
the same signature using :class:`Circuit`). Variables are the node names in PySpice's order,
lower-cased; for the half-adder::

    0 vdd nand1_out a b n_1 nand2_out n_2 nand3_out n_3 sum n_4 carry

Inputs: ``input_names`` (``a``, ``b``), defaulting to 0 V when absent.

Semantics, exactly as the original: every node present in ``inputs`` **or** ``interventions``
(and every declared input) is driven by an ideal voltage source, i.e. clamped; so inputs and
interventions behave identically for the circuit. ``vdd`` is always reported as 5.0 (an
intervention on it is ignored, as the original drives it from the analysis), ``0`` is never
driven but is reported as the given value if one is supplied (0.0 otherwise).

Equivalence with the original is limited by ngspice's own convergence tolerances
(``reltol = 1e-3``, ``vntol = 1e-6``): weakly driven internal nodes (``n_*`` between stacked
NMOS in cutoff) differ from our tightly converged solution by up to a few mV, and even between
symmetric cases of the original itself. The test therefore uses ``atol = 2e-2 V``.

Differentiable: yes, every node voltage with respect to every driven voltage (inputs and
interventions), through the implicit solve. The device equations are piecewise (cutoff /
saturation / linear, diode branch at ``-3 vt``) but C^1 across the region boundaries;
``sign(vds)`` (drain/source role swap) is a measure-zero kink.

High-level abstraction of ``test/02_transistor_circuit.py`` (half-adder): :data:`COARSE_GRAINING`,
:data:`INTERNAL_NODES`, :data:`BIT_RANGES` (volts), :data:`HIGH_LEVEL_MODELS`
(``valid``: sum = a XOR b, carry = a AND b; ``fail``: sum = a OR b; ``noise``: valid model with
Gaussian output noise of std 1.5 V).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from cae_torch.base import TorchSystem

# ngspice constants (src/include/ngspice/const.h, cktdefs defaults)
_CONST_K_OVER_Q = 8.617087e-5  # CONSTKoverQ (V/K) as used by ngspice
_TEMP_K = 25.0 + 273.15
_VT = _CONST_K_OVER_Q * _TEMP_K
_GMIN = 1e-12
_MAX_EXP_ARG = 709.0
_DEFAULT_W = _DEFAULT_L = 1e-4


# --------------------------------------------------------------------------------------
# Minimal netlist container mirroring the PySpice calls of the original
# --------------------------------------------------------------------------------------


class Circuit:
    """Just enough of ``PySpice.Spice.Netlist.Circuit`` for the original's topology builders."""

    gnd = "0"

    def __init__(self, name: str):
        self.name = name
        self.models: dict[str, dict] = {}
        self.vsources: list[tuple[str, str, str, float]] = []
        self.mosfets: list[tuple[str, str, str, str, str, str]] = []
        self._nodes: list[str] = ["0"]

    def _touch(self, *nodes: str) -> None:
        for n in nodes:
            n = str(n).lower()
            if n not in self._nodes:
                self._nodes.append(n)

    def model(self, name: str, kind: str, **params) -> None:
        kind = kind.lower()
        if kind not in ("nmos", "pmos"):
            raise ValueError(f"only level-1 nmos/pmos models are supported, got {kind!r}")
        self.models[name.lower()] = {"type": 1 if kind == "nmos" else -1, **params}

    def V(self, name: str, nplus: str, nminus: str, value: float) -> None:
        self._touch(nplus, nminus)
        self.vsources.append((name, str(nplus).lower(), str(nminus).lower(), float(value)))

    def MOSFET(self, name: str, drain: str, gate: str, source: str, bulk: str, model: str) -> None:
        self._touch(drain, gate, source, bulk)
        self.mosfets.append((name, *(str(n).lower() for n in (drain, gate, source, bulk)), model.lower()))

    def C(self, name: str, n1: str, n2: str, value: float) -> None:  # open in DC
        self._touch(n1, n2)

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes)


def create_base_circuit(name: str) -> Circuit:
    circuit = Circuit(name)
    circuit.model("pmos", "pmos", Vto=-1, Kp=20e-6)
    circuit.model("nmos", "nmos", Vto=1, Kp=20e-6)
    circuit.V("dd", "vdd", circuit.gnd, 5.0)
    return circuit


def add_nand(circuit: Circuit, name: str, in1: str, in2: str, out: str) -> None:
    circuit.MOSFET(f"p1_{name}", out, in1, "vdd", "vdd", model="pmos")
    circuit.MOSFET(f"p2_{name}", out, in2, "vdd", "vdd", model="pmos")
    circuit.MOSFET(f"n1_{name}", f"n_{name}", in1, circuit.gnd, circuit.gnd, model="nmos")
    circuit.MOSFET(f"n2_{name}", out, in2, f"n_{name}", circuit.gnd, model="nmos")
    circuit.C(f"load_{name}", out, circuit.gnd, 1e-15)


def add_inverter(circuit: Circuit, name: str, in1: str, out: str) -> None:
    circuit.MOSFET(f"p_{name}", out, in1, "vdd", "vdd", model="pmos")
    circuit.MOSFET(f"n_{name}", out, in1, circuit.gnd, circuit.gnd, model="nmos")


def nand_topology(circuit: Circuit, input_names) -> None:
    add_nand(circuit, "1", "a", "b", "output")


def half_adder_topology(circuit: Circuit, input_names) -> None:
    add_nand(circuit, "1", "a", "b", "nand1_out")
    add_nand(circuit, "2", "a", "nand1_out", "nand2_out")
    add_nand(circuit, "3", "b", "nand1_out", "nand3_out")
    add_nand(circuit, "4", "nand2_out", "nand3_out", "sum")
    add_inverter(circuit, "5", "nand1_out", "carry")


# --------------------------------------------------------------------------------------
# ngspice MOS level-1 DC equations
# --------------------------------------------------------------------------------------


def _junction(v: Tensor, isat: Tensor, vt: float, gmin: float) -> Tensor:
    """Bulk junction current (``mos1load.c``): ``gmin*v - Is`` below ``-3 vt``, ideal diode above."""
    ev = torch.exp(torch.clamp(v / vt, max=_MAX_EXP_ARG))
    return torch.where(v <= -3.0 * vt, gmin * v - isat, isat * (ev - 1.0) + gmin * v)


def _mos1_terminal_currents(
    vd: Tensor, vg: Tensor, vs: Tensor, vb: Tensor,
    mtype: Tensor, vto: Tensor, beta: Tensor, lam: Tensor, isat: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Currents *leaving the circuit node into* the drain, source and bulk terminals."""
    vgs = mtype * (vg - vs)
    vds = mtype * (vd - vs)
    vbs = mtype * (vb - vs)
    vbd = mtype * (vb - vd)
    vgd = vgs - vds
    cbs = _junction(vbs, isat, _VT, _GMIN)
    cbd = _junction(vbd, isat, _VT, _GMIN)
    mode = torch.where(vds >= 0, torch.ones_like(vds), -torch.ones_like(vds))
    von = mtype * vto  # gamma = 0: no body effect
    vgst = torch.where(mode > 0, vgs, vgd) - von
    vds_eff = vds * mode
    betap = beta * (1.0 + lam * vds_eff)
    sat = betap * vgst * vgst * 0.5
    lin = betap * vds_eff * (vgst - 0.5 * vds_eff)
    cdrain = torch.where(vgst <= 0, torch.zeros_like(vgst), torch.where(vgst <= vds_eff, sat, lin))
    i_d = mtype * (mode * cdrain - cbd)
    i_s = mtype * (-mode * cdrain - cbs)
    i_b = mtype * (cbd + cbs)
    return i_d, i_s, i_b


# --------------------------------------------------------------------------------------
# Low-level model
# --------------------------------------------------------------------------------------


class GenericSpiceModel(TorchSystem):
    """Batched torch translation of ``GenericSpiceModel(build_fn, input_names)``."""

    max_iter: int = 500
    step_limit: float = 0.5  # volts per Newton step and node (damping)
    tol: float = 1e-13

    def __init__(self, build_fn: Callable[[Circuit, list], None], input_names: list[str]):
        self.build_fn = build_fn
        self.input_names = [str(n) for n in input_names]
        circuit = create_base_circuit("Temp")
        self.build_fn(circuit, self.input_names)
        self.circuit = circuit
        self.all_nodes = [n.lower() for n in circuit.node_names]
        self.variables = tuple(self.all_nodes)
        self.input_variables = tuple(n.lower() for n in self.input_names)
        self._index = {n: i for i, n in enumerate(self.all_nodes)}
        n_nodes = len(self.all_nodes)

        # device parameter tensors
        T = len(circuit.mosfets)
        self._term = {
            k: torch.zeros(n_nodes, T, dtype=self.dtype) for k in ("d", "g", "s", "b")
        }  # incidence (node x device) matrices
        mtype, vto, beta, lam, isat = ([] for _ in range(5))
        for t, (_, d, g, s, b, model) in enumerate(circuit.mosfets):
            for key, node in zip("dgsb", (d, g, s, b)):
                self._term[key][self._index[node], t] = 1.0
            p = circuit.models[model]
            w, l = p.get("W", _DEFAULT_W), p.get("L", _DEFAULT_L)
            mtype.append(float(p["type"]))
            vto.append(float(p.get("Vto", 0.0)))
            beta.append(float(p.get("Kp", 2e-5)) * w / l)
            lam.append(float(p.get("Lambda", 0.0)))
            isat.append(float(p.get("Is", 1e-14)))
        self._params = tuple(torch.tensor(x, dtype=self.dtype) for x in (mtype, vto, beta, lam, isat))
        self._fixed_sources = {np_: val for (_, np_, nm, val) in circuit.vsources if nm == "0"}

    # -- physics ---------------------------------------------------------------------
    def _node_currents(self, v: Tensor) -> Tensor:
        """Sum of device currents leaving each node; ``v`` is ``(N,)`` (one sample)."""
        vd, vg, vs, vb = (v @ self._term[k] for k in "dgsb")
        i_d, i_s, i_b = _mos1_terminal_currents(vd, vg, vs, vb, *self._params)
        return self._term["d"] @ i_d + self._term["s"] @ i_s + self._term["b"] @ i_b

    def _solve(self, v_drive: Tensor, drive_mask: Tensor) -> Tensor:
        """Newton solve of KCL on the undriven nodes. ``v_drive``: ``(B, N)`` with the driven
        values in place (zeros elsewhere); returns the full node vector ``(B, N)``."""
        unk = (~drive_mask).nonzero().flatten()
        if unk.numel() == 0:
            return v_drive
        B = v_drive.shape[0]

        def residual(vu: Tensor, vdr: Tensor) -> Tensor:
            v = vdr.index_put((unk,), vu)
            return self._node_currents(v)[unk]

        jac = torch.func.vmap(torch.func.jacfwd(residual, argnums=0))
        res = torch.func.vmap(residual)

        with torch.no_grad():
            vdr = v_drive.detach()
            vu = torch.full((B, unk.numel()), 2.5, dtype=self.dtype)
            for _ in range(self.max_iter):
                f, j = res(vu, vdr), jac(vu, vdr)
                dv = torch.linalg.solve(j, f.unsqueeze(-1)).squeeze(-1)
                dv = dv.clamp(-self.step_limit, self.step_limit)
                vu = vu - dv
                if dv.abs().max() < self.tol:
                    break
            j = jac(vu, vdr)
        # Final step with autograd: v* = vu - J^{-1} F(vu, v_drive) has the implicit gradient.
        f = res(vu, v_drive)
        vu = vu - torch.linalg.solve(j, f.unsqueeze(-1)).squeeze(-1)
        return v_drive.index_put((unk,), vu.transpose(0, 1).reshape(-1, B).transpose(0, 1)) \
            if False else self._scatter(v_drive, unk, vu)

    @staticmethod
    def _scatter(v_drive: Tensor, unk: Tensor, vu: Tensor) -> Tensor:
        out = v_drive.clone()
        out[:, unk] = vu
        return out

    # -- interface -------------------------------------------------------------------
    def forward_with_interventions(
        self, inputs: dict[str, Tensor], interventions: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        B = self.batch_size(inputs, interventions)
        N = len(self.all_nodes)
        current: dict[str, Tensor] = {}
        for inp in self.input_names:
            current[inp.lower()] = torch.zeros(B, dtype=self.dtype)
        for k in self.all_nodes:
            if k in inputs:
                current[k] = torch.as_tensor(inputs[k]).to(self.dtype).reshape(B)
        for k, v in interventions.items():
            current[k.lower()] = torch.as_tensor(v).to(self.dtype).reshape(B)

        v_drive = torch.zeros(B, N, dtype=self.dtype)
        drive_mask = torch.zeros(N, dtype=torch.bool)
        for node, val in self._fixed_sources.items():  # Vdd
            v_drive[:, self._index[node]] = val
            drive_mask[self._index[node]] = True
        drive_mask[self._index["0"]] = True
        for node, val in current.items():
            if node in ("0", "gnd", "vdd") or node not in self._index:
                continue
            v_drive = v_drive.index_put((torch.tensor([self._index[node]]),),
                                        val.unsqueeze(1), accumulate=False) \
                if False else self._set_col(v_drive, self._index[node], val)
            drive_mask[self._index[node]] = True

        v = self._solve(v_drive, drive_mask)
        out = {name: v[:, i].reshape(B, 1) for name, i in self._index.items()}
        # the original reports the analysis value for every analysed node (all but ground)
        # and the supplied value for ground, which is never driven
        if "0" in current:
            out["0"] = current["0"].reshape(B, 1)
        return out

    @staticmethod
    def _set_col(v: Tensor, col: int, val: Tensor) -> Tensor:
        cols = [v[:, :col], val.unsqueeze(1), v[:, col + 1:]]
        return torch.cat(cols, dim=1)


# --------------------------------------------------------------------------------------
# High-level abstraction (half-adder, from test/02_transistor_circuit.py)
# --------------------------------------------------------------------------------------

COARSE_GRAINING: dict[str, list[str]] = {"a": ["a"], "b": ["b"], "sum": ["sum"], "carry": ["carry"]}
INTERNAL_NODES: list[str] = ["nand1_out", "nand2_out", "nand3_out", "n_1", "n_2", "n_3", "n_4", "vdd", "0"]
#: Value map: logic value -> voltage interval (V) for each of a, b, sum, carry.
BIT_RANGES: dict[int, tuple[float, float]] = {0: (-0.5, 0.5), 1: (4.5, 5.5)}
HL_DISTRIBUTIONS: dict[str, dict[int, float]] = {k: {0: 0.5, 1: 0.5} for k in ("a", "b", "sum", "carry")}


def hl_valid(a: int, b: int) -> dict[str, int]:
    return {"sum": a ^ b, "carry": int(a and b)}


def hl_failing(a: int, b: int) -> dict[str, int]:
    return {"sum": int(a or b), "carry": int(a and b)}


HIGH_LEVEL_MODELS: dict[str, Callable[[int, int], dict[str, int]]] = {
    "valid": hl_valid,
    "fail": hl_failing,
    "noise": hl_valid,  # low-level outputs get Gaussian noise, std 1.5 V
}
