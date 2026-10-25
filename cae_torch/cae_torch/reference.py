"""Load the original numpy systems from a CAE clone, for equivalence tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def cae_dir() -> Path:
    return Path(os.environ.get("CAE_DIR", Path.home() / "scratch" / "third_party" / "CAE"))


def load_numpy_system(filename: str) -> ModuleType:
    """Import ``systems/<filename>`` from the CAE clone by path (the names start with digits).

    ``filename`` is for example ``"01_logic_circuit.py"`` or ``"09_grn/grn.py"``.
    Raises ``FileNotFoundError`` if the clone is absent; tests turn that into a skip.
    """
    path = cae_dir() / "systems" / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set CAE_DIR to a clone of MelouxM/CAE")
    name = "cae_numpy_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def numpy_forward(model, inputs: dict, interventions: dict) -> dict:
    """Run a numpy ``LowLevelModel`` on numpy inputs and return ``{name: array}``."""
    from causal_abstraction import SystemState

    state = SystemState(values=dict(inputs))
    return dict(model.forward_with_interventions(state, dict(interventions)).values)
