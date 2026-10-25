"""Command-line entry points."""

from __future__ import annotations

import importlib
import platform
import sys


def check_env() -> int:
    """Print versions of the interpreter and every optional dependency.

    Exits non-zero only if a layer-1 dependency (numpy, scipy) is missing; the
    optional layers report MISSING and the command still succeeds.
    """
    print(f"python {platform.python_version()} on {platform.platform()}")
    required = ["numpy", "scipy", "yaml"]
    optional = ["torch", "pandas", "pyarrow", "transformer_lens", "pyvene", "gudhi", "pytest"]
    missing_required = []
    for name in required + optional:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            extra = ""
            if name == "torch":
                extra = f"  cuda={module.cuda.is_available()}"
            print(f"  {name:<18} {version}{extra}")
        except ImportError:
            print(f"  {name:<18} MISSING")
            if name in required:
                missing_required.append(name)
    if missing_required:
        print(f"required packages missing: {missing_required}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(check_env())
