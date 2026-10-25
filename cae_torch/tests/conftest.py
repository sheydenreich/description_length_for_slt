import pytest
import torch

torch.set_default_dtype(torch.float64)


@pytest.fixture
def cae_module():
    """Load a numpy system module by file name or skip if the CAE clone is missing."""
    from cae_torch.reference import load_numpy_system

    def _load(filename: str):
        try:
            return load_numpy_system(filename)
        except (FileNotFoundError, ImportError) as exc:
            pytest.skip(f"numpy reference unavailable: {exc}")

    return _load
