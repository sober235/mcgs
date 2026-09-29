"""dataset.py retrospective normalisation: torch.quantile caps at 2**24 elements. Volumes above
16,000,000 elements must use numpy (same fallback as the presampled branch, dataset.py:306-313);
volumes at or below it keep the torch path bit for bit (every brain topic is 11,534,336 voxels).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= $PY -m pytest tests/test_dataset_quantile_gate.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
from data.dataset import MRIDataset  # noqa: E402


def _stub(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.complex(torch.rand(n, generator=g), torch.rand(n, generator=g))
    return SimpleNamespace(normalize_kspace=True, normalization_percentile=99.9,
                           ground_truth_image=gt.clone(), kspace_multicoil=torch.ones(2, 4, dtype=torch.complex64))


def test_brain_sized_volume_keeps_torch_quantile_bit_for_bit():
    s = _stub(1_000_000)
    expected = float(torch.clamp(torch.quantile(torch.abs(s.ground_truth_image), 0.999), min=1e-8).item())
    MRIDataset._normalize_intensity_scale(s)
    assert s.normalization_scale == expected


def test_volume_above_16m_elements_uses_numpy_quantile():
    s = _stub(17_000_000)                      # > 2**24 = 16,777,216: torch.quantile raises on this
    expected = float(np.quantile(torch.abs(s.ground_truth_image).numpy(), 0.999))
    MRIDataset._normalize_intensity_scale(s)
    assert s.normalization_scale == pytest.approx(expected, rel=1e-6)
