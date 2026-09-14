#!/usr/bin/env python3
"""Per-contrast weight of the data term in the joint objective (paired_gs/losses/balance.py::data_term).

The shared geometry receives the gradients of both data terms; Adam makes each channel's own
densities blind to a constant factor on that channel's objective, so this weight decides how the
shared geometry is split between the two contrasts. Pins: weight 1 is exactly the old sum (the same
float operation, so older runs stay bit-identical), another weight scales only the contrast term,
and the value reported for dc_c is the one actually added.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -m pytest tests/test_data_term.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from losses.balance import data_term


def test_weight_one_is_the_old_sum_bit_for_bit():
    p = torch.tensor(0.0063123, dtype=torch.float32, requires_grad=True)
    c = torch.tensor(0.0320931, dtype=torch.float32, requires_grad=True)
    added, reported_c = data_term(p, c, 1.0)
    assert torch.equal(added, p + c)
    assert torch.equal(reported_c, c)


def test_other_weight_scales_only_the_contrast_term():
    p = torch.tensor(0.006, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.032, dtype=torch.float64, requires_grad=True)
    added, reported_c = data_term(p, c, 0.25)
    assert abs(float(added) - (0.006 + 0.25 * 0.032)) < 1e-15
    assert abs(float(reported_c) - 0.25 * 0.032) < 1e-15
    added.backward()
    assert float(p.grad) == 1.0 and abs(float(c.grad) - 0.25) < 1e-15
