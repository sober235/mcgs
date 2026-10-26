"""Seed all RNGs (extracted from the MC-GS train.py so the paired framework
does not drag in the interactive test/train harness). See train_paired.py."""
from __future__ import annotations
import os, random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs. When ``deterministic`` is True, also lock cuDNN / TF32 so a
    single machine + single GPU reproduces bit-identically. Default False keeps the
    fast path used by sweeps and the ARC autoresearch loop; paper-grade runs set
    ``training.deterministic: true`` in the pair config."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print(f"[set_seed] deterministic=True (seed={seed}, cuDNN locked, TF32 off)")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[set_seed] deterministic=False (seed={seed}, cuDNN fast, TF32 on)")
