import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
TRAINER = Path(os.environ.get("RETRO_TRAINER", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/trainer/_trainer_e778bf2"))


def test_trainer_snapshot_and_gauge_imports():
    assert (TRAINER / "arc/cycle.py").exists() and (TRAINER / "SOURCE.txt").exists()
    for p in (str(TRAINER / "paired_gs"), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from metrics import evaluate_reconstruction          # noqa: F401
    from diag_artefact.headonly import score_headonly, head_mask
    gt = np.zeros((16, 16, 16), np.float32); gt[4:12, 4:12, 4:12] = 1.0
    s = score_headonly(gt, gt)
    assert s["head_psnr"] == float("inf") and abs(s["head_ssim"] - 1.0) < 1e-6
    assert head_mask(gt).sum() == 8 ** 3
