#!/usr/bin/env python3
"""register_mcinr_arm.py -- put the MC-INR reproduction's run (mcinr/) into this topic's table.

The reproduction scored itself through this topic's own gauge module after re-checking that gauge
against the recorded CG-SENSE row, so this only copies the row it printed and links the volume it
scored. Run it after mcinr/logs/score_af12.json exists, then rescore.py, summarize.py, readme_tables.py.

  ./scripts/py scripts/register_mcinr_arm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")
from uih_mtp.register_mcinr import register  # noqa: E402

R = Path(__file__).resolve().parents[1]
for c, info in register(R / "mcinr", R).items():
    print(f"{c}: head PSNR {info['head_psnr']:.3f}  volume {info['volume']}  row {info['log']}")
