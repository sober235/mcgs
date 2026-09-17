"""arms.py -- the eleven arms of this topic in figure order, and where each scan's registered volume lives.
Swept families (CS-SENSE o200, joint CS-SENSE, L1-SPIRiT) resolve to the lambda / kernel kept in logs/summary.json."""
import json
import os
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
ORDER = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense_o200": "CS-SENSE", "jcssense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR", "gs_single": "GS-single", "gs_joint": "MC-GS"}


def arm_files(scan, summary=None):
    rows = summary if summary is not None else json.load(open(R / "logs/summary.json"))
    sel = {r["method"]: r for r in rows if r["contrast"] == scan}
    out = []
    for key in ORDER:
        if key == "cssense_o200" and key in sel:
            p = R / f"recon/cssense_o200_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        elif key == "jcssense" and key in sel:
            p = R / f"recon/jcssense_o200_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        elif key == "spirit" and key in sel:
            p = R / f"recon/spirit_ks{sel[key]['ksize']}_lam{sel[key]['lam']:g}_{scan}_complex.npy"
        else:
            p = R / f"recon/{key}_{scan}_complex.npy"
        out.append((key, LABEL[key], p))
    return out
