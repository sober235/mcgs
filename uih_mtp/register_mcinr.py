"""Register the MC-INR reproduction (Niessen et al. 2025) as an arm of a retrospective topic.

The reproduction scores itself through the topic's own gauge module (its score.py re-checks that gauge
against the topic's recorded CG-SENSE row first), so registering is a copy of the row it printed plus a
link to the volume it scored -- no second scoring path that could disagree with the rest of the table.
"""
import json
from pathlib import Path


def register(mcinr_root, topic, af=12, contrasts=("T1w", "T2s"),
             note="multi-contrast INR of Niessen et al. 2025, per-slice 2-D, paper recipe; one network "
                  "outputs both contrasts, so `seconds` is the whole two-contrast run"):
    """Write topic/logs/mcinr_<c>.json and link topic/recon/mcinr_<c>_complex.npy for each contrast."""
    mcinr_root, topic = Path(mcinr_root), Path(topic)
    score = json.load(open(mcinr_root / "logs" / f"score_af{af}.json"))
    volumes, rows = {}, {}
    for c in contrasts:
        entry = score["contrasts"][c]
        bad = [k for k, v in entry["gauge_check"].items() if not v["ok"]]
        if bad:
            raise ValueError(f"{c}: the run's gauge re-check failed on {bad}; the row is not comparable")
        src = mcinr_root / "recon" / f"af{af}" / f"mcinr_{c}_complex.npy"
        if not src.exists():
            raise FileNotFoundError(src)
        volumes[c], rows[c] = src, entry["row"]
    written = {}
    for c in contrasts:
        link = topic / "recon" / f"mcinr_{c}_complex.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(volumes[c])
        out = topic / "logs" / f"mcinr_{c}.json"
        json.dump({"contrast": c, "rows": [rows[c]], "source": str(volumes[c]),
                   "gauge_scale": score["contrasts"][c].get("gauge_scale"), "note": note},
                  open(out, "w"), indent=1)
        written[c] = {"log": str(out), "volume": str(link), "head_psnr": rows[c]["head_psnr"]}
    return written
