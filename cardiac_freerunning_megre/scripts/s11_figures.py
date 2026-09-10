"""Three-plane figure of one reconstructed set, one row per echo (optionally with the heart box)."""
import argparse
import json

import numpy as np

from common import TOPIC, recon_dir, three_plane

ap = argparse.ArgumentParser()
ap.add_argument("subject")
ap.add_argument("--tag", required=True)
ap.add_argument("--echoes", default="0,1,2,3,4,5,6,7")
ap.add_argument("--center", default=None, help="i,j,k voxel the three planes pass through")
ap.add_argument("--box", action="store_true", help="overlay gauge.json[subject]['heart_box']")
ap.add_argument("--name", default=None)
a = ap.parse_args()
vols = {f"e{e}  TE {1.25 + 2.05 * e:.2f} ms": np.load(recon_dir(a.subject) / f"{a.tag}_e{e}.npy")
        for e in map(int, a.echoes.split(","))}
center = tuple(map(int, a.center.split(","))) if a.center else None
box = json.loads((TOPIC / "gauge.json").read_text())[a.subject]["heart_box"] if a.box else None
three_plane(vols, a.name or f"{a.subject}_{a.tag}.png", center=center, title=f"{a.subject} {a.tag}", box=box)
