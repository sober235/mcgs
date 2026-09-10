"""Regenerate the results tables of the three topic READMEs from logs/summary.json (best per column in bold)."""
import json, re, sys
from pathlib import Path
R = Path("/data0/congcong/code/GS/Results_20260908")
WALL = {}
def wall(r):
    s = r.get("seconds")
    return f"{s/60:.0f} min" if s else ""
def table(rows, cols, hdr):
    rows = sorted(rows, key=lambda r: -r["head_psnr"])
    best = {}
    for c, key, lower in cols:
        vals = [r[key] for r in rows if r.get(key) is not None]
        best[key] = (min if lower else max)(vals)
    out = ["| " + " | ".join(["method"] + [c for c, _, _ in cols] + ["wall"]) + " |", "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        cells = []
        for c, key, lower in cols:
            v = r[key]; fmt = "%.3f" if key in ("psnr", "head_psnr") else ("%.5f" if key == "nmse" else ("%.4f" if key in ("ssim", "head_ssim", "hfen", "head_hfen", "grain") else "%.3f"))
            s = fmt % v
            cells.append(f"**{s}**" if abs(v - best[key]) < 1e-9 else s)
        name = r["label"]
        out.append("| " + " | ".join([name] + cells + [wall(r)]) + " |")
    return "\n".join(out)
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE (TV, 35 outer, as run 09-09)", "cssense_o200": "CS-SENSE (TV, converged)",
         "jcssense": "joint CS-SENSE (JTV)", "l1espirit": "L1-ESPIRiT", "ploraks": "P-LORAKS", "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR",
         "gs_single": "MC-GS single (old cold start)", "gs_joint": "MC-GS joint (old cold start)", "gs_single_v2": "MC-GS single v2", "gs_joint_v2": "MC-GS joint v2", "gs_single_v3": "MC-GS single v3", "gs_joint_v3": "MC-GS joint v3", "gs_joint_v3x2": "MC-GS joint v3 ×2 seeds (2× compute)", "gs_joint_v32": "**MC-GS joint v3.2**", "gs_joint_v32x2": "MC-GS joint v3.2 ×2 seeds (2× compute)"}
topic = sys.argv[1]
rows = json.load(open(R / topic / "logs/summary.json"))
for r in rows: r["label"] = LABEL.get(r["method"], r["method"])
p = R / topic / "README.md"; s = p.read_text()
if topic == "T1T2_AF12_retro":
    cols = [("PSNR", "psnr", False), ("SSIM", "ssim", False), ("NMSE", "nmse", True), ("HFEN", "hfen", True), ("head PSNR", "head_psnr", False), ("head SSIM", "head_ssim", False), ("head HFEN", "head_hfen", True), ("sharp", "sharp", False), ("grain", "grain", True)]
    for c, marker in (("T1", "**T1** — sorted by head-only PSNR, which is the number that reflects image quality here:\n\n"), ("T2", "**T2**:\n\n")):
        i = s.index(marker) + len(marker); j = s.index("\n\n", i)
        s = s[:i] + table([r for r in rows if r["contrast"] == c], cols, None) + s[j:]
else:
    cols = [("PSNR", "psnr", False), ("SSIM", "ssim", False), ("head PSNR", "head_psnr", False), ("head SSIM", "head_ssim", False)]
    for c in ("T1", "FLAIR"):
        m = re.search(r"\| method \| PSNR \| SSIM \| head PSNR \| head SSIM \|( wall \|)?\n\|---\|---\|---\|---\|---\|(---\|)?\n(\|.*\n)+", s)
        blocks = list(re.finditer(r"\| method \| PSNR \| SSIM \| head PSNR \| head SSIM \|( wall \|)?\n\|---\|---\|---\|---\|---\|(---\|)?\n(\|.*\n)+", s))
        # the README has one table per contrast in contrast order; replace by position
    blocks = list(re.finditer(r"\| method \| PSNR \| SSIM \| head PSNR \| head SSIM \|( wall \|)?\n\|---\|---\|---\|---\|---\|(---\|)?\n(\|.*\n)+", s))
    assert len(blocks) >= 1, "no table found"
    order = ["FLAIR", "T1"] if topic == "T1FLAIR_AF12_retro" else ["T1", "FLAIR"]
    for b, c in reversed(list(zip(blocks, order[:len(blocks)]))):
        s = s[:b.start()] + table([r for r in rows if r["contrast"] == c], cols, None) + "\n" + s[b.end():]
p.write_text(s); print("tables regenerated", topic)
