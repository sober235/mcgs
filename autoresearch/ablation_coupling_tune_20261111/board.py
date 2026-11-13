#!/usr/bin/env python3
"""board.py -- stage-three scoreboard (program.md section 11), read-only, CPU.

For task t1t2tb: every full model run (the registered V32 reference and the tn_t1t2tb_* full runs of the ledger)
against the pool of every independent model run so far (registered fw_t1t2tb_nojoint_* and the ledger's
tn_t1t2tb_* independents), per contrast and metric:
  PSNR / SSIM margin = full - best independent; NMSE margin = best independent - full (positive = full better).
m_best = the smaller PSNR margin of T1 and T2 (the decision metric). cells = how many of the six joint-row cells
(two contrasts x three metrics) full wins. Also iso / noprior rows with the same symmetric settings, if run.

    gpu.py run --cpu -- @python autoresearch/ablation_coupling_tune_20261111/board.py
Writes board.md to MCGS_ARTIFACT_DIR. Seed 42 only unless --seed is given.
"""
import argparse
import json
import os
from pathlib import Path

OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_framework_ablation_20261110")
RES, LEDGER = OUT / "trainer/results_arc", OUT / "tune/experiments.jsonl"
C = ("T1", "T2")
M = (("head_psnr", "PSNR", 1), ("head_ssim", "SSIM", 1), ("nmse", "NMSE", -1))
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()


def score(exp):
    s = json.load(open(RES / f"{exp}_score.json"))
    return s


def joint_row(exp, label, spec, ov):
    s = score(exp)
    return dict(exp=exp, label=label, spec=spec, ov=ov, T1=s["plain"], T2=s["contrast"])


fulls = [joint_row("fw_t1t2tbv32_full", "V32 reference (registered)", [], [])]
others = {"iso": [joint_row("fw_t1t2tb_iso", "registered (V32 + guide 0.5)", ["guide0.5"], []),],
          "noprior": [joint_row("fw_t1t2tb_noprior", "registered (V32 + guide 0.5)", ["guide0.5"], [])]}
pool = {"T1": [dict(exp="fw_t1t2tb_nojoint_T1", spec=[], m=score("fw_t1t2tb_nojoint_T1")["plain"])],
        "T2": [dict(exp="fw_t1t2tb_nojoint_T2", spec=[], m=score("fw_t1t2tb_nojoint_T2")["plain"])]}
rows = [json.loads(l) for l in open(LEDGER)] if LEDGER.exists() else []
for r in rows:
    if r.get("task") != "t1t2tb" or not r.get("ok") or r.get("smoke") or r.get("seed") != args.seed:
        continue
    spec, ov = r.get("sym_spec", []), r.get("overrides", [])
    if r["kind"] == "full":
        fulls.append(dict(exp=r["exp"], label=r["hypothesis"], spec=spec, ov=ov, T1=r["T1"], T2=r["T2"]))
    elif r["kind"] in ("iso", "noprior"):
        others[r["kind"]].append(dict(exp=r["exp"], label=r["hypothesis"], spec=spec, ov=ov, T1=r["T1"], T2=r["T2"]))
    else:
        c = "T1" if r["kind"] == "nojoint_p" else "T2"
        pool[c].append(dict(exp=r["exp"], spec=spec, m=r[c]))

best = {c: {k: max(pool[c], key=lambda p: sgn * p["m"][k]) for k, _, sgn in M} for c in C}
lines = ["## Independent pool (best per metric)", ""]
for c in C:
    for k, name, _ in M:
        b = best[c][k]
        lines.append(f"- {c} {name}: {b['m'][k]:.5g} ({b['exp']}, {' '.join(b['spec']) or 'V32'}); pool size {len(pool[c])}")
lines += ["", "## Full models", "",
          "| exp | change | T1 PSNR | T2 PSNR | dPSNR T1 | dPSNR T2 | dSSIM T1 | dSSIM T2 | dNMSE T1 | dNMSE T2 | m_best | cells |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
board = []
for f in fulls:
    d = {(c, k): sgn * (f[c][k] - best[c][k]["m"][k]) for c in C for k, _, sgn in M}
    mb = min(d[(c, "head_psnr")] for c in C)
    cells = sum(v > 0 for v in d.values())
    board.append(dict(exp=f["exp"], spec=f["spec"], ov=f["ov"], m_best=round(mb, 4), cells=cells,
                      margins={f"{c}_{n}": round(d[(c, k)], 5) for c in C for k, n, _ in M}))
    ch = " ".join(f["spec"] + f["ov"]) or "V32"
    lines.append(f"| {f['exp']} | {ch} | {f['T1']['head_psnr']:.3f} | {f['T2']['head_psnr']:.3f} | "
                 f"{d[('T1', 'head_psnr')]:+.3f} | {d[('T2', 'head_psnr')]:+.3f} | {d[('T1', 'head_ssim')]:+.4f} | "
                 f"{d[('T2', 'head_ssim')]:+.4f} | {d[('T1', 'nmse')]:+.5f} | {d[('T2', 'nmse')]:+.5f} | {mb:+.3f} | {cells}/6 |")
lines += ["", "## w/o anisotropy and w/o priors runs (full must beat the one with the same settings)", ""]
for kind, lst in others.items():
    for o in lst:
        lines.append(f"- {kind} {o['exp']} ({' '.join(o['spec'] + o['ov']) or 'V32'}): "
                     f"T1 {o['T1']['head_psnr']:.3f} / {o['T1']['head_ssim']:.4f} / {o['T1']['nmse']:.5f}; "
                     f"T2 {o['T2']['head_psnr']:.3f} / {o['T2']['head_ssim']:.4f} / {o['T2']['nmse']:.5f}")
text = "\n".join(lines)
print(text, flush=True)
art = os.environ.get("MCGS_ARTIFACT_DIR")
if art:
    Path(art, "board.md").write_text(text + "\n")
    Path(art, "board.json").write_text(json.dumps(board, indent=1))
print("BOARD " + json.dumps(board))
