#!/usr/bin/env python3
"""draw_lead_slice.py -- the slices of the pediatric TSC T1w + T2* AF12 line on which MC-GS leads by most.

Selection (from the topic's own per-slice records, logs/find_slice_all_first_{T1w,T2s}.json, MC-GS = gs_joint_p5):
among the slices on which BOTH contrasts are first on PSNR, SSIM and NMSE against every comparison arm, rank by
min(T1w PSNR lead, T2* PSNR lead), the lead being over the best other arm of that slice.

  default            one plane: the top slice of that ranking (--slice PLANE IDX draws another slice instead)
  --three            three planes: the top slice of each plane (--slices TRA COR SAG overrides them)

Drawing: the topic's house script scripts/fig_three_planes_inset.py with the flags of the figure of record
(46 / 149 / 171). The zoom box sits where the three slices meet; in the one-plane figure the other two indices are
the best slices of their planes and only place the box. One figure per contrast, then both stacked (T1w above T2*).

The topic stays read-only: the house script is copied into the artifact directory with three lines changed
(topic root given explicitly, both output folders replaced by the artifact directory). Each replaced line must
occur exactly once. CPU only.

  gpu.py run --cpu -- @python autoresearch/peds_af12_lead_fig_20261111/draw_lead_slice.py [--three]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TOPIC = Path("/data0/congcong/code/GS/Results_20260908/Peds_TSC_MTP_AF12_retro")
SRC = TOPIC / "scripts/fig_three_planes_inset.py"
SERVED = Path("/home/congcongliu/figs/peds_tsc_af12")
KEY = "gs_joint_p5"
SINGLE = {"T1w": "gs_single_p3f", "T2s": "gs_single_ar"}
ORDER = ("transverse", "coronal", "sagittal")
SHORT = {"transverse": "tra", "coronal": "cor", "sagittal": "sag"}
CONTRASTS = ("T1w", "T2s")

ap = argparse.ArgumentParser()
ap.add_argument("--slice", nargs=2, metavar=("PLANE", "IDX"), help="one-plane figure of this slice instead of the selected one")
ap.add_argument("--three", action="store_true", help="three-plane figure: the best slice of each plane")
ap.add_argument("--slices", type=int, nargs=3, metavar=("TRA", "COR", "SAG"), help="with --three: these slices instead")
ap.add_argument("--inset_center", type=int, nargs=3, default=[-1, -1, -1], metavar=("RO", "PE", "SPE"))
ap.add_argument("--top", type=int, default=5, help="how many ranked slices to print")
args = ap.parse_args()
assert not (args.three and args.slice), "--slice is the one-plane mode; with --three use --slices"
assert args.three or not args.slices, "--slices needs --three"

os.nice(10)
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "lead_fig_out")).resolve()
art.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# --- selection ---------------------------------------------------------------------------------
J = {c: json.load(open(TOPIC / f"logs/find_slice_all_first_{c}.json")) for c in CONTRASTS}
for c in CONTRASTS:
    assert J[c]["key"] == KEY, (c, J[c]["key"])
M = {c: {(e["plane"], e["idx"]): e for e in J[c]["slices"]} for c in CONTRASTS}
both = [k for k in M["T1w"] if k in M["T2s"] and M["T1w"][k]["wins_all"] and M["T2s"][k]["wins_all"]]


def lead(k):
    return min(M["T1w"][k]["d_psnr"], M["T2s"][k]["d_psnr"])


def show(k):
    a, b = M["T1w"][k], M["T2s"][k]
    return (f"{k[0]:<10} {k[1]:>3}  area {a['area_rel']:.2f}  T1w {a['d_psnr']:+.3f} dB (vs {a['runner_psnr']})"
            f"  T2s {b['d_psnr']:+.3f} dB (vs {b['runner_psnr']})")


ranked = sorted(both, key=lead, reverse=True)
plane_best = {pl: next(k for k in ranked if k[0] == pl) for pl in ORDER}

log(f"slices with both contrasts first on PSNR, SSIM and NMSE against every arm: {len(both)}")
log("rank by min(T1w, T2s) PSNR lead over the best other arm:")
for k in ranked[:args.top]:
    log("  " + show(k))
log("best slice of each plane:")
for pl in ORDER:
    log("  " + show(plane_best[pl]))

if args.three:
    slices = list(args.slices) if args.slices else [plane_best[pl][1] for pl in ORDER]
    drawn = [(pl, slices[i]) for i, pl in enumerate(ORDER)]
    cbar = "0"                                   # as the figure of record: bars beside the transverse rows
else:
    chosen = (args.slice[0], int(args.slice[1])) if args.slice else ranked[0]
    assert chosen[0] in ORDER, chosen
    slices = [chosen[1] if pl == chosen[0] else plane_best[pl][1] for pl in ORDER]
    drawn = [chosen]
    # --cbar_planes counts in the full display order (0 transverse, 1 coronal, 2 sagittal), also when one plane is drawn
    cbar = str(ORDER.index(chosen[0]))
for k in drawn:
    assert k in M["T1w"] and k in M["T2s"], f"{k} is not in the per-slice records of both contrasts"
    log(f"drawn: {k[0]} {k[1]}" + ("" if k in both else "  (NOT a both-contrasts-first slice)"))

# --- the house script, outputs redirected ---------------------------------------------------------
src = SRC.read_text()
src_sha = hashlib.sha256(src.encode()).hexdigest()
PATCHES = (
    ('R = Path(__file__).resolve().parents[1]', f'R = Path({str(TOPIC)!r})'),
    ('F = Path("/home/congcongliu/figs/peds_tsc_af12"); F.mkdir(parents=True, exist_ok=True)', f'F = Path({str(art)!r})'),
    ('for d in (F, R / "figures"):', 'for d in (F,):'),
)
for old, new in PATCHES:
    assert src.count(old) == 1, f"expected exactly one occurrence of: {old}"
    src = src.replace(old, new)
patched = art / "fig_three_planes_inset_patched.py"
patched.write_text(src)


def listing(d):
    return set(os.listdir(d)) if d.is_dir() else set()


before = {d: listing(d) for d in (TOPIC / "figures", SERVED)}
env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONNOUSERSITE="1")
RANK = re.compile(r"(PSNR|SSIM|NMSE) MC-GS #(\d+) \(first: ([^)]+)\)")
record = dict(topic=str(TOPIC), mcgs_key=KEY, single_key=SINGLE, rule="min over contrasts of the PSNR lead, both-first slices",
              n_both_first=len(both), drawn=[dict(plane=k[0], idx=k[1], both_first=k in both) for k in drawn],
              slices_arg=slices, house_script=str(SRC), house_script_sha256=src_sha, contrasts={})
pngs = []
for c in CONTRASTS:
    cmd = [sys.executable, str(patched), "--contrast", c, "--slices", *map(str, slices),
           "--planes", *[SHORT[k[0]] for k in drawn],
           "--inset", "48", "48", "16", "--inset_center", *map(str, args.inset_center), "--inset_corner", "tr", "bl", "bl",
           "--mask_error", "--gray_cbar", "--cbar_planes", cbar, "--l1", "spirit", "--with_zf", "--caps",
           "--mcgs_key", KEY, "--single_key", SINGLE[c], "--tag", "_lead_p5"]
    log("\n$", " ".join(cmd))
    out = subprocess.run(cmd, env=env, cwd=str(art), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log(out.stdout.rstrip())
    assert out.returncode == 0, f"{c}: the figure script returned {out.returncode}"
    m = re.search(r"^wrote (\S+\.png)", out.stdout, re.M)
    assert m and Path(m.group(1)).is_file(), f"{c}: no figure written"
    pngs.append(Path(m.group(1)))
    per = {}
    for k in drawn:
        line = next(ln for ln in out.stdout.splitlines() if ln.strip().startswith(f"{k[0]} {k[1]} "))
        e = M[c][k]
        others = {a: v for a, v in e["psnr"].items() if a != KEY}
        best_other = max(others, key=others.get)
        rec = dict(panel_ranks={met: dict(rank=int(r), first=first) for met, r, first in RANK.findall(line)},
                   area_rel=e["area_rel"], psnr_mcgs=e["psnr"][KEY], psnr_best_other=others[best_other],
                   psnr_best_other_arm=best_other, d_psnr=e["d_psnr"])
        if e.get("ssim"):
            so = {a: v for a, v in e["ssim"].items() if a != KEY}
            sb = max(so, key=so.get)
            rec.update(ssim_mcgs=e["ssim"][KEY], ssim_best_other=so[sb], ssim_best_other_arm=sb, d_ssim=e.get("d_ssim"))
        per[f"{k[0]} {k[1]}"] = rec
    record["contrasts"][c] = dict(figure=pngs[-1].name, slices=per)

# --- both contrasts in one image --------------------------------------------------------------------
from PIL import Image  # noqa: E402

ims = [Image.open(p).convert("RGB") for p in pngs]
W, H = max(i.width for i in ims), sum(i.height for i in ims)
canvas = Image.new("RGB", (W, H), "white")
y = 0
for i in ims:
    canvas.paste(i, ((W - i.width) // 2, y))
    y += i.height
stem = ("three_" + "_".join(map(str, slices))) if args.three else f"{SHORT[drawn[0][0]]}{drawn[0][1]}"
combined = art / f"peds_af12_lead_{stem}_T1w_T2s.png"
canvas.save(combined)
record["combined"] = combined.name

# --- checks -----------------------------------------------------------------------------------------
new = {str(d): sorted(listing(d) - before[d]) for d in before}
record["new_files_in_topic_or_served"] = new
assert not any(new.values()), f"the topic or the served folder gained files: {new}"
all_first = all(r["panel_ranks"].get(m, {}).get("rank") == 1
                for c in CONTRASTS for r in record["contrasts"][c]["slices"].values() for m in ("PSNR", "SSIM", "NMSE"))
record["all_first_on_drawn_panels"] = all_first
(art / "record.json").write_text(json.dumps(record, indent=2))

log("\nsummary")
for c in CONTRASTS:
    for name, r in record["contrasts"][c]["slices"].items():
        s = (f"  SSIM {r['ssim_mcgs']:.4f} vs {r['ssim_best_other']:.4f} ({r['ssim_best_other_arm']}, {r['d_ssim']:+.4f})"
             if "ssim_mcgs" in r else "")
        log(f"  {c} {name}: PSNR {r['psnr_mcgs']:.3f} vs {r['psnr_best_other']:.3f} ({r['psnr_best_other_arm']}, {r['d_psnr']:+.3f} dB){s}"
            f"  panel ranks " + " ".join(f"{m} #{r['panel_ranks'][m]['rank']}" for m in r["panel_ranks"]))
log("MC-GS first on PSNR, SSIM and NMSE on every drawn panel of both contrasts:", all_first)
log("no file added to the topic's figures/ or to the served folder")
log("wrote", combined)
log("DONE")
