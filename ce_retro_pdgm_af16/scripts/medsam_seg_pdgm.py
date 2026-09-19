#!/usr/bin/env python3
"""medsam_seg_pdgm.py -- one public segmentation model, one expert-derived prompt, every arm of the PDGM AF16 line.

WHY THIS LINE AND NOT THE UIH ONE. On `../ce_retro_cssense_af16` the truth is itself a TV-regularised CS-SENSE
reconstruction, so a downstream segmenter reads the CS-SENSE family as most truth-like for a structural reason and
the ranking says nothing about reconstruction quality (SEGMENTATION.md there has the numbers). Here the truth is a
real clinical image -- UCSF-PDGM T1 / T1c, put through the same synthetic acquisition -- and it ships an expert
tumour segmentation, so the reference is a radiologist's outline rather than any algorithm's output.

TASK. Target = the tumour core, labels 1 (necrotic) | 4 (enhancing), on the post-contrast volume. The target and
the prompt protocol were both fixed by the model's competence ON THE TRUTH ALONE, before any arm was scored:
  * target: core 0.787 / 0.765 / 0.523 / 0.604 (cases 0004 / 0023 / 0031 / 0053) against enhancing-only
    0.614 / 0.437 / 0.489 / 0.507 and whole-tumour 0.603 / 0.520 / 0.339 / 0.508 -- the core wins on all four;
  * prompt: a per-slice box beats one 3-D box projected onto every slice on all four cases
    (0.787 / 0.765 / 0.523 / 0.604 against 0.773 / 0.727 / 0.398 / 0.589), and it is the protocol MedSAM is
    evaluated with in its own paper. Near the poles of the mass a single 3-D box is many times the cross-section.

WHAT IS IDENTICAL FOR EVERY ARM. The model, its weights, and the prompt: one box per axial slice, taken from the
EXPERT LABEL (bounding box of the target's cross-section grown by MARGIN, only on slices holding at least MIN_AREA
of it). No arm's own image influences the prompt. The image is put in white-matter units by its own white-matter
median and windowed to [0, WINDOW_TOP] -> 8-bit, the same fixed window for all. Unlike the topic's own scorer this
does NOT scale an arm against the truth first: nothing but the white-matter mask should reach the arm's image.

SCORED TWICE, and they answer different questions:
  * vs the expert label   -- absolute: whose reconstruction lets the segmenter land closest to the radiologist;
  * vs the model's own mask on the truth -- consistency: how much of the gap is the reconstruction rather than the
    model's own error, which is large here (the truth itself only reaches Dice 0.52-0.79 against the label).
Report both; neither alone is the story.

  env PYTHONNOUSERSITE=1 PYTHONPATH= $MEDSAM_PY scripts/medsam_seg_pdgm.py --scan post
  env PYTHONNOUSERSITE=1 PYTHONPATH= $MEDSAM_PY scripts/medsam_seg_pdgm.py --scan pre

$MEDSAM_PY is the venv carrying `transformers` (see ../ce_retro_cssense_af16/SEGMENTATION.md for how it was built
and where the weights came from). Writes seg_medsam/<case>/medsam_<arm>_<scan>_<target>.npy and
logs/medsam_seg_<scan>_<target>.json under the PDGM topic.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

T = Path(os.environ.get("PDGM_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_af16"))
MEDSAM_DIR = os.environ.get("MEDSAM_DIR", "/home/congcongliu/models/medsam-vit-base")

ORDER = ["zf", "cgsense", "cssense", "jsense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]
LABEL = {"zf": "ZF", "cgsense": "CG-SENSE", "cssense": "CS-SENSE", "jsense": "Joint SENSE", "spirit": "L1-SPIRiT",
         "zsssdu": "ZS-SSDU", "dip": "3D-DIP", "inr": "3D-INR", "mcinr": "MC-INR", "gs_single": "GS-single",
         "gs_joint": "MC-GS"}
REAL_CASES = ["UCSF-PDGM-0004", "UCSF-PDGM-0023", "UCSF-PDGM-0031", "UCSF-PDGM-0053"]
TARGETS = {"tc": (1, 4), "et": (4,), "wt": (1, 2, 4)}
AXIS = 2                     # (240, 240, 155): axis 2 is the axial slice index
MARGIN = 4
MIN_AREA = 20
WINDOW_TOP = 2.0

# These four helpers are transcribed from ../ce_retro_cssense_af16/scripts/medsam_seg.py rather than imported: the
# topics are independent result trees and the repo's habit is to copy a short helper instead of building a
# cross-topic import chain. tests/test_medsam_seg_pdgm.py pins the two copies to the same behaviour.


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else float("nan")


def hd95(a, b):
    if not a.any() or not b.any():
        return float("nan")

    def surface(m):
        return m & ~ndi.binary_erosion(m, structure=np.ones((3,) * m.ndim, bool))

    sa, sb = surface(a), surface(b)
    da = ndi.distance_transform_edt(~sb)[sa]
    db = ndi.distance_transform_edt(~sa)[sb]
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def largest_component(mask):
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == int(np.argmax(sizes)) + 1


def to_uint8_window(img, top=WINDOW_TOP):
    return np.clip(img / top, 0, 1).__mul__(255).round().astype(np.uint8)


def slice_boxes(ref, margin=MARGIN, min_area=MIN_AREA):
    """{axial index: (row0, col0, row1, col1)} from the expert label -- the prompt, identical for every arm"""
    boxes = {}
    for k in range(ref.shape[AXIS]):
        s = ref[:, :, k]
        if s.sum() < min_area:
            continue
        rr, cc = np.nonzero(s)
        boxes[k] = (max(int(rr.min()) - margin, 0), max(int(cc.min()) - margin, 0),
                    min(int(rr.max()) + margin, s.shape[0] - 1), min(int(cc.max()) + margin, s.shape[1] - 1))
    return boxes


def more_sel(case):
    f = T / "recon" / case / "more_arms_selection.json"
    return json.load(open(f)) if f.exists() else {}


def cs_lambda(case, scan, default=2e-3):
    f = T / "recon" / case / "cs_lambda.json"
    return json.load(open(f))[scan]["lam"] if f.exists() else default


def paths(arm, case, scan):
    """where an arm's volume lives; transcribed from the topic's own scripts/retro_score.py, including the flag
    saying the GS trainer writes (readout, PE2, PE1) and needs a (0, 2, 1) transpose."""
    R = T / "recon" / case
    sel = more_sel(case)
    if arm == "jsense":
        lam = sel.get("jsense", {}).get("lam")
        return (R / "tv" / f"jsense_o200_lam{lam:g}_full_{scan}_complex.npy") if lam else Path("/nonexistent"), False
    if arm == "spirit":
        q = sel.get(f"spirit_{scan}")
        return ((R / "spirit" / f"spirit_ks{q['ksize']}_i150_lam{q['lam']:g}_full_{scan}_complex.npy")
                if q else Path("/nonexistent")), False
    if arm == "zsssdu":
        return R / f"zsssdu_{scan}_complex.npy", False
    if arm == "mcinr":
        return R / "mcinr" / f"mcinr_full_{scan}_complex.npy", False
    if arm == "cssense":
        return R / "tv" / f"cssense_o200_lam{cs_lambda(case, scan):g}_full_{scan}_complex.npy", False
    if arm == "gs_single":
        return R / f"gs_single_{scan}" / f"gs_single_{scan}_raw_complex.npy", True
    if arm == "gs_joint":
        return R / "gs_joint" / f"gs_joint_{scan}_raw_complex.npy", True
    return R / f"{arm}_{scan}_complex.npy", False


def wm_units(vol, wm):
    return np.abs(vol).astype(np.float64) / float(np.median(np.abs(vol)[wm]))


class AxialMedSAM:
    """MedSAM with a per-slice box prompt, slicing along this grid's axial axis"""

    def __init__(self, boxes, device=None):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.boxes = boxes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = SamProcessor.from_pretrained(MEDSAM_DIR)
        self.model = SamModel.from_pretrained(MEDSAM_DIR).to(self.device).eval()

    def segment(self, vn):
        out = np.zeros(vn.shape, bool)
        for k, (r0, c0, r1, c1) in self.boxes.items():
            rgb = np.repeat(to_uint8_window(vn[:, :, k])[:, :, None], 3, axis=2)
            inp = self.proc(rgb, input_boxes=[[[c0, r0, c1, r1]]], return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                pred = self.model(**inp, multimask_output=False)
            m = self.proc.image_processor.post_process_masks(
                pred.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            out[:, :, k] = m[0, 0].numpy()
        return largest_component(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", choices=["pre", "post"], default="post")
    ap.add_argument("--target", choices=sorted(TARGETS), default="tc")
    ap.add_argument("--cases", default=",".join(REAL_CASES))
    ap.add_argument("--arms", default=",".join(ORDER))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cases, arms = a.cases.split(","), a.arms.split(",")
    labs = TARGETS[a.target]

    rows = []
    for case in cases:
        g = np.load(T / "sim" / case / "gt.npz")
        wm, tum = g["wm"], g["tumor"]
        ref = np.isin(tum, labs)
        if not ref.any():
            print(f"{case}: no target voxels, skipped")
            continue
        boxes = slice_boxes(ref)
        net = AxialMedSAM(boxes, a.device)
        out_dir = T / "seg_medsam" / case
        out_dir.mkdir(parents=True, exist_ok=True)

        tmask = net.segment(wm_units(g[a.scan], wm))
        np.save(out_dir / f"medsam_truth_{a.scan}_{a.target}.npy", tmask)
        d_t, h_t = dice(tmask, ref), hd95(tmask, ref)
        print(f"\n{case} [{a.scan}, {a.target}] expert {int(ref.sum())} vox on {len(boxes)} slices; "
              f"truth mask {int(tmask.sum())} vox, Dice {d_t:.3f}, HD95 {h_t:.2f} mm", flush=True)
        rows.append(dict(case=case, arm="truth", label="truth", vox=int(tmask.sum()), dice_expert=d_t,
                         hd95_expert=h_t, dice_truthmask=1.0, hd95_truthmask=0.0, expert_vox=int(ref.sum())))
        for arm in arms:
            p, gs = paths(arm, case, a.scan)
            if not p.exists():
                print(f"  {LABEL.get(arm, arm):>12}  missing: {p}")
                continue
            v = np.load(p)
            if gs:
                v = np.transpose(v, (0, 2, 1))
            m = net.segment(wm_units(v, wm))
            np.save(out_dir / f"medsam_{arm}_{a.scan}_{a.target}.npy", m)
            r = dict(case=case, arm=arm, label=LABEL.get(arm, arm), vox=int(m.sum()),
                     dice_expert=dice(m, ref), hd95_expert=hd95(m, ref),
                     dice_truthmask=dice(m, tmask), hd95_truthmask=hd95(m, tmask), expert_vox=int(ref.sum()))
            rows.append(r)
            print(f"  {r['label']:>12}  {r['vox']:6d} vox   vs expert Dice {r['dice_expert']:.3f} "
                  f"HD95 {r['hd95_expert']:5.2f}   vs truth-mask Dice {r['dice_truthmask']:.3f} "
                  f"HD95 {r['hd95_truthmask']:5.2f}", flush=True)

    out = T / f"logs/medsam_seg_{a.scan}_{a.target}.json"
    json.dump(dict(scan=a.scan, target=a.target, labels=list(labs), model=MEDSAM_DIR, margin=MARGIN,
                   min_area=MIN_AREA, window_top=WINDOW_TOP, prompt="per-slice box from the expert label",
                   rows=rows), open(out, "w"), indent=1)

    real = [r for r in rows if r["case"] in REAL_CASES and r["arm"] != "truth"]
    if real:
        print(f"\nmean over {len({r['case'] for r in real})} cases, [{a.scan}, {a.target}]:")
        print(f"{'arm':>12} {'Dice vs expert':>15} {'Dice vs truth-mask':>19}")
        agg = []
        for arm in arms:
            rs = [r for r in real if r["arm"] == arm]
            if len(rs) == len({r["case"] for r in real}):
                agg.append((arm, float(np.mean([r["dice_expert"] for r in rs])),
                            float(np.mean([r["dice_truthmask"] for r in rs]))))
        for arm, de, dt in sorted(agg, key=lambda t: -t[1]):
            print(f"{LABEL.get(arm, arm):>12} {de:15.3f} {dt:19.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
