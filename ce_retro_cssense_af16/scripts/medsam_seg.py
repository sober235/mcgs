#!/usr/bin/env python3
"""medsam_seg.py -- one public segmentation model, one fixed prompt, every arm of this topic.

WHAT THIS MEASURES. The eleven arms are reconstructions of the same AF 16.00 retrospective undersampling; the truth
is this topic's truth volume (`sim/gt.npz`), i.e. the CS-SENSE reconstruction of the full AF 5.33 acquisition. The
model is run identically on the truth and on every arm, and each arm is scored against the model's own output on
the truth. So the number answers "whose reconstruction does the downstream segmenter read the same way it reads the
truth", not "whose reconstruction is closer to a radiologist's outline" -- there is no expert outline here.

READ THIS BEFORE QUOTING THE RANKING. The truth is itself a TV-regularised CS-SENSE reconstruction, and 62.7% of the
AF16 samples were never measured in the original acquisition, so the truth's lesion boundary there is TV
extrapolation rather than data. A segmenter keys on exactly that boundary texture, which hands the CS-SENSE family a
structural advantage that PSNR largely escapes. Measured, both scans: CS-SENSE and Joint SENSE take the top two
places and MC-GS lands 4th (pre) / 5th (post). Report the ranking with this caveat attached, or move the claim to a
line whose truth is an independent image (the UCSF-PDGM topics).

MODEL. MedSAM (Ma et al., Nat Commun 2024), the published `wanglab/medsam-vit-base` checkpoint -- SAM ViT-B
fine-tuned on ~1.5M medical image-mask pairs, prompted by a box. Weights live at $MEDSAM_DIR (default
/home/congcongliu/models/medsam-vit-base); the HF CDN is unreachable from this machine, so they were pulled from
https://hf-mirror.com and converted to safetensors (transformers 5 refuses .bin under torch < 2.6). It needs
`transformers`, which is not in the topic's GSTMI environment, so THIS script runs under $MEDSAM_PY
(default /home/congcongliu/.venvs/medsam_probe/bin/python, a venv made with
`<nvgen python> -m venv --system-site-packages` plus `pip install transformers`). Every other script of the topic
keeps running under GSTMI.

PROMPT AND PREPROCESSING, identical for every arm -- this is what makes the comparison fair:
  * each volume is divided by the median of its own white matter (SynthSeg labels 2/41 of the truth), so a global
    gain difference between arms cannot move the result;
  * a fixed window [0, WINDOW_TOP] white-matter units -> 8-bit, NOT per-slice min-max, which would give every arm a
    different contrast stretch;
  * one box prompt, the truth lesion core's 3-D bounding box grown by MARGIN, applied to every axial slice of that
    box's z-range and to both scans. The box comes from the truth, exactly as the threshold operator's hand-set box
    does; it is the same box for all arms.
The 2-D masks are stacked into a volume and reduced to their largest 26-connected component, the one post-process
step, applied identically to the truth and to every arm: the raw stack carries a handful of stray specks (14
components on the truth, the largest holding 99.7% of it) that no measure should be paid for. MedSAM outlines the
whole T1-dark lesion, which is a larger object than the stored `core` mask (the darkest 26-connected blob under
threshold 0.32) -- Dice against that stored mask is reported for context only, it is not the reference.

A property of box-prompted SAM to keep in mind when reading the volumes: the model returns a mask for every slice it
is asked about, with no way to answer "nothing here", so the mask fills the prompt box's whole z-range and overruns
the lesion's threshold-defined extent (on the truth, 255-339 voxels per slice at z = 186 and 221, where the stored
core has none). Every arm is asked exactly the same question, so the comparison between arms is unaffected; the
absolute volumes are not the lesion's volume.

  env PYTHONNOUSERSITE=1 PYTHONPATH= $MEDSAM_PY scripts/medsam_seg.py --scan pre
  env PYTHONNOUSERSITE=1 PYTHONPATH= $MEDSAM_PY scripts/medsam_seg.py --scan post

Writes seg/medsam_<arm>_<scan>.npy (bool volumes, truth included) and logs/medsam_seg_<scan>.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16"))
PROS = Path(os.environ.get("CE_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast"))
LABELS = PROS / "lesion_candidates/cssense_labels.npy"
ROI_NPZ = PROS / "lesion_candidates/enhance_rois_cssense.npz"
MEDSAM_DIR = os.environ.get("MEDSAM_DIR", "/home/congcongliu/models/medsam-vit-base")

WM_LABELS = (2, 41)
MARGIN = 4
WINDOW_TOP = 2.0


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else float("nan")


def hd95(a, b):
    """symmetric 95th-percentile surface distance in voxels (1 mm isotropic here). NaN if either mask is empty."""
    if not a.any() or not b.any():
        return float("nan")

    def surface(m):
        return m & ~ndi.binary_erosion(m, structure=np.ones((3,) * m.ndim, bool))

    sa, sb = surface(a), surface(b)
    da = ndi.distance_transform_edt(~sb)[sa]
    db = ndi.distance_transform_edt(~sa)[sb]
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def prompt_box(ref, margin=MARGIN):
    """the reference mask's bounding box grown by `margin`, clipped to the grid; inclusive (z0, z1, y0, y1, x0, x1)"""
    idx = np.argwhere(ref)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin, np.array(ref.shape) - 1)
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])


def to_uint8_window(img, top=WINDOW_TOP):
    """fixed window [0, top] -> 8-bit. Fixed, not per-slice min-max, so every arm gets the same contrast stretch."""
    return np.clip(img / top, 0, 1).__mul__(255).round().astype(np.uint8)


def wm_units(vol, wm):
    return np.abs(vol) / float(np.median(np.abs(vol)[wm]))


def largest_component(mask):
    """the largest 26-connected component; an all-false mask is returned unchanged"""
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == int(np.argmax(sizes)) + 1


class MedSAM:
    """the model plus the one prompt; `segment` is the whole of what an arm is put through"""

    def __init__(self, box, device=None):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.box = box
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = SamProcessor.from_pretrained(MEDSAM_DIR)
        self.model = SamModel.from_pretrained(MEDSAM_DIR).to(self.device).eval()

    def segment(self, vn):
        z0, z1, y0, y1, x0, x1 = self.box
        out = np.zeros(vn.shape, bool)
        for z in range(z0, z1 + 1):
            rgb = np.repeat(to_uint8_window(vn[z])[:, :, None], 3, axis=2)
            inp = self.proc(rgb, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                pred = self.model(**inp, multimask_output=False)
            m = self.proc.image_processor.post_process_masks(
                pred.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            out[z] = m[0, 0].numpy()
        return largest_component(out)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from arms import arm_files

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", choices=["pre", "post"], required=True)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    a = ap.parse_args()

    core = np.load(ROI_NPZ, allow_pickle=True)["core"]
    wm = np.isin(np.load(LABELS), WM_LABELS)
    box = prompt_box(core)
    print(f"[{a.scan}] prompt box z[{box[0]},{box[1]}] y[{box[2]},{box[3]}] x[{box[4]},{box[5]}] "
          f"(truth core {int(core.sum())} vox grown by {MARGIN})", flush=True)

    seg_dir = R / "seg"
    seg_dir.mkdir(exist_ok=True)
    net = MedSAM(box, a.device)
    print(f"MedSAM from {MEDSAM_DIR} on {net.device}", flush=True)

    truth = net.segment(wm_units(np.load(R / "sim/gt.npz")[a.scan], wm))
    np.save(seg_dir / f"medsam_truth_{a.scan}.npy", truth)
    print(f"  {'truth':>12}  {int(truth.sum()):6d} vox   (Dice vs stored core {dice(truth, core):.3f})", flush=True)

    rows = [dict(key="truth", label="truth", vox=int(truth.sum()), dice=1.0, hd95=0.0, vol_ratio=1.0,
                 dice_vs_core=dice(truth, core))]
    for key, lbl, path in arm_files(a.scan):
        m = net.segment(wm_units(np.load(path), wm))
        np.save(seg_dir / f"medsam_{key}_{a.scan}.npy", m)
        rows.append(dict(key=key, label=lbl, vox=int(m.sum()), dice=dice(m, truth), hd95=hd95(m, truth),
                         vol_ratio=float(m.sum() / truth.sum()), dice_vs_core=dice(m, core)))
        r = rows[-1]
        print(f"  {lbl:>12}  {r['vox']:6d} vox   Dice {r['dice']:.3f}   HD95 {r['hd95']:.2f} mm   "
              f"vol {r['vol_ratio'] - 1:+.1%}", flush=True)

    out = R / f"logs/medsam_seg_{a.scan}.json"
    json.dump(dict(scan=a.scan, model=MEDSAM_DIR, box=list(box), window_top=WINDOW_TOP, margin=MARGIN,
                   reference="medsam on this topic's truth volume (CS-SENSE reconstruction of the AF 5.33 acquisition)",
                   rows=rows), open(out, "w"), indent=1)
    arms_only = sorted([r for r in rows if r["key"] != "truth"], key=lambda r: -r["dice"])
    print(f"\n[{a.scan}] ranking by Dice vs the model's truth mask:")
    for i, r in enumerate(arms_only, 1):
        print(f"  {i:2d}. {r['label']:>12}  {r['dice']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
