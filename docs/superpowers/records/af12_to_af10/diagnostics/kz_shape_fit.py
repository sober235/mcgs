"""Fit axis-aligned ellipse / rectangle to the whole-line-tiny PE map (oracle) and to the acquired samples only."""
import sys, numpy as np, yaml, torch
ROOT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework"; sys.path.insert(0, ROOT + "/paired_gs")
from trainers.joint_trainer import _load_dataset
from data.private_paths import resolve_subject_paths
cfg = yaml.safe_load(open(ROOT + "/configs/winner_v2_edgeTV.yaml")); REL = 1e-10
def fit(zero2, weight, c, label):
    P0, P1 = zero2.shape; ii, jj = np.meshgrid(np.arange(P0) - c[0], np.arange(P1) - c[1], indexing="ij")
    out = {}
    for shape in ("ellipse", "rect"):
        best = None
        for a in np.arange(40, 170, 0.5):
            for b in np.arange(40, 170, 0.5):
                pred = ((ii / a) ** 2 + (jj / b) ** 2 >= 1.0) if shape == "ellipse" else ((np.abs(ii) >= a) | (np.abs(jj) >= b))
                mis = int(((pred != zero2) & weight).sum())
                if best is None or mis < best[0]: best = (mis, a, b)
        out[shape] = best
        print(f"    {label:24s} {shape:8s} centre={tuple(int(x) for x in c)} (a,b)=({best[1]:.1f},{best[2]:.1f}) px = rho ({best[1]/(P0/2):.3f},{best[2]/(P1/2):.3f})  misclassified {best[0]} / {int(weight.sum())}", flush=True)
    return out
for sub, con in (("plain", "T1"), ("contrast", "FLAIR")):
    d = dict(cfg["data"][sub]); [d.__setitem__(k, resolve_subject_paths(d[k])) for k in ("data_path", "csm_path", "mask_path")]
    ds = _load_dataset(d); yf = ds.kspace_multicoil; m = ds.mask; ro = int(ds.readout_axis)
    a3 = torch.movedim(yf.abs().amax(0), ro, -1); tiny = (a3 < REL * float(a3.max())).numpy(); full = tiny.mean(-1) >= 0.999
    e_pe = torch.movedim((yf.abs() ** 2).sum(0), ro, -1).sum(-1).numpy(); dc = np.unravel_index(int(np.argmax(e_pe)), e_pe.shape)
    P0, P1 = full.shape; ac = (P0 // 2, P1 // 2); m2 = torch.movedim(m > 0.5, ro, -1).any(-1).numpy()
    print(f"\n===== {con}: whole-line-zero fraction {full.mean():.4f}; DC {dc}; array centre {ac} =====", flush=True)
    # raw extents of the nonzero region along the axes through DC and through the array centre
    for cname, c in (("DC", dc), ("array-centre", ac)):
        row = ~full[:, c[1]]; col = ~full[c[0], :]
        i_nz = np.nonzero(row)[0]; j_nz = np.nonzero(col)[0]
        print(f"  nonzero extent through {cname}: axis0 [{i_nz.min()}..{i_nz.max()}] (half-widths -{c[0]-i_nz.min()}/+{i_nz.max()-c[0]}), axis1 [{j_nz.min()}..{j_nz.max()}] (half-widths -{c[1]-j_nz.min()}/+{j_nz.max()-c[1]})", flush=True)
    print("  oracle map fits:", flush=True); fit(full, np.ones_like(full), dc, "oracle/DC"); fit(full, np.ones_like(full), ac, "oracle/array-centre")
    print("  sampled-only fits:", flush=True); fit(full & m2, m2, dc, "sampled/DC"); fit(full & m2, m2, ac, "sampled/array-centre")
