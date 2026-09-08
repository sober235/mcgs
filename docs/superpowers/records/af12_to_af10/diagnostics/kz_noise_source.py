import sys, numpy as np, yaml, torch
ROOT = "/data0/congcong/code/GS/Paired_ARC_GS_general_framework"; sys.path.insert(0, ROOT + "/paired_gs")
from trainers.joint_trainer import _load_dataset
from data.private_paths import resolve_subject_paths
cfg = yaml.safe_load(open(ROOT + "/configs/winner_v2_edgeTV.yaml"))
for sub, con in (("plain", "T1"), ("contrast", "FLAIR")):
    d = dict(cfg["data"][sub]); [d.__setitem__(k, resolve_subject_paths(d[k])) for k in ("data_path", "csm_path", "mask_path")]
    ds = _load_dataset(d); y = ds.kspace_undersampled_complex; m = ds.mask; ro = int(ds.readout_axis)
    a3 = torch.movedim(y.abs().amax(0), ro, -1); m3 = torch.movedim(m > 0.5, ro, -1); P0, P1, R = a3.shape
    smax = float(a3[m3].max()); v = a3[m3]
    print(f"\n===== {con} (sampled only): max {smax:.3e}; fraction of sampled voxels by |y|max/max decade:", flush=True)
    for lo, hi in ((0, 1e-12), (1e-12, 1e-10), (1e-10, 1e-8), (1e-8, 1e-6), (1e-6, 1e-4), (1e-4, 1e-2), (1e-2, 1)):
        print(f"    [{lo:.0e},{hi:.0e}) : {float(((v >= lo*smax) & (v < hi*smax)).float().mean()):.5f}", flush=True)
    # readout-end profile of mean energy over sampled nonzero lines and coils
    e = torch.movedim((y.abs() ** 2).sum(0), ro, -1)              # (P0,P1,R)
    line_nz = (a3.amax(-1) >= 1e-10 * smax) & m3.any(-1)
    prof = e[line_nz].mean(0).numpy() / y.shape[0]
    print("  readout mean energy per (coil,sample), nonzero sampled lines, by 16-index band:", " ".join(f"{prof[i:i+16].mean():.2e}" for i in range(0, R, 16)), flush=True)
    # ellipse-radius profile (FLAIR ellipse a=123,b=88 at DC; T1: rectangle -> use normalized rect distance) of per-line mean energy
    e_pe = e.sum(-1).numpy(); dc = np.unravel_index(int(np.argmax(e_pe)), e_pe.shape)
    ii, jj = np.meshgrid(np.arange(P0) - dc[0], np.arange(P1) - dc[1], indexing="ij")
    r = np.sqrt((ii / 123.0) ** 2 + (jj / 88.0) ** 2) if con == "FLAIR" else np.maximum(np.abs(ii) / 128.0, np.abs(jj) / 88.0)
    per_line = (e.sum(-1) / (y.shape[0] * R)).numpy()
    print(f"  per-(coil,sample) mean energy of sampled NONZERO lines by shape-radius band ({'ellipse' if con=='FLAIR' else 'rect-max-norm'}):", flush=True)
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .7), (.7, .8), (.8, .85), (.85, .9), (.9, .95), (.95, 1.0), (1.0, 1.05)):
        s = line_nz.numpy() & (r >= lo) & (r < hi)
        if s.sum(): print(f"    r[{lo:.2f},{hi:.2f}) n={int(s.sum()):5d} mean={per_line[s].mean():.3e} median={np.median(per_line[s]):.3e}", flush=True)
