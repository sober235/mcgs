import json, sys
import numpy as np
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import bart
from cardiac_fr.gauge import image_metrics, norm_scale
D = "/data2/congcong/data/cardiac_3d_cine_public/derived/megre/V07"; W = D + "/bart"
S = "/tmp/claude-1002/-home-congcongliu--claude/4016652d-b661-45ca-ba05-927ae4a7ee12/scratchpad/l1diag"
T = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE"; R = T + "/recon/V07"
w = json.load(open(R + "/ref_scaling.json"))["0"]
ref = np.abs(np.load(R + "/ref_e0.npy")); sc = norm_scale(ref)
cg = np.abs(np.load(R + "/cg_af12_e0.npy"))
box = json.load(open(T + "/gauge.json"))["V07"]["heart_box"]
m = image_metrics(cg / sc, ref / sc, heart_box=box)
print(f"CG AF12 (lambda_rel 1e-3): body_psnr {m['body_psnr']:.2f} ssim {m['body_ssim']:.3f} hfen {m['body_hfen']:.3f}", flush=True)
for lam, it in ((1e-4, 100), (1e-3, 100), (1e-2, 100), (1e-3, 200)):
    out = f"{S}/fista_lam{lam:g}_it{it}"
    info = bart.pics(f"{W}/ksp_af12_e0", f"{D}/sens", out, f"{W}/traj_af12", reg="l1", lam=lam, iters=it, w=w)
    x = bart.read_cfl(out); ax = np.abs(x)
    m = image_metrics(ax / sc, ref / sc, heart_box=box)
    print(f"FISTA absolute lambda {lam:g}, {it} steps: body_psnr {m['body_psnr']:.2f} ssim {m['body_ssim']:.3f} "
          f"hfen {m['body_hfen']:.3f}  |L1|/|CG| {np.linalg.norm(ax) / np.linalg.norm(cg):.3f}  ({info['seconds']:.0f} s)", flush=True)
    np.save(f"{S}/fista_lam{lam:g}_it{it}.npy", x.astype(np.complex64))
a = np.load(f"{S}/fista_lam0.001_it100.npy"); b = np.load(f"{S}/fista_lam0.001_it200.npy")
print(f"lambda 1e-3: 100 vs 200 steps rel diff {np.linalg.norm(b - a) / np.linalg.norm(b):.4f}", flush=True)
print("FISTA_ABS_DONE", flush=True)
