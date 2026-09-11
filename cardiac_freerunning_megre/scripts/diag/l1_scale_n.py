# Diagnostic (scratchpad), 2026-09-11: does scaling the L1 lambda with the number of spokes make the L1 arm of the
# self-check monotone? AF12 keeps the selected lambda; AF8 and AF16 get lambda x N_af / N_af12.
import json, sys
import numpy as np
sys.path.insert(0, "/data0/congcong/code/GS/mcgs_cardiac_wt")
from cardiac_fr import bart
from cardiac_fr.gauge import image_metrics, kspace_nrmse, norm_scale
D = "/data2/congcong/data/cardiac_3d_cine_public/derived/megre/V07"; W = D + "/bart"
S = "/tmp/claude-1002/-home-congcongliu--claude/4016652d-b661-45ca-ba05-927ae4a7ee12/scratchpad/l1diag"
T = "/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE"; R = T + "/recon/V07"
g = json.load(open(T + "/gauge.json")); box = g["V07"]["heart_box"]
N = {a: g["V07"]["subsets"][a]["n"] for a in ("af8", "af12", "af16")}
LAM12 = g["l1_lambda"]
scal = json.load(open(R + "/ref_scaling.json"))


def held(img, e):
    bart.forward(f"{W}/traj_heldout", img, f"{D}/sens", f"{S}/pred", S)
    return kspace_nrmse(bart.read_cfl(f"{S}/pred"), bart.read_cfl(f"{W}/ksp_heldout_e{e}"))


for e in (0, 4):
    ref = np.abs(np.load(f"{R}/ref_e{e}.npy")); sc = norm_scale(ref)
    bart.write_cfl(f"{S}/official_l1_af12", np.load(f"{R}/l1_af12_e{e}.npy"))
    print(f"echo {e}: official l1_af12 through this held-out path {held(f'{S}/official_l1_af12', e):.5f}", flush=True)
    for a in ("af8", "af16"):
        lam = LAM12 * N[a] / N["af12"]
        out = f"{S}/l1N_{a}_e{e}"
        info = bart.pics(f"{W}/ksp_{a}_e{e}", f"{D}/sens", out, f"{W}/traj_{a}", reg="l1", lam=lam, iters=100, w=scal[str(e)])
        x = np.abs(bart.read_cfl(out))
        m = image_metrics(x / sc, ref / sc, heart_box=box)
        print(f"echo {e} {a}: lambda {lam:.5g} (= {LAM12:g} x {N[a]}/{N['af12']}) body_psnr {m['body_psnr']:.3f} "
              f"ssim {m['body_ssim']:.4f} hfen {m['body_hfen']:.4f} heldout {held(out, e):.5f} "
              f"mean/ref {x[x > 0].mean() / ref[ref > 0].mean():.4f} ({info['seconds']:.0f} s)", flush=True)
print("L1_SCALE_N_DONE", flush=True)
