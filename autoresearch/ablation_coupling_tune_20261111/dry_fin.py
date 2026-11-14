"""Dry check (CPU, trains nothing) of the five finalisation commands of stage three (program.md 11.6).

Final configuration (R5) = V32 + T2 wavelet weight 0.08 + gaussian.rotation_lr 0.008 + gaussian.scale_lr 0.004
(symmetric) + common-edge prior 0.001 with edge_kappa 100 (joint only).
1. w/o anisotropy and w/o intra-contrast priors with the final settings (seed 42, volumes saved).
2. Seed 7: the full model, and each contrast's best independent setting of the pool
   (T1: rotation_lr 0.008; T2: wavelet 0.08 + rotation_lr 0.008 + scale_lr 0.004).
"""
import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
R = "--set loss.lambda_wav=0.08 --set gaussian.rotation_lr=0.008"
S = f"{R} --set gaussian.scale_lr=0.004"
J = "--lambda_edge 0.001 --edge_kappa 100"
runs = [("fin_iso", "iso", f"--sym={S}", f"--ov={J}", ["--volumes"]),
        ("fin_noprior", "noprior", f"--sym={S}", f"--ov={J}", ["--volumes"]),
        ("s7_full", "full", f"--sym={S}", f"--ov={J}", ["--seed", "7"]),
        ("s7_rot008_t1", "nojoint_p", f"--sym={R}", "--ov=", ["--seed", "7"]),
        ("s7_scl004_t2", "nojoint_c", f"--sym={S}", "--ov=", ["--seed", "7"])]
bad = 0
for name, kind, sym, ov, extra in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, ov, *extra, "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-2600:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
