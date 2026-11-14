"""Dry check (CPU, trains nothing) of the R5 command of stage three.

Keeper after R4 = T2 wavelet weight 0.08 + gaussian.rotation_lr 0.008 + gaussian.scale_lr 0.004 (all symmetric)
+ common-edge prior 0.001 (joint only). R5: + edge_kappa 100 (joint only; kept in R3c, reproduced in R4a on the
rotation_lr keeper) = the combined configuration. Independents are not rerun (joint-only change).
"""
import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
S = "--set loss.lambda_wav=0.08 --set gaussian.rotation_lr=0.008 --set gaussian.scale_lr=0.004"
runs = [("r5_scl004_ekappa100_full", "full", f"--sym={S}", "--ov=--lambda_edge 0.001 --edge_kappa 100")]
bad = 0
for name, kind, sym, ov in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, ov, "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-1800:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
