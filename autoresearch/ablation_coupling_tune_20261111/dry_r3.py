"""Dry check (CPU, trains nothing) of the R3c / R3d commands of stage three.

Keeper = R2a: T2 wavelet weight 0.08 (symmetric) + common-edge prior 0.001 (joint only).
R3a / R3b (lambda_edge 0.002 / 0.004) were launched before this file was written.
R3c: edge_kappa 10 -> 100 (joint only; the gate of the common-edge prior). R3d: gaussian.rotation_lr 0.0003 -> 0.008
(symmetric: full, independent T1, independent T2).
"""
import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
K = "--set loss.lambda_wav=0.08"
E = "--lambda_edge 0.001"
R = f"{K} --set gaussian.rotation_lr=0.008"
runs = [("r3_ekappa100_full", "full", f"--sym={K}", f"--ov={E} --edge_kappa 100"),
        ("r3_rot008_full", "full", f"--sym={R}", f"--ov={E}"),
        ("r3_rot008_t1", "nojoint_p", f"--sym={R}", "--ov="),
        ("r3_rot008_t2", "nojoint_c", f"--sym={R}", "--ov=")]
bad = 0
for name, kind, sym, ov in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, ov, "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-1800:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
