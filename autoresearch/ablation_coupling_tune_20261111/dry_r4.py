"""Dry check (CPU, trains nothing) of the R4 commands of stage three.

Keeper after R3 = T2 wavelet weight 0.08 + gaussian.rotation_lr 0.008 (both symmetric) + common-edge prior 0.001
(joint only). R4a: + edge_kappa 100 (joint only) = the combined verification of the two R3 keepers.
R4b: + gaussian.scale_lr 0.002 -> 0.004 (symmetric: full, independent T1, independent T2).
"""
import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
K = "--set loss.lambda_wav=0.08 --set gaussian.rotation_lr=0.008"
E = "--lambda_edge 0.001"
S = f"{K} --set gaussian.scale_lr=0.004"
runs = [("r4_rot008_ekappa100_full", "full", f"--sym={K}", f"--ov={E} --edge_kappa 100"),
        ("r4_scl004_full", "full", f"--sym={S}", f"--ov={E}"),
        ("r4_scl004_t1", "nojoint_p", f"--sym={S}", "--ov="),
        ("r4_scl004_t2", "nojoint_c", f"--sym={S}", "--ov=")]
bad = 0
for name, kind, sym, ov in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, ov, "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-1800:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
