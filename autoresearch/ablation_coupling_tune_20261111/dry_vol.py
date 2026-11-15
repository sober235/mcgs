"""Dry check (CPU, trains nothing) of the three reruns that save volumes for the ablation figure (tangbing T1+T2).

Same commands as the scored runs, seed 42, plus --volumes: the final full model (R5), the pool's best independent
T1 (r3_rot008_t1) and the pool's best independent T2 (r4_scl004_t2). They double as same-seed reproducibility checks.
"""
import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
R = "--set loss.lambda_wav=0.08 --set gaussian.rotation_lr=0.008"
S = f"{R} --set gaussian.scale_lr=0.004"
runs = [("fin_full_vol", "full", f"--sym={S}", "--ov=--lambda_edge 0.001 --edge_kappa 100"),
        ("fin_rot008_t1_vol", "nojoint_p", f"--sym={R}", "--ov="),
        ("fin_scl004_t2_vol", "nojoint_c", f"--sym={S}", "--ov=")]
bad = 0
for name, kind, sym, ov in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, ov, "--volumes", "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-1800:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
