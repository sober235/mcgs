"""Dry check (CPU, trains nothing) of the six R2 commands of stage three; keeper = R1b (T2 wavelet weight 0.08)."""
import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
K = "--set loss.lambda_wav=0.08"
runs = [("r2_edge001_full", "full", f"--sym={K}", "--ov=--lambda_edge 0.001"),
        ("r2_wav012_full", "full", "--sym=--set loss.lambda_wav=0.12", "--ov="),
        ("r2_wav012_t2", "nojoint_c", "--sym=--set loss.lambda_wav=0.12", "--ov="),
        ("r2_geo005_full", "full", f"--sym={K} --lambda_geo 0.05", "--ov="),
        ("r2_geo005_t1", "nojoint_p", f"--sym={K} --lambda_geo 0.05", "--ov="),
        ("r2_geo005_t2", "nojoint_c", f"--sym={K} --lambda_geo 0.05", "--ov=")]
bad = 0
for name, kind, sym, ov in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, ov, "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-1800:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
