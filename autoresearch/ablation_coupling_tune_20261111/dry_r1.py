import subprocess, sys
T = "autoresearch/ablation_coupling_tune_20261111/tune.py"
KH = "--sym=--kspace_high 3.0 --kspace_high_plain 2.2"
WV = "--sym=--set loss.lambda_wav=0.08"
runs = [("r1_khi3_full", "full", KH), ("r1_khi3_t2", "nojoint_c", KH), ("dry_khi3_t1", "nojoint_p", KH),
        ("dry_khi3_iso", "iso", KH), ("dry_khi3_noprior", "noprior", KH),
        ("r1_wav008_full", "full", WV), ("r1_wav008_t2", "nojoint_c", WV), ("dry_wav008_t1", "nojoint_p", WV),
        ("dry_v32_full", "full", "--sym="), ("dry_v32_iso", "iso", "--sym=")]
bad = 0
for name, kind, sym in runs:
    r = subprocess.run([sys.executable, T, "--task", "t1t2tb", "--name", name, "--kind", kind, sym, "--dry"],
                       capture_output=True, text=True)
    print(f"--- {name} {kind} rc={r.returncode}\n{(r.stdout + r.stderr)[-1600:]}", flush=True)
    bad += r.returncode != 0
print("DRY DONE failures", bad)
