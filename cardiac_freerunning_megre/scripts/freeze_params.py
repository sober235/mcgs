"""Spec 5.8: write params_frozen.json from the V07 results (checks, REF lambda, self-check L1 lambda).
run_rest.sh reads ref_iters and ref_lam_rel from it."""
import json

from common import LOGS, TOPIC, recon_dir

it = json.loads((LOGS / "checks_V07.json").read_text())["ref_iters"]
rel = json.loads((recon_dir("V07") / "ref_lambda.json").read_text())["lam_rel"]
l1 = json.loads((TOPIC / "gauge.json").read_text())["l1_lambda"]
p = {"frozen_on": "V07", "frozen_at": "2026-09-12 after the 224^3 self-check (spec corrections 5-7)",
     "recon_grid": 224, "fov_mm": 448, "eval_crop": [56, 168], "look_grid": 112,
     "ref_iters": it, "ref_lam_rel": rel, "l1_lambda": l1,
     "readout_samples_kept": [3, 220], "t_steady_ms": 10000,
     "gating": "end-expiration third of the steady non-SI spokes; polarity = histogram mode (a) and the SI shift "
               "between the two thirds in a liver-dome box and a heart box (b, polarity_boxes.json, valid at >= 1 voxel); "
               "(a) alone if (b) is undetermined and the dwell asymmetry is >= 2 (subject flagged gating=weak)",
     "coil_maps": "112^3 (R0, polarity): echo 0, r0 spokes: bart nufft -i -x 112:112:112 -> bart fft -u 7 -> "
                  "bart ecalib -t 0.001 -c 0.0 -r 24 -m 1; 224^3 (REF, subsets): central |k| < 56 samples on 112^3 over "
                  "448 mm, ecalib -r 48, Fourier-interpolated to 224^3, unit sum of squares",
     "r0": f"bart pics -t <traj> -l2 -r 1e-3 -i {it} -S (BART's automatic scaling)",
     "ref": f"bart pics -t <traj> -l2 -r <ref_lam_rel x lambda_max> -i {it} -S -d5; lambda_max = largest eigenvalue "
            "of this subject's REF operator (bart pics -e -d5), recon/<Vxx>/ref_lmax.json",
     "ref_scaling": "BART's own (-d5 'Inverse scaling of the data'), reused as -w by every subset of the same echo",
     "cg_subsets": f"bart pics -t <traj> -l2 -r <ref_lam_rel x lambda_max> -i {it} -S -w <ref scaling>",
     "l1_subsets": f"bart pics -t <traj> -l1 -r {l1} -i {it} -e -S -w <ref scaling> "
                   "(absolute lambda in BART's data scaling; early-stopped FISTA)",
     "heldout": "5 % of REF, seed 0", "subsets": "whole interleaves, target AF 8/12/16, AF = 19704 / spokes",
     "threads": 32, "parallel_jobs": 1, "nice": 10}
(TOPIC / "params_frozen.json").write_text(json.dumps(p, indent=1))
print(json.dumps(p, indent=1))
