Diagnostics behind two corrections in ../../README.md. They ran from the session scratchpad on 2026-09-11 and
are kept unchanged, so their output paths still point there; the outputs were copied to ../../logs/diag/.

iter_diag.py      REF echo 0, absolute lambda 1e-3: split of |x100 - x60| into low-pass and high-pass parts
lam_scan.py       (started by run_lam_scan.sh) REF echo 0, absolute lambda 0.01 and 0.03, 60 vs 100 CG steps
                  -> logs/diag/ref_abs_lambda_scan_V07.log   (correction 2)
fista_abs.py      AF 12 echo 0: FISTA with absolute lambda 1e-4 / 1e-3 / 1e-2 at 100 steps and 1e-3 at 200 steps,
                  against CG with lambda_rel 1e-3 -> logs/diag/l1_fista_steps_V07.log   (corrections 3 and 4)
error_split.py    self-check reconstructions against REF: low-pass / high-pass error, high-pass content and mean
                  intensity relative to REF -> logs/diag/selfcheck_error_split_V07.log   (second self-check failure)
l1_scale_n.py     AF8 and AF16 with the L1 lambda scaled by the number of spokes (AF12 keeps 3e-3), echoes 0 and 4
                  -> logs/diag/l1_lambda_per_spoke_V07.log   (second self-check failure)
delay_test.py     echo-0 peak shift of the REF spokes fitted with a gradient-delay model, trajectories shifted by it, REF echo 0
                  rerun: in-sample held-out NRMSE 0.452 -> 0.534, i.e. worse; the peak shifts are phase effects, not a
                  trajectory error -> logs/diag/delay_model_V07.log
outfov.py         REF echo 0 gridded (low-pass) on a 2x FOV grid (224^3, 448 mm): 62 % of the coil-combined energy lies
                  outside the 220 mm reconstruction FOV; per coil 0.19-0.87, rank correlation with the per-coil in-sample
                  residual 0.86 -> logs/diag/out_of_fov_V07.log, figures/V07_fov2x_gridding.png
fov2x_test.py     V07 echo 0 on a 2x FOV grid (224^3, 448 mm, same 2 mm voxels): coil maps from a 64^3 ESPIRiT over
                  448 mm interpolated to 224^3, REF with lambda_rel 1e-3 x its own lambda_max (2,204.6), 100 CG steps
                  (1,108 s). In-sample held-out NRMSE 0.452 -> 0.130, k-centre residual 38 % -> 11 %, per-coil
                  0.17-0.81 -> 0.08-0.26, FOV-edge band / body 1.84 -> 0.93 -> logs/diag/fov2x_test_V07.{log,json},
                  figures/V07_fov2x_ref_e0.png. Basis of the 2026-09-11 evening decision to reconstruct REF and the
                  subsets on 224^3 (spec 5.6/7 correction 5).
