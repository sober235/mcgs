# CE Retro CS-SENSE AF16

Topic: `/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/`
Design spec: `docs/superpowers/specs/2026-09-17-ce-retro-cssense-af16-design.md`

## Synthetic k-space (Task 5)

`scripts/sim_cssense_af16.py` synthesises noisy pre/post k-space from the CS-SENSE truth
(`CE_pre_post_contrast/tv_sense/cssense_o200_lam0.002_full_{pre,post}_complex.npy`), the real 42-coil
maps, and the AF16 Poisson-disc mask. Per scan, sigma is fit from the residual on the prospective
mask (alpha = least-squares complex scale); the run stops unless the residual reproduces the archived
CS-SENSE kres (0.18487 pre / 0.14969 post) to < 1e-3 -- the FFT/CSM convention gate.

Result (2026-09-17): kres 0.184868 pre / 0.149690 post (both within 2e-6 of reference); sigma 1.36686
pre / 1.44523 post; WM SNR 34.4 pre / 33.1 post.

## check_sim.py operator gates (Task 5)

`scripts/check_sim.py` checks, on one GPU: (a) the noiseless adjoint identity `x_hat == x_GT` inside
the head, (b) CG-SENSE (Tikhonov 1e-6, 60 iters) noiseless head PSNR > 60 dB, (c) the noise actually
written matches meta sigma to < 2%.

**Adaptation (brief Step 7, pre-authorised):** the raw adjoint identity `x_hat = sum_c conj(S_c)
ifft3c(F(S_c x)) / sum_c|S_c|^2` is singular wherever coil coverage `sum_c|S_c|^2` is near zero. The
first run failed gate (a) with relative error 1.90e-3 (pre) / 1.13e-3 (post) -- the failure mode the
brief predicted. Per the brief, gate (a) is restricted to head voxels with `sum_c|S_c|^2 > 0.25`
(recorded as `adjoint_voxel_condition` / `adjoint_head_voxels_total` / `adjoint_head_voxels_used` in
`logs/check_sim.json`); this excludes 223/4,495,671 head voxels for pre and 74/4,520,978 for post
(< 0.006% of the head either way). With the restriction: adjoint rel err 3.81e-7 pre / 3.84e-7 post
(both < 1e-4); CG head PSNR 69.19 dB pre / 75.91 dB post (both > 60); sigma written vs meta within
4.3e-5 / 1.6e-5 relative. `CHECK_SIM OK`, both scans, all three gates.

## Bugfix vs the task-5 brief (documented, not a gate-threshold change)

`sigma_from_residual`'s `n` (used only inside `sigma = sqrt(res / (2n))`) as given in the brief was
`m0.sum() * S.shape[0]` (mask points times coils), omitting the readout axis: `[..., m0]`
fancy-indexing with a 2D mask on a 4D `(coil, readout, ky, kz)` tensor keeps the readout axis, so `res`
actually sums over `m0.sum() * S.shape[0] * S.shape[1]` terms. Left as given, `sigma` would be
inflated by `sqrt(S.shape[1])` (16x for the real 256-readout grid), while `kres` (which does not
depend on `n`) and the check_sim sigma gate (a self-consistency check between the written noise and
`meta.json`, both derived from the same `sigma`) would both still pass -- so the bug would have
silently written ~16x-too-noisy synthetic k-space into the 14 GB of study data, undetected by either
named gate. Confirmed by dimensional analysis, an empirical `torch` indexing check, and the brief's own
unit test (`test_sigma_from_residual_recovers_known_noise`), which failed against the unfixed code
(measured sigma/sigma0 = 3.96 = sqrt(16), matching the missing factor exactly) and passes once fixed.
The one-line fix `n = m0.sum() * S.shape[0] * S.shape[1]` was applied identically in
`sim_cssense_af16.py` and in the test's own `expect_kres` formula (`test_sim.py`), which had the same
omission.
