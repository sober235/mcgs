Reconstruction time per arm, one T1 + FLAIR pair, jiangyizhou AF 12 (26 coils, 256 x 256 x 176), one NVIDIA A800 80GB per job, solver only (data loading 5.04 s / 4.95 s and scoring excluded). Fast arms re-timed on an exclusive card; long arms from the runs of record (one card per job). No foreign process on the card during the fast-arm timing.

| method | contrast coupling | T1 | FLAIR | one T1 + FLAIR pair | notes |
|---|---|---|---|---|---|
| ZF | single | 0.04 s | 0.04 s | **0.09 s** | adjoint SENSE of the zero-filled k-space; median of 3 |
| CG-SENSE | single | 5.24 s | 5.26 s | **11 s** | 60 CG iterations, Tikhonov 0.01; median of 3 |
| CS-SENSE | single | 145 s | 145 s | **289 s** | ADMM isotropic 3-D TV, 200 outer x 8 CG, lambda 7e-4 |
| Joint SENSE | joint | -- | -- | **289 s** | vectorial TV over both contrasts, 200 outer x 8 CG, lambda 1e-3; one run for the pair |
| L1-SPIRiT | single | 93 s | 93 s | **186 s** | FISTA, 150 iterations, kernel 5, lambda 2e-3 / 1e-3, per-readout calibration included; re-timed on the exclusive card |
| ZS-SSDU | single | 21.9 min | 36.1 min | **57.9 min** | self-supervised training to early stop (epochs 40 / 66) + inference; run of record minus the measured load; spot check T1 1331 s |
| 3D-DIP | single | 50.3 min | 50.2 min | **100.5 min** | 5000 iterations of the convolutional decoder, held-out selection online; run of record |
| 3D-INR | single | 57.2 min | 57.2 min | **114.4 min** | 1500 iterations of the hash-grid INR, held-out selection online; run of record |
| MC-INR | joint | -- | -- | **11.5 min** | Niessen 2025 per-slice 2-D multi-contrast INR, 256 readout slices x 200 epochs; run of record (A800, torch 1.13) |
| GS-single | single | 62.8 min | 59.2 min | **122.0 min** | P14 recipe, 4000 iterations, 3.6M Gaussians; training loop of the run of record (FLAIR single ran on an A100) |
| MC-GS | joint | -- | -- | **61.2 min** | P14 recipe, 4000 iterations, 3.6M Gaussians, both contrasts in one run; training loop of the run of record; spot check on the exclusive card 3712 s |
