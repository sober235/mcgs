# CE AF16: pre/post figures of every passing lesion plane (2026-11-07)

Goal (user, 2026-11-07): MC-GS must beat every opponent on any lesion-containing slice. The three fixed
views are no longer the target. In the frozen evaluator (`evaluate.py`, lesion_planes block) this is a
plane where MC-GS strictly wins all six of pre/post × PSNR/SSIM/NMSE against all ten opponents. There
are 59 frozen planes containing the lesion core (≥150 px): 17 transverse, 24 coronal, 18 sagittal.

Pure MC-GS C77 (`runs/C77_bm4dpos_lp2c4_once_loc0_s0165_a1`, milestone 625, α = 1) meets it:

| seed | passing planes (all coronal) | best plane, S6 |
|---|---|---|
| 42 | 127, 129, 130, 131, 132, 133 | 130, +0.0256 |
| 7 | 121, 127, 129, 130, 131, 132, 133 | 130, +0.0285 |

Coronal 121 on seed 7 passes by a hair: its pre-SSIM margin is 1.2e-5.

The independent-start repeat R1 had S6 > 0 in 17/17 members. Margins are small (about 0.1–0.2 dB PSNR
over ZS-SSDU). No transverse or sagittal plane passes, and the whole-head score is still negative.

## Figures: `fig_lesion_plane_prepost.py`

House style of the T1+FLAIR AF12 / AF16 figures of record
(`Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires`, `fig_three_planes_compact.py`) and of their
CE adaptation (`CE_retro_cssense_af16_vdpois_ar/scripts/fig_three_planes_compact.py`).

- **Rows** (one plane per figure): pre image, pre error, post image, post error.
- **Columns:** GT, ZF, CG-SENSE, CS-SENSE, Joint SENSE, L1-SPIRiT, ZS-SSDU, 3D-DIP, 3D-INR, MC-INR,
  GS-single, MC-GS.
- **Error maps:** |recon − GT| / max GT, jet [0, 0.1], head only.
- **Mask cell:** the cell left of each error row shows the k-space mask in that plane.
- **Zoom:** a 48 × 48 box centred on the lesion core is magnified into a corner.
- **Grey window:** P99.9 / 1.5, as in the CE figures.
- **Output:** PNG at 200 dpi plus a 2000 px JPEG preview.

Each panel prints the evaluator's own lesion-plane numbers from `lesion_planes.json`. The script checks
that the drawn volumes are the scored ones in two ways:

- It verifies every volume's SHA-256 against `input_hashes.json`.
- It recomputes all 66 numbers per plane with `evaluate.plane_metrics` arithmetic and stops if any
  differs (tolerance: PSNR 1e-6 dB, SSIM 1e-9, NMSE 1e-10 relative).

All 13 figures passed both checks.

Run with CPU only, read-only (`gpu.py run --cpu`, jobs j-20261107T134609-84a76464 and
j-20261107T134629-b06034a3):

```sh
W=/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_losstune_20261101/runs/C77_bm4dpos_lp2c4_once_loc0_s0165_a1/scores
O=/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_lesionfigs_20261107
python fig_lesion_plane_prepost.py --score-dir $W/s42_m0625 --label "C77 seed 42 (m625)" --out $O/C77_s42
python fig_lesion_plane_prepost.py --score-dir $W/s7_m0625  --label "C77 seed 7 (m625)"  --out $O/C77_s7
```

Server output is `$O/C77_s{42,7}/`: PNG, preview JPEG and `record_*.json` (hashes, margins, zoom box).
Local copies are in `../results/figs_lesion_prepost/`. Add `--pdf` for a vector PDF (images rasterised
at `--pdf_dpi`, TrueType fonts).

## Three-plane figure, pre and post (`--three TRA COR SAG`, user 2026-11-07)

Rows top to bottom: pre transverse, post transverse, pre coronal, post coronal, pre sagittal, post
sagittal. Each image row is followed by its error row, in the same house style as above. Sagittal rows
use the house sagittal window: the in-head median / P99.9 ratio of the other planes of the same scan.

The planes are transverse 196, coronal 130 and sagittal 43, the best lesion plane of each axis for C77
(highest S6, the same on both seeds). Only coronal 130 is a passing plane. No transverse or sagittal
lesion plane passes, so in those rows the bold numbers go mostly to ZS-SSDU.

| plane | seed 42 S6, wins | seed 7 S6, wins | MC-GS vs best opponent, PSNR pre / post |
|---|---|---|---|
| transverse 196 | −0.1310, 0/6 | −0.1328, 0/6 | 36.79 vs 37.33 / 38.72 vs 39.28 (ZS-SSDU) |
| coronal 130 | +0.0256, 6/6 | +0.0285, 6/6 | 37.05 vs 36.93 / 38.48 vs 38.31 (ZS-SSDU) |
| sagittal 43 | −0.0221, 1/6 | −0.0183, 1/6 | 35.82 vs 35.89 / 38.22 vs 38.31 (ZS-SSDU) |

PSNR figures are seed 42. Jobs j-20261107T135917-437a9c84 and j-20261107T135938-27adb0fc. Output is in
`$O/C77_s{42,7}_three/` (PNG at 200 dpi, about 40 MB; preview JPEG; record JSON), with local copies in
`../results/figs_lesion_prepost/C77_s{42,7}_three/`.

## September 2026 data: one lesion slice per plane, pre and post (2026-11-09)

User question: is there a lesion-containing slice in each of the transverse, coronal and sagittal planes where
MC-GS beats every other method on both pre and post? In the current vdpois campaign the answer is no: no
transverse lesion plane passes (best TS01 plane 196, ZS-SSDU ahead by about 0.2 dB pre / 0.3 dB post). A scan
of all 1344 `lesion_planes.json` files on the server found no run where all three axes pass.

The historical topic `Results_20260908/CE_retro_cssense_af16` (2026-09-17/18) does meet it:

- **Setup:** BART Poisson-disc AF16 (seed 0, no shutter), CS-SENSE truth, 42 real coils.
- **MC-GS:** the frozen jyz recipe (trainer e778bf2, 4000 iterations), not tuned on this data.
- **Slices:** transverse 201, coronal 131 and sagittal 48. All three contain lesion core: 446, 279 and 384 px of
  `enhance_rois_cssense.npz`.
- **Result:** MC-GS is best on PSNR, SSIM and NMSE against all ten arms in all six panels (3 planes × pre/post).
  The runner-up is GS-single on all six (+0.19 to +0.43 dB).

These numbers come from `sept_slice_metrics.py` (read-only, CPU), which uses the topic's own slice arithmetic: head
mask |GT| > 0.05 P99.9, PSNR at max|GT|, 7×7 SSIM averaged over the head, head NMSE.

### Caveat: not the same setup as the current campaign

`sept_vs_now.py` (read-only, CPU) scores every candidate against the same GT with the same metric code. Head PSNR,
pre / post:

| volume | mask | head PSNR |
|---|---|---|
| Sept recipe | vdpois | 34.45 / 36.60 |
| Oct ar_i1_k30p90 | vdpois | 34.69 / 36.94 |
| TS01 s42 (current keeper) | vdpois | 35.30 / 37.53 |
| strong ZS-SSDU (15 blocks, 400 epochs) | vdpois | 35.32 / 37.53 |
| Sept recipe | Sept BART mask | 33.62 / 35.66 |
| old ZS-SSDU (5 blocks, ≤100 epochs) | Sept BART mask | 32.10 / 33.50 |

- **The current method is better.** On the same mask, TS01 beats the Sept recipe on every slice, by 0.6–1.2 dB.
- **The September sweep relies on a weaker opponent.** The old ZS-SSDU is 1.5–2.2 dB behind.
- **The masks differ.** The BART mask is harder for every method.
- **Splicing is not valid.** Putting TS01 (vdpois-mask input) into the September table would compare methods on
  different inputs. A fair version reruns the TS01 pipeline on the September mask; this has not been run yet.

### Figure: `--topic`, `--err_layout zoom`, `--cbar single`

New options of `fig_lesion_plane_prepost.py`. The defaults keep the previous behaviour.

- `--topic DIR`
  - Draws a CE retro topic that has no frozen evaluator.
  - Arms are resolved from `logs/summary.json`, as in the topic's `arms.py`; GS arms use `*_mag.npy`.
  - Panel numbers are computed by `plane_metrics`, and every volume's SHA-256 goes into the record JSON.
- `--err_layout zoom`
  - The row under each image row shows, per method, the magnified 48×48 lesion box (left) and the same box of
    the error map (right).
  - In the GT column it shows the GT box and the sampling mask.
  - Image rows keep the box outline and drop the corner inset.
- `--cbar single`
  - One grey bar (intensity as a fraction of each row's own window, 0–1) and one error bar, beside the first
    row pair.

```sh
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py \
  --topic /data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16 \
  --label "Sept 2026-09-18 recipe (BART Poisson-disc AF16)" \
  --out /data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_lesionfigs_20261107/sept_bart_three \
  --three 201 131 48 --err_layout zoom --cbar single
```

- **Final job:** j-20261109T091219-7e4357df.
- **Output:** PNG (about 16 MB), preview JPEG and record JSON in the `--out` directory.
- **Local copy:** `../results/hist_ce_retro_af16_201_131_48/sept_prepost_201_131_48_zoomerr.png`; the symlink
  `../results/sept_prepost_zoomerr.png` points to it. The full-error-map version is `sept_prepost.png`.
- Figures are not committed.

## September data: 3D-DIP / 3D-INR rerun with output support (2026-11-09/10, done)

In the September figure, 3D-DIP (and, less so, 3D-INR) shows bright signal outside the head. A read-only check
(coil coverage sum|S|² > 0.25) found that this signal lies outside the coil coverage, where the k-space data do not
constrain the output:

| arm | mean \|x\| outside coverage, pre | share of error energy outside coverage |
|---|---|---|
| 3D-DIP | 0.187 (head 0.216) | 97 % |
| 3D-INR | 0.074 | 63 % |
| ZS-SSDU | 0.009 | 7 % |
| CG-SENSE / MC-GS | 0.000 | 0 % |

- **Effect on metrics:** none on the head-masked numbers (the head is 99.99 % inside the coverage), but the figure is
  misleading.
- **Fix (user's choice 1):** rerun both arms with the October comparison-arm audit winners. These were picked by each
  arm's pre-registered held-out k-space rule, with no GT:
  - DIP `--support 6 --dip_base 96 --tv 0.03`, 12000 iterations;
  - INR `--support 6 --inr_nmax 512 --tv 0.03`, 1500 iterations.
- **Support:** the zero-filled head support dilated by 6 voxels, computed from acquired data only.

`sept_cmp_support.py` wraps the audited `recon_dip_inr.py` (mcgs_ce_mask_wt @ b01fadf) unchanged:

- **Inputs and outputs:** it writes a new topic `codex_artifacts/mcgs_ce_sept_cmp_20261109/`, with `sim/` symlinked
  to the September `sim/`. The September topic is never written, and existing outputs are refused.
- **GPU:** it passes the gpu.py-granted physical id through as `--gpu`.

Runs:

- **GPU smoke:** j-20261109T095257-80f9016e passed. It showed a peak of about 70 GB on the card.
- **Pre-contrast:** j-20261109T100902-c9266335 on GPU 2. DIP took 13 336 s; INR took 2 180 s.
- **Post-contrast:** run on GPU 2 from 22:58 CST by a parallel session, using the same wrapper, arguments and script
  revision (logs `run_post*.log` in the topic dir). A later duplicate launch, j-20261110T014653-faca26c1, was refused
  by the no-overwrite guard and changed nothing.

Results, head-masked PSNR / SSIM / NMSE:

| | outside-coverage mean \|x\|, pre | head PSNR, pre / post |
|---|---|---|
| DIP, registered | 0.187 | 29.32 / 30.24 |
| DIP, with support | 0.0004 | 30.76 / 32.07 (post: the new run's own score) |
| INR, registered | 0.074 | 28.08 / 30.61 |
| INR, with support | 0.0006 | 32.39 / (see record) |

- **Background:** with the support, the background is gone; GT outside coverage is 0.0002.
- **Stronger opponents:** both arms got stronger. INR with support now beats ZS-SSDU on transverse 201 PSNR/NMSE.
- **MC-GS still wins:** MC-GS still wins all six pre/post panels on slices 201/131/48 against all ten arms. The
  runner-up stays GS-single, with unchanged margins (PSNR +0.19 to +0.43 dB).
- **Figure:** j-20261110T015143-54201809 (`--arm` for 3D-DIP and 3D-INR). The output is
  `codex_artifacts/mcgs_ce_lesionfigs_20261107/sept_bart_three_supfix/`; the local copy is
  `../results/sept_prepost_supfix.png`.

`fig_lesion_plane_prepost.py --arm LABEL=PATH` (topic mode) swaps an arm's volume for the redraw. `{s}` in PATH is
replaced by the scan.

## Record for plain-scan / enhanced reconstruction (user decision, 2026-11-11)

The user designated the September three-plane result above as the record of the best plain-scan (pre) and enhanced
(post) joint reconstruction. When the best CE pre/post result is asked for, cite this one, not the vdpois TS01 figure.

- **Data and method:** topic `Results_20260908/CE_retro_cssense_af16` (BART Poisson-disc AF16, CS-SENSE truth);
  MC-GS = the untuned September recipe (trainer e778bf2, 4000 iterations).
- **Slices:** transverse 201, coronal 131, sagittal 48, all with lesion core.
- **Result:** MC-GS first on PSNR, SSIM and NMSE in all six panels against all ten arms. PSNR margins over the
  runner-up (GS-single in every panel), pre / post: transverse +0.25 / +0.43 dB, coronal +0.19 / +0.30 dB,
  sagittal +0.21 / +0.38 dB.
- **Figure of record:** the support-fixed redraw (3D-DIP / 3D-INR rerun with support, zoom error layout):
  - PNG: `codex_artifacts/mcgs_ce_lesionfigs_20261107/sept_bart_three_supfix/` (SHA-256 `fed2f5d1…adc36`);
  - 600-dpi PDF: `codex_artifacts/mcgs_completed_report_20261110/figs_rerender/sept_supfix/`;
  - local: `../results/ce_prepost_record.png` and `.pdf` (symlinks into `../results/hist_ce_retro_af16_201_131_48/`).
- **Limits that go with the record:** ZS-SSDU here is the 5-block version; GS-single is the registered arm, and
  against the independently trained single-contrast GS of the framework ablation MC-GS wins 5 of these 18
  conditions (`../ce_af16_ablation_20261110/RESULTS.md`); one subject.
