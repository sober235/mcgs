# Diff-mask study on jiangyizhou T1 + FLAIR, nominal AF 16: the three sampling patterns, redone (2026-11-11)

**Status: masks only. Approved by the user on 2026-11-11 (about 21:00 PST) from `preview/masks_af16_redo_preview.png`.
No reconstruction has run on them and nothing is registered in `Results_20260908`.**

## Why

The September diff-mask study (server worktree `/data0/congcong/code/GS/mcgs_diffmask_wt`, topics
`Results_20260908/T1FLAIR_AF16_{gauss,radial}_v3_jiangyizhou`, GitHub branch `jyz-af16-diffmask`) compared three rows:
the house vd-Poisson mask, a Gaussian variable-density random mask, and a Cartesian pseudo-radial mask that carried the
24 x 16 ACS block. On 2026-11-11 the user said these three masks were not the intended ones and asked for:

1. the vd-Poisson mask at AF 16;
2. a mask whose centre ACS is fully sampled and whose other points are spread over k-space without a density that
   rises towards the ACS, then (second instruction, same day) "re-generate the second mask from the ESPIRiT tool in
   Michael Lustig's work";
3. the conventional radial mask.

## The masks

PE plane (ky, kz) = (256, 176), DC at (128, 88). AF = PE-plane points / sampled points that carry that contrast's data
(user definition, 2026-09-14). T1 acquires the whole plane; FLAIR's vendor shutter leaves 34139 of 45056 points.

| row | mask | how | points | T1 AF | FLAIR AF | points outside FLAIR's shutter | fully sampled centre (ky x kz) |
|---|---|---|---|---|---|---|---|
| 1 | vd-Poisson disc | the house AF16 mask, unchanged: `vdPoisMex(256,176,256,176, 4, 4, 24, 0, 1)` | 2966 | 15.19 | 16.49 | 233 | 24 x 16 |
| 2 | uniform density + ACS | the same generator with `pp = 0`: `vdPoisMex(256,176,256,176, 4.1979, 4, 24, 0, 0)` | 2964 | 15.20 | 19.48 | 651 | 24 x 16 |
| 3 | conventional radial | 16 full-diameter spokes through DC, every 11.25 deg, one point wide, no ACS block | 2964 | 15.20 | 15.45 | 48 | 14 x 6 |
| (3 alt) | radial + ACS | 15 spokes every 12 deg plus the 24 x 16 block | 2964 | 15.20 | 15.46 | 49 | 24 x 16 |

- **Row 1** is read from `Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/mask_vdpois_256x176_af16.mat`
  (md5 `2d7ed0001214aec2af8f6dce5d52ce70`), not regenerated.
- **Row 2** comes from `vdPoisMex` in Michael Lustig's ESPIRiT toolbox (`/data0/congcong/code/ESPIRiT/utils`, the
  compiled mex, not rebuilt). `pp` is the polynomial order of the variable density; the toolbox's help says
  "0 = uniform". Everything else is the house call. The budget is the September study's, exactly 2964 points:
  `accelx` was fitted on ten draws (n = 353.1 + 10960.3 / accelx, so 4.1979) and draws were repeated until the first one
  with exactly 2964 points, draw 43, taken untrimmed and chosen by its point count alone
  (`preview/mask2_vdpoismex_pp0_draws.txt`). The points still keep the generator's minimum spacing: it is a
  uniform-density Poisson-disc pattern, not an i.i.d. random one. `vdPoisMex` seeds from `time(0)`, so this mask
  **cannot be regenerated bit for bit**; the saved file is the record.
- **Row 3** has angles uniform in normalised k-space (same k-max in ky and kz), equal-length spokes, one sample per step
  along each spoke's major axis. 16 is the smallest spoke count that reaches the budget; the 143 excess points are
  removed at the spoke tips, so every spoke ends at normalised radius 0.957. Deterministic.

The first version of row 2 in this session was an i.i.d. uniform random draw in numpy (2964 points, FLAIR 19.38x). The
user replaced it with the ESPIRiT-toolbox mask above; it is not kept.

## Open points for whoever runs the reconstructions

- **Row 3, ACS block or not.** The user approved the preview, which shows row 3 without an ACS block and the variant
  with one beside it; they did not say which of the two the study uses. Pure spokes leave a 14 x 6 fully sampled
  centre, and L1-SPIRiT calibrates its kernel from the mask's fully sampled centre block (`recon_spirit.py`,
  `acs_box()`), which was 24 x 16 in every earlier topic. Ask before running that arm.
- **FLAIR sits at 19.48x on row 2**, against 16.49x on row 1: a uniform density spends 651 points in the corners FLAIR
  never acquires. Reported, not tuned, as in the September study.
- A full-diameter spoke is point-symmetric: every sampled point of row 3 has its conjugate partner sampled too.

## Files

| file | what |
|---|---|
| `make_masks_preview.py` | reads row 1, calls the MATLAB script for row 2, draws row 3 and its variant, writes the figure, the masks and the read-out to `$MCGS_ARTIFACT_DIR` |
| `gen_mask2_vdpoismex_pp0.m` | row 2: the `vdPoisMex` draws (output directory from `MASK2_OUT`) |
| `preview/masks_af16_redo_preview.png` | the figure the user approved |
| `preview/masks_af16_redo_preview.json` | every number quoted above, with md5 sums |
| `preview/mask2_vdpoismex_pp0_draws.txt` | all 43 draws of row 2 with their point counts |

Run (CPU, about two minutes; it draws a NEW row 2, so do not use it to "restore" the approved one):

```sh
/usr/bin/python3 ../gpu-tools/gpu.py run --cpu -- @python autoresearch/diffmask_af16_masks_20261111/make_masks_preview.py
```

The masks themselves are not in git (`*.mat` is ignored). Approved files, job `j-20261112T035450-6ed722ac`, source
digest `24b6d0dc591f7bf381cf9b09a81675351b6789a03dc683b4449d4e2a4794de54`, kept read-only on the server in
`/home/congcongliu/codex_artifacts/mcgs_diffmask_af16_redo_20261111/` (key `mask`, (256, 176)):

| file | md5 |
|---|---|
| `mask_vdpoismex_pp0_256x176_af16_preview.mat` (row 2) | `d0796ac1a198ae2aba8983146b3830fc` |
| `mask_radial_256x176_af16_preview.mat` (row 3) | `a18824d8448161b687ccfd8ae06e3321` |
| `mask_radial_acs_256x176_af16_preview.mat` (row 3 variant) | `25de9fa0b6aee01a3d179f5fb77b2e97` |
