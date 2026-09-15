# Record 2026-09-15: the L1-SPIRiT arm registered on the 16yo MTP AF 12 topic (L1-ESPIRiT no longer counted)

Branch `16yo-mtp`, worktree `/data0/congcong/code/GS/mcgs_peds_wt`, topic `Results_20260908/Peds_TSC_MTP_AF12_retro/`
(README section "The L1-SPIRiT arm" is the outward version of this record). Session claude-e7 (the stage-3 session
after its `/clear`), registration 22:54–23:15; the arm itself was built and swept by the stage-4 session (claude-eb,
`autoresearch_peds_p4.md`, commits b9b0729 and 43bfe56).

## 1. Decision

User, 16:05: "是的,更换成l1-spirit; gs-single也算对比方法" — L1-SPIRiT replaces L1-ESPIRiT as the L1 arm, GS-single
stays a comparison method. The jyz T1/FLAIR topics made the same swap the same day (branch
`spirit-arm-jyz-af12-16-20`), and this topic follows their convention: the replaced arm's rows stay in the jsons and are
printed under each table as "not counted"; the shown tables and the figures of record carry L1-SPIRiT.

## 2. What was swept (claude-eb) and what was kept (this session)

`scripts/recon_spirit.py` = the jyz AF16 script (Lustig's toolbox transcribed, aligned to the MATLAB original to 1e-12
there), changed only in the contrast names and in measuring the ACS block on Ω. Ω's block is 24 x 8 here (jyz 24 x 16),
under-determining a 5 x 5 x 32-coil kernel, so both calibrations ran: `acquired` and `full24` (centre 24 x 24 of the
full k-space, the block the topic's ESPIRiT maps came from). 50 points: `acquired` ks {3, 5} x λ {0, 5e-4, 1e-3, 2e-3,
4e-3, 8e-3} (T1w also 2.5e-4), `full24` ks {3, 5} x λ {0, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3}, both contrasts, 150 FISTA
iterations, db2 / 4 levels, Tikhonov 0.01; data consistency on Ω ∪ Z like every arm. Sweep finished 20:03
(`logs/spirit_sweep.log`, `logs/spirit_<c>_*.json`).

Kept by `summarize.py` (one family over every spirit json, best head PSNR per contrast — my own scan of the 50 jsons
agrees with claude-eb's report to every digit):

| contrast | kept | full volume (own scorer) head PSNR / SSIM / HFEN | best acquired-ACS point |
|---|---|---|---|
| T1w | full24, ks 5, λ 2.5e-4 (`spirit_full24_ks5_lam0.00025`) | 38.384 / 0.9629 / 0.0689 | ks 3, λ 5e-4: 38.114 / 0.9642 / 0.0780 |
| T2s | full24, ks 3, λ 5e-4 (`spirit_full24_ks3_lam0.0005`) | 32.144 / 0.8642 / 0.2302 | ks 3, λ 1e-3: 32.095 / 0.8682 / 0.2384 |

Both optima are interior to the λ grid. `full24` is the fair choice, not a privilege: every other arm and the ground
truth already rest on that calibration block.

## 3. Registration

- `summarize.py`: new family `("spirit", "logs/spirit_{c}_*.json")`; family rows now carry `src` (the run's own
  name → `recon/<src>_<c>_complex.npy`), `ksize`, `calib`.
- `rescore.py`: the spirit arm found through `src`; new `--only KEY` scores the named arms and merges them into the
  existing json (the gauge is deterministic; the other 26 rows are untouched). `scripts/rescore_spirit.sh` runs both
  gauges; 23:04:58 → 23:06:25.
- Check: the full-volume rescore reproduces the sweep's own JSON to every printed digit on both contrasts
  (38.384 / 0.9629 / 0.0689 / 41.629 / 0.00410; 32.144 / 0.8642 / 0.2302 / 35.656 / 0.01765).
- `readme_tables.py`: `spirit` labelled "L1-SPIRiT (full-centre calibration, ks 5, λ 0.00025)"; `DROP = {"l1espirit"}`
  removes L1-ESPIRiT from the shown rows and the bold, and prints its numbers under each table as "Not counted".
  `logs/readme_tables.md` regenerated and re-injected; the tables as they were are in `logs/archive_pre_spirit_20260915/`.
- Figure scripts (`fig_three_planes_compact.py`, `fig_three_planes_inset.py`, `fig_row_final.py`): `--l1 {spirit,espirit}`,
  default spirit. `scripts/figs_spirit.sh` redrew the two figures of record for both contrasts (`_p3f_spirit`,
  `_af16style_spirit` with the 48-voxel zoom, grey colorbar on the transverse row, vector PDF); looked at, layout intact.

## 4. Numbers (primary region, head inside partitions 12..92)

| contrast | L1-SPIRiT | rank among counted rows | L1-ESPIRiT (not counted) |
|---|---|---|---|
| T1w | 38.310 / 0.9607 / **0.0679**, whole 41.225, sharp 0.982 | 5 of 14 on head PSNR; **first on head HFEN** | 38.830 / 0.9674 / 0.0756 |
| T2s | 32.017 / 0.8630 / 0.2264, whole 34.971, sharp 0.951 | 8 of 12 | 33.860 / 0.9050 / 0.1882 |

Consequence for the six-cell target (user: MC-GS first in every head cell against every arm, GS-single included):
the T1w head PSNR and SSIM bars stay GS-single's (38.993 / 0.9687; MC-GS 38.936 / 0.9685), the three T2s cells stay
MC-GS's own, and the **T1w head HFEN bar moves from GS-single's 0.0764 to L1-SPIRiT's 0.0679; MC-GS's 0.0768 is 0.0089
short of it** — before the swap it was 0.0004 short. The lightly regularised k-space solution keeps the fine grain
(noise included) that both GS arms smooth away, which is what HFEN rewards; it pays for it with 0.68 dB of head PSNR
against GS-single. On T2s the arm sits below CS-SENSE, Joint SENSE and ZS-SSDU and changes nothing.

## 5. Left as is

- `slice_quality.py`, `find_mcgs_first_slice.py`, `best_slice_mcgs.py` still list L1-ESPIRiT; swap when next used.
  The stage-4 lesion tools were switched by claude-eb (43bfe56).
- The per-slice figures drawn on 2026-09-15 before the swap (`_headerr_p3f`, `_af16style`, the per-plane picks
  `tra91` / `cor110` / `sag183`) keep L1-ESPIRiT in their L1 column; they are labelled as such in the README.
- Commit b9fcf07's message quotes "466 of 561 slices" from this session's context before its `/clear`; that count was
  never written down and is not reproducible from the tree. Treat it as NOT_EVIDENCE until recomputed.
