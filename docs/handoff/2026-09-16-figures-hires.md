# Handoff 2026-09-16 — the AF 12/16/20/24 figures at publication resolution (branch `autoresearch-jyz-af24`, worktree `mcgs_af16_wt`)

User request: "将前期的 af=12,16,20,24 对应重建结果全部整理成 600 dpi 高清图以及对应的矢量图 pdf".
No reconstruction was re-run. Only rendering changed.

## 1. Done and verified

Commits `5ea6189` (main pass) and `ac33c8a` (the last two mask figures), both pushed to
`origin/autoresearch-jyz-af24`.

**The eight figures of record** — `t1flair_v3_af{12,16,20,24}_{T1,FLAIR}_row3planes_compact_180_124_76_inset48_headerr_{p14|p3}_mcinr_spirit_zf_caps.{png,pdf}`:

| check | result |
|---|---|
| size | 8/8 at 6453 x 4352 px, 200 dpi (12 columns; the no-ZF layout is 5928 px at the same dpi) |
| PDF fonts | 8/8 CID TrueType, no Type 3 |
| title | 8/8 `nominal AF 12/16/20/24, actual 11.71 / 15.19 / 18.56 / 24.00x` |
| L1 label | 8/8 PDF text layer reads `L1-SPIRiT` |
| content unchanged | 8/8 PSNR / lead lines diff IDENTICAL against the previous figure of record |
| MC-GS rank | 24/24 cells (4 AF x 2 contrasts x 3 planes x 3 metrics) first |

```
$ R=/data0/congcong/code/GS/Results_20260908
$ diff <(grep -o "PSNR MC-GS #[0-9].*lead [-+0-9.]* dB" $R/T1FLAIR_AF24_v3_jiangyizhou/logs/fig_mcinr_spirit_T1.log) \
       <(grep -o "PSNR MC-GS #[0-9].*lead [-+0-9.]* dB" $R/T1FLAIR_AF24_v3_jiangyizhou/logs/hires/fig_T1.log) && echo IDENTICAL
IDENTICAL
```

**Four script changes**, all with the default preserved so every documented command still reproduces
its own figure:

1. `--png_dpi` (default 130). The set is rendered at 200 = 6453 px wide = 911 dpi at a 180 mm page
   width, 600 dpi across A4 landscape. The user picked the page-width reading of "600 dpi"; a literal
   `savefig(dpi=600)` would be 19356 x 13056 px and 150-300 MB per PNG. The panels hold about 256
   voxels either way, so resolution buys sharper text and boxes, not image content.
2. `matplotlib.rcParams["pdf.fonttype"] = 42` in every figure script. matplotlib's default is Type 3,
   which IEEE and Elsevier submission systems reject.
3. `--caps` no longer upper-cases a label containing `SPIRiT`: the lower-case i is part of the method
   name, so `L1-SPIRiT` and `L1-ESPIRiT` keep their spelling while the rest stay upper case.
4. The AF 12 title gained the word `nominal`, which only that topic was missing.

**Auxiliary figures**, PNG at 600 dpi with a vector PDF beside each (these are scatter and line plots,
where vector is what matters): AF 20 mask x2; AF 24 mask, mask series, raw k-space support x4;
AF 16 `probe_c_mask_T1_vs_FLAIRc` (its generator was always there, inside
`prep_04c_export_af16c.py`'s `figure()`, which `--fig_only` redraws without re-exporting data);
AF 12 `mask_check_af12_af11_af16`, which had no script anywhere on disk — a filesystem-wide search
found it only in old session transcripts, so it had been drawn by an inline heredoc and never landed
in the repository. `scripts/fig_mask_check_af12_af11_af16.py` rebuilds it from the three mask files,
recomputing every number rather than copying it, and all of them agree with the 2026-09-13 figure:
3847 / 4173 / 2966 points of 45056 (AF 11.71 / 10.80 / 15.19), centre 56x56 sampled
39.3 / 40.8 / 36.1 %, AF12 and AF16 sharing 812 points = 21 % / 27 %.

**Delivery folder** `Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires/`: `png/`, `pdf/`,
`README.md`, `manifest.md5`, `scripts/{render_hires_final.sh,collect_delivery.sh}`. Versioned here as
`t1flair_af12_16_20_24_figures_hires/` minus the binaries — those are the per-topic `figures/` files
already versioned, and a second copy would add 222 MB. The archive root README has a row for it (that
index lives only on disk; it is not mirrored into this repository).

## 2. Decisions waiting for the user

- Whether the 2026-09-14 per-contrast AF gauge still on the earlier `_spirit` set and on the
  re-archived figures of record should be cascaded to gauge B (plane points / mask points, one number
  for both contrasts). Raised in `14016df`, unchanged since.
- The AF 24 FLAIR label, option A (24.00x, one number for both contrasts) or option B (25.51x). This
  set is rendered under A.
- Whether the delivery folder's PNG and PDF should also go into the repository (+222 MB).

## 3. Next steps if the line stays open

Nothing is queued. Every figure of all four topics now has a 600-dpi-class PNG and a vector PDF.

## 4. Pitfalls and do-not-redo

- **One 600-dpi PDF render holds about 100 GB of RAM.** Two at a time is the ceiling that
  `render_hires_final.sh` enforces; four in parallel took this shared server to 6 GB free earlier the
  same day.
- **Two sessions rendered the same file names at once** (this one at 200 dpi, another at 130 dpi via
  `render_zf_caps_af12_16_20.sh`), leaving `figures/` half one and half the other plus a PDF truncated
  to 0.2 MB by an interrupted write. Before starting a batch, `pgrep -fa` for the script and check the
  target directory's mtimes. Do not kill another session's processes: the auto-mode classifier refuses
  it as `Interfere With Workloads`, and it is not this session's work to stop.
- Do not re-render to "fix" the resolution: the figures are at the reading of "600 dpi" the user
  chose. A literal `dpi=600` was offered and declined.
- `mask_check_af12_af11_af16.png` is now script-generated. Do not go back to drawing it inline.

## 5. Why

The four AF topics are meant to be laid side by side, so the set has to be uniform: same layout, same
title wording, same resolution, same font encoding. The three cosmetic fixes in this pass (SPIRiT
spelling, the missing `nominal`, Type 3 fonts) were each the kind of thing that only shows up when the
figures sit next to each other or reach a submission system.
