# autoresearch phase 2 — jiangyizhou T1+FLAIR at AF16: MC-GS best under BOTH gauges

Started 2026-09-11 (user: "启动karpathy的autoresearch调参，使得mc-gs效果最好相对于对比方法at af=16倍加速。请注意参数要和af=12点实验隔离开").

## Objective

MC-GS joint first against the seven comparison arms (CG-SENSE, CS-SENSE converged, Joint SENSE, L1-ESPIRiT,
ZS-SSDU, 3D-DIP, 3D-INR) on head-only PSNR under **both** gauges, on this subject at AF16:

* magnitude gauge `| |recon| - |GT| |` — the table's gauge and the field's convention. **Already won** by P6.
* complex gauge `|recon - GT|` on the complex volumes — **not won**: P6 is 0.251 dB (T1) and 0.517 dB (FLAIR) behind.

Ledger objective: maximise `min_margin_cplx = min(dT1_cplx, dFLAIR_cplx)` subject to `min_margin_mag >= 0`.

## The bar (`../Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/logs/bar_complex_gauge.json`, 2026-09-11)

| gauge | T1 best of the seven | FLAIR best of the seven | P6 (MC-GS joint) | margin |
|---|---|---|---|---|
| magnitude | L1-ESPIRiT 35.2092 | L1-ESPIRiT 34.5028 | 35.5521 / 34.8066 | +0.343 / +0.304 |
| complex | L1-ESPIRiT 32.2011 | L1-ESPIRiT 31.7175 | 31.9502 / 31.2011 | -0.251 / -0.517 |

Phase agreement inside the head, magnitude-weighted cos: P6 0.9911 / 0.9907, L1-ESPIRiT 0.9925 / 0.9930,
CS-SENSE 0.9921 / 0.9927, ZS-SSDU 0.9909 / 0.9935, CG-SENSE 0.9859 / 0.9898.

## Parent

P6, the final recipe of the first loop (`autoresearch_jyz_af16.md`, verdict 2026-09-11 03:50), baked into
`autoresearch_af16_p2.sh`: v2 edge-TV working point (λ_geo 0.10, κ_q 0.85, w_floor 0.10, ε 0.04,
kspace_high 2.2, warmup 500, max_radius 20, init head / 3-NN / κ from ZF) + wavelet sparsity T1 0.10 /
FLAIR 0.07 + 3.6M points. Candidates differ from the parent only by their extra args.

## Isolation from the AF12 experiment (user requirement)

* `autoresearch_af16_p2.sh` hard-codes `configs/t1flair_v3_af16_jyz.yaml`; there is **no** CONFIG override.
* EXP names must start with `p2_`; the runner refuses anything else and refuses to overwrite outputs.
* Rows go to `autoresearch_af16_p2.tsv` through `scripts/autoresearch_af16_p2_row.py` (flock-protected);
  the first loop's `autoresearch_jyz_af16.tsv` is never touched.
* The AF12 configs, runs (`ar17_P6_tf_jyz_af12_full4k`) and topic are out of scope for this line.

## Diagnosis going in

With hard DC the sampled k-space matches the data exactly, so all image error lives in the unobserved
region, and the complex gauge counts the part of it that the magnitude gauge drops: the phase component.
In the trainer, TV, directional TV, edge-align and vectorial TV all act on `|x|` (`losses/edge_tv.py`
`tv_magnitude`), so the only complex-aware terms are the k-space fit loss and the wavelet prior
(`loss.wav_on` defaults to `complex`). `e_unobs` in the scorer is the unobserved-region complex error and
is the cheap proxy for this objective. First levers: the complex-aware prior and the high-band fit.

## Protocol

* Screen at 1500 iterations, one trainer per GPU (the first loop stalled twice when a card got a second).
* Promote a winner at 4000 iterations with `SAVE_VOLUMES=1`, then re-check both gauges.
* Seed floor from the first loop: 0.002 dB on the magnitude gauge; treat anything under it as noise.
* New scorer fields (`arc/cycle.py`, validated against an offline computation on the P6 checkpoint:
  31.950 / 31.201 exactly): `head_psnr_cplx`, `cos_phase`.

## What the first loop already ruled out (do not retry)

Directional TV (λ_dtv), prewhitening, FLAIR-only high-band weight, faster LR decay, stronger TV,
λ_geo 0.07. Wavelet sparsity, capacity 1.2M → 3.6M and per-contrast wavelet strength are what won there.

## Round log

### Round 1 — 2026-09-11, five runs at 1500 iterations, GPUs 0-4

| exp | GPU | change from the parent | rationale |
|---|---|---|---|
| `p2_r1_parent` | 0 | none | the 1500-iteration reference for both gauges |
| `p2_r1_wavup` | 1 | wavelet T1 0.14 / FLAIR 0.10 | the only complex-aware prior; stronger may fix phase |
| `p2_r1_kh30` | 2 | kspace_high 3.0 | better fit of the sampled high band -> better extrapolation |
| `p2_r1_n50` | 3 | 5.0M points | capacity for finer structure, and for phase detail |
| `p2_r1_curric` | 4 | `--curriculum` | progressive high-frequency schedule, untried on this subject |

Results appended below when the round finishes.
