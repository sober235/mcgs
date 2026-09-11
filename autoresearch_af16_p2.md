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

Results (1500 iterations; bar on the complex gauge: T1 32.2011, FLAIR 31.7175):

| exp | T1 mag | FLAIR mag | T1 cplx | FLAIR cplx | min margin cplx | cos phase T1 / FLAIR |
|---|---|---|---|---|---|---|
| `p2_r1_parent` | 35.515 | 34.724 | 31.981 | 31.187 | -0.531 | 0.9913 / 0.9908 |
| **`p2_r1_wavup`** | **35.546** | **34.771** | **32.257** | **31.499** | **-0.218** | **0.9922 / 0.9918** |
| `p2_r1_n50` | 35.520 | 34.803 | 31.848 | 31.148 | -0.570 | 0.9908 / 0.9905 |
| `p2_r1_kh30` | 34.718 | 34.109 | 30.880 | 30.271 | -1.447 | 0.9883 / 0.9881 |
| `p2_r1_curric` | 33.958 | 33.180 | 31.028 | 30.526 | -1.192 | 0.9902 / 0.9910 |

**The complex-aware prior is the lever, as the diagnosis predicted.** Wavelet T1 0.14 / FLAIR 0.10 gains
+0.276 (T1) and +0.312 (FLAIR) on the complex gauge over the control at the same budget, passes the bar on
T1 (+0.056) and lifts the phase agreement 0.9913 -> 0.9922 and 0.9908 -> 0.9918. The magnitude gauge moves
with it (+0.031 / +0.047), so nothing is traded away. FLAIR is now the binding contrast (-0.219 left).
Both high-band levers lose badly: kspace_high 3.0 costs 0.8 dB on the magnitude gauge and 1.1 on the
complex one, and the frequency curriculum costs 1.5 dB of magnitude. Capacity 5.0M is flat at this budget
(the first loop's capacity gain needed the full budget), so it is deferred to a promotion run, not dropped.

### Round 2 — 2026-09-11, three runs at 1500 iterations, serial on GPU 0

The other user took GPUs 1-7 (60-78 GB each) right after round 1 finished, so this round runs one card at a
time. Candidates all attack the binding contrast, on top of round 1's winner where stated:

| exp | change | rationale |
|---|---|---|
| `p2_r2_wav14` | wavelet T1 0.14 / FLAIR 0.14 | push the winning direction on the binding contrast |
| `p2_r2_ptv05` | round 1 winner + `--lambda_ptv 0.05` | the new phase-only TV term, aimed straight at the gauge gap |
| `p2_r2_wav18f` | wavelet T1 0.14 / FLAIR 0.18 | FLAIR-heavy, to find where the wavelet stops paying |

Results (1500 iterations; bar complex T1 32.2011 / FLAIR 31.7175, magnitude 35.2092 / 34.5028):

| exp | T1 mag | FLAIR mag | T1 cplx | FLAIR cplx | min margin cplx | min margin mag | cos phase |
|---|---|---|---|---|---|---|---|
| **`p2_r2_ptv05`** | 35.320 | 34.682 | **32.365** | **31.830** | **+0.112** | +0.111 | **0.9929 / 0.9931** |
| `p2_r2_wav18f` | 35.543 | 34.593 | 32.255 | 31.690 | -0.028 | +0.090 | 0.9922 / 0.9927 |
| `p2_r2_wav14` | 35.545 | 34.714 | 32.257 | 31.664 | -0.053 | +0.211 | 0.9922 / 0.9925 |
| `p2_r3_wavT118` | 35.460 | 34.712 | 32.332 | 31.662 | -0.056 | +0.209 | 0.9925 / 0.9925 |

**Objective met at the screening budget.** `p2_r2_ptv05` = round-1 winner + the new phase-only TV at 0.05 is
above the bar on BOTH gauges and BOTH contrasts: complex +0.164 / +0.112, magnitude +0.111 / +0.179. Its
phase agreement, 0.9929 / 0.9931, matches and slightly beats L1-ESPIRiT's 0.9925 / 0.9930, which is what the
term was built for. The cost is on the T1 magnitude side: the margin there falls from +0.336 (wav14) to
+0.111, so the term trades a little magnitude accuracy for phase.

The wavelet path alone saturates: FLAIR 0.18 gains 0.026 dB of FLAIR complex over 0.14 while giving up
0.121 dB of FLAIR magnitude, and T1 0.18 gains nothing over 0.14. So the wavelet is done at T1 0.14 /
FLAIR 0.10-0.14 and the phase term carries the rest.

### Round 3 — 2026-09-11, from 20:35

| exp | GPU | change from `p2_r2_ptv05` | rationale |
|---|---|---|---|
| `p2_r3_wav14_ptv05` | 0 | FLAIR wavelet 0.10 -> 0.14 | stack the two winners; FLAIR was the binding contrast |
| `p2_r3_ptv10` | 2 | phase TV 0.05 -> 0.10 | find where the phase term stops paying |
| `p2_promo_wav14_full4k` | 1 | wav14 at 4000 iterations, volumes | full-budget reference for the wavelet-only path (launched 20:10) |
