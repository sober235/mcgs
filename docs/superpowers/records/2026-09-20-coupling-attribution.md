# Record — Where does the MC-GS margin come from? Coupling attribution, Stage 1 (jiangyizhou AF 16, 2026-09-20)

Branch `coupling-attribution` (worktree `mcgs_hparam_wt`, cut from `hparam-sensitivity` 69e7daa). Spec
`docs/superpowers/specs/2026-09-20-coupling-attribution-design.md`, plan
`docs/superpowers/plans/2026-09-20-coupling-attribution.md`. **Stage 1 only.** Stages 2 (λ_edge dose-response),
3 (flipped-FLAIR placebo) and 4 (equal-budget search on the single arm) are specified and not run.

## 1. The question, and what Stage 1 does to answer it

A reviewer asks: *how do you know the MC-GS margin over GS-single comes from joint modelling and not from the fact
that you tuned the joint arm?*

Spec §2 read `paired_gs/trainers/joint_trainer.py` on this branch and established that the frozen phase-3 recipe has
exactly **two** cross-contrast channels:

1. **shared geometry** — positions, scales and rotations shared across the two channels, densities per channel. No λ
   on it; it cannot be switched off without turning MC-GS into two independent clouds.
2. **`R_edge`**, the symmetric common-edge cross-gradient prior, entered at `--lambda_edge 0.001`. A tunable λ, and
   the one the phase-3 ladder added.

Everything else in the recipe — edge-preserving TV, phase-only TV, wavelet L1 — is per channel and applies identically
to both arms; the other two coupled terms in the code (`R_dtv`, `R_vtv`) are off in every registered run. So switching
`R_edge` off removes the only tunable coupling while every per-channel knob keeps its tuned value, and the published
margin splits into two measurable parts:

```
M1    = J1 - S1     joint - single at the published lambda_edge 0.001   (the published margin)
M0    = J0 - S0     joint - single at lambda_edge 0                     (what shared geometry alone buys)
delta = M1 - M0                                                         (what the common-edge prior buys)
```

Spec §2.1 established that **both** arms have to be re-run at λ_edge 0, not just the joint one. GS-single is the joint
config with the same contrast in both channel slots (`configs/t1flair_v3_af16_jyz_singleT1.yaml`); its two channels
carry independent densities, so their gradients do not align by construction and `R_edge` still pulls on them —
measured at iteration 150, `R_edge` = 6.60e-01 in the joint arm and 4.06e-01 in the single arm. A joint-only ablation
would have compared against a baseline still paying a prior the joint arm no longer pays. §7.1 below shows what that
would have cost.

## 2. What ran

Five runs, queue `scripts/ca_queue_stage1.txt` via `scripts/ca_queue.sh` (free cards only, one trainer per card,
launches 90 s apart). All `rc=0`; queue drained 2026-09-20 20:23.

| exp | arm | config | λ_edge | card | wall |
|---|---|---|---|---|---|
| `ca_s_t1_p3` | single T1 | `…_singleT1.yaml` | 0.001 | 1 | 65.8 min |
| `ca_s_fl_p3` | single FLAIR | `…_singleFLAIR.yaml` | 0.001 | 2 | 62.8 min |
| `ca_j_edge0` | joint | `t1flair_v3_af16_jyz.yaml` | 0 | 3 | 63.6 min |
| `ca_s_t1_edge0` | single T1 | `…_singleT1.yaml` | 0 | 6 | 64.8 min |
| `ca_s_fl_edge0` | single FLAIR | `…_singleFLAIR.yaml` | 0 | 7 | 62.3 min |

The joint anchor J1 is `hp_n36` from the hyper-parameter study (already in `results_arc/`, 65.4 min), which reproduces
the registered `gs_joint_p3` (35.706 / 34.928) **to the three decimals the topic registers**, so it was not re-run.
Measured, it is 35.706153 / 34.928080 — deltas +0.00015 / +0.00008 dB, inside that reference's own ~5e-4 dB rounding
resolution; the "0.000 / 0.000 dB" quoted for this anchor in the hparam study is the reference's rounding rather than
a demonstrated zero. Everything else is fixed:
jiangyizhou T1+FLAIR, the AF 16 house vd-Poisson mask (md5 `d08179b8…`, T1 15.19× / FLAIR 16.49×), the phase-3 recipe
with its recipe block byte-identical to `run_hp_mcgs.sh` except λ_edge, 4000 iterations, 3.6M points, seed 42,
`SAVE_VOLUMES=1`. Metrics are head-only (`|GT| > 0.05 · P99.9|GT|`), the same scorer as the hparam study.

## 3. The table (`results_arc/ca_summary.md`, verbatim)

| arm | lambda_edge | contrast | exp | channel | head PSNR / SSIM / HFEN / NMSE | whole PSNR / NMSE | cplx head PSNR |
|---|---|---|---|---|---|---|---|
| joint | 0.001 | T1 | hp_n36 | plain | 35.706 / 0.9396 / 0.1139 / 0.01295 | 39.268 / 0.01542 | 32.497 |
| joint | 0.001 | FLAIR | hp_n36 | contrast | 34.928 / 0.9261 / 0.1121 / 0.01372 | 38.585 / 0.01626 | 31.883 |
| single | 0.001 | T1 | ca_s_t1_p3 | plain | 35.558 / 0.9371 / 0.1186 / 0.01340 | 39.144 / 0.01587 | 32.400 |
| single | 0.001 | FLAIR | ca_s_fl_p3 | plain | 34.886 / 0.9249 / 0.1137 / 0.01385 | 38.498 / 0.01659 | 31.905 |
| joint | 0.0 | T1 | ca_j_edge0 | plain | 35.645 / 0.9388 / 0.1150 / 0.01314 | 39.223 / 0.01558 | 32.468 |
| joint | 0.0 | FLAIR | ca_j_edge0 | contrast | 34.876 / 0.9253 / 0.1134 / 0.01389 | 38.543 / 0.01641 | 31.854 |
| single | 0.0 | T1 | ca_s_t1_edge0 | plain | 35.599 / 0.9381 / 0.1164 / 0.01327 | 39.210 / 0.01563 | 32.433 |
| single | 0.0 | FLAIR | ca_s_fl_edge0 | plain | 34.904 / 0.9255 / 0.1128 / 0.01380 | 38.521 / 0.01650 | 31.919 |

| contrast | M1 (joint - single @ 0.001) | M0 (joint - single @ 0) | delta (M1 - M0) |
|---|---|---|---|
| T1 | +0.148 | +0.046 | +0.102 |
| FLAIR | +0.042 | -0.029 | +0.070 |

Anchor gate (single arm at lambda_edge 0.001 vs registered GS-single, tolerance 0.02 dB): ca_s_t1_p3 T1 35.558 vs 35.558 (delta -0.000); ca_s_fl_p3 FLAIR 34.886 vs 34.886 (delta +0.000) -> PASS

Reproduce: `env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/ca_summarize.py`
(reads the six score jsons and the saved volumes; exits 1 if either anchor misses).

## 4. The three numbers per contrast

Head-only PSNR, dB, at the full precision of `results_arc/ca_summary.csv`:

| | T1 | FLAIR |
|---|---|---|
| J1 joint @ 0.001 (`hp_n36`) | 35.706153 | 34.928080 |
| S1 single @ 0.001 (`ca_s_t1_p3` / `ca_s_fl_p3`) | 35.557999 | 34.886404 |
| J0 joint @ 0 (`ca_j_edge0`) | 35.644656 | 34.875808 |
| S0 single @ 0 (`ca_s_t1_edge0` / `ca_s_fl_edge0`) | 35.598942 | 34.904343 |
| **M1** the published margin | **+0.148154** | **+0.041676** |
| **M0** shared geometry alone | **+0.045714** | **−0.028535** |
| **delta** what the common-edge prior buys | **+0.102439** | **+0.070211** |

In words: on **T1**, shared geometry supplies +0.046 dB of the published +0.148 dB margin (31 %) and the common-edge
prior supplies +0.102 dB (69 %). On **FLAIR**, shared geometry supplies **−0.029 dB** — it costs FLAIR — and the
common-edge prior supplies +0.070 dB, which is more than the whole published +0.042 dB margin because it first has to
pay back what shared geometry took. The percentage framing is not usable on FLAIR and is not given for it.

### Anchor gate

Both single-arm re-runs reproduce the registered GS-single values far inside the 0.02 dB tolerance:
`ca_s_t1_p3` = 35.55799935 against the registered 35.558 (delta −0.00000065 dB) and `ca_s_fl_p3` = 34.88640392 against
34.886 (delta +0.00040 dB). M1 = +0.148 / +0.042 reproduces the topic's registered margin **to the three decimals the
topic registers**, which is the resolution the comparison actually has: the reference values 35.558 / 34.886 are
themselves published to three decimals, so the T1 delta of −6.5e-7 dB is an artefact of the reference's own rounding
and the FLAIR delta of +0.0004 dB shows the real resolution (about 5e-4 dB). So the four cells come out of one code
tree and are comparable with each other and with the AF 16 record.

### Are these numbers above noise?

The spec pre-registered an indistinguishability band of **0.02 dB** (§4), which is the line's anchor tolerance and 20×
the measured seed floor. T1's M0 = +0.046 is 2.3× that band; FLAIR's M0 = −0.029 is 1.4× that band in magnitude.
Against the seed floor itself — repeated runs at the same recipe on this data differ by **0.001 / 0.000 dB**
(`T1FLAIR_AF16_v3_jiangyizhou/README.md:195`, read from that topic) — T1's +0.046 is about 46× the T1 floor, and
FLAIR's −0.029 is far above a floor that rounds to 0.000 dB at three decimals. **Neither M0 is rounding, and FLAIR's
negative sign is not rounding either.**

Two honesty qualifiers on that paragraph. The seed floor is imported from the AF 16 topic, where it was measured on
the joint arm at the phase-3 recipe; no seed repeat was run in this study, so the single arms' floor is assumed, not
measured. And each of the four λ_edge-0 and S1 cells is a single run.

### The other three metrics agree (not pre-registered)

The spec reports head PSNR; SSIM, HFEN and head NMSE are in the table above and were computed by the same summariser.
All three move the same way as PSNR in all four margins and in all four per-arm λ_edge effects. In particular the
FLAIR M0 is negative on every one of them — SSIM −0.00021, HFEN +0.00063 worse, head NMSE +0.00009 worse — so the
unflattering FLAIR cell is not an artefact of the PSNR gauge.

## 5. Numbers read from the topic READMEs, not measured here

Quoted as context. These were read from the result topics; this study did not re-run them.

| quantity | value | source |
|---|---|---|
| MC-GS − GS-single at AF 16, phase-3 recipe (= M1) | +0.148 / +0.042 | `T1FLAIR_AF16_v3_jiangyizhou/README.md:195` |
| the same margin at the phase-1 recipe (P6) | +0.037 / **−0.003** | same line |
| seed floor, repeat at the same recipe | 0.001 / 0.000 dB | same line |
| the same margin at AF 12, P13 → P14, when common-edge and phase TV were added | +0.066 / **−0.038** → +0.150 / **+0.036** | `T1FLAIR_AF12_v3_jiangyizhou/README.md:643` |

The AF 12 line is the precedent for what Stage 1 found on FLAIR: there too, before the common-edge term was added, the
joint arm was **behind** the single arm on FLAIR, and adding the term flipped the sign. That history is why a negative
or near-zero M0 on FLAIR was pre-registered as the expected case rather than as a failure.

## 6. Reading against the pre-registered rule (spec §4)

The decision table was written before the runs. Applied literally, with the arithmetic shown.

### T1 → row 2: `0.02 dB < M0 < 0.6 · M1`

```
M0        = +0.045714
0.02      = the pre-registered indistinguishability band
0.6 * M1  = 0.6 * 0.148154 = +0.088892

  0.02 < 0.045714      TRUE   (M0 is 2.3x the band)
  0.045714 < 0.088892  TRUE   (M0 is 0.31 x M1, below the 0.6 threshold)
```

Row 1 (`M0 ≥ 0.6 · M1 on both contrasts`) is ruled out on T1 by itself, before FLAIR is even considered: 0.045714 is
below 0.088892. Rows 3 and 4 do not apply (M0 is neither ≤ 0.02 nor < 0).

In the spec's own words, the reading is **"both mechanisms contribute"** and what the paper must say is
**"report the split; name R_edge as part of the method, not as a tuning detail."**

### FLAIR → row 4: `M0 < 0`

```
M0 = -0.028535    <  0          TRUE
                  <= 0.02       TRUE  (row 3's condition is satisfied trivially by a negative M0)
0.6 * M1 = +0.025005 ;  M0 = -0.028535 < +0.025005, so row 1 fails here too
```

Rows 3 and 4 are nested and FLAIR satisfies both; row 4 is the strictly more specific description and is the operative
row. **Both of their obligations bind.**

- Row 4, in the spec's own words: the reading is **"shared geometry costs that contrast, the prior pays it back"** and
  the paper must **"report it; this is the AF 12 FLAIR story (−0.038 → +0.036) and must not be hidden."**
- Row 3, in the spec's own words: **"on that contrast the margin *is* the common-edge prior"**, and the paper must
  **"say so. It is still cross-contrast modelling, but the mechanism sentence changes and the shared-geometry claim
  must be dropped for that contrast."**

So on FLAIR at AF 16 the shared-geometry claim is not available. Sharing one geometry between T1 and FLAIR, with the
common-edge prior removed, leaves the joint arm **0.029 dB behind** two independent clouds run under the same recipe.
The published +0.042 dB FLAIR margin exists because the common-edge prior contributes +0.070 dB on top of that
deficit. This is a real result of the study, is above the pre-registered band and above the seed floor, and is the
same sign flip the AF 12 line recorded at P13 → P14.

### What no row of the table covers

Row 1 is ruled out on both contrasts, so the current claim — "the gain is structural: one geometry, two densities" —
does not stand as written on either contrast. Stage 1 does not, however, license the opposite claim either: on T1 the
shared geometry does carry a third of the margin, well above noise, and on both contrasts the coupling prior is doing
cross-contrast work (§7.1). The spec's own summary of this outcome applies: none of these outcomes is a failure of the
study, and the one failure mode is running it and then not writing down the unflattering cell.

## 7. Two things in the cells that the decision table does not ask for

Both are arithmetic on the same four cells per contrast; both were checked against `results_arc/ca_summary.csv`.

### 7.1 The prior moves the two arms in **opposite directions**

Switching `R_edge` on, per arm, head PSNR in dB:

| | λ_edge 0 → 0.001 | effect |
|---|---|---|
| joint, T1 | 35.644656 → 35.706153 | **+0.061497** |
| joint, FLAIR | 34.875808 → 34.928080 | **+0.052272** |
| single T1 | 35.598942 → 35.557999 | **−0.040943** |
| single FLAIR | 34.904343 → 34.886404 | **−0.017939** |

The term helps the joint arm on both contrasts and **hurts both single arms**. SSIM, HFEN and head NMSE agree on all
four rows. This is the strongest pro-joint-modelling evidence Stage 1 produced, and it answers the reviewer's question
more directly than M0 does: a knob that improves a reconstruction when the two channels carry *different* contrasts,
and degrades it when they carry the *same* contrast, is behaving as a cross-contrast mechanism, not as a generic
regularisation gain that happened to be tuned on the joint arm. A knob that merely happened to be well-tuned for the joint
arm would have no reason to be actively harmful in the single arm, which runs the same code, the same recipe, the same
iterations and the same capacity.

That inference holds **on this subject, at this acceleration, with λ_edge sampled at two points**. The alternative a
referee will raise is that 0.001 is simply too strong for a differently-conditioned arm whose own optimum is above 0
but below 0.001 — in which case the single arm's loss would be a step-size effect rather than evidence that the term
is contrast-specific. Two sample points cannot exclude that; **Stage 2 is what closes it**, by measuring the shape of
each arm's curve in λ_edge.

What it does **not** show: it does not show that the effect needs *shared anatomy* rather than merely a second,
differently-conditioned channel. That is exactly what the spec's Stage 3 placebo (a real but grossly misregistered
flipped FLAIR) was designed to test, and Stage 3 is not run.

### 7.2 A free, partial equal-budget comparison — and it is lower than M1

Because λ_edge is actively bad for GS-single, quoting M1 credits MC-GS with a margin that is partly made of a λ value
that hurts the baseline. At **each arm's own best λ_edge of the two values Stage 1 measured** — joint at 0.001, single
at 0 — the margin is:

```
T1     35.706153 - 35.598942 = +0.107211   ->  +0.107 dB   (published M1 +0.148)
FLAIR  34.928080 - 34.904343 = +0.023737   ->  +0.024 dB   (published M1 +0.042)
```

The gap between the two is exactly what λ_edge costs the single arm: 0.148154 − 0.107211 = 0.040943 on T1 and
0.041676 − 0.023737 = 0.017939 on FLAIR, the two single-arm penalties in §7.1. **+0.107 / +0.024 is the honest
equal-budget number for this one knob** and should be reported beside M1.

It is **only one knob**, and this must be said in the same breath. λ_edge was searched over two points here (0 and
0.001), not optimised; and the spec's **Stage 4** — a search of the same size on the single arm over λ_geo, wavelet λ,
λ_ptv and the edge-TV `w_floor` / κ_q, the knobs that mean anything to a single-contrast method — is **not run**.
Stage 1 holds the recipe fixed for both arms, and the recipe itself was selected by joint performance. **+0.107 /
+0.024 does not substitute for Stage 4**, and quoting it as "the equal-budget margin" without that qualifier would be
the same overclaim in a smaller package.

## 8. A correction owed to the AF 12 topic README

`T1FLAIR_AF12_v3_jiangyizhou/README.md:652-654` explains that line's sign flip like this:

> The common-edge term is a symmetric cross-contrast gradient-direction prior: in a single-contrast run it compares an
> image with itself and is inert, so it lifts only the joint arm — which is how it flips the sign of joint − single.

Stage 1 measures that the premise is wrong. `R_edge` is **not** inert in GS-single: its two channels carry independent
densities, the term's value at iteration 150 is 4.06e-01 there (spec §2.1), and turning it on costs the single arm
0.041 dB on T1 and 0.018 dB on FLAIR (§7.1). The README's *conclusion* survives and is in fact strengthened — the term
does flip the sign of joint − single — but it does so by **two** mechanisms, lifting the joint arm and lowering the
single arm, and the second one is the one a reviewer will ask about, because it is the reason M1 overstates the
equal-budget margin.

The premise is false in the AF 12 line too, not merely in this one. That line's GS-single uses the same
duplicated-channel construction: `configs/t1flair_v3_af12_hyw_singleT1.yaml` and `…_singleFLAIR.yaml` put the same
contrast in both the `plain` and `contrast` slots, with identical `name`, `data_path` and `csm_path` in each — so
there too `R_edge` compares two independently-densitied reconstructions rather than an image with itself, and is not
inert. What has not been measured is the *size* of the AF 12 single-arm penalty; only the AF 16 penalty
(0.041 / 0.018 dB) is measured here. This record does not edit the AF 12 topic; the sentence should be corrected when
that topic is next touched.

## 9. What this does not cover

The spec's §5 caveats apply unchanged:

- **One subject, one mask, one acceleration.** Stage 1 is jiangyizhou at AF 16 only. The AF 12 line is where the FLAIR
  sign flip happened, so a repeat of Stage 1 at AF 12 is the obvious extension and is not in this design.
- **The peds counter-example is evidence already and is not re-run.** `Peds_TSC_MTP_AF12_retro/README.md:398`: T1w +
  T2* are the two least similar contrasts of the ten (r = 0.57), MC-GS is not ahead there, and Joint SENSE moves the
  same way — a cross-family dose-response that tuning does not explain. It belongs in the rebuttal as-is.
- **Noise.** The frozen recipe already fails at 2σ₀ (`T1FLAIR_AF16_noise2x_v3_jiangyizhou`). Attribution is measured at
  1σ₀ here.
- **Lesion contrast.** The T1+FLAIR subjects are healthy volunteers, as in the hparam study.

Specific to Stage 1, on top of those:

- **λ_edge takes two values here, 0 and 0.001.** The shape of the margin in λ_edge is unmeasured. A margin that peaks
  exactly at the tuned 0.001 and falls off on both sides is the signature the reviewer is worried about, and Stage 1
  cannot see it; that is Stage 2.
- **Both mechanisms tested are still joint modelling.** Neither M0 nor delta tests whether the gain needs shared
  *anatomy* or merely a second well-posed channel; that is Stage 3.
- **The recipe was selected by joint performance and is held fixed for both arms.** §7.2 removes that asymmetry for
  one knob only; the strict form of the reviewer's question is Stage 4.
- **Single runs, magnitude gauge as the primary.** No seed repeat in this study (the floor is imported from the AF 16
  topic); complex-gauge head PSNR is in the table for all eight cells and was not used for any reading above.

## 10. One paragraph, for the paper

At AF 16 on jiangyizhou, the published MC-GS − GS-single margin of +0.148 / +0.042 dB splits into shared geometry
(+0.046 / **−0.029** dB) and the common-edge cross-gradient prior (+0.102 / +0.070 dB). On T1 both mechanisms
contribute and the split should be reported, with `R_edge` named as part of the method rather than as a tuning detail.
On FLAIR the shared-geometry claim must be dropped: one shared geometry costs FLAIR 0.029 dB at this acceleration, and
the margin exists because the coupling prior more than pays that back — the same sign flip the AF 12 line recorded when
the term was added. The prior nonetheless behaves as a cross-contrast mechanism rather than a tuning artefact: it
improves the joint arm on both contrasts and degrades both single arms, which run the identical recipe with one
contrast in both channel slots — shown here at two λ_edge values on one subject, so a step-size explanation is not
yet excluded and the dose-response of spec Stage 2 is what would close it. Because λ_edge is bad for the single
arm, the equal-budget margin for that one knob is +0.107 / +0.024 dB rather than +0.148 / +0.042; a full
equal-budget search on the single arm (spec Stage 4) has not been run.
