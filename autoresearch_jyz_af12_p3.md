# autoresearch phase 3 — jiangyizhou T1+FLAIR at AF 12: MC-GS first over EVERY arm, GS-single included

_Opened 2026-09-13. Same Karpathy loop. Runner `autoresearch_jyz_af12_p3.sh` (P13 baked in, arena = the all-AF12
config), ledger `autoresearch_jyz_af12.tsv` continued with the `p3_` prefix, branch `autoresearch-jyz-af12-phase3`
(from `autoresearch-jyz-af12-phase2` at 721b80e), worktree `/data0/congcong/code/GS/mcgs_af12_wt`._

## Why this phase exists (user decisions 2026-09-13)

1. **GS-single counts as a comparison method**, not as our own ablation. Under that gauge phase 2 leaves T1 done
   (first over every arm on 335/335 full-brain slices) and **FLAIR open**: MC-GS is 0.037 dB behind GS-single,
   flat in sampling (−0.030 at 9.05x, −0.038 at 9.74x, −0.035 at 10.56x — the user's FLAIR-at-10.5 experiment
   falsified the "less data makes the coupling pay" hypothesis), and behind on 310 of 332 slices.
2. **Borrow from the AF16 line.** Its phase-3 recipe (`autoresearch_af16_p3.md`, another session, branch
   `autoresearch-jyz-af16-phase3`) is *our P10* plus two terms we do not have, and at AF16 those two terms are
   exactly what turned joint − single positive on both contrasts (T1 +0.037 → +0.148, FLAIR −0.003 → +0.042).

## The two borrowed levers, and why our own sweeps missed them

- **Common-edge prior `--lambda_edge` at 0.001.** A symmetric cross-contrast gradient-direction prior. `pair_prior.py`
  and the trainer's call site are **byte-identical on both branches** (verified by diff against the shared ancestor
  3a44aba), so the AF16 λ transfers without rescaling. The AF16 bracket: 0.0005 slightly worse, **0.001 ≈ 0.002 best**,
  0.005 collapses (−0.12 dB, every cell worse). **Phase 1 here tested only 0.02 — twenty times past that optimum — and
  recorded "loses by 1.33 dB". That verdict is about the strength, not the term.** It is the only coupling term that
  helped both contrasts at AF16, which is what makes it the candidate for our FLAIR problem.
- **Phase-only TV `--lambda_ptv` around 0.01–0.03.** Not in this branch's code; port from
  `autoresearch-jyz-af16-phase2` (`paired_gs/losses/phase_tv.py`, the trainer hook, the cycle.py flag). Phase 1 here
  dismissed it on the AF16 **phase-2** result (0.05 costs 0.24 dB of T1). The AF16 phase-3 dose curve on their db4/1-level
  base is gentler and is **positive for FLAIR**: 0.01 → T1 −0.005 / FLAIR +0.019; 0.02 → −0.033 / +0.013; 0.03 →
  −0.05 / sharpness gates met; 0.05 → −0.182 / −0.069. Our sharpness baseline differs from theirs, so the operating
  point must be re-measured here, not copied.

## Already closed by the AF16 line — do not spend a run on these

Contrast-private residual Gaussians (0.3M for FLAIR): **null**, +0.005 / +0.015 with every other cell unchanged —
"the shared cloud already carries everything FLAIR needs". This retires the structural fix proposed in chat before
the AF16 program was read. Also closed there: non-local RED with a BM4D target (three settings, only smooths),
adaptive density control (both split scores, null), vectorial TV (smooths). Plus everything phase 1 and 2 closed here.

## Arena and gauge

All arms at AF 12 (`configs/t1flair_v3_af12_hyw.yaml`); design B and the actual-AF-10.5 FLAIR mask are side studies
and take no part in selection. Parent **P13** at 4000 iterations: T1 36.969 / FLAIR 36.310. The arms to beat, per
head cell, are every non-MC-GS arm **including `gs_single_p2`** (T1 36.903 / 0.9495 / 0.0968, FLAIR 36.347 / 0.9408 /
0.0943) and L1-ESPIRiT (36.426 / 0.9469 / 0.1036, 35.757 / 0.9368 / 0.1000). So today:

| head cell | T1: P13 / best other | margin | FLAIR: P13 / best other | margin |
|---|---|---|---|---|
| PSNR | 36.969 / 36.903 (GS-single) | +0.066 | 36.310 / **36.347 (GS-single)** | **−0.037** |
| SSIM | 0.9504 / 0.9495 (GS-single) | +0.0009 | 0.9401 / **0.9408 (GS-single)** | **−0.0007** |
| HFEN | 0.0948 / 0.0968 (GS-single) | +0.0020 | 0.0956 / **0.0943 (GS-single)** | **−0.0013** |

**Objective: all six margins above zero at 4000 iterations, i.e. FLAIR's three cells turned positive without losing
T1's three.** The single number the loop maximises is the smallest of the six.

## Decision rules (pre-registered before the first candidate ran)

1. *Reference rows first*: `p3_P13_i1500` (parent at the screening budget in this arena) and `p3_P13_i1500_s43`
   (seed floor). No candidate is read before both exist.
2. *Keep* at 1500: FLAIR head PSNR up by more than its floor, AND no other of the six cells down by more than its
   own floor, AND neither sharpness down by more than 0.01. A FLAIR gain bought with T1 is a `reshape`, not a keep.
3. *Promote* at 4000 with volumes, same rule against `p3_P13_full4k` (the parent at 4000 = the existing
   `p2_P13_af12_full4k`, T1 36.969 / FLAIR 36.310).
4. **The verdict needs its own GS-single.** A promoted recipe is only judged against GS-single after both
   single-contrast arms are re-run at that recipe (rule 5 of the earlier phases), because `lambda_edge` is inert in a
   single-contrast run (a contrast compared with itself) while `lambda_ptv` is not.
5. Two-seed averages and figures come after a promotion, never inside the loop.
6. Whole-volume SSIM reported, never optimised.

## Round 0 — queued 2026-09-13, before any candidate ran

| exp | what | args | card |
|---|---|---|---|
| `p3_P13_i1500` | parent reference at 1500 | (none) | 0 |
| `p3_P13_i1500_s43` | seed floor at 1500 | `--set training.seed=43` | 1 |
| `p3_edge0001` | common-edge 0.001 (the AF16 optimum) | `--lambda_edge 0.001` | 3 |
| `p3_edge0002` | common-edge 0.002 (the other end of their plateau) | `--lambda_edge 0.002` | 4 |

Then, on the read: `p3_edge00005` / `p3_edge0005` to bracket ours independently of theirs, and the phase-TV port with
0.01 / 0.02 / 0.03 screens. Combinations of the keeps at 1500, then one promotion at 4000, then the two GS-single arms.

## Round log

### Port — phase-only TV, 2026-09-13 17:45
`paired_gs/losses/phase_tv.py` copied verbatim from `autoresearch-jyz-af16-phase2`; trainer hook and the
`--lambda_ptv` flag re-written here in the same shape. `tests/test_phase_tv.py` (6 tests) pins what the term is for:
zero on a real volume, zero (to the module's own 1e-12 floor) on any constant-phase volume however rough its
magnitude, positive on a phase ramp, invariant to a global phase rotation, linear in |x| at fixed phase, finite
gradients. **Two test corrections, disclosed**: the first draft asserted exact zeros and 1e-5 invariance; measured,
the term carries a float32 cancellation residue of 1.1e-4 on a constant-phase volume (against 2.3e-1 on a realistic
phase ramp, so 2000x below the signal) and a 1e-4 relative wobble under global rotation, both of which vanish in
float64 (1.7e-11) — the tolerances now match the measurement and a float64 assertion pins that the formula itself is
exact. **Default path bit-identical**: with `lambda_ptv` unset the first iteration prints
`loss=1.2371e-01 dc_p=7.1636e-02 dc_c=5.0637e-02 R_geo=6.5341e-03 R_wav=7.8751e-03`, the same digits as the
pre-port reference run `p3_P13_i1500` launched at 17:36.

### Round 0 read — 2026-09-13 18:02
References at 1500 in this arena: `p3_P13_i1500` T1 36.948 / 0.9505 / 0.0949, FLAIR 36.233 / 0.9394 / 0.0962,
sharp 0.857 / 0.822. Seed 43: 36.937 / 36.228. **Seed floor at 1500: PSNR 0.011 / 0.005, SSIM 0.0001 / 0.0001,
HFEN 0.00001 / 0.00002, sharp 0.0003 / 0.0004.**

`p3_edge0001` (common-edge 0.001): **36.997 / 36.275** — T1 **+0.049**, FLAIR **+0.041**, SSIM +0.0005 / +0.0005,
HFEN better by 0.0008 / 0.0012, sharpness −0.002 / −0.003 (inside the 0.01 tolerance). **KEEP, and it lifts both
contrasts**, the size the AF16 line reported (+0.056 / +0.051) — the term behaves the same at both accelerations.
Phase 1's verdict on this lever ("loses by 1.33 dB") was a verdict on λ = 0.02, twenty times the optimum.
FLAIR still has 0.037 dB to find before it passes GS-single, and full-budget gains here have been running at about
half the screening ones, so the promotion and its own GS-single arms decide it, not this row.

### Round 1 — queued 18:05, before the runs started
Bracket the lever on our own data rather than inheriting the AF16 bracket, and open the phase-TV curve in parallel:

| exp | args | why |
|---|---|---|
| `p3_edge00005` | `--lambda_edge 0.0005` | lower end of their plateau |
| `p3_edge0005` | `--lambda_edge 0.005` | where theirs collapsed; confirms the optimum is ours too |
| `p3_ptv001` | `--lambda_ptv 0.01` | phase-TV dose curve, the point that was free at AF16 |
| `p3_ptv002` | `--lambda_ptv 0.02` | the point where FLAIR gained and T1 paid 0.03 there |
| `p3_ptv003` | `--lambda_ptv 0.03` | their chosen operating point |
Read by rule 2 against `p3_P13_i1500` with the floor above. Then the combination of the keeps, one promotion at
4000, and the two GS-single arms at that recipe.
