# results_crisp — what must travel with the numbers

This file is tracked in git (unlike `.superpowers/sdd/**`, which is blanket-gitignored and
disappears on merge) specifically so the caveats below survive alongside the committed JSONs.
Anyone citing a number from this directory should read the entry for that number first.

## G2 — voxel-domain joint-TV gate (`g2_voxel_huangyuwei.json`, commit `10c04be`)

**Result:** joint − independent = **−0.088 dB** (same-mask) and **−0.098 dB** (complementary),
i.e. no gain in either condition. The pre-registered prediction (same-mask gain ≈ 0,
complementary shows a real gain) is **refuted** — the verdict was declared to rest on the
complementary arm, and that arm shows no gain either.

Caveats (do not cite the headline numbers without these):

- **n = 1 subject / 1 seed / 1 acceleration factor.** This is not the pre-registered 5-subject
  protocol; it is a single-instance gate check.
- **The bound is on vectorial-TV coupling as the sharing mechanism, not on "shared information"
  generally.** The only coupling tested is a shared gradient-magnitude normaliser (edges tend to
  occur at the same voxels across contrasts); it carries no intensity relationship and no learned
  prior. A null result here does not license a null result for shared-information methods in
  general.
- **Both arms sit at E_unobs 0.645–0.68, far from any ceiling.** Two-thirds of the unobserved
  k-space energy is still wrong at the best operating point for either arm. This experiment
  excludes a *large* joint-TV effect much more strongly than it excludes a small one.
- **λ does not transfer from the 64³ phantom** — the optimum found here is roughly 2.5 decades
  away from where the phantom-derived starting grid assumed it would be; at the phantom's
  optimum, the reconstruction on this real data actually *loses* to zero-fill.
- **The secondary endpoint (brain PSNR) is reported at the E_unobs-optimal λ, not the
  PSNR-optimal λ.** At each arm's own PSNR optimum the gap is **−0.025 dB** instead of −0.098 dB
  (same sign, smaller magnitude — the qualitative conclusion is unchanged, but the two numbers
  are not interchangeable).

## G0 — representation-ceiling gate (`g0_ceiling_huangyuwei_{T1,FLAIR}.json`, commit `8662e1a`)

**Result:** Gaussian `E_coil_full` 0.3667 (T1) / 0.2535 (FLAIR) against a voxel (CG-SENSE)
reference of 0.2843 / 0.1830 — ratios **1.29×** and **1.39×** against the pre-registered ≤2×
line, at 200k primitives. **PASS** on both contrasts.

Caveats (do not cite the headline numbers without these):

- **The reference floor is noise plus CSM model error, not voxel-grid representation power.**
  Real 12-coil k-space is not perfectly consistent with any single coil-combined image under a
  rank-1-per-voxel CSM model even at full sampling, which is why the reference `E_coil_full` is
  well above zero (0.18–0.28) rather than near machine precision. The error attributable to the
  *Gaussian representation itself*, after removing this floor in quadrature, is
  `sqrt(0.3667² − 0.2843²) ≈ 0.23` (T1) and `sqrt(0.2535² − 0.1830²) ≈ 0.18` (FLAIR) of the data
  norm — **1.29×/1.39× must not be read as "the representation is nearly free."**
- **Coil-domain PASS is not visual parity.** Image-domain PSNR is 27.5 dB (T1) / 25.5 dB
  (FLAIR); the Gaussian reconstruction is visibly blurred next to the near-perfect GT/zero-fill
  pair (mask ≡ 1 here, so zero-fill == GT). "Passes the aggregate coil-domain ceiling" and
  "matches voxel-grid image quality" are different claims — only the former is established.
  See `g0_ceiling_huangyuwei_{T1,FLAIR}_panel.png` (not committed, regenerable).
- **Training had not plateaued at 4000 iterations.** Training loss was still slowly decreasing
  at the end of the run, not flat. 4000 iterations matches `winner_v2_aligned.yaml`'s
  `max_iterations`, not an independently verified plateau.
- **200k primitives was never swept toward 1e6.** This establishes that 200k *suffices* to
  clear the pass line at the fixed ≤1e6 gate; it does not establish where a ceiling (if any)
  sits as primitive count increases. The finding is "200k suffices," not "here is where the
  ceiling sits."
- **The Gaussian arm ran with the shipped winner config's regularisers** (`tv_weight 0.06`,
  `kspace_freq_adaptive true`, centre weight 0.3 / high weight 2.2 — see
  `configs/g0_fullsample.yaml`), **not** the plan's Task 8 Step 4 settings
  (`--lambda_geo 0.0 --kspace_high 1.0`). This deviation was never amended against the written
  plan. Its direction is conservative: the winner config's regularisers were tuned for
  undersampled reconstruction, not this full-sampling capacity probe, so if anything they
  handicap the Gaussian arm here — and a handicapped arm still passed. **1.29×/1.39× is
  therefore a lower bound on the representation's capability, not its ceiling.**

## Cross-experiment warning — G2 and G0 are not on a common scale

G2 loads CSM via `crisp.io.load_trainer_mat` (the trainer `.mat`'s embedded `map` field). G0
deliberately does **not** — it loads the ESPIRiT CSM (`csm_espirit_{T1,FLAIR}.mat`), the same
file the trainer config's `csm_path` points at. The two CSM estimates differ by **111% relative
Frobenius norm** (verified empirically; see `crisp/experiments/g0_ceiling.py`'s module
docstring). G2's `E_unobs` (≈0.65 at optimum) and G0's `E_coil_full` (≈0.28) are therefore
computed under **different forward operators** (`A = FS` differs between the two experiments)
and are not comparable on a common scale. **Do not compute a "gap to the ceiling" by subtracting
or ratio-ing a G2 number against a G0 number.**

## Void artifacts — commits `782b3e9` and `e7bbbf6`

These two commits carry G2 numbers that were later voided: `782b3e9` is a preliminary run
explicitly marked "PRELIMINARY RUN INADMISSIBLE, do not cite" in its own commit message (a
2-lambda × 50-iteration timing probe, not a completed sweep, plus a since-retracted apparent
FISTA-divergence finding). `e7bbbf6` is the sweep that superseded it, itself later found to use
a lambda grid that never bracketed the true optimum (every cell in that sweep lost to
zero-fill) and was itself superseded by the sweep actually reported in `g2_voxel_huangyuwei.json`
(commit `10c04be`). **Numbers from `782b3e9` and `e7bbbf6` are void — do not cite them.**

`results_crisp/g2_voxel_huangyuwei.png` and `results_crisp/diagnostics/{anatomy_grid,diff_grid}.png`
were leftover images from those discarded runs (stale filenames that don't match the current
script's `_panel.png`/`_curves.png`/shard-JSON naming convention) and have been deleted.

## CE (contrast-enhancement pair check)

Parked **INCONCLUSIVE** by human ruling — see `results_crisp/ce_check.json`'s `verdict_status`
field and the prominent block at the top of `crisp/experiments/ce_check.py`'s docstring for the
full text. In short: criterion (b) (`positive_fraction < 0.15`) measures the fraction of
positively-signed voxels rather than the concentration of positive difference energy, so genuine
focal enhancement leaves ~50% positive fraction (this run measured 0.546) and the criterion has
no discriminative power; the difference image is additionally confounded by inter-scan motion (7
minutes apart, a saturated ring artifact along the whole scalp contour); and the bundled CSM has
near-zero sensitivity over 1/3–1/2 of the brain. **Do not cite `ce_check.json`'s `verdict` field
(REPEAT)** — it is not admissible evidence pending external DICOM SeriesDescription /
ContrastBolusAgent confirmation. The criterion was deliberately left unchanged rather than
retro-fitted after seeing the data.
