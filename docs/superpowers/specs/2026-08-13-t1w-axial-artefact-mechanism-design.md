# Why the T1W axial view carries the most visible artefact — mechanism, not adjudication

Date: 2026-08-13
Status: design approved, not yet implemented
Branch: `diag-t1-artefact`

## 1. The question

The user, looking at `results_arc/LOOK_T1_axial.png`: why does T1W show so much aliasing
artefact, *especially* in the transverse (axial) view?

## 2. Scope, and what this explicitly is NOT

**In scope.** A reproducible, numeric explanation of the two asymmetries in the observation:
why T1W rather than FLAIR, and why the axial view rather than coronal or sagittal.

**Out of scope, deliberately.**

- **Re-adjudicating the cause.** §16 adjudicated four pre-registered propositions on this exact
  image and left P1 (white-matter speckle = residual incoherent aliasing) at `insufficient`.
  This work *accepts* the undersampling-residual reading as its premise and explains its
  distribution. It therefore cannot be cited as evidence that the speckle IS aliasing; that
  question stays open where §16 left it. The new `e_unobs` oracle could settle it, and doing so
  was offered and declined for this round.
- **Reducing the artefact.** The knob directions were closed the same day (memory §18): `tv_eps`
  is a pure trade and `lambda_geo` cannot translate it. Nothing here proposes a new working point.

**Deliverable size (user decision):** two figures plus a memory section. One script, no new
`diag_artefact/` module, no unit tests. This is an accepted trade recorded here rather than
silently taken.

## 3. The three claims

Each claim carries the measurement that produces it and the result that would refute it.

### C1 — Undersampling moves no information along S-I. (Geometry; already verified)

The shared mask is bit-constant along the readout axis, so the undersampling operator factorises
as `M(k0,k1) ⊗ 1(k2)` and its point-spread function is `psf2d ⊗ δ(x2)`. Coil combination is a
per-voxel weighted sum (ESPIRiT maps with SoS ≡ 1) and the magnitude is per-voxel, so neither step
transports information along S-I either. Consequently, in the zero-filled pipeline, perturbing one
voxel changes only its own axial slice. This is exact, not approximate.

*Measurement.* Off-plane energy fraction of `ifft3c(mask)`.
*Refuted if.* Off-plane energy > 1e-12.
*Status.* Measured **0.0** exactly, with `mask == mask[:, :, c:c+1]` bitwise true. **Passes.**

### C2 — The axial view is the only view that puts both fast-varying axes in the display plane

The axial image plane is spanned by axes (0, 1) — both phase-encode axes. The coronal plane is
(1, 2) and the sagittal plane is (0, 2); each contains one PE axis and the fully-sampled readout
axis, along which the undersampling injects no structure at all.

*Measurement.* In-plane high-pass fraction, per view `v`:

```
e   = rec - gt                       # both stored as magnitude float32; signed difference
A_v = RMS(e - blur_v(e, sigma=2)) / RMS(e),   both RMS taken over brain voxels only
```

`blur_v` is a 2-D Gaussian applied **only within the two axes spanning view v** (never along the
slicing axis), computed on the full volume and only then restricted to the brain mask, so the
filter never sees a mask edge. The brain mask is `diag_artefact.volumes.brain_mask(gt)` of the
same contrast. `A_v` is the share of error energy that varies
rapidly *within the displayed plane* — which is what the eye reads as artefact texture, as
opposed to slow in-plane variation, which reads as blur or shading.

Computed for the full grid: `{zf, rec} x {T1, FLAIR} x {axial, coronal, sagittal}` = 12 values.

*Predicted.* `A_axial > A_coronal` and `A_axial > A_sagittal`, for `zf` on both contrasts.
*Refuted if.* Either inequality reverses. Then "axial is worse" is not a geometric consequence,
the explanation collapses, and the refutation is what gets reported.

**Pre-declared alternative outcome.** The reconstruction uses 3-D Gaussians and a 3-D TV
regulariser; both couple across S-I. The prior may therefore isotropise the residual, giving the
predicted anisotropy in `zf` and little or none in `rec`. That is not a failure — it is the
sharper finding ("geometry sets the direction in which aliasing is injected; the prior smears it
across that direction"). Figure 1(c) places `zf` and `rec` side by side precisely so this outcome
cannot be quietly dropped.

### C3 — Equal relative extrapolation failure times a larger unmeasured base

`e_unobs` (oracle complex multicoil error on truly unobserved k) reads T1 0.6443 vs FLAIR 0.6467 —
essentially equal. The unobserved-k energy reads T1 102397 vs FLAIR 69207, a factor **1.48**.
Equal relative failure on a 1.48x larger base gives a residual-energy ratio of **1.47**. The
algorithm is not worse at T1; T1 simply leaves more energy for it to guess.

*Measurement.* Read from `results_arc/autoresearch_af12/tvs_e040_kspace.json`, with source path
and SHA echoed into the output record.
*Refuted if.* The two `e_unobs` values differ materially, in which case the mechanism must be
rewritten as "T1 is harder to extrapolate" rather than "T1 has more left to extrapolate".

## 4. Gauge discipline

Two traps this project has already been caught by twice (memory.md §3, `.claude/memory.md` §14c):

1. **The multicoil unobserved-k quantities in C3 may only come from the checkpoint replay.** The
   NPZ holds magnitude volumes; re-encoding them discards phase and cannot reproduce a valid
   unobserved-k quantity. The AF12 round-2 protocol already forbids this. The NPZ is used only for
   descriptive magnitude-image statements (the radial spectrum), and every panel states which of
   the two it is showing.
2. **Every T1-vs-FLAIR panel uses a single shared normaliser**, stated in the caption (T1's GT
   peak). Per-volume `@max` normalisation inflates T1 by ~1.6 dB and manufactures the opposite
   conclusion.

Error maps follow the house convention: `jet`, fixed range [0, 0.1], never autoscaled.

## 5. Figures

**Figure 1 — `results_arc/t1_axial_why_view.png`**

- (a) Mask PSF: in-plane `psf2d` on a log scale (diffuse blue-noise sidelobes rather than discrete
  replicas), beside the S-I energy profile showing a single line at x2 = 0 and exact zero
  elsewhere. Note this is a property of the **mask**, not of the reconstruction: it is consistent
  with §8b's "incoherent grain, not coherent wrap-around", but it does not test §8b, which is a
  claim about the recon residual.
- (b) The same T1 error volume in all three views, `jet` [0, 0.1], each annotated with its `A_v`.
- (c) Bars: `A_v` for `{zf, rec} x {T1, FLAIR} x 3 views`. The falsifiable core of C2.

**Figure 2 — `results_arc/t1_axial_why_contrast.png`**

- (a) Radial k-space energy of T1 vs FLAIR (**labelled: magnitude-image spectrum**) with the
  mask's radial sampling density on a twin axis.
- (b) Bars: unobserved-k energy, `e_unobs`, residual energy, T1 vs FLAIR (**labelled: complex
  multicoil, checkpoint replay**).
- (c) T1 and FLAIR axial error maps side by side under the single shared normaliser.

Slice indices are fixed by rule, not by eye, and recorded in the output JSON: axial z = 160
(matching `LOOK_T1_axial.png`); the coronal and sagittal indices are the axis-0 and axis-1
coordinates of the brain-mask centroid **within that axial slice**, so all three displayed planes
intersect in one point of the same anatomy.

## 6. Implementation and outputs

One script, `scripts/fig_t1_axial_mechanism.py`, reading:

- `results_arc/tvs4k_e040_volumes.npz` (magnitude volumes: rec/gt/zf, both contrasts)
- `/data0/congcong/code/GS/3DGS/GS_TMI/3DGS/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat`
- `results_arc/autoresearch_af12/tvs_e040_kspace.json`

producing the two PNGs and `results_arc/diag_artefact/t1_axial_mechanism.json` containing every
number, the fixed slice indices, the input paths with SHA-256, and the C1/C2/C3 verdicts.

Brain masking reuses `diag_artefact.volumes.brain_mask`; no new measurement module is added.

## 7. Verification criteria

1. C1 is a **hard assertion** in the script (off-plane energy < 1e-12). A failure means the
   implementation contradicts the algebra, which is a bug and should stop the run.
2. C2 is **recorded, never asserted**. A reversal is a finding, not a crash; the JSON records
   `verdict: refuted` and the memory section says so.
3. C3's numbers are read from the replay JSON with provenance echoed; they are not recomputed
   from magnitude data.
4. A memory section (§19) records all three claims with their numbers, and — if C2 is refuted or
   the prior turns out to isotropise the residual — records that instead of the predicted story.

## 8. Accepted limitations

- No unit tests and no new module; the script's own C1 assertion is the only automated check.
  Proportionate to the "understand it and file it" scope, but it means `A_v` has no test pinning
  its behaviour on synthetic inputs, unlike SEAC or selectivity.
- `A_v` is a new index and this project's record with new indices is poor: twelve pre-registered
  bounds and two normalised metrics have already failed on contact with reality (§16c, §14c, §18).
  `A_v` is used here only for an **ordering between views of the same volume**, never across
  acquisitions or contrasts on an absolute scale, which is the comparison that has repeatedly gone
  wrong. It should not be reused outside that ordering without a gauge check.
- The explanation is conditional on the premise in §2. If P1 is later settled the other way — that
  the white-matter speckle is not aliasing at all — C2 and C3 still describe where the
  undersampling residual goes, but they stop being an explanation of what the user is looking at.
