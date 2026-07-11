# Layer 1 validation — shared anatomy latent + response decoders (§5–§6)

**Claim under test (§3.2 → §6).** The original MC-GS represents the two contrasts with two
*independent* per-Gaussian density tables — which §3.2 critiques as "a shared spatial
dictionary with independent coefficients", not a genuine *shared anatomy → contrast-specific
response* model. Layer 1 replaces those tables with a per-Gaussian anatomy latent `z_A` and a
tiny shared decoder `D_ψ(z_A, e_m)` that emits each contrast's density from the **same**
latent. The Layer-1 requirement: this coupling costs **no** representational power — the
decoder must match or beat independent tables at matched geometry/iterations, on both pairs.

## Setup (controlled comparison)
Both arms share everything except the density parameterization:
- shared ACS structural init (§13-S1: `E_A = √(E₁²+E₂²)` → importance-sampled positions),
- 200 000 Gaussians, geometry **frozen** (Stage 2 only — no densification, no phase/change/deform),
- 800 iterations, identical LRs, §12 noise-normalized DC + geometry reg only.

Arm A = `decoder_enable=False` (independent tables, = §17 ablation row 2, "原 MC-GS").
Arm B = `decoder_enable=True` (response decoder, = §17 ablation row 3, **Layer 1**).

## Metric — GT-free held-out k-space NRMSE
The CE pair is clinical data with **no fully-sampled reference**; its stored "ground truth"
is the zero-filled reconstruction of the *acquired* (undersampled) k-space. A full-reference
PSNR against that is circular — it rewards a **null** model (a zero reconstruction scores
~58 dB / SSIM 1.0 because hard-DC just replays the acquired samples). So the primary metric
is a **held-out k-space NRMSE**: 10 % of the acquired samples are reserved, never fit, and

    NRMSE_ho = ‖ M_test (F(S x) − y) ‖ / ‖ M_test y ‖

is measured on that held-out set. It cannot be gamed by data-consistency replacement, a null
model scores ≈ 1.0, and a perfect model scores 0 — monotone with genuine reconstruction
quality, and valid on **both** pairs. (Full-reference PSNR/SSIM is additionally reported for
T1/FLAIR only, where the GT is genuinely fully-sampled.)

## Results

| Pair | Contrast | Indep tables (row 2) | Response decoder (row 3) | Δ | full-ref PSNR/SSIM (decoder) |
|---|---|---|---|---|---|
| T1/FLAIR | T1    | 0.5485 | **0.5483** | −0.0002 | 28.13 dB / 0.646 |
| T1/FLAIR | FLAIR | 0.8364 | **0.8332** | −0.0032 | 25.68 dB / 0.655 |
| CE | pre  | 0.3548 | **0.3527** | −0.0021 | — (GT-free) |
| CE | post | 0.3361 | **0.3336** | −0.0026 | — (GT-free) |

*(held-out k-space NRMSE, lower = better; null model ≈ 1.0)*

**Verdict — 4/4 contrasts: decoder ≤ independent tables.** The response decoder matches
independent tables on T1 and is marginally better on FLAIR, pre, and post. A second signal:
at initialization the decoder's held-out NRMSE is ≈ 1.00 (a clean near-zero output) whereas
independent random tables sit at 1.03–1.23 (they inject broadband noise) — the low-capacity
decoder is a better-behaved starting point as well as an equal endpoint.

**Interpretation.** Coupling both contrasts through one shared anatomy latent + a 516-parameter
decoder (for 200 k Gaussians — far too small to memorize either image, §6.2) sacrifices no
representational power versus the independent tables the spec critiques, and does so on a real
second pair (pre/post-contrast) as well as T1/FLAIR. This is the Layer-1 result: §17 ablation
row 3 established, row 2 reproduced as its in-model baseline.

**Scope / what these numbers are not.** These are Stage-2-only numbers with frozen ACS-init
geometry and no phase, change, or deformation modules — they are deliberately far below the
fully-trained MC-GS winner-v2 (T1 37.7 dB / FLAIR 38.1 dB, 4000 iters with densification and
geometry optimization). Layer 1 validates the *response-decoder substitution in isolation*;
absolute quality is the job of Layers 2–4 (audited change, phase, deformation) and full-length
training. The bit-for-bit MC-GS baseline remains available as the `paired_gs` package (§17 row
1/2 anchor).

Artifacts: `results_arc_layer1/ablation_layer1.json` (raw), `layer1_ablation.png` (figure).
