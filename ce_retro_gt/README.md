# Retrospective validation with ground truth — UCSF-PDGM pre/post-contrast

> New session: this topic exists because the prospective UIH pair (`../CE_pre_post_contrast`) has no fully sampled
> reference, so nothing there can be scored against truth. Read that topic's `STATUS.md` first for the claims this
> one is meant to test.

## Why this data

UCSF-PDGM (501 cases, `/data2/congcong/data/FM_data/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5`) gives
co-registered T1 and T1c at 1 mm isotropic on a 240 x 240 x 155 grid, with a tumour segmentation whose labels are
exactly the two situations the prospective pair can only ask about qualitatively:

- **label 4, enhancing**: the arm must keep the uptake it is given;
- **label 1, necrotic core**: it does not enhance, so any uptake an arm reports there is invented -- the same
  situation as the non-enhancing lesion of the prospective pair;
- label 2 is oedema, not used for scoring.

Eight cases span the range: 0004 (enhancing 15.7 mL), 0012 (48.9), 0017 (24.3), 0023 (8.8, the smallest), 0026
(19.2), 0031 (3.7, a thin rim with a 45 mL necrotic core), 0041 (32.5) and 0053 (64.6, the largest). The first four
were picked by enhancing volume alone; the second four were additionally screened for a genuinely enhancing label 4
and a genuinely non-enhancing core. 0031's label 4 turns out to enhance by only -0.046 in white-matter units, so it
cannot score "uptake kept" and is excluded from that average (the scorer marks it).

## The simulation (`scripts/retro_sim.py`)

One case becomes the same kind of acquisition as the UIH pair:

| | |
|---|---|
| object | T1 / T1c divided by the median of tumour-free white matter, so enhancement is in the same units as the prospective work |
| phase | a smooth synthetic phase per scan (different fields), so no arm can exploit real-valued images |
| coils | 24 loops on a cylinder, near-field fall-off and a spatially varying phase, sum-of-squares 1; the same maps for both scans |
| sampling | readout axis 0 fully sampled; the phase-encode plane gets the UIH recipe -- 24 x 24 fully sampled centre, elliptical shutter, uniform random density elsewhere -- at AF 5.34, one mask for both scans |
| noise | complex Gaussian in k-space, scaled to a white-matter SNR of 120 in the fully sampled SENSE image |

It writes `work_{pre,post}/{ksp_full,csm,mask2d}.npy` (what the reconstruction scripts read), `gs_{pre,post}/{ksp,csm,mask}.h5`
(what the GS trainer reads, in the CE layout: coils, readout, PE2, PE1, complex as a (real, imag) compound) and
`gt.npz` (both truth volumes, the masks, the labels). About 14 GB per case.

**Two settings were found by probing, not assumed:**
- Coils as smooth random fields gave maps that are too alike (mean correlation 0.44 but no spatial structure), and
  CG-SENSE came out *worse* than zero-filled. Loops on a cylinder fixed it.
- At the first guess of SNR 30, the error was noise-dominated and CG-SENSE reached only 27 dB. The prospective pair's
  background suggests a fully sampled SNR of order 120; at that level CG-SENSE is 31.3 dB against 26.6 dB for ZF,
  which is the expected spread.

## Reconstruction (`scripts/retro_run.py`)

Every arm runs the *prospective topic's own script*, pointed at a simulated case through environment variables
(`CE_WORK_PRE/POST` for `ce_classical.py`, `ce_tv_sense.py`, `ce_dip_inr.py`; `CE_GS_PRE/POST` for `ce_gs.py`;
`CE_SPLIT` for the held-out split), so the code path is identical to the prospective work and only the data differ.
Arms: ZF, CG-SENSE, CS-SENSE (ADMM-TV, 200 outer), 3D-DIP (5000 iters), 3D-INR (1500 iters), GS-single and MC-GS in
`retro_run.py`; Joint SENSE, L1-SPIRiT, ZS-SSDU (60 epochs) and MC-INR in `scripts/retro_more_arms.py`, which was
added on 2026-09-17 after the first pass had left them out -- the registered comparison set is nine methods besides
ZF, plus MC-INR. All hyper-parameters are the prospective ones except the lambdas, which are re-selected here. One job per card; a card is taken only when no process of this
user is on it, its utilisation is below 30%, it is not in `--exclude`, and it holds no live lock from another
session's scheduler (`--lockglob`, see Gotchas). Jobs write pid files, so a restart adopts what is still running.

CS-SENSE's lambda is **not** inherited from the prospective topic: 2e-3 was far too strong here (it washed out a
quarter to a half of the true uptake and scored below zero-filled). `scripts/retro_cs_lambda.py` re-selects it per
case on 5% held-out phase-encode lines, the same GT-free protocol as the prospective work -- tuning one arm on truth
while the others get nothing would tilt the comparison.

## Scoring (`scripts/retro_score.py` -> `retro_scores.json`)

Against truth, after the best complex scale per volume:

- **PSNR / NRMSE** inside the brain, on magnitudes;
- **uptake kept** = mean post minus pre inside label 4, over the true one;
- **false uptake** = mean post minus pre inside label 1 minus the true one -- what the arm invents where there is none;
- **white matter** the same, where the truth is near zero;
- **Dice** of the region recovered by thresholding post minus pre at half the true uptake.

`scripts/retro_figure.py` draws one case at the slice with the largest enhancing cross-section: truth plus every arm,
pre / post / post minus pre, with the enhancing label in cyan and the necrotic core in orange.

## Results (8 cases, 10 arms, 2026-09-17)

`retro_scores.json`, printed by `scripts/retro_score.py`. The column set is the prospective topic's: ZF plus the
nine registered comparison methods, plus MC-INR. Every lambda is selected on this data on 5% held-out lines, never
on truth (CS-SENSE and Joint SENSE per case, both 2.5e-05 everywhere; L1-SPIRiT's kernel and lambda settled once on
UCSF-PDGM-0004 with the full kernel 3/5 x lambda grid -- kernel 5, 1e-4 -- and reused for the other seven, since the
grid costs about 3 h per case under contention and every other selection had been identical across cases).

```
arm            PSNR    kept  |uptake err|  false(nec)   Dice     (mean of 8 cases; kept over the 7 with real uptake)
ZF            23.18     80%         0.121      +0.138  0.664
CG-SENSE      27.97     92%         0.049      +0.063  0.829
CS-SENSE      32.88     96%         0.024      +0.034  0.894
Joint SENSE   32.59     96%         0.026      +0.037  0.890
L1-SPIRiT     31.32     97%         0.019      +0.025  0.880
ZS-SSDU       30.35    105%         0.031      -0.042  0.884
3D-DIP        29.23    100%         0.005      +0.015  0.833
3D-INR        31.48     98%         0.012      +0.011  0.896
MC-INR        30.69     95%         0.030      +0.028  0.846
GS-single     32.62     99%         0.008      +0.010  0.906
MC-GS         31.47     99%         0.006      +0.012  0.880
```

What it says:

- **Undersampling biases the derived quantity, not just the noise.** Zero-filling keeps 80% of the true uptake and
  reports +0.138 of uptake inside a necrotic core that does not enhance; CG-SENSE 92% and +0.063. Both failure modes
  come from the same blur across the rim-core boundary, and both change what a reader would call the lesion.
- **GS-single is the most faithful arm on the enhancement itself**: the highest Dice (0.906) and the lowest false
  uptake (+0.010), with PSNR second (32.62) behind CS-SENSE (32.88). 3D-INR is close behind on both.
- **ZS-SSDU trades the two errors against each other**: it exaggerates uptake in the enhancing label (105%) and
  under-reports it in the core (-0.042), i.e. it sharpens the contrast in both directions rather than reproducing it.
- **Coupling the two scans does not help, in either family.** Joint SENSE against CS-SENSE: PSNR 32.59 vs 32.88,
  false +0.037 vs +0.034, Dice 0.890 vs 0.894. MC-GS against GS-single: 31.47 vs 32.62, +0.012 vs +0.010, 0.880 vs
  0.906, with identical uptake (99%). Two independent joint-versus-single comparisons agree, and so do the
  prospective held-out kres (MC-GS 0.30510 vs GS-single 0.30343, Joint SENSE 0.29210 vs CS-SENSE 0.29159) and the
  seed null test. In a pre/post pair the two scans differ only by uptake, so a shared prior adds little; what MC-GS
  does buy is compute -- one joint run (about 21 min here) produces both volumes, against about 42 min for two
  single runs.

Figures (`/home/congcongliu/figs/ce_retro/`): `retro_UCSF-PDGM-0053_z134.png` (largest enhancing volume),
`retro_UCSF-PDGM-0017_z117.png`, `retro_UCSF-PDGM-0031_z141.png` (thin rim, large necrotic core). Twelve columns:
truth plus the eleven arms, pre / post / post minus pre.

Caveats to carry into any write-up: synthetic phase, simulated coils, Gaussian noise, no motion between the two
scans and perfect registration -- none of the real pre/post difficulties are here. The truth itself is two separate
acquisitions, so its own difference map has edge structure that enters the "false uptake" number. PSNR is not
diagnostic quality; there is no reader study. L1-SPIRiT's kernel and lambda are transferred within the simulation
family rather than selected per case.

## Gotchas (do not re-discover)

- The frozen 95/5 split of the prospective topic is 232 x 176 and crashes the scoring of a 240 x 155 simulation.
  `ce_common.drop_foreign_planes` now drops planes whose shape does not match, and `CE_SPLIT` points at a per-case
  split made by `scripts/retro_split.py`.
- Sharing a card with another session's job costs about 8x. On 2026-09-17 this scheduler took GPU 1 while a peer
  session's job was still loading data on the CPU and therefore invisible on the card. Utilisation is not enough:
  `--exclude` and `--lockglob` (lock directories `.gpulock.<index>` holding the owner pid in `pid`) are.
- The GS trainer renders (readout, PE2, PE1); every consumer needs `np.transpose(v, (0, 2, 1))`.
- Do not carry a regularisation weight across datasets. See CS-SENSE above.
