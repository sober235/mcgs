# Ceiling probes — how much can ANY non-learned cross-contrast prior add to MC-GS over GS-single? (jiangyizhou AF16, 2026-09-26)

Branch `widen-mcgs-af16-20-24`, worktree `mcgs_widen_wt`. Context: after rounds 0–5 of the widen loop
(`autoresearch_widen.md`), the fair margin MC-GS − GS-single is +0.106 / +0.064 dB (T1 / FLAIR) at AF16 and about the
same at AF20 / AF24. The user (2026-09-26) asked for a stronger method or prior, without learned priors, that lifts the
margin to something a reviewer notices; scope = reconstruction side first, acquisition co-design allowed later
(choice C). Before designing anything, this record measures the ceilings.

Pre-registered reading (written before the GPU runs scored): a family whose *oracle* (perfect partner) gain over
GS-single is below 0.3 dB is closed on this data; a family with a ceiling of 1 dB or more is worth a real design.

## 1. Intensity coupling — closed (numpy, GT pair, no GPU)

Question: how much of a contrast's UNSAMPLED k-space is predictable from the other contrast's *perfect* image through
a hand-crafted intensity relation (ratio field, K-class piecewise-linear map = the joint-histogram / joint-entropy
family, local linear map)? Script `scratchpad/predictability{,2,3,4}.py` (session scratch, not committed; the numbers
below are reproducible from the recipe in this section).

Gauge: predictor gets the target's sampled k-space back (hard DC), phase from the target's GS-single reconstruction,
scored as head-only PSNR of |x| against |GT| (head = |GT| > 0.05, peak = 1 at the p99.9 normalisation; this gauge sits
~7 dB below the topic scorer's head PSNR, only differences are read). The registered `gs_single_wd` / `gs_joint_wd`
volumes are the anchors. "Leak-free" = the relation is fitted with the target's own reconstruction (ZF or GS-single),
never its GT.

| target FLAIR (source = T1 GT) | dB | target T1 (source = FLAIR GT) | dB |
|---|---|---|---|
| ZF | 20.47 | ZF | 18.45 |
| GS-single (DC re-applied) | 27.89 | GS-single | 28.66 |
| MC-GS, current recipe | 27.95 | MC-GS | 28.77 |
| K-class map, K=32, fitted on GS-single | 25.92 | | 26.33 |
| local-linear map, 3-voxel window, on GS-single | 24.46 | | 24.23 |
| K-class / local-linear fitted on ZF | 20.9–21.7 | | 20.0–20.5 |
| GS-single + local-linear model of its residual, 5-voxel window (**leaks the true residual**, upper bound) | **28.23 (+0.34)** | | **29.08 (+0.41)** |

Reading: with the partner contrast known *exactly*, every intensity-coupling predictor lands 2–5 dB below what the
single-contrast prior already recovers, and even a residual model that is allowed to see the true residual through a
5-voxel coefficient field adds only +0.34 / +0.41 dB. Ratio fields (sigma 2–8 voxels) are worse than no prediction at
all on the unsampled band. **Ratio / class-mixture / joint-entropy priors are closed on this data.** (A leaky
K-class fit with 16 classes on the GT pair "captures" 82 % of the unsampled energy — that number is the class labels
leaking the target and is reported here only so nobody re-derives it as a positive result.)

## 2. Literature numbers, and why they do not transfer

| work | setting | reported gain | what differs here |
|---|---|---|---|
| Ehrhardt & Betcke 2016 (structure-guided / directional TV, arXiv 1511.06631) | side information = the other contrast **fully sampled**; Shepp-Logan, BrainWeb, one patient; Cartesian / radial / spiral, 5 % noise; baseline plain TV (mean 26.9 dB) | dTV over TV: mean +5.8 dB (T1) / +6.5 dB (T2), SSIM +8 % (their Table 1) | perfect guide, weak baseline, simulated noise; "the quantitative results for patient do not indicate much improvement" (their §5.3) |
| Bilgic et al. 2018 MRM 80:619 (JVC-GRAPPA, J-LORAKS) | parallel imaging regime, contrasts sampled with **complementary** (shifted) patterns, phase priors | "2-fold reduction in reconstruction error compared to conventional GRAPPA" (abstract, Europe PMC) | complementary sampling + GRAPPA baseline; not a CS-strength single-contrast baseline |
| Gong et al. 2015 MRM 73:523 (PROMISE) | sequential: one contrast reconstructed first, its structure feeds the next; L1-SPIRiT base | "lower error level and better detail preservation" (abstract; no numbers in the abstract) | sequential / asymmetric, L1-SPIRiT baseline |
| joint TV / vector TV / group sparsity (Huang 2014, Majumdar 2011, Esfahani 2022) | both undersampled, same or different masks | small; on this line vector TV at matched strength is identical to independent TV (CRISP note) | already closed on this line |

The common pattern: large multi-contrast gains in the literature come from (a) a fully sampled or lightly accelerated
partner, (b) complementary sampling, or (c) a weak single-contrast baseline. None holds for GS-single at the same mask.

## 3. GPU ceilings (AF16, 4000 it, 3.6M points, seed 42; queue `scripts/wd_queue_oracle.txt`)

Fair baselines (best single run, lambda_edge 0): T1 35.744, FLAIR 34.945. Current MC-GS: 35.850 / 35.009.

Ceiling 1, perfect PARTNER DATA — joint arm, the other contrast fully sampled (`configs/*_oracle_{flairfull,t1full}.yaml`,
loader accepts it with `acceleration_factor: 1`): the current coupling family (shared geometry + R_edge + dtv + guided
weights) given a partner it can trust completely. `js` = the same with every coupling knob turned up (lambda_edge 0.003,
dtv 0.03, both guide alphas 1).

Ceiling 2, perfect STRUCTURAL GUIDE — single arm, the edge-TV and phase-TV weights read a GT volume
(`loss.guide_path_plain`, alpha 1): `gfl` = T1 guided by GT FLAIR, `gt1` = FLAIR guided by GT T1, `gself` = guided by
its own GT (the ceiling of any edge-location prior whatsoever), `gself_geo03` = own GT with 3x the TV strength.
`dtvp` = Ehrhardt-Betcke directional TV (projection form, `loss.lambda_dtv_proj`) with the GT partner, guided weights off.

| run | arm / guide | scored contrast head PSNR / SSIM | vs fair single | vs current MC-GS |
|---|---|---|---|---|
| `wd_o_ffull_j` | joint, FLAIR fully sampled, current recipe | T1 35.831 / 0.9409 | **+0.087** | −0.019 |
| `wd_o_ffull_js` | joint, FLAIR fully sampled, all coupling knobs up | T1 35.816 / 0.9398 | +0.072 | −0.034 |
| `wd_o_t1full_j` | joint, T1 fully sampled, current recipe | FLAIR 35.006 / 0.9253 | **+0.061** | −0.003 |
| `wd_o_t1full_js` | joint, T1 fully sampled, all coupling knobs up | FLAIR 35.010 / 0.9240 | +0.065 | +0.001 |
| `wd_o_s_t1_gfl` | single T1, weights guided by GT FLAIR (alpha 1) | T1 35.780 / 0.9401 | +0.036 | −0.070 |
| `wd_o_s_t1_gself` | single T1, weights guided by its OWN GT | T1 35.872 / 0.9412 | **+0.128** | +0.022 |
| `wd_o_s_t1_gself_geo03` | same, TV strength x3 | T1 35.598 / 0.9345 | −0.146 | −0.252 |
| `wd_o_s_fl_gt1` | single FLAIR, weights guided by GT T1 (alpha 1) | FLAIR 34.966 / 0.9253 | +0.021 | −0.043 |
| `wd_o_s_fl_gself` | single FLAIR, weights guided by its OWN GT | FLAIR 35.048 / 0.9267 | **+0.103** | +0.039 |
| `wd_o_s_t1_dtvp005_gfl` | single T1, directional TV with GT FLAIR, lambda 0.05 | T1 35.413 / 0.9342 | −0.331 | −0.437 |
| `wd_o_s_t1_dtvp015_gfl` | single T1, directional TV with GT FLAIR, lambda 0.15 | T1 34.109 / 0.9156 | −1.635 | −1.741 |
| `wd_o_s_fl_dtvp010_gt1` | single FLAIR, directional TV with GT T1, lambda 0.10 | FLAIR 34.284 / 0.9118 | −0.661 | −0.725 |

All twelve runs rc=0, one seed each (seed floor on this line 0.001 dB; the pre-registered band is 0.02 dB). The
directional TV of Ehrhardt & Betcke, given the *true* partner, is monotonically harmful on top of the tuned single
prior: in guide-flat regions it is an extra isotropic vector TV, and the edge-direction information it carries is worth
less than the smoothing it adds (R_dtvp stays at 7e-3 → 1.2e-2 through training, i.e. it never "finds" a low-cost
configuration). The 6 dB of their paper is what this term does against plain TV with no other prior; here it is −0.3 dB
at its mildest setting.

Two things stand out already. (i) A **perfectly known partner adds nothing** to what MC-GS already gets from an
undersampled one: T1 35.831 with FLAIR fully sampled against 35.850 with FLAIR at AF16, FLAIR 35.006 against 35.009.
The margin is set by what the coupling mechanism can transfer, not by the partner's quality — the same thing the
placebo (40 % of the margin survives a flipped partner) and the small shared-geometry term M0 said. Turning every
coupling knob up makes it slightly worse. (ii) Even a **perfect map of the target's own edges** lifts the single arm by
only +0.128 dB (T1), and using that map to smooth 3x harder costs 0.15 dB: the residual of GS-single is not on edges
and is not removable by smoothing — it is unresolved texture / high-frequency content that a structure prior does not
describe. The cross-contrast versions of the same prior (+0.036 T1, +0.021 FLAIR) sit below the own-edge ceiling, as
they must.

## 3b. Regime checks read from existing topics (no new runs)

Noise (topics `T1FLAIR_AF16_noise{0p5x,0p75x,1x,2x}_v3_jiangyizhou`, registered recipe, GS-single at lambda_edge 0.001,
so subtract ~0.04 / 0.02 dB for the fair baseline), MC-GS − GS-single head PSNR T1 / FLAIR:
0.5σ₀ +0.171 / +0.036, 0.75σ₀ +0.194 / +0.025, 1σ₀ +0.217 / +0.007, 2σ₀ +0.231 / **−0.116**. More noise buys T1 a
few hundredths and costs FLAIR the margin. Acceleration (this loop, fair): AF16 +0.106 / +0.064, AF20 +0.112 / +0.069,
AF24 +0.117 / +0.089. Neither regime moves the margin to the 0.5 dB scale.

## 3c. Asymmetric-acceleration probe (acquisition side, user nod 2026-09-26; queue `scripts/wd_queue_asym.txt`)

Hypothesis: when the target's own data is too sparse to place its edges, the partner's structure starts to pay.
Masks: house generator `vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1)`, one draw each (`scripts/prep_asym_masks*.m`,
files in `data/`, not in git): accelx 8 → 1689 points = **26.7x** on the plane, accelx 2 → 5704 = 7.9x, accelx 16 → 1040
= **43.3x**, accelx 12 → 1246 = 36.2x. Probe masks, not topic masks (no pool / exact-count pick). FLAIR is the heavily
accelerated contrast; recipes as in §3; the single arm is not re-tuned at these AFs (a ≥ 1 dB verdict does not hinge on it).

| FLAIR at 26.7x (T1 partner) | FLAIR head PSNR / SSIM / HFEN | vs GS-single FLAIR@26.7x |
|---|---|---|
| GS-single | 32.634 / 0.8825 / 0.1822 | — |
| MC-GS, T1 at AF16 | 32.766 / 0.8856 / 0.1757 | +0.132 |
| MC-GS, T1 at AF8 | 32.784 / 0.8864 / 0.1755 | +0.150 |
| MC-GS, T1 **fully sampled** | 32.802 / 0.8860 / 0.1726 | +0.168 |
| MC-GS, T1 fully sampled, all coupling knobs up | 32.880 / 0.8857 / 0.1745 | +0.246 |

At 26.7x a fully sampled T1 is worth +0.17 dB to FLAIR (+0.25 with the coupling turned all the way up). The margin
does grow as the partner improves — the ordering AF16 < AF8 < full is monotone — but the whole range is 0.1 dB wide.
Below the 1 dB line; closed at this level. 43.3x: (pending, `wd_a_{s_fl64,j_tfull_fl64,j_t16_fl64}`).

## 4. Verdict (against the pre-registered 0.3 dB / 1 dB rule)

| prior family (non-learned) | ceiling measured | how | status |
|---|---|---|---|
| the current MC-GS coupling (shared geometry + R_edge + dtv + guided weights) | +0.087 / +0.061 with a **perfectly known** partner — no better than with the AF16 partner | ceiling 1 | saturated |
| edge-location / edge-weight priors, cross-contrast | +0.036 / +0.021 (perfect guide) | ceiling 2 | closed |
| edge-location priors of ANY kind | +0.128 / +0.103 (the target's own GT as guide) | ceiling 2, gself | closed |
| directional TV (Ehrhardt-Betcke), perfect guide | −0.33 … −1.6 | ceiling 2, dtvp | closed |
| intensity coupling (ratio field, class mixture / joint entropy, local linear) | ≤ +0.34 / +0.41 even leaking the true residual | §1 | closed |
| joint / vector TV, non-local RED, private Gaussians, densification, capacity, complementary in-shutter mask | closed on earlier lines (AF16 phase 3, CRISP) | history | closed |
| more noise / more acceleration as a regime | margin stays at the 0.1–0.2 dB scale | §3b | closed |

**On this pair (jiangyizhou T1 + FLAIR, 1 mm, one vd-Poisson mask for both contrasts, AF 16–24) there is no
non-learned cross-contrast prior whose ceiling reaches 0.5 dB over the fair GS-single.** The reason is measurable: the
residual of a strong single-contrast reconstruction is neither on edges (own-edge oracle +0.13 / +0.10) nor a local
function of the partner's intensities (§1), and the partner's quality is already irrelevant to the current joint model
(ceiling 1). Whatever the two contrasts share at this resolution, GS-single already reconstructs from its own data.

What this does not cover: (a) other subjects / pairs — the T1w + T2* peds pair was even less similar (r 0.57, MC-GS
not ahead); (b) an **asymmetric** protocol, one contrast lightly and the other extremely accelerated, where the target's
own data may no longer place its edges — ceiling 1 was measured at target AF16 only; (c) learned priors, excluded by
the user. (b) is the one open door on the acquisition side (user choice C) and is the proposed next probe.
