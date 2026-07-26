# MC-GS edge-TV — algorithm, parameters and hyperparameters

> 本文件 = 当前方法(**MC-GS edge-TV**)的完整算法与全部参数/超参数清单。每一条都对着代码逐行核对过
> (2026-07-26),不是凭记忆复述;真源见每节标注的 `file:line`。数值以那次胜出运行的**生效配置快照**
> `results_arc/edgeTV_best/edgeTV_best/config.yaml` 为准(YAML 里的默认值会被 CLI 覆盖,见 §7.2)。
>
> Method chosen by the user on 2026-07-24 on the SSIM / perceptual criterion; the PSNR/NMSE-optimal
> sibling (isotropic-TV `reg_aligned_4k`) differs **only** in §4.2 and is kept as the alternative.
> **No learned or generative prior anywhere in this pipeline** (verified: §8).

---

## 1. Problem setup

A dual-contrast 3D brain acquisition of one subject — T1W (`plain`, the high-SNR guide channel) and
FLAIR (`contrast`) — is retrospectively undersampled with a shared variable-density Poisson mask at
AF ≈ 12 and reconstructed **jointly**.

| item | value |
|---|---|
| volume shape `(D, H, W)` | `256 × 220 × 256` (readout axis = 2, S–I, fully sampled) |
| receive coils `C` | 12 |
| phase-encode plane | axes (0, 1) |
| contrasts | `plain` = T1, `contrast` = FLAIR (same session, co-acquired, **voxel-aligned — no registration is performed or needed**, see `memory.md §5`) |

**Forward operator** (per contrast `m ∈ {plain, contrast}`, `paired_gs/trainers/joint_trainer.py:430`):

```
A_m(x) = M_m ⊙ F3c( S_m ⊙ x )                      x ∈ C^{D×H×W},  S_m ∈ C^{C×D×H×W}
```

* `S_m` — coil sensitivity maps (ESPIRiT, 1-map, SoS ≡ 1), §5.2
* `F3c` — centred 3D FFT, `ifftshift → fftn → fftshift`, **norm = backward** (`data/transforms.py`)
* `M_m` — binary sampling mask broadcast over coils, §5.1
* Measurements `y_m ∈ C^{C×D×H×W}` (zero outside the mask)

---

## 2. Model — MC-GS (`JointGaussianModel3D`)

`paired_gs/gaussian/joint_gaussian_model.py`, `paired_gs/gaussian/gaussian_model.py`

One set of `N` anisotropic 3D Gaussians describes the **shared anatomy**; each Gaussian carries **two
complex amplitudes**, one per contrast. This is the whole cross-contrast mechanism: joint k-space
supervision double-constrains a single geometry.

| parameter | symbol | shape | parametrisation | trainable |
|---|---|---|---|---|
| positions | `μ_n` | `(N, 3)` | normalised coords ∈ [−1, 1]³ | ✔ |
| scales | `s_n` | `(N, 3)` | stored as `log s`, used as `exp(·)` (`gaussian_model.py:69,85`) | ✔ |
| rotations | `q_n` | `(N, 4)` | quaternion, normalised at use (`tile_voxelizer.py:578`), init `(1,0,0,0)` | ✔ |
| density, plain | `α_n^p` | `(N,)` complex | separate real/imag `nn.Parameter`s | ✔ |
| density, contrast | `α_n^c` | `(N,)` complex | separate real/imag `nn.Parameter`s | ✔ |

Covariance and rendering (`tile_voxelizer.py:896-912`, CUDA kernel `:196-210`):

```
Σ_n^{-1} = R(q_n) diag(1/s_n²) R(q_n)ᵀ                       (built in NORMALISED coords)

x_m(v) = Σ_n α_n^m · exp( -½ (v̂ - μ_n)ᵀ Σ_n^{-1} (v̂ - μ_n) ) · 1[ mahal ≤ 9 ]
```

* `v̂` = voxel index mapped to normalised coords; centre in voxels `= (μ+1)·½·shape − 0.5`
* **3σ truncation**: contributions with Mahalanobis distance > 9 are dropped (hard cutoff, both CUDA
  and reference paths — `tile_voxelizer.py:205, 849`)
* per-Gaussian splat radius `= 3 · max_axis(s · half_shape)`, **clamped to `max_radius = 20` voxels**
* tiled rasterisation, `tile_size = 8` (8³ voxel tiles)
* **the render is complex** (`joint_trainer.py:411` `torch.complex(vol_r, vol_i)`) — the model carries
  its own phase; this matters for anyone re-using the hard-DC operator (§6)
* `has_private` residual Gaussians: **not used** (`num_private_points` unset ⇒ 0)

Rendering is called twice per iteration, once per channel, through the **same** voxelizer with the
**same** geometry — only `α^p` vs `α^c` differ (`joint_trainer.py:402-425`).

---

## 3. Warm start (geometry initialisation)

`joint_trainer.py:176-201`

| item | value |
|---|---|
| checkpoint | `…/3dgsVC/results_dualwarm_v2_t1_flairTV5/checkpoints/latest.pth` |
| Gaussians loaded `N` | **576 902** (fixed for the whole run — densification is OFF) |
| what is loaded | `positions`, `scales` (log), `rotations`, `density_real/imag` |
| contrast channel init | `α^c ← α^p` (cloned, then trained to differ) |

Provenance: this cloud comes from single-scan MC-GS reconstructions of **this same subject's own
undersampled measurements** — no external data, no population prior, no pretrained network. A
controlled A/B (`memory.md`, warm-start study) showed the 3000–4000-iteration joint optimisation
washes out the warm start's initial geometry quality to within ±0.08 dB, i.e. it is an initialiser,
not a lever.

> When a warm-start checkpoint is present, `gaussian.initial_num_points (200000)` and
> `gaussian.init_scale (0.008)` in the YAML are **ignored** — they only apply to the cold-start branch
> (`joint_trainer.py:203-221`).

---

## 4. Loss

`joint_trainer.py:774-824`. Total objective, per iteration:

```
L = L_dc^plain + L_dc^contrast + λ_geo · [ R_edgeTV(x_p) + R_edgeTV(x_c) ]
```

All other implemented regularisers are **disabled by config** (`λ_dtv = λ_edge = λ_vtv =
λ_private_sp = 0`; `lambda_diff` / `lambda_enh` are dead keys that the v2 trainer ignores with a
warning). They are documented negative results — see §9.

### 4.1 Data consistency `L_dc` (`joint_trainer.py:428-434`)

```
L_dc^m = ‖ W ⊙ M_m ⊙ ( F3c(S_m ⊙ x_m) − y_m ) ‖²_F  /  Σ M_m
```

Note the denominator is `mask.sum()` (number of sampled k-points, **not** multiplied by coils).
`W` is a static frequency-adaptive weight (`joint_trainer.py:518-540`) built on the normalised
k-radius `r = ‖(k_d, k_h, k_w)‖₂` with each axis mapped linearly to `[−1, 1]` (so `r ∈ [0, √3]`):

| band | condition | weight | value |
|---|---|---|---|
| centre | `r < kspace_center_fraction` | `kspace_center_weight` | **0.3** |
| mid | otherwise | `kspace_mid_weight` | **1.0** |
| high | `r > 0.5` | `kspace_high_weight` | **2.2** |

with `kspace_center_fraction = 0.05`. Order of assignment matters: the high band overwrites mid, and
the centre band is applied before it, so the three regions are `r<0.05` → 0.3, `0.05 ≤ r ≤ 0.5` → 1.0,
`r > 0.5` → 2.2. The k-space curriculum (`kspace_freq_curriculum`) is **off**; the per-contrast
sharpness decoupling (`kspace_high_weight_plain`) is **unset** — both channels share `W`.

### 4.2 Edge-preserving TV `R_edgeTV` — *this is the "edge-TV" of the method name*

`paired_gs/losses/edge_tv.py`, called at `joint_trainer.py:789-791`

Anisotropic per-axis TV on the **magnitude** of the render, each axis down-weighted at its own sharp
gradients (a detached Perona–Malik weight ⇒ purely spatial, no extra gradient path):

```
u = |x|                                    (magnitude of the complex render)
g_a = finite difference of u along axis a ∈ {0,1,2}
w(g) = clamp( 1 / (1 + (|g|_stopgrad / κ)²),  w_floor,  1 )
R_edgeTV(x) = Σ_a  mean( w(g_a) · |g_a| )
```

* `w → 1` in flat tissue (full smoothing), `w → w_floor` across sharp edges (smoothing spared)
* **κ is resolved once at iteration 0** and then frozen (`edge_tv.py:28-41`,
  `joint_trainer.py:753-757`): κ = the `edge_tv_kappa_q` quantile of the in-brain per-axis gradient
  magnitude of the *initial* render, where "in-brain" = voxels above `0.05 · max|x|`. Computed
  **separately for each contrast** (`κ_plain`, `κ_contrast`); the resolved values are printed to
  stdout at run start as `[edge_tv] kappa_q=… -> kappa_plain=… kappa_contrast=…`.
* the isotropic-TV sibling replaces this whole term with
  `R_TV(x) = Σ_a mean(|g_a|)` (`joint_trainer.py:436-441`) — that is the **only** difference between
  MC-GS edge-TV and the iso-TV winner, besides `λ_geo` (0.10 vs 0.06).

---

## 5. Data pipeline

### 5.1 Undersampling mask

| item | value |
|---|---|
| file | `…/3dgsVC/mask/mask3d_vdpois_af12_transverse_ALIGNED.mat` |
| type | variable-density Poisson disc in the (0,1) PE plane, replicated along the fully-sampled readout |
| nominal AF | 12 · **net AF 11.832** (measured sampled fraction **0.084517**) |
| ACS | contained in the vd-Poisson centre; **aligned to the data's k-space DC** |

> ⚠️ The `_ALIGNED` suffix is load-bearing. The original mask's ACS sat ~22 voxels off the data's true
> DC on the PE axis; realigning it was worth **+0.88 dB** on T1 and is the single largest fix in this
> project's history (`memory.md`). Never swap in the non-aligned mask.

### 5.2 Coil sensitivity maps

| item | value |
|---|---|
| files | `…/3dgsVC/csm_espirit/csm_espirit_{T1,FLAIR}.mat` |
| method | BART `ecalib -t0.001 -c0.0 -r24 -m1` (ESPIRiT, **1 map**, soft-SENSE off) |
| normalisation | SoS ≡ 1 over coils |

2-map soft-SENSE and DC-centred recalibration were both tested and rejected (§9).

### 5.3 Normalisation and ground truth

`paired_gs/data/dataset.py:304-378`

1. GT image built with the full k-space and the same CSM: `x_GT = Σ_c conj(S_c) · ifft3c(y_c^full)`
2. scale `= P99.9` of `|x_GT|`; both `y` and `x_GT` divided by it (`normalization_percentile: 99.9`)
3. undersampled data = `y ⊙ M` (`dataset.py:754`)
4. `center_kspace_dc` = **off** (tested, no gain — §9)

---

## 6. Inference / scoring operator

`arc/cycle.py:180-207` (scoring) and `:256-269` (volume export). **Hard data consistency** is applied
to the render before any metric is computed:

```
k_dc  = M ⊙ y + (1 − M) ⊙ F3c(S ⊙ x_model)        ← measured k is kept exactly
x_dc  = Σ_c conj(S_c) · ifft3c(k_dc)               ← SENSE combine
report on |x_dc| vs |x_GT|
```

Consequence worth internalising: the sampled 8.45 % of k-space is **exact by construction**, so every
residual error lives in the unsampled 91.5 % — which is why all remaining levers are about the null
space (§9).

**Metric gauges** — two are in use, do not mix them:

| gauge | definition | used for |
|---|---|---|
| pipeline (`@max`) | `paired_gs/metrics/metrics.py:30-55`: both volumes divided by `max|x_GT|`, clipped to [0,1], `data_range = 1`; SSIM `win_size = 7`, 3D | the scorer, `*_score.json`, TSV logs |
| brain-restricted | same normalisation, but PSNR/NMSE/SSIM averaged **only over `|x_GT| > 0.05·max`** (SSIM via `full=True` map, valid crop `[3:-3]³`) | cross-method comparison, figures |

⚠️ `@max` PSNR is **not comparable across contrasts** (T1's peak 1.887 vs FLAIR's 1.568 inflates T1 by
~1.6 dB). Use the brain gauge, or a unified peak, whenever T1 and FLAIR are compared.

Auxiliary artifact metrics used by the autoresearch scorer (`arc/cycle.py:76-121`): `hfen` (3D-Laplacian
high-frequency error norm), `sharp` (edge-gradient ratio vs GT), `grain` (std of the high-pass residual
over GT-flat voxels), combined as
`comp = psnr + 25·ssim − 60·nmse − 25·hfen − 40·max(0, 1−sharp) − 200·grain`.

---

## 7. Complete parameter table

### 7.1 Effective values of the shipped run (`edgeTV_best`)

| group | key | value | note |
|---|---|---|---|
| **loss** | `tv_weight` (= `λ_geo`) | **0.10** | edge-TV's optimum; iso-TV sibling uses 0.06 |
| | `edge_tv` | **true** | swaps iso-TV → edge-weighted TV |
| | `edge_tv_kappa_q` | **0.85** | quantile → κ, resolved once at init |
| | `edge_tv_wfloor` | **0.1** | residual smoothing at edges (damps Gibbs) |
| | `loss_type` | `l2` | |
| | `kspace_freq_adaptive` | true | |
| | `kspace_center_fraction` / `_center_weight` | 0.05 / **0.3** | |
| | `kspace_mid_weight` | **1.0** | |
| | `kspace_high_weight` | **2.2** | sharpness lever, opposes TV |
| | `lambda_dtv`, `lambda_edge`, `lambda_vtv`, `lambda_private_sp` | **0** | all off (§9) |
| | `lambda_diff`, `lambda_enh` | 0.01 / 0 | **dead keys**, ignored by the v2 trainer |
| **gaussian** | `position_lr` | **3e-4** | Adam |
| | `scale_lr` | **2e-3** | |
| | `rotation_lr` | **3e-4** | |
| | `density_lr` | **1.5e-2** | applies to all four density tensors |
| | `warm_start_checkpoint` | dual-warm 576 902 | §3 |
| | `initial_num_points` (2e5), `init_scale` (8e-3) | — | **ignored**, warm start present |
| **training** | `max_iterations` | **4000** | |
| | `seed` | **42** | `deterministic: false` |
| | `max_grad_norm` | **1.0** | `clip_grad_norm_` over all params |
| | `lr_scheduler` | exponential, `gamma = 0.9995` | stepped **every iteration** |
| | `shared_warmup_iters` | 500 | **INERT — see 7.3** |
| | `log_every` / `save_every` / `eval_every` | 50 / 1000 / 1000 | |
| **voxelizer** | `type` | `tile_cuda` (default) | custom CUDA kernel |
| | `max_radius` | **20** voxels | splat radius clamp |
| | `tile_size` | **8** | |
| **adaptive_control** | `enable` | **false** (empty dict) | **densification/pruning OFF** — `N` fixed at 576 902 |
| **data** | `normalization_percentile` | 99.9 | |
| | `acceleration_factor` | 12 (net 11.83) | |
| | `phase_axes` / `readout_axis` | (0,1) / 2 | |
| | `center_kspace_dc` | false | |

Optimiser: **Adam** with PyTorch defaults (`β = (0.9, 0.999)`, `eps = 1e-8`, no weight decay), one
param group per tensor with the lrs above (`joint_gaussian_model.py:198-226`).

### 7.2 ⚠️ The CLI clobbers the YAML

`arc/cycle.py:124-160` **overwrites** these keys unconditionally from its own defaults, whatever the
YAML says: `max_iterations`, `shared_warmup_iters`, `tv_weight`, `lambda_dtv`, `dtv_eta_scale`,
`lambda_edge`, `edge_p`, `edge_kappa`, `lambda_vtv`, `kspace_high_weight`, `voxelizer.max_radius`.
Its bare defaults are **not** the winner (`--lambda_geo 0.03`, `--kspace_high 1.0`). So the flags in
§7.4 are mandatory — passing only `--base_cfg configs/winner_v2_edgeTV.yaml` silently reconstructs a
different, worse model.

### 7.3 `shared_warmup_iters` is inert

The warmup-tying branch it used to gate was deleted with the v1 cross-contrast priors; the training
loop does `del warmup` at `joint_trainer.py:770` and never reads it again. It survives only as a
config sanity guard (`joint_trainer.py:99-106`: must be `< 0.5 · max_iterations`, else `ValueError`).
Passing `--warmup 500` therefore changes nothing about the optimisation — keep it only to satisfy the
guard. (Earlier project notes list it as a "key hyperparameter"; that is no longer true of this
trainer.)

### 7.4 Exact reproduce command

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONPATH= \
/home/congcongliu/anaconda3/envs/GSTMI/bin/python arc/cycle.py \
  --exp edgeTV_best --gpu <N> --iters 4000 \
  --lambda_geo 0.10 --kspace_high 2.2 --warmup 500 --max_radius 20 \
  --edge_tv --edge_tv_kappa_q 0.85 --edge_tv_wfloor 0.1 \
  --base_cfg configs/winner_v2_edgeTV.yaml --save_volumes
```

Cost: ~16.1 GB VRAM, 4000 iterations on one GPU. Outputs `results_arc/edgeTV_best/…`
(`config.yaml` snapshot, `joint_train.log`, `checkpoints/{latest,best}.pth`, TB events),
`results_arc/edgeTV_best_score.json`, `results_arc/edgeTV_best_volumes.npz`
(`recT1, gtT1, zfT1, recFL, gtFL, zfFL`, all `|·|` float32).

---

## 8. Results (verified)

**Pipeline gauge `@max`** (`results_arc/edgeTV_best_score.json`):

| contrast | PSNR | SSIM | NMSE | HFEN | sharp | grain | beats ZF anchor |
|---|---|---|---|---|---|---|---|
| T1 (plain) | 38.464 | 0.8980 | 0.02035 | 0.767 | 0.768 | 0.00904 | ✔ |
| FLAIR (contrast) | 38.184 | 0.9093 | 0.00863 | 0.675 | 0.810 | 0.01132 | ✔ |

**Brain-restricted gauge** (the one to quote when comparing methods or contrasts):

| method | T1 PSNR / SSIM / NMSE | FLAIR PSNR / SSIM / NMSE |
|---|---|---|
| **MC-GS edge-TV (this method)** | **33.90 / 0.9103 / 0.01365** | **33.31 / 0.9271 / 0.00593** |
| MC-GS iso-TV (`reg_aligned_4k`) | 34.03 / 0.9111 / 0.01323 | 33.53 / 0.9293 / 0.00565 |
| zero-filled (anchor) | 25.24 / 0.7058 / 0.10025 | 25.21 / 0.7118 / 0.03830 |

The edge-TV ↔ iso-TV difference is the deliberate perceptual↔fidelity trade the user selected on
2026-07-24 (edge-TV wins the whole-volume SSIM that the scorer reports, 0.8980 / 0.9093 vs
0.8936 / 0.9017; iso-TV wins PSNR/NMSE by ~0.1–0.2 dB).

**Robustness / provenance checks already run on this exact reconstruction:**

* **N4 bias-field normalisation** (fit once on GT, same field divided into both — the only fair
  protocol): deep-interior NMSE is *bit-for-bit unchanged* (T1 0.0044 → 0.0044, FLAIR 0.0023 →
  0.0023); the error map is the same error re-weighted by `1/B`. Independent per-volume N4 fits
  double the interior NMSE and are an artifact, not a measurement. Figure
  `results_arc/n4_errormap_edgeTV_T1_FLAIR.png`, details `memory.md §6`.
* **Null-space error** `‖(1−M)(F(x) − F(|GT|))‖ / ‖(1−M)F(|GT|)‖` = **0.196 (T1) / 0.128 (FLAIR)**,
  versus 0.430 / 0.288 for zero-fill — i.e. the model supplies real, correct unsampled-k content.
  Figure `results_arc/maisi_nullspace_gate.png`, details `memory.md §8`.

---

## 9. What is deliberately OFF (each one a measured negative result)

Everything below is implemented and config-gated to zero/false. Re-enabling any of them has been
tested and does **not** improve this reconstruction — see the cited section before spending GPU time.

| switch | mechanism | verdict |
|---|---|---|
| `lambda_dtv` | plain→contrast directional TV | marginal, PSNR↔SSIM reshape (`memory.md`) |
| `lambda_edge` | symmetric cross-gradient edge alignment | **catastrophic** (−10 dB even at 0.03) |
| `lambda_vtv` | joint vectorial TV | pure PSNR↔SSIM trade |
| P1 / P3 (branch `priors-p1p3`) | self-supervised k-extrapolation / cross-contrast local low-rank | both null or harmful |
| `adaptive_control.enable` | densification / pruning | worse at fixed budget |
| `center_kspace_dc` | k-space DC re-centring | no gain (`memory.md §6`) |
| 2-map ESPIRiT CSM | soft-SENSE second map | zero gain, loses under any regularisation (`§4`) |
| deformation module (ARC only) | inter-contrast registration | out of scope by user decision (`§5`); MC-GS has no such module |
| frozen MAISI VAE as a prior | latent generative prior, `L_maisi` / DDNM / DPS | **inadmissible** — the oracle prior is 49 % (T1) / 89 % (FLAIR) worse than MC-GS in the null space (`§8`) |

**No learned prior is used.** Grepping `maisi|diffus|vae|unet|pretrain` over the config, trainer,
losses and `arc/cycle.py` returns nothing; the only `torch.load` is the Gaussian warm start of §3,
which is derived from this subject's own measurements. The residual error at AF ≈ 12 is unsampled
high-frequency content (physics), and the only remaining levers are a **measurement-supervised**
learned reconstruction or a lower acceleration factor.

---

## 10. File map

| what | where |
|---|---|
| trainer | `paired_gs/trainers/joint_trainer.py` |
| model | `paired_gs/gaussian/joint_gaussian_model.py`, `…/gaussian_model.py` |
| rasteriser | `paired_gs/gaussian/tile_voxelizer.py` (+ `cuda_ext/`) |
| edge-TV loss | `paired_gs/losses/edge_tv.py` |
| data / forward | `paired_gs/data/dataset.py`, `paired_gs/data/transforms.py` |
| metrics | `paired_gs/metrics/metrics.py`, scorer in `arc/cycle.py` |
| config (template) | `configs/winner_v2_edgeTV.yaml` |
| config (**effective, authoritative**) | `results_arc/edgeTV_best/edgeTV_best/config.yaml` |
| edge metrics / verdict tooling | `scripts/score_edge.py` |
| project record | `memory.md` (§1–§8), `.claude/memory.md` (§0 = method card) |
| branch | `edge-preserving-tv` (PR #1); `edge_tv` defaults to **off** on `master` |
