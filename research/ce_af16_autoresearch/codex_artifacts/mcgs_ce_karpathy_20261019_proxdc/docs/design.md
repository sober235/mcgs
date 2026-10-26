# CE AF16 MC-GS autoresearch design

The user requested execution of the previously researched directions on 2026-10-19. This campaign retains shared 3D Gaussian geometry, independent complex contrast signals and the original multi-coil forward operator. It uses no external pretrained prior. A small network trained only on this case is the selected first candidate from the user's requested design; the exact residual parameterization below is an implementation choice.

## Frozen benchmark

Use the existing G2 iteration4500 checkpoint (SHA256 282e0303c46a6c18775d6224c032611072bd465b4ef975ba7a7e2ea636230fd0), 3.6M Gaussian points, AF16 measurements, original masks and CSM, fixed normalization and all ten complete comparators. Preserve the existing three-metric contract at ../mcgs_ce_autoresearch_20261017/reassessment_without_hfen_20261019/metric_contract_v1.json.

Primary ranking is the minimum of the six normalized whole-head margins against each metric's best frozen opponent. Use the existing normalization formula: PSNR difference times ln(10)/10; log((1-best_ssim)/(1-candidate_ssim)); log(best_nmse/candidate_nmse). Success requires all six strict inequalities. Report the original best-plane S6 and all59 planes separately. Do not confuse a local enhanced-plane win with whole-head superiority. Report six enhancement ROIs and background without fitting output scales or selecting ROI-specific parameters.

## First model family

Let z0_i,c be the original G2 complex coefficient. Use z_i,c = z0_i,c * exp(g_i,c) * exp(i*dphi_i,c), where dphi is independent per point and contrast. A shared small Fourier-feature MLP with two output heads produces bounded log-amplitude gains g from detached Gaussian positions. Its final layer and dphi start at zero, so the initial complex coefficients exactly equal G2. The G2 coefficient buffers remain frozen. This is a residual spatial amplitude model, not a claim that a fully implicit field has reproduced every original coefficient.

The matched explicit control replaces the shared MLP with independent per-point log-amplitude gains, both initialized at zero. Both branches use a new, labeled optimizer and checkpoint schema, independent phase, identical loss/forward/export, seed and update budget. Do not present them as original seven-group Adam continuation. The first bounded comparison keeps geometry fixed to isolate amplitude parameterization; joint geometry updates can be a later declared intervention.

Keep the existing G2 flat frequency weights and original likelihood, magnitude TV, phase, wavelet and edge priors. Use only acquired data for optimization. Do not use GT/ROI or full-data diagnostic checkpoints for training. Zero-step reproduction must agree with saved canonical G2 arrays before any model trial.

## Work budgets and route order

First run small CPU mathematics/gradient checks, a real zero-step G2 reproduction, then a two-step CUDA smoke for each new parameterization. A first pilot may use200 updates; each compared formal arm uses500 updates and records wall time, peak memory, loss history and actual steps. MRI iterations are not substituted for the upstream five-minute nanochat budget or described as equal compute. A bounded stage has a60-minute numerical timeout; a failed or unfinished run remains a failure.

Start with a64-wide two-hidden-layer shared field, Fourier bands4, bounded log-gain0.25, network LR0.001 and phase LR0.001. The matched explicit amplitude LR is0.001, using the same Adam betas(0.9,0.999). These are predeclared engineering starting values, not reference-selected best settings. Change one substantive mechanism per candidate. If the shared-amplitude family fails, the next previously researched alternatives are fixed-budget Gaussian relocation and partial frequency carriers; do not silently stack them into the first candidate.

The separately labeled MC-GS+proxDC control remains available after a G2/three-metric integration and review; it is a general output correction, not evidence of a joint-model invention. Any quality gain must be measured under the same frozen scoring.

## Validation and recovery

Test identity at zero including complex values, phase/amplitude gradients, channel independence, coordinate detachment, bounded gains, checkpoint round-trip and fixed buffers. The zero-step exported arrays must have the exact canonical spatial shape/order, complex64 dtype and finite values; complex relative L2 must be <=1e-5 for both scans, retaining the previously calibrated CUDA tolerance. Initial per-point complex coefficients must equal G2 elementwise. Bind the whole-head computation and all ten opponents separately in whole_head_contract.json, including the original headonly/Pair formulas and source hashes; the old plane contract is not a whole-head definition. A CPU rescore of saved G2 must reproduce its recorded head_* values before candidate scoring is used. Validate source/data hashes and exact recipe at launch and completion. Use a fresh-context review before GPU launches. Discard by recording rejection and retaining the candidate revision; the original G2 source and results stay untouched. A provisional overall win triggers repeat plus a matched GS-single variant before a final claim.
