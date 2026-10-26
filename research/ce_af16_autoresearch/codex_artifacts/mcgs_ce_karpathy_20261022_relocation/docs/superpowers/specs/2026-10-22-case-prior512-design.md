# Single-factor CNN fitting-budget experiment

The previous case-prior trial completed500 Gaussian updates and ten128-update
CNN fits. Its whole-head minimum-six score -0.12526398563890848 is below both
native500 (-0.11718558579803065) and the keeper (-0.11118128532897047), with
paired-plane success0/59. All evidence remains in `docs/caseprior_outcome.md`.
Several target changes were small and several first/last random-patch losses
were close; these are observations, not proof that underfitting caused the MRI
failure. Test that possibility with one declared budget increase, not a sweep.

## Scientific contract

Change only CNN updates per invocation from128 to512. Keep fresh independent
CNN initialization and fresh Adam for every invocation; do not warm weights or
optimizer state, change seeds, noise, patch selection, architecture or loss.
The full recipe remains width32, five3x3x3 convolutions, batch4, patch32,
tile32/halo5, learning rate0.001, seed42+1009*j, original NonLocalRED coefficients
0.1, sigma_rel0.015, start4501 and refresh100. Gaussian updates remain500 from
original G2 at4500 with all seven original states and gamma0.9995 restored.
Learned smoke has two512-update fits (1024 total); learned500 has ten (5120).
Zero/control constructs no CNN. Higher compute cost is explicit and measured.

All data,42coils, AF16, historical fixed mask/CSM, normalization, original
renderer, canonical export, ten opponents and PSNR/SSIM/NMSE evaluator remain
fixed. No scoring reference/ROI, opponent image, external data or pretrained
weight enters fitting. No HFEN decision, mask generation, source deletion,
dependency installation or renderer rebuilding. Preserve all completed outputs
and their original absolute-path source bindings. This is same-case development.
Existing normalization/reference and legacy-mask disclosures still apply.

Use route `g2_case_internal_magnitude_prior_fit512_v1`. Public stages/modes
remain zero0/control,smoke2/control or learned,trial500/control or learned.
Actual new zero and both smokes are required. Reuse the true old native500 only
after this route's genuine ordinary smoke and strict new receipt validate;
otherwise use a newly versioned control500. The current128 candidate remains an
additional observed comparison, never relabeled or counted as a new control.
Promote only above keeper+0.0001; overall success still requires all six strict
inequalities plus later repeat/matched improved GS-single/visual inspection.

## Minimal implementation boundary

Keep every existing numerical file byte-identical, including `run_caseprior.py`,
`caseprior_runner.py`, `case_denoiser.py`, inherited validators and evaluator.
Continue in the existing isolated worktree on the new caseprior512 branch.
Add only a small fixed variant loader, parent/child entry points, focused tests
and command documentation. There is no new optimizer or training-loop code.

A fixed private module copy of the reviewed parent source may be loaded under
a private registered module name, with only ROUTE,CHILD_RUNNER,SETTINGS/POLICY
and executing-source closure adjusted. The private child copy binds its `route`
to that same variant protocol. Do not mutate original imported modules or
rewrite source/bytecode. This explicit adapter is confined to these two known
local sources; it is not a user-configurable plugin or generic loader framework.
Ensure repeated loader access and the original `main()` lookup via sys.modules
resolve the correct module; no alias can accidentally dispatch the128 protocol.

The new source closure includes all original221 executing files plus every new
executing adapter/entry point. Preserve the original old218-source native
closure for its receipt; derive common old inputs from the new fresh snapshot
without redundant MRI hashing. Parent must remain stdlib-only. The unchanged
original functions must actually read512 through their private module globals
and send it to the real denoiser, validate512/1024/5120 evidence, publish the new
route, and preserve all original lifecycle/state/physics/artifact/RNG/readback
checks. Never merely relabel an old128 result or weaken a validator.

## Acceptance and execution

Focused tests must demonstrate: original module constants/functions/state and
bytes remain unchanged; correct variant parser/selected-policy/source hashes;
stdlib parent import; child dispatch and actual original producer settings;
zero/control bypass; positive default512 setting reaching the actual constructor;
wrong128 evidence rejected by512 policy (and vice versa); actual small original
trainer plus real CNN records accepted under private test settings; valid
new-route zero/smoke/trial and native-reuse semantics remain enforced. Reuse
existing real-producer fixtures without changing their restore evidence.

Private small CPU fixtures are acceptable; public N3.6M/CUDA/42coil/constants
must stay hard-bound. Run focused RED/GREEN, then one frozen complete CPU suite
after changes settle, with observed source-bound stdout. Existing577-pass
source is the unchanged baseline; do not repeat it before modifying code.
Obtain a fresh independent complete-change review, honoring the user's three
fix-round cap. Then create actual review/selected recipes and execute genuine
zero, both smokes, receipt/fallback and the one learned500 trial. CPU fixtures
cannot produce scientific passes. Preserve failed attempts and all evidence.

The user already authorized this autonomous experiment loop. Budget512 and
the fixed private-copy adapter are researcher choices within that scope; a
negative result rejects this bounded choice, without changing the benchmark.
