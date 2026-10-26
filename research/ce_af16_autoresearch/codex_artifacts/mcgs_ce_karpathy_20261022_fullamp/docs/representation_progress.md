# Full-amplitude representation continuation, 2026-10-22

This worktree continues the MC-GS plain/enhanced AF16 campaign using the official
Karpathy autoresearch workflow. Upstream HEAD was checked directly on October22:
228791fb499afffb54b46200aca536f79142f117. The installed reference program,
README and image hashes match upstream.json; no update was needed.

## Scientific starting point

There are28 formal MRI training experiments,65 historical reconstruction
rescorings and one separate proximal output-correction experiment. The latter
completed numerically but was discarded: whole-head PSNR34.740702/37.098449dB
for plain/enhanced, minimum-six margin-0.1364181617003603. It improved its own
measurement objective while worsening all six image metrics versus G2.

The whole-head keeper remains lowreg_f075_solve_v1, minimum-six margin
-0.11118128532897047. G2 remains the initialization and slice anchor, with
PSNR34.80795582896276/37.15822506984288dB. All prior candidates still have0/59
joint plane wins. Only PSNR/SSIM/NMSE decide MRI quality; the full ten opponents,
fixed data/mask/CSM and existing normalization limitation remain in force.

## Current implementation

The field, strict original raw renderer and bounded fitting/protocol are
implemented at scientific source revision131bb0d. One complex-modulus overflow
defect was fixed and scoped re-review round1 found no remaining actionable
issue. The final CPU suite passed304 checks in33.10s; a fresh independent
whole-branch review found0 actionable issues. Its report is
review_representation_round1.md. Reviewed source/input bindings cover31 code
files and29 input files.

The genuine G2 raw-render control completed on GPU2. Both reporting-gauge
arrays exactly equal the frozen raw anchors, with relative complex L2 zero.
It executed2 forwards/0 VJPs/0 updates in5.595520s numerical time, with3.197255GiB
peak allocated memory (37.802010s supervisor wall time). Its completed gate is
outputs/g2_fullamp_raw_control_v1/completed.json. Actual dtype was float32,
autocast disabled, CUDA matmul TF32 false and cuDNN TF32 true.

The MLP fit g2_fullamp_mlp_rep_v1 completed all2000/128 updates,264 forwards and
256 VJPs with finite gradients and fixed source/phase. Final raw relative L2
errors are0.3247297186805714 plain and0.3593065686077883 enhanced; it fails
the predeclared.01 bound and is discarded as a warm-start scheme. Numerical
time120.696723s, supervisor136.187275s, peak allocated3.493444GiB. Completed gate
SHA17471d4f8d10612b285beccd084b3e1e22ed8f29b375d81eaa8612b87be8e1cd.

The grid fit g2_fullamp_grid_rep_v1 also completed all2000/128 updates and
264 forwards/256 VJPs. Its final errors0.273462019725601/0.2974255657616767
fail the same.01 bound, so it is discarded. Numerical time141.326509s,
supervisor156.772738s, peak allocated3.504954GiB. Both stages' owned children
were reaped and no representation run is active. See representation_outcome.md
for all endpoints, counts, hashes and limitations. Each
fit uses2000 coefficient updates and128 raw-image updates with frozen geometry
and phase. Admission requires both final raw relative complex L2 errors<=0.01;
this is a predeclared initialization criterion, not an MRI metric or a quality
claim. These diagnostics add zero formal MRI trials. Actual baseline/outcomes are
recorded in results.tsv and autoresearch.jsonl; parent/child evidence stays in
each unique output directory. A running job is not a completed result.

No MRI continuation may be inferred from a software review or a low coefficient
loss. It requires actual raw-render admission and a separately reviewed route
using the unchanged MRI evaluator, all59 planes and the enhancement ROI reports.
All code revisions, negative outputs and review evidence are retained.

The detailed plan ledger is
.superpowers/sdd/2026-10-22-full-amplitude-representation/progress.md.
