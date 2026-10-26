# Completed G2 proximal-DC outcome

The fixed raw-G2 proximal correction completed and was discarded. The queue is
`candidate_complete_goal_unmet`, with no active child and no infrastructure
failure. This supersedes the October19 launch-time narrative; it is not a
currently running experiment.

| Whole-head metric | Plain | Enhanced |
| --- | ---: | ---: |
| PSNR | 34.740702046513924 | 37.098449094854196 |
| SSIM | 0.9384908080101013 | 0.9527453184127808 |
| NMSE | 0.010089105865455228 | 0.009733180193994778 |

All six values worsen relative to G2 and the historical whole-head keeper.
Minimum-six margin is -0.1364181617003603, versus G2 -0.11778703065863405 and
keeper lowreg_f075_solve_v1 -0.11118128532897047. Paired-plane wins remain0/59;
plain0/59 and enhanced0/59. Best-plane S6 is -0.07942140898997206 at axis1,index130.
G2 retained one enhanced passing plane before this correction.

The numerical physics solve succeeded: each scan used11 CG iterations,12 CG
normal evaluations and one separate endpoint normal. Objective decreased from
1434.288975 to1412.264797 for plain and896.576843 to886.275964 for enhanced;
acquired-data error also decreased. Train/optimizer/scheduler calls were zero,
as were optimizer updates; training_reference_access was false. Numerical time
was126.2525703s and completed supervisor time222.8613017s. Lower physical fitting
error therefore did not improve the frozen image-quality measures.

Relative to G2, absolute enhancement fidelity errors improve for white matter,
cortex and plexus, but worsen for core, rim, vessels and background. Relative to
the historical keeper, only white matter and cortex improve. The signed errors,
in core/rim/WM/cortex/vessels/plexus/background order, are
[+.00292979553342,-.0000790860503912,-.000457666115835,+.0000213906168938,
-.0878257751465,-.125094413757,+.00244822958484]. These diagnostics do not
replace the primary quality rule.

Counts remain28 formal MRI training experiments:25 historical plus3 new
500-update trials. Keep the one zero-update output correction separate. The65
historical rescoring cases, six total smoke updates and the CPU coefficient
approximation probe do not inflate the formal MRI count.

Read-only continuation audit verified170 small referenced source/descriptor
files and both actual candidate arrays: no missing or mismatching bytes. It did
not rehash all MRI acquisitions or recompute full-volume scores. No newer CE
scientific result superseded the existing keeper in the shallow workspace scan.
Initial campaign_state.json fields and inherited sibling progress documents
remain historical setup snapshots; completed loop states, ledgers and scores
are the scientific record.

Authoritative evidence is in sibling mcgs_ce_karpathy_20261019_proxdc:

| Evidence | SHA256 |
| --- | --- |
| loop_state.json | cfa408cb5dd005f1203b872ca485e65afe79465b470cd0e23f2b29d3e7937efa |
| scores/g2_proxdc_candidate_v1/objective.json | bb5f9928358bf3dbf3846f1cbfc6650b925bc8e39086bf56b42ca1e359d2adcb |
| outputs/g2_proxdc_candidate_v1/completed.json | 53db1828e8653a7e1b65a0e2399cb4e9dd848727ba0682509cd783a074a27cc3 |
| scores/g2_proxdc_candidate_v1/lesion_planes.json | 5b173e7c7cc0c131ae718b6cd59ecbc8b9e410f80c62047896b2f6bf347306bb |
| scores/g2_proxdc_candidate_v1/enhancement.json | 75c8d8d170b2890efe9bef0308647588011083c9ebbaf4af13d96b5425c13497 |
| outputs/g2_proxdc_candidate_v1/candidate_pre_complex.npy | 3c948d51a4cda49f173562a1e70d21fe6786c8dc1022af8cf1d892a08f0f6968 |
| outputs/g2_proxdc_candidate_v1/candidate_post_complex.npy | 27edf6369e19eeda4af053f99109d9be4dd0a9e59b8bd37e436f43c5c572e829 |

Continuation now develops the approved full shared-amplitude function in sibling
mcgs_ce_karpathy_20261022_fullamp. Its first experiments fit only G2 coefficients
and checksum-bound G2 raw renders. They are representation diagnostics, not MRI
quality results or new formal MRI trials. A predeclared1% two-scan raw-complex
fidelity threshold precedes any separately reviewed MRI continuation.
