0 actionable findings. No verified scoring, artifact-binding or keep/discard defect was found in this outcome audit.

The original coordinate-only family completed both real 500-update arms. Both
are correctly **discarded** against the frozen historical whole-head keeper
`lowreg_f075_solve_v1`. The explicit control has slightly better observed values
than the shared MLP in all six whole-head conditions, so these runs provide no
evidence of a shared-network advantage. Neither arm achieves the user's target.
Discard means rejecting the candidate for promotion; all code, checkpoints,
arrays, traces and scores remain evidence to preserve.

This was a read-only audit after both completed objectives were published.
`loop_state.json` now reports `family_complete_goal_unmet`, no active process,
and unchanged best margin `-0.11118128532897047`. The pre-loop
`campaign_state.json` supplies the pinned historical anchor; its old progress
fields are not the completed family outcome. Only this report was authored.

The primary score is the minimum of six margins, across pre/post and
PSNR/SSIM/NMSE, against each condition's best of the same ten complete frozen
opponents. The formulas are `(candidate-best)*ln(10)/10` for PSNR,
`ln((1-best)/(1-candidate))` for SSIM, and `ln(best/candidate)` for NMSE. Higher
scores are better. Success requires all six raw strict inequalities, with no
equality tolerance. HFEN has no role in ranking, acceptance or comparisons.
ZS-SSDU supplies the best frozen value in each whole-head condition here.

Exact whole-head values from the completed scores and pinned historical
`metrics.json` files:

| Condition | Frozen G2 | Shared500 | Explicit500 | Historical keeper | Best frozen opponent |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre PSNR, dB ↑ | 34.80795582896276 | 34.8098923910014 | 34.81066745868763 | 34.83664421852701 | 35.3194982813409 |
| pre SSIM ↑ | 0.9415072798728943 | 0.9415384531021118 | 0.9415454268455505 | 0.9412524700164795 | 0.9463346004486084 |
| pre NMSE ↓ | 0.009934071948278786 | 0.009929643243982158 | 0.009927871293368992 | 0.009868666263386654 | 0.008830250899305128 |
| post PSNR, dB ↑ | 37.15822506984288 | 37.15982920237576 | 37.1603895396045 | 37.14367982759124 | 37.527184786498296 |
| post SSIM ↑ | 0.9541497230529785 | 0.9541682004928589 | 0.9541727304458618 | 0.9538549780845642 | 0.9578452110290527 |
| post NMSE ↓ | 0.009600131154803154 | 0.009596585856137695 | 0.009595347758660406 | 0.009632337481701822 | 0.008818225694366846 |

| Arm | Primary minimum-six margin | Change from G2 | Change from keeper | Correct outcome |
| --- | ---: | ---: | ---: | --- |
| Frozen G2 | -0.11778703065863405 | 0 | -0.006605745329663584 | Baseline |
| Shared500 | -0.11734112169073506 | +0.0004459089678989919 | -0.006159836361764592 | Discard |
| Explicit500 | -0.11716265518481501 | +0.000624375473819036 | -0.005981369855844548 | Discard |
| Historical keeper | -0.11118128532897047 | +0.006605745329663584 | 0 | Retain existing anchor |

Both candidates improve all six values over their G2 initialization, but neither
exceeds the historical keeper plus the predeclared `0.0001` keep threshold.
Using G2 as the promotion incumbent would incorrectly revive weaker candidates.
The binding whole-head condition for both candidates is pre NMSE. Both fail all
six comparisons against the best frozen opponent. Against the historical keeper,
they trade worse pre PSNR/NMSE for better pre SSIM and all three post metrics;
the keeper is retained by the declared minimum-six objective, not by six-metric
dominance. No provisional win, repeat-success claim or joint-model attribution
is warranted. The frozen GS-single opponent is not a new matched GS-single
experiment with either new parameterization.

The whole-head arithmetic remains the frozen original `headonly`/`Pair` subset:
the head mask is GT magnitude greater than `0.05*P99.9`; PSNR uses the global GT
maximum and float64 squared differences of float32 magnitudes; SSIM averages
the original 7-cubed uniform SSIM map over that head mask; NMSE uses the original
CPU complex128-candidate magnitude path and complex64-GT magnitude promoted to
float64. There is no fitted amplitude scale. Original reference-based scalar
normalization is retained and disclosed; this is not a newly strict GT-free
benchmark. All six rescored G2 values equal the original saved `head_*` values
exactly, and its exported array hashes equal the canonical frozen G2 hashes.

Each formal arm has 500 consecutive trace rows, steps 1–500 / iterations
4501–5000, and `steps_observed=500` in metrics and provenance. Both start afresh
from G2 iteration4500 with new optimizer moments; the separate two-update smokes
are validation runs, not two updates of each formal 500 or additional updates
to its scored checkpoint. The zero baseline has no updates. All five completed
gates report successful numerical completion; every nonzero-stage trace records
finite gradients. That gate success is distinct from scientific success.

| Stage | Actual updates | Training seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Shared smoke | 2 | 4.038095180876553 | 34.32045936584473 |
| Explicit smoke | 2 | 3.6193140260875225 | 34.60749626159668 |
| Shared formal trial | 500 | 584.2421843451448 | 34.32051992416382 |
| Explicit formal trial | 500 | 423.5851254002191 | 34.60749626159668 |

The arms have equal update budgets, not equal elapsed compute. Provenance
records frozen original geometry/base coefficients and
`training_reference_access=false` for both. The model schema is the original
`g2_residual_amplitude_v1`, not the separately prepared paired-feature v2 branch.

All 59 frozen lesion-containing planes were inspected. Their metrics use the
head mask on each plane, not a lesion-only ROI. A passing plane must have both
scans' three metrics strictly beat all ten opponents on that same plane. The
secondary S6 is the maximum over the 59 planes of their minimum-six margins;
it is separate from the primary whole-head score.

| Arm | Best-plane S6 | Best axis,index (zero based) | Both scans pass | Pre passes | Post passes |
| --- | ---: | --- | ---: | ---: | ---: |
| Frozen G2 | -0.03381844992347395 | 1,130 | 0/59 | 0/59 | 1/59 |
| Shared500 | -0.0339382374367295 | 1,133 | 0/59 | 0/59 | 1/59 |
| Explicit500 | -0.03372242734014797 | 1,133 | 0/59 | 0/59 | 1/59 |
| Historical keeper, existing plane evidence with HFEN excluded | -0.03949034023797147 | 1,133 | 0/59 | 0/59 | 0/59 |

For G2 and both current arms, per-condition passing counts in the ordered six
conditions are `[0,0,0,3,1,3]`. The historical keeper's counts are all zero.
Historical S6/counts above were derived only from its existing 59-plane JSON
comparisons after excluding HFEN; no historical volume was rescored. Thus a
stronger whole-head keeper need not be the best local-plane candidate. Shared
S6 also worsens slightly from G2 even though all six whole-head values improve.

The six unchanged enhancement ROIs and post background are mandatory diagnostic
reports. Each scan's float32 magnitude is divided by its own white-matter median;
enhancement is the mean of normalized post minus normalized pre. Background is
the mean normalized post magnitude outside the GT-post head mask dilated six
voxels. Signed error is arm minus the frozen truth row. These errors do not
replace the primary score. Values below are rounded to 12 significant digits.

| Region | G2 signed error | Shared500 signed error | Explicit500 signed error | Historical keeper signed error |
| --- | ---: | ---: | ---: | ---: |
| Lesion core | +0.00253874994814 | +0.00252849794924 | +0.00253596715629 | +0.00281319394708 |
| Rim 0–3 mm | +0.0000442583113909 | +0.0000689961016178 | +0.00007688626647 | +0.0000277832150459 |
| White matter | -0.000657458556816 | -0.000695824390277 | -0.000685091828927 | -0.000916003948078 |
| Cortex | +0.000160731375217 | +0.000131942331791 | +0.000140734016895 | -0.000192373991013 |
| Vessels | -0.0849323272705 | -0.0851268768311 | -0.0850396156311 | -0.0828557014465 |
| Plexus | -0.127899885178 | -0.128078818321 | -0.127993106842 | -0.121160984039 |
| Post background | -0.000620405189693 | -0.000622688792646 | -0.000621730461717 | -0.000651197042316 |

Relative to G2, both formal arms reduce absolute core/cortex error but increase
absolute rim/WM/vessel/plexus/background error. The shared model has slightly
smaller core/cortex errors than explicit, while explicit is closer on WM,
vessels, plexus and background; shared is closer on rim. Against the historical
keeper, both are closer on core/WM/cortex/background and farther on
rim/vessels/plexus. Lower background is farther below this nonzero reference;
its clinical desirability is not determined by the signed fidelity error alone.
The ROI counts are unchanged: core6039, rim7522, WM486769, cortex580587,
vessels39620, plexus1079, and background4774646 voxels.

Artifact verification streamed SHA256 bytes rather than recomputing full-volume
scores: **40** artifacts bound by the five `completed.json` gates, all **18**
ledger-bound score artifacts, gate control bindings/post-run descriptors,
reviewed production sources, relevant metric/ROI helper sources and historical
keeper evidence were checked (**90 unique files** total). Candidate array bytes
match completed-gate artifacts, objective candidate hashes, the separate input
manifest and enhancement inputs. Metrics/provenance/trace/checkpoint hashes
match each completed gate. All arrays have the recorded `[256,232,176]`
complex64 header. Frozen scoring input maps and whole-head contracts match
across baseline/shared/explicit. The three current scores' 354 comparisons per
arm, plane order/core counts, best margins, strict flags and pass-count
summaries agree with the ten-opponent frozen plane baseline. ROI signed errors
equal the reported values minus the unchanged truth row.

Key SHA256 anchors:

```text
G2 zero completed  892ec336a78a4fdd96c19f7286239cdb00d9a8f0ae41239eb84dc1039582a4b8
G2 score objective 55fcbfc170e27d8d1be6c6b0013743037228f237baf25ccc6032da42edbcc0c2
G2 pre array       57df744a30b690c5ac61ba87e268882f71501b2bad043fdb9452f99fce6bff42
G2 post array      e64a0f97d4bb4e27ef64024708c16373898bba11d4157fd9f8214d0a37da91dd

Shared completed   99b8e38bd6d2cb8f4d0e838619fee48e955822cf84801af40f45a6d64ab39a91
Shared provenance  6a6766ba45399288a588e37b1d56c71dcacd451d1e0178a32b6fe47658a780fc
Shared objective   3ae656cb178ac18fefac4a57bd6442b1bf0a2ff901f567fb2faecdff51ca61a3
Shared pre array   8c0f4dcf59d3d952b5694575fbcd86d7ae718256ee11fa020e90726eee0c6d4a
Shared post array  66eb3b677805db18b1e1b6e684e49c8626b843d3a16a815b1c99b203e2c02930

Explicit completed c0d28ec4e75858326c1bcd8b3d5f7af3f5616ee550439055f2bea21476a72afb
Explicit provenance 7451c93c97bd5162a3cc25637eb4dff3dd7c02192b27c9752b6c2c1d8ff36ed3
Explicit objective c13016e5054261723de25ce4814282a68307b526750d7af1e19dad9b747b4e5f
Explicit pre array 471dab59f129c55fa6e658de5fe22cb72fbb6da43d36b528a26c96002ff2d6a8
Explicit post array fb7096412a7e454dc3b504b07dfd1222653593d8324da1192d40d57ed732a289

Keeper metrics     8ceeeee1f0d44aff6c080f572714be9382fc7f4789c6f052510bc1f32bd26805
Keeper old objective e61ee8f9dbf74aa3cb0eee8314f8cddc76987c0d407ed5a96d3e7d0b66caf75b
Keeper enhancement d3f5d50ed8f9d836686d57cb2ad0090d67ec203756dde8448173532b32bf8635
Whole-head contract a576014ad4839097ed260c4f69b144dcba3d47bae050e2a0c41316b004541699
Scorer evaluate.py 9a2c38d1942d6d6ad2331f3364dfad5be86a03cb316fec96f27deb7c5e40a875
```

Current evidence is in `outputs/g2_zero_v2`, both
`outputs/family1_*_trial_v1` directories and the corresponding three `scores/`
directories; the append-only ledger is `autoresearch.jsonl`. Historical metrics
come from `../mcgs_ce_autoresearch_20261017/frequency_lowreg_mm/outputs/lowreg_f075_solve_v1/metrics.json`,
with existing ROI/plane evidence in that project's
`scoring/lowreg_f075_solve_v1_v1/`. The historical old objective's own `score`
includes HFEN and is not used as the current primary score. The keeper's
whole-head margin was independently calculated from its six saved `head_*`
values against the present fixed whole-head opponents and matches the pinned
anchor exactly.

No GPU was used and no full-volume metric or clinical image score was recomputed
in this audit. It verifies existing evidence and recorded arithmetic, not
independent replication, generalization or a new matched GS-single result. No
production code, gate, queue state, ledger, score or candidate array was edited.
