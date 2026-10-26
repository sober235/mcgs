# SDD ledger — plan: docs/superpowers/plans/2026-10-22-carrier-operator.md

Branch autoresearch/ce-af16-carrier-20261022, base61daed6, design/plan3764b5c. Existing isolated worktree retained. Historical224 source map frozen before implementation. Current formalMRI32/historyrescore65; keeperlowreg-.11118128532897047; CNN512-.11867860310970936 discarded with independent auditPASS. No activeGPU/scorer.

- [x] Task1 CPU reference/derivatives and fresh review.
- [x] Task2 isolatedCUDA implementation, source review, build/microgate and outcome review.
- [ ] Separate MRI integration design only after numerical acceptance.

## Preflight scan

|Boundary|Producer/consumer checked|Finding/ruling|
|---|---|---|
|Task1 self|Selected rows exactpublicsignature -> CPUdouble complexvolume and five-input autograd|Geometry formula/hard support/tolerances pinned; no MRI/CUDA work. EmptyK connected zeros required.|
|Task1->Task2|Reference identical selectedrow shapes/voxel frequencies -> CUDA boundedfixtures/VJPs|Oracle remains unchanged; same logscale/wxyz conversion. CPUdouble vsGPUfloat deliberate.|
|Task2 self|Explicitloader provenance -> CUDA wrapper/customautograd -> microgate evidence|No implicitcompile/fallback, separate source/binary; actualGPU onlyafterreview. Source-level approval is not numericpass.|
|Task2->futureMRI|Acceptedoperator -> future separate carrierstate/optimizer/rowselection|No integration/quality claim now; fullscalecontrols later.|

Ruling: retain existing isolated worktree and new branch, rather than duplicate unchanged large route — preserves original numerical source/outputs and allold224 hashes — path still saysrelocation, so explicitbranch/state filenames carry identity.
Ruling: scoped two-task numerical component before MRI integration — zero-carrier derivative and newCUDA correctness are load-bearing — adds upfront checks but avoids invalid expensive reconstruction.
Ruling: user3-round review/fix cap and scientific preservation override skill5-round/cleanup defaults — unresolved substantive defects block dependentGPU/MRI — preserves disk and may require redesign.
Ruling: only focused newcomponent suites, no unrelated legacy606 repeat without new affectedsource — oldsource stays frozen — historical fullsuite remains previousevidence rather than rerun.
Builder check freshcheck_carrier_operator_design: nothing important left. All technical choices researcher-owned under standingautoresearch authorization.

Task1 ready atBASE3764b5c; rootalone commits/ledger/state. Worker owns carrier_reference.py, tests/test_carrier_reference.py and report only.

Task1 dispatched to implement_carrier_reference(gpt-6.1-sol/high) BASE3764b5c. Read-only toolchain preflight found defaultCUDA12.6 incompatible with GSTMIcu118; compatibleCUDA11.8nvcc11.8.89 andGNUc++10.5/ninja installed. Ruling: newbuild process explicitly pinsCUDA_HOME11.8 beforecpp_extension import — matchesexistingTorchABI withoutenvironment reinstall — default12.6 must not silently build. No compilation or GPU initialized.

Task1 source frozen/rootcommitteddb841cb:93CPUtests passed2.05s,exit0,empty stderr. Controller readactualevidence and verifiedownedbeforeafter/currenthashes and historical224unchanged. Fresh astra/high review_carrier_reference reviews exact3764b5c..db841cb package; tasknotcomplete untilverdict.

Task 1: complete (commits3764b5c..db841cb, independent specPASS/qualityAPPROVED,0findings). Actual93CPUtests retained; no rerun. Task2 atBASEdb841cb, new5fileownership; CPU/source phase first, actualcompile/GPU deferred untilcleanreview.

Task2 mathematical boundary ruling: newcorrection uses actualinverse-M conservative axisbounds, with historical floor/ceil/clamp/stablepair order; reliableinverseabsent forfinitegeometry -> explicitfullvolumebounds thenexactmahal test. Rejectderivednonfinite andclampbeforeintconversion. Spec exactmahal<=9 support takes precedence over bit-identical historicalapproximatebbox; zeroequivalence remainsrequired. Ruling: preservefullmathematicalsupport and documentnonzero-support distinction — oldquaternionnorm+eps/scaleeps make covariancebboxapproximate — pathologicalbroadatoms costmoretiles and laterMRI must test/discloseboundarypolicy. Newboundary/degeneratefixtures required; oldrendererunchanged.

Task2 initial focusedCPU21pass/13GPUskipped, no build/GPU. Worker adding finalboundary/provenancechecks before one fullcomponentCPUrun. This interimresult is notfinalsourceboundacceptance and noGPUapproval.

Ruling beforeanyCUDA: near-singular tiny-q pathological fixture checks support/output/localgeometryVJPs withidentical roundedM; ordinaryboundedfixtures retainallpublicinputVJPdoublethresholds2e-4/2e-5 — q->R cancellation makes relativeparameteraccuracy illconditioned — limitedpathologicalgradientclaim explicit, no thresholdrelaxation or failedGPUgate reinterpretation. Worker may add samefloatTorch chain internalconsistency ifuseful.

Task2 sourceCPU frozen/rootcommit198db2d. Fullcomponent118pass/14explicitGPUskips3.85s exit0,empty stderr; controllerverifiedcurrent7files matchfinalCPUsourcebeforeafter andhistorical224unchanged. CombinedfreshTask2+wholecomponent astra/high review_carrier_component examining61daed6..198db2d. No nativecompile/GPU yet; Task2notcomplete.

Task 2: source review round 1 has 4 Important findings: finite log-scale exponential overflow gives NaN VJP; effective compiler overrides escape identity; required nondefault-stream gate absent; int32 tile ceil arithmetic can overflow. Whole-component source approval withheld. Fix round 1/3 dispatched to original implementer_carrier_cuda, owning the same five files. No build or GPU permitted before scoped clean re-review. Report: final-source-review-round1.md.

Related F1 advisory assessed before handoff: finite p~1e30/logscale80 at zero frequency does not create a standalone d*d overflow in the kernel; left-to-right -.5*w*Q*d[x]*d[y] retains zero with Q0 and finite d, M*d is zero, and exp80 stays finite in geometry backward. Worker source inspection resolves that specific concern; no arbitrary new range restriction or redundant rerun added. Nonzero pathological float32 gradients remain outside bounded-fixture accuracy guarantees.

Fix round 1/3 source frozen/root commit3775681. Affected CPU67 passed/15 explicit GPU skips3.47s exit0, empty stderr; current7 component hashes match test and historical224 unchanged. Task1 reference93 not repeated. Scoped review198db2d..3775681 dispatched to existing reviewer; all4 findings await independent disposition. No native build or GPU run.

Source round2 scoped review: all4 findingsADDRESSED,0remaining; specPASS/qualityAPPROVED/source readyforactualbuild. Read fullreport final-source-review-round2.md. Initial sandbox resource query failed nvidia-smi exit9 beforebuild; identical command escalated per environment permissions and began successfully. Fresh resource query selected idle GPU5 UUIDGPU-18b3d14a-7ef2-870d-d438-1d6d09755a2d,capability8.0; otherjobs untouched. Actual isolated microgate tool session85950 running. No MRI count increment.

Actual microgate v1: isolated C++/CUDA build succeeded, publicexit1 after1failed/15passed/66deselected plus completeness teardownerror (98.49s). Noncontiguous fixture failed its layout precondition before operator call; other13 numerical fixture records and invalid-input check retained. Historical binary unchanged. Failed evidence carrier_verification/20261023T011403Z-bc86c383e901 preserved. Fix round2/3 dispatched to original worker for CPU layout reproduction and test-only correction; no tolerance/source-kernel changes authorized. Actual component still unaccepted.

2026-10-26 resume: test-only correction6a3e8df independently approved,0findings. Current resources: GPUs0/1/5/6 occupied; GPU3A100idle andcapability8.0 selected. Corrected complete GPUgate launched in fresh report directory, reusing dedicated artifact only through unchanged loader identity checks. Tool session51756. First failed gate remains preserved; formalMRI32/history65 unchanged.

Task 2: complete (source6a3e8df, source reviews closed, actual GPU v2 accepted, independent evidence auditPASS/0findings). Actual16tests includes14numerical+1GPUinvalid+1CPUcompleteness,67deselected,5.48s; artifact reuse no work. No newMRI count. Preserve whole workspace/failedgate under scientific evidence policy. Next is independent MRIintegrationdesign.
