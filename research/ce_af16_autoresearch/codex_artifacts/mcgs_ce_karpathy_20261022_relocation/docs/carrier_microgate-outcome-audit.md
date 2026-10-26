0 actionable findings. No actionable findings were identified in the observed microgate evidence.

Observed acceptance verdict: PASS for the bounded carrier numerical component on the recorded A100/CUDA 11.8 environment. The evidence supports the declared output/first-VJP acceptance, corrected noncontiguous inputs, nondefault-stream ordering, invalid-input checks, and validated reuse of the dedicated native binary. It does not establish full-scale performance, MRI integration safety, optimizer trajectory equivalence, reconstruction quality, or metric improvement.

Audit scope and execution evidence:

- Audited `carrier_verification/20261026T063900Z-7a66d90c3cf1/{command.json,microgate.json,stdout.log,stderr.log}`, the saved v2 launcher/reuse request, and the dedicated artifact's provenance manifest. Reused the completed source reviews and their source-bound test interpretation; no source review or numerical run was repeated.
- The actual command was GSTMI Python 3.11 running `-B -m pytest -q -s -p no:cacheprovider tests/test_carrier_cuda.py -k gpu`. The command record gives exit 0. Stdout records `16 passed, 67 deselected in 5.48s`, with no skipped or failed tests. These sixteen passes comprise fourteen numerical fixtures, the native/public invalid-input test, and the CPU report-completeness test selected by its name. Deselected CPU tests are not counted as GPU acceptance.
- Stdout explicitly records `ninja: no work to do.` Stderr contains the expected verbose Torch extension setup/load messages, with no warning, compiler diagnostic, traceback, or error. It is not empty; its contents agree with successful artifact reuse rather than a new compilation.
- The report has `status=passed`, exactly the fourteen declared fixture names, every fixture `passed=true`, and `invalid_inputs=true`. No required case is missing. The corrected noncontiguous case is present in this fresh report; it was not copied into the earlier failed report.

Numerical results:

The frozen acceptance tolerances remain rtol=2e-4 and atol=2e-5. Thirteen ordinary fixtures record all five public-input VJPs: positions, log_scales, rotations, density, and frequencies. The degenerate case explicitly records only centers, M, density, and frequencies for identical rounded geometry, alongside its public output/support check; it excludes the ill-conditioned quaternion parameter chain as approved before GPU execution. All reported output/VJP error summaries and repeated-variation values are finite.

| Fixture | Output max absolute error | Largest VJP max absolute error | Three-run VJP max variation |
|---|---:|---:|---:|
| ragged | 2.173255e-7 | 1.539318e-5 | 1.144409e-5 |
| single | 8.053702e-8 | 3.057621e-7 | 3.576279e-7 |
| overlap | 4.406158e-8 | 2.124919e-6 | 1.788139e-7 |
| outside | 0 | 0 | 0 |
| empty | 0 | 0 | 0 |
| noncontiguous | 4.346510e-7 | 1.913905e-5 | 3.433228e-5 |
| zero | 0 | 2.769221e-7 | 0 |
| quarter_d | 1.065309e-7 | 1.348136e-6 | 2.384186e-7 |
| quarter_h | 1.098680e-7 | 1.532994e-6 | 2.384186e-7 |
| quarter_w | 1.088869e-7 | 1.255274e-6 | 7.152557e-7 |
| tiny_phase | 1.514436e-13 | 1.149440e-6 | 4.768372e-7 |
| near_bound | 1.450610e-7 | 2.110711e-6 | 0 |
| degenerate, local VJPs | 3.206899e-7 | 1.802462e-5 | 4.577637e-5 |
| nondefault_stream | 1.289312e-7 | 3.010877e-5 | 9.536743e-6 |

All individual VJP summary fields were checked, not only the maxima above. The largest reported ordinary VJP relative summary is 1.308572e-4 for overlap/log_scales, below rtol. The stream frequency VJP's absolute error is 3.010877e-5, above atol alone, but its relative summary is 8.210213e-7; it satisfies the unchanged combined tolerance. The degenerate output's relative summary is 0.004578339, but its absolute error is only 3.206899e-7, below atol. Neither value is a failed check or a reason to change tolerance. The report's relative summaries use the reviewed denominator floor of 2e-5; they must not be presented as an independent strict relative-error contract. The source-bound passing tests performed the actual elementwise comparisons; raw output/gradient arrays are not persisted in this report and were not regenerated for audit.

For zero frequency, output and the four original-input VJP error summaries are exactly zero. The reviewed test asserts exactly zero correction and original-input correction gradients before marking this fixture passed, and separately asserts a finite, nonzero frequency gradient for the asymmetric imaginary upstream example. Its observed pass therefore supports those checks; the frequency VJP error is 2.769221e-7. The report does not store the raw frequency-gradient magnitude. The tiny-phase passing test also includes the explicit retained nonzero quadratic real-term assertion.

The stream fixture records `default_delay_completed_at_capture=false`, `cuda_launch_blocking="0"`, and producer-event → operator-forward/backward-event → consumer pinned-host snapshot ordering. Its passing result is bound to the reviewed test that synchronizes the consumer event and evaluates snapshots before global cleanup synchronization. Consumer output max absolute error is 9.932023e-8; output and all five VJPs passed. This is actual evidence for the tested stream sequence, not just source inspection of the launch stream.

Atomic backward variation is measured and nonzero. The largest observed value is 4.57763671875e-5 in the degenerate case; the largest ordinary-case variation is 3.4332275390625e-5 in the corrected noncontiguous case. The contract required disclosure rather than a separate absolute repeatability threshold. Three repetitions and comparison of the first VJP result with the oracle do not establish bitwise determinism, an exhaustive variation bound, or behavior at MRI scale.

Source and native provenance:

- The five source/test hashes in `command.json` are identical before and after execution. The three compiled-source identity entries match those command hashes and the approved production sources. The reference hash remains the accepted Task 1 hash. The report's test hash is `4a936e9b6642e65b8abe008d640cbfceda4f8b316dc3acf4ee6b1951e7f9a0e8`, matching the corrected test's approved focused GREEN evidence and this run's command record.
- The complete reported provenance equals the dedicated artifact's persisted manifest. Recomputing the digest of the recorded identity yields `7e4074ab699211d0b99e28f0766c6c6b2f12ad9f3057105fe6993faa6d95e719`, matching the report, artifact subdirectory, and module naming. The recorded Ninja text hashes to its reported `build_ninja_sha256`. These are consistency checks over saved evidence, not a compiler or GPU probe.
- The loaded module is `carrier_cuda_7e4074ab699211d0b99e28f0`, from the dedicated first-build directory under `carrier_builds/20261023T011403Z-bc86c383e901/`. Its recorded binary SHA256 is `b8d08c0aeead3c4c6d2e9f0fd803929c4ea5252bdd6de692053da6ced4c4c95d`, equal to the first run. Successful explicit loading, unchanged identity/manifest, and Ninja's no-work result support actual verified reuse.
- The recorded toolchain is Torch 2.7.1+cu118 with CXX11 ABI enabled, CUDA/nvcc 11.8.89, GNU C++/gcc 10.5.0, and Ninja 1.10.1. Effective Ninja commands use the pinned C++/nvcc paths, explicit gcc host compiler, C++17, `--fmad=false`, and compute_80/sm_80. The environment records the required PATH, CXX, CUDA_HOME, architecture, and asynchronous launch mode; CPATH and LD_LIBRARY_PATH were removed only from the child and their names were recorded.
- The historical binary hash before and after is exactly `3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5`, the required frozen hash. This is the run's recorded before/after evidence; the historical binary was not reread or rehashed during this audit.

Resource observation and evidence boundary:

The actual command binds CUDA visibility to `GPU-a9b716ea-de2e-5d23-217d-db82a1d48efb`, and the report identifies an NVIDIA A100 80GB PCIe with capability 8.0, consistent with the compiled target. The controller reports that its immediately preceding live check showed GPU3 at 0% utilization and 2 MiB usage, while VLLM jobs on GPUs 0/1 and other Python jobs on GPUs 5/6 were left running. No separate pre-v2 utilization/PID snapshot was saved. Idle selection and preservation of those jobs are therefore attributed to the controller's observed tool outputs, not claimed as independently persisted evidence in this run directory. This recording limitation does not invalidate the numerical results or change their scope.

Negative evidence and final limits:

The earlier `20261023T011403Z-bc86c383e901/microgate.json` was inspected and still says `failed`. Its absence of a passing noncontiguous case and original failure evidence remain separate from this successful fresh run. The successful run establishes native compatibility and bounded first-derivative numerical consistency for the recorded binary/device/toolchain, including the fixed layout case and tested stream sequence. It is not a new MRI trial and supplies no reconstruction, PSNR/SSIM/NMSE, full-scale speed/memory, full-range input-accuracy, original-renderer trajectory, checkpoint, or clinical/scientific quality claim. The approved pathological-gradient limitation and actual-M support policy remain unchanged.

No GPU work, test rerun, rebuild, new numerical probe, MRI read, source change, commit, or subagent dispatch occurred. This audit only read recorded artifacts and checked their internal bindings. This report is the sole write.
