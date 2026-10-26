### Task 1: Double precision reference and derivative acceptance

**Files:** Create carrier_reference.py and tests/test_carrier_reference.py. Report owned task-1-report.md in this plan's SDD workspace.

**Interfaces:** Produces `carrier_correction_reference(positions, log_scales, rotations, density, frequencies, volume_shape)` CPU float64/complex128 autograd output; selected K rows in shapes[K,3],[K,3],[K,4],[K],[K,3]. `volume_shape` three positive ints. Reject invalid shapes/dtypes/devices/nonfinite/frequencies outside strict Nyquist. EmptyK supported with connected zero gradients. Geometry/internal helpers may be private. No CUDA import/build side effect.

- [ ] Write failing tests for direct-Gabor-minus-Gaussian algebra, all-five-input derivative correctness, zero-frequency old-input gradient zeros/nonzero frequency example, anisotropic quarter-cycle units, emptyK/overlap/outside support, noncontiguous inputs and validation refusal. Use exact acceptance tolerances and central differences avoiding support boundary crossings. Independent direct formula must not call implementation geometry helpers; analytic VJP checks include center phase term.
- [ ] Run RED with `env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q -p no:cacheprovider tests/test_carrier_reference.py`; record actual failure and exit.
- [ ] Implement minimal reference in carrier_reference.py using native Torch float64 autograd, stable correction and original geometry convention. No hardcoded test output or production fallback.
- [ ] Run same focused command GREEN once; self-review calculations/input error paths; fixes rerun affected tests only. Freeze source/test hashes and output/exit evidence. Confirm historical224 source closure unchanged.
- [ ] Controller inspects report/current diff, commits only owned two files, generates exact BASE..HEAD package and dispatches fresh task spec/quality review. Resolve substantive findings within3 rounds before Task2. No independent suite rerun absent a named doubt.

