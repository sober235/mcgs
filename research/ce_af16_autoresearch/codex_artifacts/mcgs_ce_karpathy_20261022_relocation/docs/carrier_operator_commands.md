# Sparse carrier component commands

Scope: selected-row numerical component only. This stage does not authorize MRI
integration, optimizer changes, reconstruction, or metric claims. No historical
renderer, source, build, mask, data or output is modified. Only first derivatives
are supported. Frequencies use cycles per voxel in D,H,W order.

The API is `carrier_operator.carrier_correction(positions, log_scales, rotations,
density, frequencies, volume_shape)`; selected rows have float32 CUDA shapes
`[K,3], [K,3], [K,4], complex64[K], [K,3]`. All tensors share one CUDA device.
Import does not build or initialize CUDA. Forward requires a successful explicit
`load_carrier_extension(build_directory)`, which returns `(module, provenance)`.
Noncontiguous inputs retain autograd through contiguous copies. Empty K remains
differentiable. Zero frequency follows the full derivative path.

CPU checks (no build, no GPU):

```sh
cd /home/congcongliu/codex_artifacts/mcgs_ce_karpathy_20261022_relocation
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_carrier_cuda.py tests/test_carrier_reference.py
```

GPU tests are explicitly skipped without `CARRIER_RUN_GPU=1`. Such a CPU run is
not a CUDA numerical pass. Compilation and GPU execution remain unverified until
independent source review and the subsequent authorized gate.

Tile pairs retain detached historical floor/ceil/clamp and stable atom ordering.
Bounds use the actual symmetric inverse covariance M: reliable positive definite
M gives axis radii `3*sqrt(diag(inv(M)))*shape/2`. Finite singular or poorly
conditioned M receives full-volume tile bounds; the kernel still applies exact
`mahal<=9` support. This conservative bound is necessary because epsilon-based
quaternion normalization can make R nonorthogonal. It does not promise bitwise
support parity with the historical approximate bbox for pathological inputs.
Derived nonfinite geometry is refused before bound conversion or pointer access.
This includes intermediate exponential scales and quaternion norms: finite input
values whose float32 conversion overflows are refused before the autograd
boundary, even when a later inverse would conceal the overflow in a finite M.
There is no dense voxel-by-all-atoms production array or backend fallback.
Positive tile ceil-division uses `1+(n-1)//8`; native tile coordinates and voxel
offset products use int64. Shapes with a voxel product above INT32_MAX remain
unsupported, while a single INT32_MAX axis is arithmetically supported without
adding seven to a signed int32 dimension.

After independent source review, inspect available resources before selecting a
GPU. Respect scheduler allocation and CUDA visibility, leave existing workloads
running, and choose an idle compatible device with sufficient memory:

```sh
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

Set `CUDA_VISIBLE_DEVICES` to the chosen allocation/device and set
`TORCH_CUDA_ARCH_LIST` to its actual compute capability (for example, `8.6` only
for a device with capability 8.6). Do not blindly use the example capability.
Use the following command only at that reviewed stage. It creates unique build
and report directories, preserves separate stdout/stderr and an exact command
record, and propagates test/report failure to its public exit status:

```sh
export CUDA_VISIBLE_DEVICES='<inspected compatible allocation/device>'
export TORCH_CUDA_ARCH_LIST='<actual compute capability>'
export CUDA_HOME=/usr/local/cuda-11.8
export CXX=/usr/bin/c++
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -B - <<'PY'
import hashlib, json, os, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path
from carrier_operator import _UNSUPPORTED_BUILD_ENV
root = Path.cwd().resolve()
run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12]
out = root / 'carrier_verification' / run
out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, CARRIER_RUN_GPU='1',
           PATH='/usr/local/cuda-11.8/bin:/usr/bin:/bin', CXX='/usr/bin/c++',
           CUDA_LAUNCH_BLOCKING='0',
           CARRIER_BUILD_DIRECTORY=str(root / 'carrier_builds' / run),
           CARRIER_REPORT_PATH=str(out / 'microgate.json'),
           CARRIER_HISTORICAL_BINARY='/home/congcongliu/codex_artifacts/mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs/gaussian/cuda_ext/build/tile_voxelize_cuda_0141c525_7aedd3b0.so')
# Sanitize only the child build/test process; record every removed selector.
removed_overrides = sorted(k for k in _UNSUPPORTED_BUILD_ENV if k in env)
for name in _UNSUPPORTED_BUILD_ENV:
    env.pop(name, None)
assert env['CUDA_HOME'] == '/usr/local/cuda-11.8'
assert env['CUDA_VISIBLE_DEVICES'] and env['TORCH_CUDA_ARCH_LIST']
command = [sys.executable, '-B', '-m', 'pytest', '-q', '-s',
           '-p', 'no:cacheprovider', 'tests/test_carrier_cuda.py', '-k', 'gpu']
record = {'command': command, 'cwd': str(root),
          'removed_build_override_names': removed_overrides,
          'environment': {k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'TORCH_CUDA_ARCH_LIST',
              'CUDA_HOME', 'CXX', 'PATH', 'CUDA_LAUNCH_BLOCKING', 'MAX_JOBS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'CARRIER_RUN_GPU', 'CARRIER_BUILD_DIRECTORY', 'CARRIER_REPORT_PATH', 'CARRIER_HISTORICAL_BINARY']},
          'sources_before': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
              ['carrier_operator.py', 'carrier_reference.py', 'carrier_cuda/carrier.cpp',
               'carrier_cuda/carrier_kernel.cu', 'tests/test_carrier_cuda.py']}}
(out/'command.json').write_text(json.dumps(record, indent=2))
with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
    result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
record['exit_code'] = result.returncode
record['sources_after'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in record['sources_before']}
(out/'command.json').write_text(json.dumps(record, indent=2))
print(out)
if result.returncode:
    raise SystemExit(result.returncode)
report = json.loads((out/'microgate.json').read_text())
assert report['status'] == 'passed'
assert record['sources_before'] == record['sources_after']
assert report['historical_binary_sha256_before'] == report['historical_binary_sha256_after']
PY
```

CUDA_HOME must be pinned before importing cpp_extension: the system default
CUDA 12.6 does not match GSTMI Torch 2.7.1+cu118. The loader refuses unowned or
historical destinations and requires a directory strictly below `carrier_builds/`.
It uses a source/toolchain/flags/Torch ABI/architecture identity for the module
and its artifact subdirectory. Its manifest binds source hashes, compiler and
nvcc paths/versions/hashes, explicit CUDA host gcc, nvcc profile, builder source,
Ninja executable, effective environment, flags, Torch configuration, ABI,
architecture, loaded binary SHA256, and the generated Ninja commands/text/hash.
Reusing a mismatched or partial artifact is refused; use a
new dedicated run directory after preserving any failed build evidence.

The loader requires `CXX=/usr/bin/c++` and the exact child PATH shown above;
nvcc's host compiler is explicitly `-ccbin /usr/bin/gcc`. It rejects inherited
`CC`, `PYTORCH_NVCC`, CUDA/nvcc compiler selectors and appended/prepended flags,
CUDA include/CUDNN/ROCM overrides, compiler/include/library search variables,
implicit C/C++/link flags, dynamic-library overrides, and Torch's ABI/dependency
check overrides before tool invocation or artifact reuse. The documented command
explicitly removes these selectors only in its child process and records their
names. It does not depend on inherited overrides being absent by accident.
This policy was scoped against the installed Torch JIT build path and nvcc/GCC
controls; the loader does not alter the parent process environment. Cached
cpp_extension toolkit/CUDNN/ROCM configuration is also checked before building.

The microgate requires all 14 declared fixtures plus invalid-input cases to run;
requesting it without GPU availability fails. Ordinary bounded fixtures compare
the output and all five public-input VJPs against the independent CPU double
oracle at frozen `rtol=2e-4, atol=2e-5`, with three backward repetitions recording
atomic variation. Zero frequency requires exactly zero correction and exactly
zero original-input correction gradients, plus a constructed nonzero frequency
gradient. Ragged, empty, overlap, outside-support, noncontiguous, axis quarter
cycles, near-bound and tiny-phase cases are included.

The required `nondefault_stream` fixture runs a producer, operator forward/all
VJPs, and consumer with explicit event dependencies. It copies output and
gradient snapshots to pinned host memory on the consumer stream and synchronizes
only the consumer event before comparison. A delayed default stream must still
be blocked when capture completes; otherwise the test fails as unisolated.
Initialization/warmup happens before that sequence, and final global sync is
cleanup after snapshots. Missing this fixture makes the report fail. Its output
and all public-input VJPs use the same frozen tolerances and report atomic
variation, consumer output, and the observed default-stream delay state.

The additional nearly singular quaternion fixture checks public output/support
and local center/M/density/frequency VJPs using identical rounded float32 geometry
converted to double. Its report names this narrower scope: it makes no claim of
double-precision relative accuracy for the ill-conditioned quaternion chain.
It does not relax ordinary-fixture tolerances. The original binary SHA must remain
`3973daeef5f55ebfed931c38538a37d83921a9287b96b69a431da238057194b5`.

A passing report establishes tiny-fixture numerical consistency only. Full-scale
performance, reproducibility of MRI optimizer trajectories, real G2 zero/control
parity, checkpoints, and MRI quality remain separate future gates.
