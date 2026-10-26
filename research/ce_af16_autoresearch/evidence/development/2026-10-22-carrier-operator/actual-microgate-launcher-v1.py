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
