"""Set the initial GPU environment, then exec an existing reviewed supervisor."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = Path('/home/congcongliu/anaconda3/envs/nvgen/bin/python')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('route', choices=('frequency_tvwave_mm', 'inner_precision_mm', 'frequency_adam', 'frequency_lowreg_mm', 'fixed_geometry_oracle', 'phase_firstorder'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    hidden = ('--reserved-gpu', '--reservation-token', '--claim-fd')
    if not extra or any(x.startswith('--') and any(h.startswith(x.split('=', 1)[0]) for h in hidden) for x in extra):
        parser.error('public supervisor arguments are required; internal ownership arguments are forbidden')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited is not None:
        devices = [x.strip() for x in inherited.split(',')]
        if not devices or any(not x.isdecimal() for x in devices) or args.gpu not in map(int, devices):
            parser.error('requested GPU conflicts with inherited CUDA_VISIBLE_DEVICES restriction')
    directory = ROOT / args.route
    command = [str(PYTHON), '-X', 'utf8', '-B', '-u', str(directory / 'run_gpu.py'), *extra]
    settings = {'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONUTF8': '1', 'PYTHONPATH': ''}
    if args.describe:
        print(json.dumps({'command': command, 'cwd': str(directory), 'environment_overrides': settings}))
        return
    os.chdir(directory)
    os.execve(PYTHON, command, {**os.environ, **settings})


if __name__ == '__main__':
    main()
