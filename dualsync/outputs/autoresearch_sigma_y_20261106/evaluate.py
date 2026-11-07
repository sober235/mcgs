"""Score one sigma_y run against the registered accepted Tuned (base seed 1234), CPU only.

Usage (server, repo root):
  python outputs/autoresearch_sigma_y_20261106/evaluate.py --tag sy0.5 --cases tune
  python outputs/autoresearch_sigma_y_20261106/evaluate.py --tag null --cases null --null

Uses the frozen evaluator (SHA-pinned) and base-protocol guards unchanged. Writes
runs/<tag>/metrics.json once; never overwrites.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as harness  # noqa: E402  (stdlib-only module)


def load_evaluator(protocol):
    if harness.digest(protocol['evaluator']) != protocol['evaluator_sha256']:
        raise ValueError('Frozen evaluator changed')
    spec = importlib.util.spec_from_file_location('_frozen_snr_evaluator', protocol['evaluator'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pair(tag, case, protocol):
    import numpy as np
    import scipy.io as sio
    cand = sio.loadmat(str(HERE / 'runs' / tag / (case + '.mat')))
    ref = sio.loadmat(str(Path(protocol['anchor_root']) / (case + '.mat')))
    for key in ('cg_sense', 'coeff', 'orig_shape'):
        if not np.array_equal(np.squeeze(cand[key]), np.squeeze(ref[key])):
            raise ValueError('Preprocessing differs from accepted anchor: %s %s' % (case, key))
    image = np.squeeze(cand['diff'])
    reference = dict(cg_sense=np.squeeze(ref['cg_sense']), tuned=np.squeeze(ref['diff']),
                     coeff=float(np.squeeze(ref['coeff'])),
                     orig_shape=tuple(np.ravel(ref['orig_shape']).astype(int)))
    return image, reference, json.loads(str(np.squeeze(cand['params_json'])))


def null_check(image, reference, protocol):
    import numpy as np
    tuned = reference['tuned']
    rel = float(np.linalg.norm(image - tuned) / np.linalg.norm(tuned))
    return dict(relative_l2=rel, exact=bool(np.array_equal(image, tuned)),
                passed=rel <= protocol['null_max_relative_l2'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--null', action='store_true')
    a = parser.parse_args(argv)
    harness.require_isolated_python()
    protocol, base = harness.load_protocols()
    cases = harness.resolve_cases(a.cases, protocol, base)
    out = HERE / 'runs' / a.tag / 'metrics.json'
    if out.exists():
        raise RuntimeError('metrics.json already exists for ' + a.tag)
    images, refs, params = {}, {}, set()
    for case in cases:
        images[case], refs[case], p = load_pair(a.tag, case, protocol)
        variable, value = harness.run_variable(p, protocol)
        if p != harness.make_params(base['params'], variable, value, protocol):
            raise ValueError('Run changed more than one approved variable: ' + case)
        params.add((variable, value))
    if len(params) != 1:
        raise ValueError('Mixed settings in one tag')
    variable, value = params.pop()
    sigma_y = value if variable == 'sigma_y' else protocol['variables']['sigma_y']['accepted']
    if a.null:
        if (variable, value) != ('sigma_y', protocol['variables']['sigma_y']['accepted']):
            raise ValueError('Null run must use accepted settings')
        result = dict(kind='null', sigma_y=sigma_y, cases=cases,
                      per_case={c: null_check(images[c], refs[c], protocol) for c in cases})
        result['passed'] = all(r['passed'] for r in result['per_case'].values())
    else:
        report = load_evaluator(protocol).evaluate_images(images, refs, cases, base)
        report.pop('metric_contract', None)
        result = dict(kind='trial', variable=variable, value=value, sigma_y=sigma_y, cases=cases, **report)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    if a.null:
        print(json.dumps(result, indent=1))
        return
    print('%s=%g score=%.5f feasible=%s' % (variable, value, result['score'], result['feasible']))
    for case, row in result['per_case'].items():
        print('  %-20s bg x%.3f  in x%.3f  ssimT %.4f  edge %.3f  p95 %.1f  fg %.3f  inSSIM %.4f/%.4f' % (
            case, row['bg_snr_gain'], row['inband_snr_gain'], row['tuned_coarse_ssim'],
            row['strong_edge_amplitude_ratio'], row['symmetric_edge_p95_pixels'], row['fg_mean_ratio'],
            row['input_coarse_ssim'], row['tuned_input_coarse_ssim']))
    for v in result['violations']:
        print('  VIOLATION', v)


if __name__ == '__main__':
    main()
