"""CPU tests for the stdlib-only guard logic in run.py (no torch, data or GPU needed)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

PROTOCOL = json.loads((Path(__file__).resolve().parent / 'protocol.json').read_text())
BASE_PARAMS = {"group": "adaptive", "mu": 11.0, "sigma_y": 0.3, "lambda_z": 0.0001, "lambda_max": 1,
               "tau_m": 0.3, "tau_a": 0.1, "a0": 0.7, "t_jac": 100, "n_steps": 50, "fd_check": True,
               "t_late": -1, "w_late": 1}
BASE = dict(cases=PROTOCOL['tune_cases'] + PROTOCOL['confirmation_cases'] + ['DYF-FSE-T/slice_03'])


class HarnessTest(unittest.TestCase):
    def test_only_sigma_y_changes(self):
        params = run.make_params(BASE_PARAMS, 'sigma_y', 0.5, PROTOCOL)
        diff = {k for k in BASE_PARAMS if params[k] != BASE_PARAMS[k]}
        self.assertEqual(diff, {'sigma_y'})
        self.assertEqual(params['sigma_y'], 0.5)
        self.assertEqual(BASE_PARAMS['sigma_y'], 0.3)  # caller dict untouched

    def test_null_params_identical(self):
        self.assertEqual(run.make_params(BASE_PARAMS, 'sigma_y', 0.3, PROTOCOL), BASE_PARAMS)
        self.assertEqual(run.make_params(BASE_PARAMS, 'eta', 1.0, PROTOCOL), BASE_PARAMS)

    def test_eta_adds_only_eta(self):
        params = run.make_params(BASE_PARAMS, 'eta', 0.5, PROTOCOL)
        self.assertEqual(params, dict(BASE_PARAMS, eta=0.5))
        self.assertEqual(run.run_variable(params, PROTOCOL), ('eta', 0.5))
        self.assertEqual(run.run_variable(BASE_PARAMS, PROTOCOL), ('sigma_y', 0.3))
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                run.make_params(BASE_PARAMS, 'eta', bad, PROTOCOL)
        with self.assertRaises(ValueError):
            run.make_params(BASE_PARAMS, 'mu', 12, PROTOCOL)

    def test_out_of_range_and_wrong_base_rejected(self):
        for bad in (0.1, 2.5):
            with self.assertRaises(ValueError):
                run.make_params(BASE_PARAMS, 'sigma_y', bad, PROTOCOL)
        with self.assertRaises(ValueError):
            run.make_params(dict(BASE_PARAMS, sigma_y=0.4), 'sigma_y', 0.5, PROTOCOL)

    def test_resolve_cases(self):
        self.assertEqual(run.resolve_cases('tune', PROTOCOL, BASE), PROTOCOL['tune_cases'])
        self.assertEqual(run.resolve_cases('null', PROTOCOL, BASE), [PROTOCOL['null_case']])
        for bad in ('', 'X/slice_99', 'DYF-FSE-T/slice_01,DYF-FSE-T/slice_01'):
            with self.assertRaises(ValueError):
                run.resolve_cases(bad, PROTOCOL, BASE)

    def test_budget_counts_reservations_including_failures(self):
        ledger = [dict(event='reserve', tag='a', sigma_y=.5, case='c%d' % i) for i in range(22)]
        ledger.append(dict(event='failed', tag='a', case='c0'))
        self.assertEqual(run.check_budget(2, PROTOCOL, ledger), 22)
        with self.assertRaises(RuntimeError):
            run.check_budget(3, PROTOCOL, ledger)

    def test_no_repeat_same_tag_or_value(self):
        ledger = [dict(event='reserve', tag='sy0.5', sigma_y=0.5, case='DYF-FSE-T/slice_01'),
                  dict(event='reserve', tag='eta0.5', variable='eta', value=0.5, sigma_y=0.3,
                       case='DYF-FSE-T/slice_01')]
        c = ['DYF-FSE-T/slice_01']
        with self.assertRaises(RuntimeError):
            run.check_no_repeat('other', 'sigma_y', 0.5, c, ledger)
        with self.assertRaises(RuntimeError):
            run.check_no_repeat('sy0.5', 'sigma_y', 0.7, c, ledger)
        with self.assertRaises(RuntimeError):
            run.check_no_repeat('other', 'eta', 0.5, c, ledger)
        run.check_no_repeat('sy0.7', 'sigma_y', 0.7, c, ledger)
        run.check_no_repeat('eta0', 'eta', 0.0, c, ledger)
        run.check_no_repeat('sy0.5x', 'sigma_y', 0.5, ['MSP-FSE-T/slice_06'], ledger)

    def test_ledger_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.jsonl'
            self.assertEqual(run.read_ledger(path), [])
            run.append_ledger(dict(event='reserve', case='c'), path)
            run.append_ledger(dict(event='complete', case='c'), path)
            self.assertEqual([e['event'] for e in run.read_ledger(path)], ['reserve', 'complete'])

    def test_bind_visibility(self):
        env = {}
        run.bind_visibility('4', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4')
        with self.assertRaises(ValueError):
            run.bind_visibility('5', {'CUDA_VISIBLE_DEVICES': '4'})
        for bad in ('0,1', 'all', ''):
            with self.assertRaises(ValueError):
                run.bind_visibility(bad, {})


class EtaSamplerTest(unittest.TestCase):
    def test_patch_is_single_line_and_pinned(self):
        import eta_sampler
        text = eta_sampler.SOURCE.read_text()
        patched = eta_sampler.patched_source(text)
        old, new = text.splitlines(), patched.splitlines()
        changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertIn('_eta_transition', new[changed[0]])
        with self.assertRaises(ValueError):
            eta_sampler.patched_source(text + '#')

    def test_transition_eta1_bit_exact_eta0_deterministic(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed (run on server)')
        import eta_sampler
        ns = {}
        exec(eta_sampler.HELPER, ns)
        step = ns['_eta_transition']
        g = torch.Generator().manual_seed(0)
        xt, eps, noise = (torch.randn(1, 2, 8, 8, generator=g) for _ in range(3))
        gt, gn = torch.tensor(0.7), torch.tensor(0.4)
        original = xt + (gt ** 2 - gn ** 2) * eps + (gt ** 2 - gn ** 2).sqrt() * noise
        self.assertTrue(torch.equal(step(xt, gt, gn, eps, noise, {}), original))
        self.assertTrue(torch.equal(step(xt, gt, gn, eps, noise, {'eta': 1.0}), original))
        flow = xt + .5 * (gt ** 2 - gn ** 2) * eps
        self.assertTrue(torch.allclose(step(xt, gt, gn, eps, noise, {'eta': 0.0}), flow, atol=0, rtol=1e-7))
        with self.assertRaises(ValueError):
            step(xt, gt, gn, eps, noise, {'eta': 1.5})


if __name__ == '__main__':
    unittest.main()
