"""Tuned sampler with a per-step transition-noise scale eta, derived from the frozen source.

The SHA-pinned functions/denoising_tianyuan.py text is loaded and exactly ONE line is
replaced: the reverse-SDE update

    z_t = xt + (g^2 - g'^2) * eps_t + sqrt(g^2 - g'^2) * noise

becomes the eta family (Song et al. reverse-SDE / probability-flow interpolation)

    z_t = xt + (1 + eta^2)/2 * (g^2 - g'^2) * eps_t + eta * sqrt(g^2 - g'^2) * noise

eta = 1 returns the original expression verbatim (bit-exact); eta = 0 is the
deterministic probability-flow (Euler) step. torch.randn_like is still called every
step, so all RNG draws (initial noise, JVP noise, later steps) stay paired with the
accepted run. Nothing else in the sampler changes.
"""
import hashlib
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'functions' / 'denoising_tianyuan.py'
SOURCE_SHA256 = '8e2db2cc0ccf3b15e4257d2b8e4e654cc22fe2ddaaec1a27b286442caa0c2d4c'
ORIGINAL_LINE = '            z_t = xt + (gt ** 2 - gt_next ** 2) * eps_t + (gt ** 2 - gt_next ** 2).sqrt() * noise\n'
NEW_LINE = '            z_t = _eta_transition(xt, gt, gt_next, eps_t, noise, opts)\n'
HELPER = '''

def _eta_transition(xt, gt, gt_next, eps_t, noise, opts):
    eta = float(opts.get('eta', 1.))
    if not 0. <= eta <= 1.:
        raise ValueError('eta must be in [0, 1]')
    if eta == 1.:
        return xt + (gt ** 2 - gt_next ** 2) * eps_t + (gt ** 2 - gt_next ** 2).sqrt() * noise
    return xt + (.5 * (1. + eta ** 2)) * (gt ** 2 - gt_next ** 2) * eps_t + eta * (gt ** 2 - gt_next ** 2).sqrt() * noise
'''


def patched_source(text):
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Frozen sampler source changed')
    if text.count(ORIGINAL_LINE) != 1:
        raise ValueError('Expected exactly one transition line')
    return text.replace(ORIGINAL_LINE, NEW_LINE) + HELPER


def load():
    """Return a module whose generalized_steps_tianyuan honours cfg['eta']."""
    module = types.ModuleType('eta_denoising_tianyuan')
    module.__file__ = str(SOURCE) + ' [eta patch]'
    exec(compile(patched_source(SOURCE.read_text()), module.__file__, 'exec'), module.__dict__)
    return module
