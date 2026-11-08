#!/usr/bin/env python3
"""threeaxis.py -- CE AF16 three-axis autoresearch driver (2026-11-07).

Goal (user, 2026-11-07): MC-GS strictly beats all ten opponents in all six of pre/post x PSNR/SSIM/NMSE on at
least one frozen lesion-core plane of EACH axis (transverse a0, coronal a1, sagittal a2), on both dev seeds 42/7.

Metric (one number, higher is better), read from the frozen evaluator's lesion_planes block only:
    M3(seed) = min over axes a of ( max over the frozen lesion planes of axis a of score_six )
    M3       = min over dev seeds {42, 7} of M3(seed)
M3 > 0  <=>  every axis has a plane that wins all six conditions on both seeds  (success; then confirm 201/207).

Numerics are the losstune chain, imported read-only: long_chain.stage2 (625 updates from the seed's iteration-4500
base, relax alpha, export) and long_chain.score (KE/evaluate.py, CPU). Extensions live in this process only and
leave every frozen file unchanged:
  * loss_edits may also set 'nl_sharpen'. The frozen trainer reads it natively (BM4D transform-domain sharpening
    alpha, losses/nonlocal_red.make_denoiser); long_chain's LOSS_EDIT_KEYS whitelist is widened, and its own
    numeric checks still apply.
  * variant {'kind': 'joint_bm'}: both contrasts' BM4D targets use ONE set of block matches, computed on a guide
    made from the two current renders, (pre/sigma_pre + post/sigma_post) / 2. The trainer refreshes pre then
    post in the same iteration before either target is used, so the pre call stores its input, the post call
    denoises both, and a wrapped NonLocalRED.refresh replaces the pre target before the loss reads it.
  * variant {'kind': 'sigma', 'scale': {'pre': a, 'post': b}, 'decay': d}: the BM4D sigma of refresh k of a
    contrast is nl_sigma_rel * scale[contrast] * d**k (k = 0 at nl_start); with nl_refresh < steps this is an
    annealed plug-and-play-style schedule instead of the constant-sigma refreshes that failed (FV16/FV17).
  * variant {'kind': 'init_scale', 'factor': c}: the stage-2 start state is the seed's iteration-4500 base with every
    Gaussian scale multiplied by c and all four densities by c**-3 (the voxelizer kernel is peak-normalised,
    exp(-mahal/2), so this keeps each Gaussian's mass, i.e. the low frequencies). Rationale: the density gradient
    is the residual filtered by the Gaussian kernel, so with sigma ~ 0.7-1 voxel the 0.3-0.5 cycles/px residual
    that separates MC-GS from ZS-SSDU on transverse planes reaches the parameters attenuated ~10-25x. The
    modified copy is written next to the run (never over the base) and its SHA-256 recorded.
  * variant {'kind': 'dc_target', 'gain': g (default 1)}: each refresh denoises the data-consistent image of the current render instead
    of the render itself, x_dc = sum_c conj(S_c) F^-1[ mf*y_dc + (1-mf)*F(S_c x) ] -- the frozen canonical export
    formula (nonlocal_autoresearch/runner.export_trainer_volume) on the trainer's own acquired data. The target thus
    carries the measured high frequencies, and block matching can spread them (one plug-and-play ISTA step per
    refresh; with nl_refresh < steps and a sigma decay it iterates). Uses the trainer's frozen
    data.known_zero.hard_dc_combine on its own acquired k-space (kspace_dc, mask_fit); no GT.
  * variant {'kind': 'profile', 'attrs': {...}}: bm4d_pos with BM4DProfile attributes overridden (PROFILE_ATTRS),
    e.g. anisotropic blocks / search windows (transverse = phase-encode plane). joint_bm
    accepts the same optional 'attrs'.
Variants compose: an item relative to the keeper runs the keeper's variant parts plus its own, a part of the same
kind replacing the keeper's ({'kind': 'stack', 'parts': [...]}); install order profile -> joint_bm -> sigma, with
profile attrs merged into joint_bm's when both are present; init_scale acts on the start checkpoint.
  * variant {'kind': 'scale_lr', 'factor': k}: the stage-2 Adam rate of the Gaussian scales only is multiplied by k
    (kernel widths otherwise stay at their initialisation; add-on round 2026-11-08).
No GT or ROI is read by any training path here; GT enters only through the frozen evaluator.

Loop (Karpathy autoresearch; program.md is the human layer):
  claim next queued item -> build recipe = current keeper + one change -> seed 42 (screen) -> if M3(42) beats the
  keeper's M3(42) by > KEEP_MARGIN, seed 7 -> keep iff M3 > keeper M3 + KEEP_MARGIN -> ledger -> next.
  An item with "screen": false runs every seed before the same keep rule (axis margins measured on both seeds even
  when the limiting axis does not move). A control item may give "reference_scores" {seed: score dir} to record
  per-seed deviations from a reference run.
  An item may give "bases" {seed: iteration-4500 checkpoint} to start stage 2 from a retrained base (add-on H2).
  A kept item with M3 > 0 is re-run on confirmation seeds 201/207; success requires both > 0 and a visual check.
  Dead or failed attempts are preserved and never resumed; a retry needs a new id.

Commands: selftest (CPU), stage2 (GPU child), worker (driver on ONE launcher-exposed GPU), summary, enqueue, skip,
probe-relax, promote (user-decided keeper change, recorded in the ledger).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = Path('/home/congcongliu/anaconda3/envs/GSTMI/bin/python')
ARTS = Path('/data0/congcong/code/GS/3DGS/outputs/codex_artifacts')
WS = ARTS / 'mcgs_ce_losstune_20261101'
LONG_CHAIN = WS / 'long_chain.py'
NONLOCAL_RED = ARTS / 'mcgs_ce_autoresearch_20261017/nonlocal_autoresearch/trainer/paired_gs/losses/nonlocal_red.py'
ROOT = ARTS / 'mcgs_ce_threeaxis_20261107'
HERE = Path(__file__).resolve()
DEV_SEEDS = (42, 7)
CONFIRM_SEEDS = (201, 207)
KEEP_MARGIN = 1e-4
EXTRA_LOSS_KEYS = frozenset({'nl_sharpen'})
VARIANTS = ('none', 'joint_bm', 'profile', 'sigma', 'init_scale', 'dc_target', 'scale_lr', 'stack')
PART_ORDER = ('profile', 'joint_bm', 'dc_target', 'sigma', 'init_scale', 'scale_lr')
# BM4DProfile attributes a variant may override (block / step / search sizes, stack sizes, matching thresholds).
# Tuples are given as JSON lists, ordered like the crop axes (a0 readout, a1, a2). 'denoise_residual' is excluded:
# bm4d 4.2.5's residual path fails with a scalar sigma (operands (2N..) vs (N..) broadcast error, tested 2026-11-07).
PROFILE_ATTRS = frozenset({'bs_ht', 'bs_wiener', 'step_ht', 'step_wiener', 'search_window_ht', 'search_window_wiener',
                           'max_stack_size_ht', 'max_stack_size_wiener', 'tau_match', 'tau_match_wiener', 'beta'})
SCRUB = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_LAUNCH_BLOCKING', 'PYTHONSTARTUP', 'PYTHONPATH',
         'PYTHONHOME', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')
NEED_FREE_GB = 50.
GPU_WAIT_LIMIT = 4 * 3600.


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def require(ok, msg):
    if not ok:
        raise RuntimeError(f'REFUSED: {msg}')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def long_chain():
    """long_chain imports only the standard library at module level; torch is imported inside stage2."""
    lc = sys.modules.get('_ta_long_chain') or load_module(LONG_CHAIN, '_ta_long_chain')
    lc.LOSS_EDIT_KEYS = frozenset(lc.LOSS_EDIT_KEYS) | EXTRA_LOSS_KEYS
    return lc


# ------------------------------------------------------------------ metric (frozen evaluator output only)
def axis_summary(score_dir):
    score_dir = Path(score_dir)
    planes = read_json(score_dir / 'lesion_planes.json')
    require(planes['eligible_planes'] == 59 and len(planes['planes']) == 59, 'unexpected frozen lesion-plane set')
    axes = {}
    for p in planes['planes']:
        row = axes.setdefault(p['axis'], dict(planes=0, passing=0, best=None))
        row['planes'] += 1
        row['passing'] += bool(p['strict_success'])
        if row['best'] is None or p['score_six'] > row['best']['s6']:
            limit = min(p['comparisons'].items(), key=lambda kv: kv[1]['margin'])
            row['best'] = dict(s6=p['score_six'], index=p['index'],
                               wins=sum(bool(c['strict_win']) for c in p['comparisons'].values()),
                               limiter=limit[0], limiter_opponents=limit[1]['labels'],
                               psnr={k.split(':')[0]: [c['candidate'], c['best_opponent']]
                                     for k, c in p['comparisons'].items() if k.endswith(':psnr')})
    require(sorted(axes) == [0, 1, 2] and [axes[a]['planes'] for a in (0, 1, 2)] == [17, 24, 18],
            'frozen lesion planes are not 17/24/18 per axis')
    for a, row in axes.items():
        require((row['best']['s6'] > 0) == (row['passing'] > 0), f'axis {a}: score_six sign and strict_success disagree')
    objective = read_json(score_dir / 'objective.json')
    return dict(m3=min(axes[a]['best']['s6'] for a in axes), axes={str(a): axes[a] for a in sorted(axes)},
                whole_head=objective['score'], score_dir=str(score_dir))


def short(summary):
    ax = summary['axes']
    return ' '.join(f"a{a}:{ax[a]['best']['s6']:+.4f}@{ax[a]['best']['index']}({ax[a]['best']['wins']}/6,"
                    f"pass{ax[a]['passing']})" for a in ('0', '1', '2'))


# ------------------------------------------------------------------ recipe building
def build_recipe(keeper_recipe, item):
    if 'recipe_path' in item:
        recipe = read_json(item['recipe_path'])
    elif 'recipe' in item:
        recipe = copy.deepcopy(item['recipe'])
    else:
        recipe = copy.deepcopy(keeper_recipe)
        recipe.update(copy.deepcopy(item.get('edits', {})))
        recipe['loss_edits'] = {**recipe.get('loss_edits', {}), **copy.deepcopy(item.get('loss_edits', {}))}
        for key in item.get('drop_loss_edits', ()):
            recipe['loss_edits'].pop(key, None)
    recipe.setdefault('milestones', [recipe['steps']])
    return recipe


def check_recipe(recipe, variant, scratch):
    """Validate with long_chain's own load_recipe (widened whitelist) on a scratch file."""
    lc = long_chain()
    path = Path(scratch) / 'recipe_check.json'
    write_json(path, recipe)
    loaded = lc.load_recipe(path)
    path.unlink()
    require(variant.get('kind', 'none') in VARIANTS, f'unknown variant {variant}')
    for part in parts(variant):
        check_part(part, loaded)
    if 'nl_sharpen' in loaded['loss_edits']:
        require(0 < loaded['loss_edits']['nl_sharpen'] <= 4, 'nl_sharpen must be in (0, 4]')
    return loaded


def parts(variant):
    """Normalised list of variant parts (no 'none'), at most one per kind, in install order."""
    kind = variant.get('kind', 'none')
    items = variant['parts'] if kind == 'stack' else ([] if kind == 'none' else [variant])
    out = {}
    for part in items:
        require(part.get('kind') in PART_ORDER, f'unknown variant part {part}')
        require(part['kind'] not in out, f'duplicate variant part {part["kind"]}')
        out[part['kind']] = copy.deepcopy(part)
    return [out[k] for k in PART_ORDER if k in out]


def compose(keeper_variant, item_variant):
    merged = {p['kind']: p for p in parts(keeper_variant or {'kind': 'none'})}
    for part in parts(item_variant or {'kind': 'none'}):
        if part.get('remove'):
            merged.pop(part['kind'], None)
        else:
            merged[part['kind']] = part
    out = [merged[k] for k in PART_ORDER if k in merged]
    return {'kind': 'none'} if not out else out[0] if len(out) == 1 else {'kind': 'stack', 'parts': out}


def check_part(variant, loaded):
    if variant.get('kind', 'none') not in ('none', 'init_scale', 'scale_lr'):
        require(loaded['loss_edits'].get('nl_denoiser') == 'bm4d_pos', 'NL variants extend bm4d_pos only')
        check_attrs(variant.get('attrs', {}))
    if variant.get('kind') == 'profile':
        require(variant.get('attrs'), 'a profile variant needs attrs')
    if variant.get('kind') == 'sigma':
        check_sigma_variant(variant)
    if variant.get('kind') == 'dc_target':
        check_dc_variant(variant)
    if variant.get('kind') == 'scale_lr':
        require(set(variant) == {'kind', 'factor'} and type(variant['factor']) in (int, float)
                and 0.25 <= variant['factor'] <= 50, 'scale_lr takes only factor in [0.25, 50]')
    if variant.get('kind') == 'init_scale':
        require(type(variant.get('factor')) in (int, float) and 0.5 <= variant['factor'] <= 1.5, 'init_scale factor in [0.5, 1.5]')
    if variant.get('attrs'):
        preflight_attrs(variant['attrs'], float(loaded['loss_edits'].get('nl_sharpen', 1.0)))


# ------------------------------------------------------------------ BM4D profile overrides
def check_attrs(attrs):
    require(isinstance(attrs, dict) and set(attrs) <= PROFILE_ATTRS, f'profile attrs must be within {sorted(PROFILE_ATTRS)}')
    for key, value in attrs.items():
        if isinstance(value, list):
            require(len(value) == 3 and all(isinstance(v, int) and 1 <= v <= 32 for v in value), f'{key} must be 3 ints')
        else:
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, f'invalid {key}')


def preflight_attrs(attrs, sharpen=1.0):
    """bm4d 4.2.5 returns NaN for some legal-looking profiles (e.g. a block size of 2 on any axis with the bior1.5
    hard-threshold transform); the frozen refresh would then stop the GPU run. Check on a small synthetic volume."""
    import numpy as np
    import bm4d
    g = np.random.default_rng(0)
    base = np.zeros((40, 44, 36), np.float32)
    base[6:34, 7:37, 5:31] = 1.0
    base[15:25, 15:25, 12:20] = 1.8
    z = np.abs(base + 0.05 * g.standard_normal(base.shape)).astype(np.float32)
    pro = bm4d.BM4DProfile()
    pro.num_threads = 8
    if sharpen != 1.0:
        pro.set_sharpen(sharpen)
    apply_attrs(pro, attrs)
    y = np.asarray(bm4d.bm4d(z, 0.05, pro))
    require(y.shape == z.shape and np.isfinite(y).all(), f'BM4D profile {attrs} gives a nonfinite or misshaped output')
    return float(np.sqrt(((y - base) ** 2).mean()))


def apply_attrs(pro, attrs):
    check_attrs(attrs)
    for key, value in attrs.items():
        setattr(pro, key, tuple(value) if isinstance(value, list) else value)
    return {key: getattr(pro, key) for key in attrs}


def install_profile(nlr, variant):
    attrs = variant.get('attrs', {})
    check_attrs(attrs)
    base_make = nlr.make_denoiser               # long_chain's bm4d_pos wrapper, installed just before

    def make_denoiser(name, profile='np', threads=32, sharpen=1.0, **kwargs):
        require(name == 'bm4d_pos', f'profile variant wraps bm4d_pos only, got {name!r}')
        inner = base_make(name, profile, threads, sharpen, **kwargs)
        applied = apply_attrs(inner.profile, attrs)   # the same profile object the BM4D closure uses

        def run(z, sigma):
            out = inner(z, sigma)
            run.last_stats = dict(getattr(inner, 'last_stats', {}), variant='profile', profile_overrides=applied)
            return out
        run.profile, run.last_stats = inner.profile, {}
        return run
    nlr.make_denoiser = make_denoiser
    return nlr


# ------------------------------------------------------------------ per-contrast / annealed sigma variant
def check_sigma_variant(variant):
    scale, decay = variant.get('scale', {}), variant.get('decay', 1.0)
    require(isinstance(scale, dict) and set(scale) <= {'pre', 'post'}
            and all(type(v) in (int, float) and 0.25 <= v <= 4 for v in scale.values()), 'sigma scale must be in [0.25, 4]')
    require(type(decay) in (int, float) and 0 < decay <= 1, 'sigma decay must be in (0, 1]')
    return {k: float(v) for k, v in scale.items()}, float(decay)


def install_sigma(nlr, variant):
    scale, decay = check_sigma_variant(variant)
    orig_refresh = nlr.NonLocalRED.refresh

    def refresh(self, name, vol, it):
        base = self.sigma_rel
        k = len(self.history.get(name, []))
        factor = scale.get(name, 1.0) * decay ** k
        self.sigma_rel = base * factor
        try:
            info = orig_refresh(self, name, vol, it)
        finally:
            self.sigma_rel = base
        info['sigma_variant'] = dict(refresh_index=k, factor=factor, sigma_rel=base * factor)
        print(f"[nl] sigma variant {name} refresh {k}: sigma_rel {base:.5g} x {factor:.4g}", flush=True)
        return info
    nlr.NonLocalRED.refresh = refresh
    return nlr


# ------------------------------------------------------------------ data-consistent (plug-and-play) target
def frozen_known_zero():
    """The trainer's own data.known_zero (hard_dc_combine = the frozen canonical export formula)."""
    import importlib
    kz = sys.modules.get('data.known_zero')
    if kz is None:
        sys.path.insert(0, str(NONLOCAL_RED.parents[1]))
        kz = importlib.import_module('data.known_zero')
    require(Path(kz.__file__).resolve().parents[1] == NONLOCAL_RED.resolve().parents[1], 'foreign data.known_zero')
    return kz


def dc_blend(x, x_dc, gain):
    """x + gain * (x_dc - x): gain 1 = the data-consistent image; > 1 over-relaxes toward the measured residual."""
    return x_dc if gain == 1.0 else x + gain * (x_dc - x)


def dc_gain(variant, k):
    """gain of refresh k (0 = the first target). 'later_gain' applies from the second refresh on: a gain > 2 iterated
    over refreshes diverges (TK01: |x_t - x|/|x| 0.13 -> 0.82 -> 3.8), a gain <= 2 is a stable PnP step."""
    return float(variant.get('gain', 1.0) if k == 0 else variant.get('later_gain', variant.get('gain', 1.0)))


def check_dc_variant(variant):
    require(set(variant) <= {'kind', 'gain', 'later_gain'}, f'dc_target takes gain, later_gain; got {sorted(variant)}')
    for key in ('gain', 'later_gain'):
        gain = variant.get(key, 1.0)
        require(type(gain) in (int, float) and math.isfinite(gain) and 0 < gain <= 16, f'dc_target {key} in (0, 16]')


def install_dc_target(nlr, variant):
    import gc
    import torch
    jt = sys.modules['trainers.joint_trainer']
    trainers = [o for o in gc.get_objects() if isinstance(o, jt.JointTrainer)]
    require(len(trainers) == 1, f'dc_target needs exactly one live JointTrainer, found {len(trainers)}')
    trainer = trainers[0]
    kz = frozen_known_zero()
    tags = {trainer.name_plain: 'p', trainer.name_contrast: 'c'}
    orig_refresh = nlr.NonLocalRED.refresh

    count = {}

    def refresh(self, name, vol, it):
        tag = tags[name]
        k = count[name] = count.get(name, -1) + 1
        gain = dc_gain(variant, k)
        x = vol.detach()
        with torch.no_grad():
            x_dc = kz.hard_dc_combine(x, getattr(trainer, f'csm_{tag}'), getattr(trainer, f'kspace_dc_{tag}'),
                                      getattr(trainer, f'mask_fit_{tag}'))
            x_dc = dc_blend(x, x_dc, gain)
            ok = tuple(x_dc.shape) == tuple(x.shape) and bool(torch.isfinite(torch.view_as_real(x_dc)).all())
            rel = float((x_dc - x).abs().norm() / x.abs().norm().clamp_min(1e-12))
        require(ok, 'dc_target image invalid')
        info = orig_refresh(self, name, x_dc, it)
        del x_dc
        info['dc_target'] = dict(dc_minus_render_relative_norm=rel, gain=gain, refresh_index=k)
        print(f"[nl] dc_target {name} iter={it} refresh={k} gain={gain:g}: |x_t - x|/|x| = {rel:.5g}", flush=True)
        return info
    nlr.NonLocalRED.refresh = refresh
    return nlr


# ------------------------------------------------------------------ joint block-matching variant
def install_joint_bm(nlr, variant):
    """Patch the frozen nonlocal_red module object in this process (see module docstring)."""
    import numpy as np
    import torch
    guide_sigma = float(variant.get('guide_sigma', 1.0))
    require(math.isfinite(guide_sigma) and guide_sigma > 0, 'guide_sigma must be positive')

    def make_denoiser(name, profile='np', threads=32, sharpen=1.0, **kwargs):
        require(name == 'bm4d_pos' and not kwargs, f'joint_bm wraps bm4d_pos only, got {name!r}')
        import bm4d
        from bm4d.profiles import BM4DProfileLC
        pro = bm4d.BM4DProfile() if profile == 'np' else BM4DProfileLC()
        pro.num_threads = int(threads)
        if float(sharpen) != 1.0:
            pro.set_sharpen(float(sharpen))
        applied = apply_attrs(pro, variant.get('attrs', {}))
        state = {}

        def denoise(z, sigma, matches):
            out = bm4d.bm4d(np.ascontiguousarray(z, dtype=np.float32), float(sigma), pro, blockmatches=matches)
            return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)

        def run(z, sigma):
            if state.get('phase') is None:                 # first call of a refresh: the plain contrast
                state.update(phase='plain_pending', zp=np.array(z, dtype=np.float32, copy=True), sp=float(sigma))
                run.last_stats = {'variant': 'joint_bm', 'role': 'plain placeholder, replaced after the contrast call'}
                return np.maximum(np.asarray(z, dtype=np.float32), np.float32(0))
            require(state['phase'] == 'plain_pending', f'joint_bm out of order: {state.get("phase")}')
            zp, sp, zc, sc = state.pop('zp'), state.pop('sp'), np.asarray(z, dtype=np.float32), float(sigma)
            require(zp.shape == zc.shape and sp > 0 and sc > 0, 'joint_bm needs two same-shape crops')
            guide = np.ascontiguousarray(0.5 * (zp / np.float32(sp) + zc / np.float32(sc)), dtype=np.float32)
            t0 = time.time()
            out = bm4d.bm4d(guide, guide_sigma, pro, blockmatches=(True, True))
            require(isinstance(out, tuple) and len(out) == 2, 'bm4d did not return block matches')
            matches = out[1]
            yp, yc = denoise(zp, sp, matches), denoise(zc, sc, matches)
            stats = {}
            for tag, y in (('plain', yp), ('contrast', yc)):
                stats[tag] = {'clipped_voxels': int((y < 0).sum()), 'clipped_fraction': float((y < 0).mean()),
                              'min_before_clip': float(y.min())}
            state.update(phase='contrast_done', plain_target=np.maximum(yp, np.float32(0)))
            run.last_stats = {'variant': 'joint_bm', 'guide_sigma': guide_sigma, 'seconds_all': time.time() - t0,
                              'profile_overrides': applied,
                              'sigma_plain': sp, 'sigma_contrast': sc, 'kernel': stats}
            return np.maximum(yc, np.float32(0))
        run.profile, run.last_stats, run.state, run.joint = pro, {}, state, True
        return run

    orig_refresh = nlr.NonLocalRED.refresh

    def refresh(self, name, vol, it):
        den = self.denoiser
        if not getattr(den, 'joint', False):
            return orig_refresh(self, name, vol, it)
        st = den.state
        if st.get('phase') is None:
            info = orig_refresh(self, name, vol, it)
            st['plain_name'], st['plain_iter'] = name, int(it)
            return info
        info = orig_refresh(self, name, vol, it)
        require(st.get('phase') == 'contrast_done' and st['plain_iter'] == int(it) and st['plain_name'] != name,
                'joint_bm: the plain refresh of this iteration is missing')
        pname = st.pop('plain_name')
        st.pop('plain_iter')
        target = torch.as_tensor(st.pop('plain_target'), device=vol.device).detach()
        placeholder = self.targets[pname]
        require(tuple(target.shape) == tuple(placeholder.shape), 'joint_bm plain target shape differs')
        self.targets[pname] = target
        diff = target - placeholder.float()          # the placeholder is the plain render's magnitude crop
        joint = dict(self.info[pname], joint_replaced=True, kernel_stats=copy.deepcopy(den.last_stats),
                     target_rms_difference=float(diff.square().mean().sqrt()),
                     target_relative_norm=float(diff.norm() / placeholder.float().norm().clamp_min(self.eps)))
        self.info[pname] = joint
        self.history[pname][-1] = joint
        st['phase'] = None
        print(f"[nl] joint_bm iter={it} replaced {pname} target_relative_norm={joint['target_relative_norm']:.6g}",
              flush=True)
        return info

    nlr.make_denoiser = make_denoiser
    nlr.NonLocalRED.refresh = refresh
    return nlr


def install_scale_lr(lc, variant):
    """Multiply the stage-2 learning rate of the Gaussian scales (Adam group 1) only, right after the frozen
    relocation restore and before long_chain applies lr_scale (multiplicative, so the order does not matter). The
    exponential scheduler then decays it like every other group. Motivation: the scale rate after lr_scale is
    ~1e-4, so 625 Adam steps can move log-scales by at most ~0.07 and the kernel widths stay at their
    normalised-isotropic initialisation (diag_gauss_width.py)."""
    factor = float(variant['factor'])
    base_load = lc.load_module

    def load(path, name):
        module = base_load(path, name)
        if name == '_long_relocation_runtime':
            orig = module.restore_training_state

            def restore(trainer, checkpoint):
                out = orig(trainer, checkpoint)
                require(list(out.get('group_names', []))[1:2] == ['scales'], f"Adam group 1 is not the scales: {out.get('group_names')}")
                group = trainer.optimizer.param_groups[1]
                before = group['lr']
                group['lr'] = before * factor
                print(f'[scale_lr] scales lr {before:.6g} -> {group["lr"]:.6g} (x{factor:g}) before lr_scale', flush=True)
                return out
            module.restore_training_state = restore
        return module
    lc.load_module = load


def install_variant(lc, variant):
    ps = {p['kind']: p for p in parts(variant)}
    if 'scale_lr' in ps:
        install_scale_lr(lc, ps.pop('scale_lr'))
    if 'joint_bm' in ps and 'profile' in ps:          # joint_bm builds its own profile: merge the overrides
        ps['joint_bm'] = dict(ps['joint_bm'], attrs={**ps.pop('profile').get('attrs', {}), **ps['joint_bm'].get('attrs', {})})
    nl = [ps[k] for k in ('profile', 'joint_bm', 'dc_target', 'sigma') if k in ps]
    if not nl:
        return
    original = lc.install_bm4d_pos

    def install():
        nlr = original()
        for part in nl:
            {'profile': install_profile, 'joint_bm': install_joint_bm, 'dc_target': install_dc_target,
             'sigma': install_sigma}[part['kind']](nlr, part)
        return nlr
    lc.install_bm4d_pos = install


# ------------------------------------------------------------------ stage2 child (one GPU)
def scaled_checkpoint(src, dst, factor):
    import torch
    require(0.5 <= factor <= 1.5 and not Path(dst).exists(), f'init_scale factor {factor} / target {dst}')
    ck = torch.load(src, map_location='cpu', weights_only=True)
    st = ck['model_state']
    before = {k: float(v.float().abs().mean()) for k, v in st.items()}
    st['scales'] = (st['scales'] + math.log(factor)).contiguous()
    for key in ('density_real', 'density_imag', 'density_real_contrast', 'density_imag_contrast'):
        st[key] = (st[key] * factor ** -3).contiguous()
    torch.save(ck, dst)
    return dict(source=str(src), source_sha256=sha256(src), path=str(dst), sha256=sha256(dst), factor=factor,
                density_factor=factor ** -3, mean_abs_before=before,
                mean_abs_after={k: float(v.float().abs().mean()) for k, v in st.items()})


def cmd_stage2(args):
    variant = read_json(args.variant)
    lc = long_chain()
    install_variant(lc, variant)
    out = Path(args.out).resolve()
    ckpt, init = Path(args.ckpt), None
    scale_part = [p for p in parts(variant) if p['kind'] == 'init_scale']
    if scale_part:
        ckpt = out.parent / f'{out.name}_init_scale.pth'
        init = scaled_checkpoint(Path(args.ckpt), ckpt, float(scale_part[0]['factor']))
        print('[init_scale]', json.dumps(init), flush=True)
    lc.stage2(argparse.Namespace(ckpt=ckpt, out=out, recipe=Path(args.recipe)))
    write_json(out / 'threeaxis_extensions.json',
               dict(variant=variant, init_scale=init, extra_loss_keys=sorted(EXTRA_LOSS_KEYS), long_chain_sha256=sha256(LONG_CHAIN),
                    nonlocal_red_sha256=sha256(NONLOCAL_RED), driver_sha256=sha256(HERE),
                    source_digest=os.environ.get('MCGS_SOURCE_DIGEST')))


# ------------------------------------------------------------------ worker
@contextlib.contextmanager
def locked(root):
    with open(Path(root) / '.lock', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ledger_rows(root):
    path = Path(root) / 'experiments.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_row(root, row):
    with open(Path(root) / 'experiments.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str, allow_nan=False) + '\n')
    tsv = Path(root) / 'results.tsv'
    new = not tsv.exists()
    with open(tsv, 'a') as f:
        if new:
            f.write('id\tstatus\tm3\ts42\ts7\twhole_head_s42\tkeeper_before\thypothesis\n')
        ps = row.get('per_seed', {})
        cell = lambda s: short(ps[s]) if s in ps else '-'  # noqa: E731
        m3 = row.get('m3')
        f.write(f"{row['id']}\t{row['status']}\t{'' if m3 is None else f'{m3:+.5f}'}\t{cell('42')}\t{cell('7')}\t"
                f"{ps['42']['whole_head'] if '42' in ps else ''}\t{row.get('keeper_before', '')}\t"
                f"{row.get('hypothesis', '')}\n")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def init_root(root, queue):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'runs').mkdir(exist_ok=True)
    with locked(root):
        if (root / 'keeper.json').exists():
            return
        k0 = queue['keeper0']
        per_seed = {str(s): axis_summary(k0['scores'][str(s)]) for s in DEV_SEEDS}
        keeper = dict(id=k0['id'], recipe=read_json(k0['recipe_path']), variant={'kind': 'none'},
                      m3=min(v['m3'] for v in per_seed.values()), per_seed=per_seed, source='existing losstune run',
                      candidate_relax=k0.get('candidate_relax'), set_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(root / 'keeper.json', keeper)
        write_json(root / 'keeper_history.json', [keeper])
        log(f"keeper0 {keeper['id']} M3 {keeper['m3']:+.5f}  s42 {short(per_seed['42'])}  s7 {short(per_seed['7'])}")


def queue_items(root, queue_file=None):
    """Items with their keep margin: from one queue file (legacy) or from every ROOT/queue/*.json in name order,
    re-read at every claim so batches can be added (enqueue) while workers run."""
    files = [Path(queue_file)] if queue_file else sorted((Path(root) / 'queue').glob('*.json'))
    out = []
    for f in files:
        q = read_json(f)
        margin = float(q.get('keep_margin', KEEP_MARGIN))
        out += [(dict(item, queue_file=str(f)), margin) for item in q['items']]
    return out


def claim(root, queue, gpu):
    """queue: a path (legacy single file) or None (queue directory). Returns (item, keeper, margin)."""
    with locked(root):
        claims_path = root / 'claims.json'
        claims = read_json(claims_path) if claims_path.exists() else {}
        done = {r['id'] for r in ledger_rows(root)}
        for eid, c in list(claims.items()):
            if eid not in done and not pid_alive(c['pid']):
                append_row(root, dict(id=eid, status='abandoned_dead_worker', m3=None, claim=c,
                                      hypothesis='(dead worker; attempt preserved, never resumed)'))
                done.add(eid)
        for item, margin in queue_items(root, queue):
            if item['id'] in done or item['id'] in claims:
                continue
            claims[item['id']] = dict(gpu=gpu, pid=os.getpid(), job=os.environ.get('MCGS_JOB_DIR'),
                                      at=time.strftime('%Y-%m-%d %H:%M:%S'))
            write_json(claims_path, claims)
            return copy.deepcopy(item), read_json(root / 'keeper.json'), margin
    return None, None, None


def status(root, gpu, **kw):
    with locked(root):
        path = root / 'status.json'
        st = read_json(path) if path.exists() else {}
        st[f'gpu{gpu}'] = dict(kw, updated=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid())
        write_json(path, st)


def wait_gpu_free(gpu):
    waited = 0.
    while True:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True).stdout
        rows = {int(r.split(',')[0]): (float(r.split(',')[1]), float(r.split(',')[2])) for r in out.strip().splitlines()}
        used, total = rows[gpu]
        if (total - used) / 1024 >= NEED_FREE_GB:
            return used
        require(waited < GPU_WAIT_LIMIT, f'GPU {gpu} has had < {NEED_FREE_GB} GB free for {waited:.0f}s')
        log(f'GPU {gpu}: {used:.0f} MiB used by others; waiting')
        time.sleep(120.)
        waited += 120.


def child_env(gpu):
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE='1', PYTHONPATH='',
               PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1', OMP_NUM_THREADS='1')
    return env


def run_seed(root, exp_dir, seed, gpu, recipe, base=None):
    lc = long_chain()
    out = exp_dir / f's{seed}'
    require(not out.exists(), f'{out} exists; attempts are never resumed')
    ckpt = Path(base) if base else lc.BASES[seed]
    require(ckpt.is_file(), f'iteration-4500 base for seed {seed} missing: {ckpt}')
    wait_gpu_free(gpu)
    argv = [str(PY), '-X', 'utf8', '-B', '-u', str(HERE), 'stage2', '--ckpt', str(ckpt), '--out', str(out),
            '--recipe', str(exp_dir / 'recipe.json'), '--variant', str(exp_dir / 'variant.json')]
    t0 = time.time()
    with open(exp_dir / f's{seed}.log', 'x') as stream:
        code = subprocess.call(argv, env=child_env(gpu), stdout=stream, stderr=subprocess.STDOUT, cwd=str(WS))
    train_s = time.time() - t0
    require(code == 0 and (out / 'metrics.json').exists(), f'stage2 seed {seed} exit {code}; see {exp_dir}/s{seed}.log')
    m = recipe['milestones'][-1]
    target = out / f'm{m:04d}' / f'relax_a{recipe["alpha"]:g}'
    (exp_dir / 'scores').mkdir(exist_ok=True)
    key = f's{seed}_m{m:04d}'
    t1 = time.time()
    lc.score(target, exp_dir / 'scores' / key, f'{exp_dir.name}_{key}', exp_dir / 'scores' / f'{key}.log')
    summary = axis_summary(exp_dir / 'scores' / key)
    summary.update(train_seconds=train_s, score_seconds=time.time() - t1,
                   candidate_sha256=read_json(target / 'relax.json')['scans'])
    return summary


def publish(root):
    art = os.environ.get('MCGS_ARTIFACT_DIR')
    if not art:
        return
    Path(art).mkdir(parents=True, exist_ok=True)
    for name in ('experiments.jsonl', 'results.tsv', 'keeper.json', 'status.json', 'keeper_history.json'):
        if (root / name).exists():
            shutil.copy2(root / name, Path(art) / name)


def run_experiment(root, item, keeper, gpu, margin=KEEP_MARGIN):
    eid = item['id']
    exp_dir = root / 'runs' / eid
    exp_dir.mkdir(parents=True, exist_ok=False)
    relative = not ('recipe_path' in item or 'recipe' in item)
    variant = compose(keeper.get('variant'), item.get('variant')) if relative else item.get('variant', {'kind': 'none'})
    recipe = build_recipe(keeper['recipe'], item)
    check_recipe(recipe, variant, exp_dir)
    write_json(exp_dir / 'recipe.json', recipe)
    write_json(exp_dir / 'variant.json', variant)
    spec = dict(item=item, keeper_before=keeper['id'], keeper_m3=keeper['m3'], recipe=recipe, variant=variant, gpu=gpu,
                source_digest=os.environ.get('MCGS_SOURCE_DIGEST'), job=os.environ.get('MCGS_JOB_DIR'),
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(exp_dir / 'spec.json', spec)
    row = dict(id=eid, hypothesis=item.get('hypothesis', ''), keeper_before=keeper['id'], recipe=recipe,
               variant=variant, gpu=gpu, per_seed={}, started=spec['started'], keep_margin=margin)
    t0 = time.time()
    seeds = item.get('seeds', list(DEV_SEEDS))
    for seed in seeds:
        status(root, gpu, item=eid, phase=f'seed {seed}')
        log(f'[{eid}] seed {seed} start on GPU {gpu}')
        if 'bases' in item:
            require(str(seed) in item['bases'], f"item {eid} gives bases but none for seed {seed}")
        s = run_seed(root, exp_dir, seed, gpu, recipe, item.get('bases', {}).get(str(seed)))
        row['per_seed'][str(seed)] = s
        log(f"[{eid}] seed {seed}: M3 {s['m3']:+.5f} {short(s)} wh {s['whole_head']:+.5f}")
        if item.get('control'):
            continue
        if seed == 42 and len(seeds) > 1 and item.get('screen', True):
            with locked(root):                # screen against the keeper this item must beat now, not at claim
                current = read_json(root / 'keeper.json')
            ref = max(keeper['per_seed']['42']['m3'], current['per_seed']['42']['m3'])
            if s['m3'] <= ref + margin:
                row.update(status='screen_discard', m3=None, m3_s42=s['m3'],
                           note=f"s42 M3 {s['m3']:+.5f} <= keeper s42 {ref:+.5f} + {margin} "
                                f"(claim keeper {keeper['id']}, current keeper {current['id']})")
                break
    row['seconds'] = time.time() - t0
    row['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if item.get('control'):
        got = row['per_seed']['42']['candidate_sha256']
        want = read_json(item['reference_relax'])['scans']
        same_hash = all(got[s]['sha256'] == want[s]['sha256'] for s in ('pre', 'post'))
        same_m3 = abs(row['per_seed']['42']['m3'] - axis_summary(item['reference_score'])['m3']) < 1e-9
        ref_m3 = axis_summary(item['reference_score'])['m3']
        row.update(status='control_identical' if same_hash else ('control_same_score' if same_m3 else 'control_differs'),
                   m3=row['per_seed']['42']['m3'], same_hash=same_hash, same_m3=same_m3,
                   reference_m3=ref_m3, deviation=row['per_seed']['42']['m3'] - ref_m3)
        refs = {s: axis_summary(p) for s, p in item.get('reference_scores', {}).items() if s in row['per_seed']}
        row['deviations'] = {s: dict(m3=row['per_seed'][s]['m3'] - r['m3'],
                                     axes={a: row['per_seed'][s]['axes'][a]['best']['s6'] - v['best']['s6']
                                           for a, v in r['axes'].items()})
                             for s, r in refs.items()}
        with locked(root):
            append_row(root, row)
        return row
    if 'status' not in row:
        m3 = min(v['m3'] for v in row['per_seed'].values())
        row['m3'] = m3
        with locked(root):
            current = read_json(root / 'keeper.json')       # another worker may have moved the keeper
            if m3 > current['m3'] + margin:
                row['status'] = 'keep'
                new = dict(id=eid, recipe=recipe, variant=variant, m3=m3, per_seed=row['per_seed'],
                           source='threeaxis', set_at=row['finished'], replaced=current['id'])
                write_json(root / 'keeper.json', new)
                hist = read_json(root / 'keeper_history.json')
                write_json(root / 'keeper_history.json', hist + [new])
            else:
                row['status'] = 'discard'
            append_row(root, row)
        if row['status'] == 'keep' and m3 > 0:
            confirm(root, item, recipe, variant, gpu)
    else:
        with locked(root):
            append_row(root, row)
    log(f"[{eid}] {row['status']} M3 {row.get('m3')} in {row['seconds']:.0f}s")
    return row


def confirm(root, item, recipe, variant, gpu):
    eid = item['id'] + '_confirm'
    exp_dir = root / 'runs' / eid
    exp_dir.mkdir(parents=True, exist_ok=False)
    write_json(exp_dir / 'recipe.json', recipe)
    write_json(exp_dir / 'variant.json', variant)
    row = dict(id=eid, hypothesis=f"confirmation seeds {CONFIRM_SEEDS} of {item['id']}", recipe=recipe,
               variant=variant, gpu=gpu, per_seed={})
    for seed in CONFIRM_SEEDS:
        status(root, gpu, item=eid, phase=f'confirm seed {seed}')
        row['per_seed'][str(seed)] = run_seed(root, exp_dir, seed, gpu, recipe)
    row['m3'] = min(v['m3'] for v in row['per_seed'].values())
    row['status'] = 'confirm_pass' if row['m3'] > 0 else 'confirm_fail'
    with locked(root):
        append_row(root, row)
        if row['status'] == 'confirm_pass':
            (root / 'STOP').write_text(f"success candidate {item['id']}: dev and confirmation M3 > 0; "
                                       f"final visual check required\n")
    log(f"[{eid}] {row['status']} M3 {row['m3']:+.5f}")


def cmd_worker(args):
    gpu = int(args.gpu)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu),
            f"launcher exposed CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, expected --gpu {gpu}")
    require(Path(sys.executable).resolve() == PY.resolve(), 'run the worker with the GSTMI interpreter (@python)')
    root = Path(args.root)
    queue = str(Path(args.queue).resolve()) if args.queue else None
    if queue:
        init_root(root, read_json(queue))
    require((root / 'keeper.json').exists(), 'no keeper.json; start once with a --queue file that has keeper0')
    failures, idle, started, drain_seen = 0, 0., time.time(), False
    while True:
        stop = root / 'STOP'
        if stop.exists():
            text = stop.read_text().strip()
            # 'drain ...' retires the workers that were running when it was written (code update between items);
            # workers started after it ignore it. Any other STOP (e.g. a success candidate) stops every worker.
            if not (text.startswith('drain') and stop.stat().st_mtime < started):
                log('STOP file present:', text)
                break
            if not drain_seen:
                log('ignoring drain STOP written before this worker started:', text)
                drain_seen = True
        item, keeper, margin = claim(root, queue, gpu)
        if item is None:
            if queue or idle >= args.idle_wait:
                log('queue exhausted' + ('' if queue else f' (idle {idle:.0f}s)'))
                break
            if idle == 0:
                status(root, gpu, item=None, phase=f'idle, waiting up to {args.idle_wait:.0f}s for ROOT/queue')
            time.sleep(60.)
            idle += 60.
            continue
        idle = 0.
        try:
            run_experiment(root, item, keeper, gpu, margin)
            failures = 0
        except Exception as exc:          # preserved attempt; recorded, never resumed
            failures += 1
            log(f"[{item['id']}] FAILED: {exc}")
            with locked(root):
                append_row(root, dict(id=item['id'], status='infrastructure_or_training_failure', m3=None,
                                      error=str(exc), gpu=gpu, hypothesis=item.get('hypothesis', '')))
            if failures >= 3:
                log('three consecutive failures; worker stops')
                break
        finally:
            publish(root)
    status(root, gpu, item=None, phase='worker exited')
    publish(root)


def cmd_enqueue(args):
    """Validate a batch file against the current keeper and copy it into ROOT/queue (CPU)."""
    root = Path(args.root)
    src = Path(args.queue).resolve()
    q = read_json(src)
    scratch = Path(os.environ.get('MCGS_ARTIFACT_DIR') or '/tmp')
    with locked(root):
        keeper = read_json(root / 'keeper.json')
        qdir = root / 'queue'
        qdir.mkdir(exist_ok=True)
        known = {r['id'] for r in ledger_rows(root)} | {i['id'] for i, _ in queue_items(root)}
        for name in ('claims.json',):
            if (root / name).exists():
                known |= set(read_json(root / name))
        for item in q['items']:
            require(item['id'] not in known, f"id {item['id']} already used")
            known.add(item['id'])
            relative = not ('recipe_path' in item or 'recipe' in item)
            variant = compose(keeper.get('variant'), item.get('variant')) if relative else item.get('variant', {'kind': 'none'})
            check_recipe(build_recipe(keeper['recipe'], item), variant, scratch)
            log(f"  ok {item['id']}: on keeper {keeper['id']} -> variant {variant}")
        n = len(list(qdir.glob('*.json')))
        # workers take queue files in name order; '0-' sorts before every '00_'.. name, so --front runs first
        dst = qdir / (f'0-{n:02d}_{src.stem}.json' if args.front else f'{n:02d}_{src.stem}.json')
        write_json(dst, dict(q, enqueued=time.strftime('%Y-%m-%d %H:%M:%S'), source=str(src),
                             source_digest=os.environ.get('MCGS_SOURCE_DIGEST'), keeper_at_enqueue=keeper['id']))
        log(f'enqueued {len(q["items"])} items as {dst}')
    publish(root)


def cmd_probe_relax(args):
    """CPU probe of the export relaxation on a finished run: candidate = raw + alpha (canonical - raw) through the
    frozen long_chain.relax, scored by the frozen evaluator (lc.score, CPU). Writes ROOT/probes/<run>/s<seed>_a<alpha>
    and never touches the run directory. A positive probe is promoted by enqueueing the recipe edit (alpha)."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('-1', ''), 'probe-relax is CPU only')
    lc = long_chain()
    root = Path(args.root)
    run = (root / 'runs' / args.run).resolve()
    require(run.is_dir() and run.parent == (root / 'runs').resolve(), f'no run {args.run}')
    out_root = root / 'probes' / run.name
    result = {}
    for seed in args.seeds:
        stage = run / f's{seed}' / 'm0625'
        base = axis_summary(run / 'scores' / f's{seed}_m0625')
        log(f'{run.name} s{seed} alpha 1 (as run): {short(base)}')
        result[str(seed)] = {'1': dict(m3=base['m3'], short=short(base))}
        for alpha in args.alpha:
            require(0 <= alpha <= 1, 'alpha in [0, 1]')
            tag = f's{seed}_a{alpha:g}'
            rdir = out_root / tag / f'relax_a{alpha:g}'
            if not (rdir / 'relax.json').exists():
                lc.relax(stage, float(alpha), rdir)
            sdir = out_root / tag / 'score'
            if not (sdir / 'objective.json').exists():
                lc.score(rdir, sdir, f'{run.name}_{tag}', out_root / tag / f'score_{time.strftime("%H%M%S")}.log')
            summ = axis_summary(sdir)
            result[str(seed)][f'{alpha:g}'] = dict(m3=summ['m3'], short=short(summ))
            log(f'{run.name} s{seed} alpha {alpha:g}: M3 {summ["m3"]:+.5f} ({summ["m3"] - base["m3"]:+.5f}) {short(summ)}')
    for alpha in [1.0] + list(args.alpha):
        vals = [result[str(s)][f'{alpha:g}']['m3'] for s in args.seeds]
        log(f'alpha {alpha:g}: M3 over seeds {min(vals):+.5f}')
    art = os.environ.get('MCGS_ARTIFACT_DIR')
    if art:
        write_json(Path(art) / f'probe_relax_{run.name}.json', result)


def cmd_skip(args):
    """Mark queued, unclaimed items as skipped (ledger row); running workers then never claim them."""
    root = Path(args.root)
    with locked(root):
        claims_path = root / 'claims.json'
        claims = read_json(claims_path) if claims_path.exists() else {}
        done = {r['id'] for r in ledger_rows(root)}
        for eid in args.ids:
            require(eid not in claims and eid not in done, f'{eid} is already claimed or recorded; cannot skip')
            append_row(root, dict(id=eid, status='skipped', m3=None, hypothesis=args.reason,
                                  at=time.strftime('%Y-%m-%d %H:%M:%S')))
            log(f'skipped {eid}: {args.reason}')
    publish(root)


def cmd_promote(args):
    """Make a finished run the keeper by user decision (e.g. an M3 tie with better secondary axes). The keep rule
    stays as it is for queued items; the promotion is recorded in the ledger and in keeper_history (CPU)."""
    root = Path(args.root)
    with locked(root):
        rows = {r['id']: r for r in ledger_rows(root)}
        require(args.id in rows, f'{args.id} is not in the ledger')
        run = rows[args.id]
        require(run['status'] in ('keep', 'discard'), f"{args.id} has status {run['status']}; only fully run items")
        require(all(str(s) in run.get('per_seed', {}) for s in DEV_SEEDS), f'{args.id} lacks a dev seed')
        require(f'{args.id}_promote' not in rows, f'{args.id} was already promoted once')
        current = read_json(root / 'keeper.json')
        require(current['id'] != args.id, f'{args.id} is already the keeper')
        m3 = min(run['per_seed'][str(s)]['m3'] for s in DEV_SEEDS)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        new = dict(id=args.id, recipe=run['recipe'], variant=run['variant'], m3=m3, per_seed=run['per_seed'],
                   source='user promotion', reason=args.reason, set_at=now, replaced=current['id'])
        write_json(root / 'keeper.json', new)
        write_json(root / 'keeper_history.json', read_json(root / 'keeper_history.json') + [new])
        append_row(root, dict(id=f'{args.id}_promote', status='promoted', m3=m3, keeper_before=current['id'],
                              keeper_before_m3=current['m3'], hypothesis=args.reason, at=now))
    log(f"keeper {current['id']} ({current['m3']:+.5f}) -> {args.id} ({m3:+.5f}): {args.reason}")
    publish(root)


def cmd_summary(args):
    root = Path(args.root)
    keeper = read_json(root / 'keeper.json')
    print(f"keeper {keeper['id']} M3 {keeper['m3']:+.5f}")
    for s, v in keeper['per_seed'].items():
        print(f'  s{s} {short(v)}')
    for r in ledger_rows(root):
        m3 = r.get('m3')
        print(f"{r['id']:34s} {r['status']:22s} {'' if m3 is None else f'{m3:+.5f}':9s} "
              + ' | '.join(f's{s} {short(v)}' for s, v in r.get('per_seed', {}).items()))
    if (root / 'status.json').exists():
        print(json.dumps(read_json(root / 'status.json'), indent=1))
    publish(root)


# ------------------------------------------------------------------ selftest (CPU only)
def cmd_selftest(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('-1', ''), 'selftest is CPU only')
    import numpy as np
    import torch
    queue = read_json(Path(args.queue).resolve())
    k0 = queue['keeper0']
    per = {s: axis_summary(k0['scores'][str(s)]) for s in DEV_SEEDS}
    m3 = min(v['m3'] for v in per.values())
    log(f"keeper0 {k0['id']} M3 {m3:+.6f}; " + '; '.join(f's{s} {short(v)}' for s, v in per.items()))
    require(abs(m3 - k0['expected_m3']) < 5e-6, f"keeper0 M3 {m3} differs from expected {k0['expected_m3']}")
    lc = long_chain()
    for seed in DEV_SEEDS + CONFIRM_SEEDS:
        require(Path(lc.BASES[seed]).is_file(), f'base checkpoint for seed {seed} missing')
    keeper_recipe = read_json(k0['recipe_path'])
    scratch = Path(os.environ.get('MCGS_ARTIFACT_DIR') or '/tmp')
    ids = set()
    for item in queue['items']:
        require(item['id'] not in ids, f"duplicate id {item['id']}")
        ids.add(item['id'])
        recipe = check_recipe(build_recipe(keeper_recipe, item), compose(queue.get('assume_keeper_variant'),
                                                                          item.get('variant')), scratch)
        diff = {k: v for k, v in recipe['loss_edits'].items() if keeper_recipe['loss_edits'].get(k) != v}
        diff.update({k: v for k, v in recipe.items() if k != 'loss_edits' and keeper_recipe.get(k) != v})
        log(f"  {item['id']:28s} variant={item.get('variant', {}).get('kind', 'none'):9s} change={diff}")
    require(read_json(k0['candidate_relax']['42'])['scans']['pre']['sha256'], 'keeper candidate hashes missing')
    jb, sg = {'kind': 'joint_bm', 'guide_sigma': 1.0}, {'kind': 'sigma', 'decay': 0.6}
    require(compose(None, None) == {'kind': 'none'} and compose(jb, None) == jb
            and compose(jb, sg) == {'kind': 'stack', 'parts': [jb, sg]}
            and compose({'kind': 'stack', 'parts': [jb, sg]}, {'kind': 'sigma', 'decay': 0.8})['parts'][1]['decay'] == 0.8
            and compose(jb, {'kind': 'joint_bm', 'remove': True}) == {'kind': 'none'}, 'variant composition wrong')
    log('  variant composition ok (keeper parts + item parts, same kind replaces)')

    # real bm4d + the frozen nonlocal_red module on a small synthetic pair (CPU)
    nlr = load_module(NONLOCAL_RED, '_ta_nonlocal_red_test')
    g = np.random.default_rng(0)
    base = np.zeros((24, 28, 20), np.float32)
    base[4:20, 5:23, 3:17] = 1.0
    base[8:14, 9:15, 6:12] = 1.8
    zp = torch.from_numpy(base + 0.05 * g.standard_normal(base.shape).astype(np.float32))
    zc = torch.from_numpy(1.3 * base + 0.05 * g.standard_normal(base.shape).astype(np.float32))
    support = torch.from_numpy(base > 0)
    sharp = nlr.make_denoiser('bm4d', 'np', 4, 1.3)
    plain = nlr.make_denoiser('bm4d', 'np', 4, 1.0)
    a, b = sharp(zp.abs().numpy(), 0.05), plain(zp.abs().numpy(), 0.05)
    hp = lambda v: float(np.abs(np.diff(v, axis=1)).mean())  # noqa: E731
    log(f'  bm4d sharpen 1.3 vs 1.0: mean |grad| {hp(a):.4f} vs {hp(b):.4f} (input {hp(zp.numpy()):.4f})')
    require(hp(a) > hp(b), 'sharpening did not raise high-frequency content')
    install_joint_bm(nlr, {'kind': 'joint_bm', 'guide_sigma': 1.0})
    red = nlr.NonLocalRED(support, nlr.make_denoiser('bm4d_pos', 'np', 4, 1.0), sigma_rel=0.03, start=1, refresh=1000,
                          dilate=2)
    info_p = red.refresh('plain', zp, 1)
    placeholder = red.targets['plain'].clone()
    info_c = red.refresh('contrast', zc, 1)
    tp, tc = red.targets['plain'], red.targets['contrast']
    require(tuple(tp.shape) == tuple(tc.shape) == red.crop_shape, 'joint targets have the wrong shape')
    require(bool((tp >= 0).all()) and bool((tc >= 0).all()), 'joint targets must be nonnegative')
    require(not torch.equal(tp, placeholder), 'plain target was not replaced')
    require(red.history['plain'][-1].get('joint_replaced') and len(red.history['plain']) == 1, 'history not updated')
    truth = torch.from_numpy(base)[red.bbox]
    err = lambda t: float((t - truth).square().mean().sqrt())  # noqa: E731
    log(f"  joint_bm: plain target rms err {err(tp):.4f} (render {err(placeholder):.4f}), contrast "
        f"{float((tc - 1.3 * truth).square().mean().sqrt()):.4f}; relnorm plain "
        f"{red.info['plain']['target_relative_norm']:.4f}, info keys ok={bool(info_p) and bool(info_c)}")
    require(err(tp) < err(placeholder), 'joint denoising did not reduce the plain error on the synthetic pair')

    # profile variant: overrides reach the profile object the BM4D closure actually uses
    nlr2 = load_module(NONLOCAL_RED, '_ta_nonlocal_red_test2')
    base_make = nlr2.make_denoiser
    nlr2.make_denoiser = lambda name, *a, **k: base_make('bm4d' if name == 'bm4d_pos' else name, *a, **k)
    attrs = {'bs_ht': [4, 8, 8], 'search_window_ht': [7, 11, 11], 'max_stack_size_ht': 32}
    try:
        preflight_attrs({'bs_ht': [2, 4, 4]})
        raise AssertionError('preflight accepted a NaN-producing profile')
    except RuntimeError:
        pass
    log(f'  preflight ok, synthetic rms err {preflight_attrs(attrs):.4f}')
    install_profile(nlr2, {'kind': 'profile', 'attrs': attrs})
    den = nlr2.make_denoiser('bm4d_pos', 'np', 4, 1.0)
    require(den.profile.bs_ht == (4, 8, 8) and den.profile.max_stack_size_ht == 32, 'profile overrides not applied')
    y = den(zp.abs().numpy(), 0.05)
    require(y.shape == tuple(zp.shape) and np.isfinite(y).all(), 'profile variant output invalid')
    require(den.last_stats.get('profile_overrides', {}).get('search_window_ht') == (7, 11, 11), 'overrides not logged')
    nlr3 = load_module(NONLOCAL_RED, '_ta_nonlocal_red_test3')
    install_sigma(nlr3, {'kind': 'sigma', 'scale': {'post': 2.0}, 'decay': 0.5})
    seen = []
    red3 = nlr3.NonLocalRED(support, lambda z, sigma: (seen.append(sigma), np.maximum(z, 0))[1], sigma_rel=0.03,
                            start=1, refresh=2, dilate=2)
    for it in (1, 3):
        red3.refresh('pre', zp, it)
        red3.refresh('post', zc, it)
    ratios = [seen[2] / seen[0], seen[1] / seen[0] * float(np.quantile(zp.abs().numpy()[red3.bbox][red3.head_crop.numpy()], .99))
              / float(np.quantile(zc.abs().numpy()[red3.bbox][red3.head_crop.numpy()], .99)), seen[3] / seen[1]]
    log(f'  sigma variant: pre k1/k0 {ratios[0]:.3f} (want 0.5), post/pre at k0 {ratios[1]:.3f} (want 2), '
        f'post k1/k0 {ratios[2]:.3f} (want 0.5); sigma_rel restored {red3.sigma_rel}')
    require(abs(ratios[0] - 0.5) < 1e-6 and abs(ratios[2] - 0.5) < 1e-6 and abs(ratios[1] - 2) < 0.02
            and red3.sigma_rel == 0.03, 'sigma variant schedule wrong')
    tmp = scratch / f'selftest_init_scale_{os.getpid()}'
    tmp.mkdir()
    fake = {'iter': 4500, 'model_state': {'positions': torch.zeros(5, 3), 'scales': torch.full((5, 3), -4.8),
            'rotations': torch.ones(5, 4), 'density_real': torch.ones(5), 'density_imag': torch.full((5,), -2.),
            'density_real_contrast': torch.ones(5), 'density_imag_contrast': torch.zeros(5)}, 'optim_state': {}}
    torch.save(fake, tmp / 'src.pth')
    rec = scaled_checkpoint(tmp / 'src.pth', tmp / 'dst.pth', 0.8)
    got = torch.load(tmp / 'dst.pth', map_location='cpu', weights_only=True)['model_state']
    require(torch.allclose(got['scales'], torch.full((5, 3), -4.8 + math.log(0.8)))
            and torch.allclose(got['density_imag'], torch.full((5,), -2. / 0.512))
            and torch.equal(got['positions'], fake['model_state']['positions']), 'init_scale transform wrong')
    shutil.rmtree(tmp)
    log(f"  init_scale: scales +log(0.8), densities x{rec['density_factor']:.4f}; other tensors unchanged")
    kz = frozen_known_zero()
    g2 = torch.Generator().manual_seed(1)
    xs = torch.randn(6, 7, 5, dtype=torch.complex64, generator=g2)
    S = torch.randn(3, 6, 7, 5, dtype=torch.complex64, generator=g2)
    S = S / S.abs().square().sum(0, keepdim=True).sqrt()
    mfit = (torch.rand(1, 7, 5, generator=g2) > 0.5).expand(6, 7, 5).float()
    ydc = kz.fft3c(S * xs.unsqueeze(0)) * mfit.unsqueeze(0)
    require(torch.allclose(kz.hard_dc_combine(xs, S, ydc, mfit), xs, atol=1e-5),
            'dc target must equal x when the data match it exactly')
    require(torch.allclose(kz.hard_dc_combine(torch.zeros_like(xs), S, ydc, mfit),
                           (kz.ifft3c(ydc) * S.conj()).sum(0), atol=1e-6), 'dc target of zero must be the ZF image')
    log(f'  dc_target: frozen {Path(kz.__file__).name}.hard_dc_combine exact-data and zero-render identities hold')
    xd = kz.hard_dc_combine(torch.zeros_like(xs), S, ydc, mfit)
    require(dc_blend(xs, xd, 1.0) is xd and torch.allclose(dc_blend(xs, xd, 2.0), 2 * xd - xs), 'dc_blend wrong')
    for bad in ({'kind': 'dc_target', 'gain': 0}, {'kind': 'dc_target', 'gain': 17}, {'kind': 'dc_target', 'beta': 1}):
        try:
            check_dc_variant(bad)
        except SystemExit:
            continue
        except Exception:
            continue
        raise RuntimeError(f'check_dc_variant accepted {bad}')
    require(dc_gain({'gain': 8}, 0) == 8 and dc_gain({'gain': 8}, 1) == 8 and dc_gain({'gain': 8, 'later_gain': 1}, 0) == 8
            and dc_gain({'gain': 8, 'later_gain': 1}, 2) == 1, 'dc_gain schedule wrong')
    log('  dc_target gain: blend, later_gain schedule and validation ok')
    log(f"  profile variant: rms err {float(np.sqrt(((y - base) ** 2).mean())):.4f} with {den.last_stats['profile_overrides']}")
    log('SELFTEST OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('selftest'); s.add_argument('--queue', required=True)
    c = sub.add_parser('stage2'); c.add_argument('--ckpt', required=True); c.add_argument('--out', required=True)
    c.add_argument('--recipe', required=True); c.add_argument('--variant', required=True)
    w = sub.add_parser('worker'); w.add_argument('--gpu', type=int, required=True); w.add_argument('--queue')
    w.add_argument('--idle-wait', type=float, default=2700.)
    e = sub.add_parser('enqueue'); e.add_argument('--root', default=str(ROOT)); e.add_argument('--queue', required=True)
    e.add_argument('--front', action='store_true', help='run before the already queued batches')
    w.add_argument('--root', default=str(ROOT))
    m = sub.add_parser('summary'); m.add_argument('--root', default=str(ROOT))
    pr = sub.add_parser('probe-relax'); pr.add_argument('--root', default=str(ROOT)); pr.add_argument('--run', required=True)
    pr.add_argument('--alpha', type=float, nargs='+', required=True); pr.add_argument('--seeds', type=int, nargs='+', default=list(DEV_SEEDS))
    k = sub.add_parser('skip'); k.add_argument('--root', default=str(ROOT)); k.add_argument('--reason', required=True)
    k.add_argument('ids', nargs='+')
    po = sub.add_parser('promote'); po.add_argument('--root', default=str(ROOT)); po.add_argument('--reason', required=True)
    po.add_argument('id')
    args = ap.parse_args()
    {'probe-relax': cmd_probe_relax, 'selftest': cmd_selftest, 'stage2': cmd_stage2, 'worker': cmd_worker, 'summary': cmd_summary,
     'skip': cmd_skip, 'enqueue': cmd_enqueue, 'promote': cmd_promote}[args.cmd](args)


if __name__ == '__main__':
    main()
