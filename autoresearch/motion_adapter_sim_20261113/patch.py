#!/usr/bin/env python3
"""Child process of the motion study: patch the frozen trainer IN MEMORY, then run arc/cycle.py unchanged.

    cd ROOT/trainer && python <this file> --motion <motion.json> -- <arc/cycle.py arguments>

No file of the trainer copy is edited (same approach as abl_framework.py --child iso). What is replaced:
  * the data-set loader: a contrast marked with `motion_ksp_path` trains on that (motion-corrupted) k-space while its
    ground truth, full k-space and intensity scale come from `data_path` (the reference-pose k-space);
  * JointTrainer._setup_optimizer / _optimizer_param_groups: create the MotionAdapter, append its parameter groups;
  * JointTrainer._render: the adapter's channel is rendered through the adapter (reference pose);
  * JointTrainer._kspace_loss: for the adapter's channel, the data term is split over acquisition segments, each
    rendered at its own pose (reference segment always, n_extra random others, importance-weighted); plus the
    adapter's regulariser;
  * edge_preserving_tv (name in trainers.joint_trainer): a cross-contrast guide is re-rendered, without gradient, in
    the frame of the volume it guides;
  * train: afterwards the motion estimate and the canonical-frame render are saved.
With `adapter: false` only the data-set loader and the saving are active (no-adapter arms on intra-scan conditions).
Nothing here prints or stores a clock value or a duration.
"""
import json
import os
import sys
from pathlib import Path

REQUIRED = ("arm", "channel", "rigid", "nonrigid", "segments", "n_extra", "schedule_path", "truth_path", "disp_path",
            "native_shift_mm", "oracle", "lrs", "reg", "stage_iters", "out_dir", "exp", "seed")
_ACTIVE = {}


def validate(cfg):
    missing = [k for k in REQUIRED if k not in cfg]
    assert not missing, f"motion config lacks {missing}"
    assert cfg["channel"] in ("plain", "contrast"), cfg["channel"]
    assert int(cfg["segments"]) >= 0 and int(cfg["n_extra"]) >= 1
    if cfg["segments"]:
        assert cfg["schedule_path"] and cfg.get("ref_seg") is not None, "segments need the schedule and ref_seg"
    if cfg["oracle"]:
        assert cfg["truth_path"], "the oracle needs the motion truth"
        assert not cfg["nonrigid"] or cfg["disp_path"], "a non-rigid oracle needs the displacement field"
    assert len(cfg["stage_iters"]) == 2
    return cfg


def has_adapter(cfg):
    return bool(cfg["rigid"] or cfg["nonrigid"] or cfg["segments"] or cfg["oracle"])


# ------------------------------------------------------------------------------------------- pure pieces
def segment_masks(mask_fit, seg2d, ref, K):
    """Split the fit mask over the K acquisition segments. seg2d: (ky, kz) segment index, -1 outside the acquired
    support. The reference segment also takes every fit point that belongs to no other segment (the known-zero
    region and anything unassigned), so the masks add up to mask_fit exactly."""
    masks = [mask_fit * (seg2d == s).to(mask_fit.dtype) for s in range(K)]
    masks[ref] = mask_fit - sum(m for s, m in enumerate(masks) if s != ref)
    return masks


def segmented_dc(loss_fn, render_seg, vol_ref, masks, ref, n_extra, gen):
    """Data term over segments: the reference segment with the reference-pose render, plus n_extra other segments
    drawn without replacement, each with its own render and weighted by (K-1)/n_extra, so the expectation equals the
    sum over all segments. loss_fn(vol, mask) must use one fixed normalisation for every mask."""
    import torch
    others = [s for s in range(len(masks)) if s != ref]
    loss = loss_fn(vol_ref, masks[ref])
    if not others:
        return loss
    n = min(int(n_extra), len(others))
    pick = torch.randperm(len(others), generator=gen)[:n].tolist()
    scale = len(others) / n
    for i in pick:
        s = others[i]
        loss = loss + scale * loss_fn(render_seg(s), masks[s])
    return loss


def append_groups(base_groups, adapter, lrs, eps=None):
    """Optimiser groups of the trainer followed by the adapter's (nothing for a frozen = oracle adapter)."""
    groups = list(base_groups)
    if adapter is not None and adapter.trainable():
        groups += adapter.param_groups(lrs, eps=eps)
    return groups


class GradStats:
    """Per adapter parameter group and per window of iterations: how large the gradient is (abs_mean, to compare with
    Adam's eps) and how consistent its sign is (consistency = mean_i |mean_t g| / mean_i,t |g|; 1 = always one sign,
    about 1/sqrt(n) = noise). Groups of at most 3 elements also report the mean gradient. Fed by tensor hooks; it
    only records."""

    def __init__(self):
        self._sum, self._abs, self._n = {}, {}, {}

    def add(self, name, g):
        g = g.detach().double().reshape(-1).cpu()
        if name not in self._n:
            self._sum[name], self._abs[name], self._n[name] = g.clone(), g.abs(), 1
        else:
            self._sum[name] += g
            self._abs[name] += g.abs()
            self._n[name] += 1

    def flush(self):
        out = {}
        for name, n in self._n.items():
            mean, a = self._sum[name] / n, float((self._abs[name] / n).mean())
            out[name] = dict(n=n, abs_mean=a, consistency=float(mean.abs().mean()) / a if a > 0 else 0.0)
            if mean.numel() <= 3:
                out[name]["mean"] = mean.tolist()
        self.__init__()
        return out


def attach_grad_stats(adapter, stats):
    """Tensor hooks that feed GradStats. A hook can only be registered while the tensor requires a gradient, and the
    stage switch may already have frozen a group, so the flag is raised for the registration and put back."""
    for name, attr in adapter.GROUPS:
        if hasattr(adapter, attr):
            p = getattr(adapter, attr)
            was = p.requires_grad
            p.requires_grad_(True)
            p.register_hook(lambda g, name=name: stats.add(name, g))
            p.requires_grad_(was)


def apply_stage(adapter, it, stage_iters):
    """Stage 1 (it < stage_iters[0]): only the inter-scan rigid pose learns. Afterwards everything learns; the decay
    of stage 3 is the trainer's exponential learning-rate schedule, which covers every group."""
    late = it >= int(stage_iters[0]) or not adapter.has_rigid
    for name in ("ctrl", "seg_theta", "seg_t_mm"):
        if hasattr(adapter, name):
            getattr(adapter, name).requires_grad_(bool(late))
    for name in ("theta", "t_mm"):
        if hasattr(adapter, name):
            getattr(adapter, name).requires_grad_(True)


def guide_frame(vol, guide, last):
    """Which re-render a cross-contrast guide needs so that it lies in the frame of `vol` (None: leave it alone)."""
    if guide is None:
        return None
    if vol is last.get("contrast") and guide is last.get("plain"):
        return "plain_in_contrast_frame"
    if vol is last.get("plain") and guide is last.get("contrast"):
        return "contrast_in_canonical_frame"
    return None


def load_with_reference_gt(orig_loader, data_cfg):
    """Data set whose ground truth, full k-space and intensity scale come from data_path (reference pose), and whose
    undersampled measurement comes from motion_ksp_path when that key is present."""
    ds = orig_loader(data_cfg)
    path = data_cfg.get("motion_ksp_path")
    if not path:
        return ds
    import h5py
    import torch
    from data.transforms import ifft3c
    with h5py.File(path, "r") as h:
        arr = h["kspace"][:]
    y = torch.from_numpy(ds._normalize_kspace_layout(arr)).to(torch.complex64) / float(ds.normalization_scale)
    assert y.shape == ds.kspace_multicoil.shape, (y.shape, ds.kspace_multicoil.shape)
    und = (y * ds.mask.unsqueeze(0)).contiguous()
    zf = torch.sum(torch.conj(ds.sensitivity_maps) * ifft3c(und), dim=0).contiguous()
    ds.kspace_undersampled_complex = und
    ds.zero_filled_complex = zf
    ds.zero_filled_image = torch.stack([zf.real, zf.imag], dim=0)
    ds.kspace_undersampled = torch.cat([und.real, und.imag], dim=0)
    print(f"[motion] measurement replaced by {path}; ground truth and scale from {data_cfg['data_path']}", flush=True)
    return ds


# ------------------------------------------------------------------------------------------- patches
def install(cfg):
    import numpy as np
    import torch
    import trainers.joint_trainer as JT
    import trainers.single_contrast_trainer as ST
    from adapter import MotionAdapter, render_transformed
    from data.transforms import fft3c, ifft3c

    J, S = JT.JointTrainer, ST.SingleContrastTrainer
    ch = cfg["channel"]
    tag = "c" if ch == "contrast" else "p"
    K, ref = int(cfg["segments"]), int(cfg.get("ref_seg") or 0)
    use = has_adapter(cfg)
    orig = dict(setup=J._setup_optimizer, groups=J._optimizer_param_groups, render=J._render, kloss=J._kspace_loss,
                ept=JT.edge_preserving_tv, load=JT._load_dataset, jtrain=J.train, strain=S.train)

    def load(data_cfg):
        return load_with_reference_gt(orig["load"], data_cfg)

    def render_with(self, density_channel, seg, adapter):
        m = self.model
        vol = render_transformed(self.voxelizer, m.positions, m.get_scale_values(), m.rotations,
                                 m.get_density(density_channel), adapter, seg)
        if density_channel == "contrast" and m.has_private:
            vol = vol + render_transformed(self.voxelizer, m.private_positions, m.get_private_scale_values(),
                                           m.private_rotations, m.get_private_density(), adapter, seg)
        return vol

    def setup(self):
        ad = None
        if use:
            ad = MotionAdapter(self.volume_shape, cfg["rigid"] or cfg["oracle"], cfg["nonrigid"], K, ref).to(self.device)
            if cfg["oracle"]:
                import geom
                truth = json.loads(Path(cfg["truth_path"]).read_text())
                inter = geom.Rigid.from_json(truth["inter"]) if truth.get("inter") else None
                dense_u = None
                if cfg["nonrigid"]:
                    g = inter if inter is not None else geom.Rigid.identity()
                    r = geom.grid_mm(self.volume_shape)
                    dense_u = g.inverse().apply(r + np.load(cfg["disp_path"]).astype(np.float64)) - r
                segs = [geom.Rigid.from_json(p) for p in truth["segment_mean_poses"]] if K else None
                brain = np.load(cfg["brain_path"]) if cfg.get("brain_path") else None
                ad.set_truth(inter=inter, dense_u=dense_u, segments=segs, native_shift_mm=cfg["native_shift_mm"],
                             brain=brain)
                ad.to(self.device)
                print(f"[motion] oracle transform set; control-grid fit residual {ad.oracle_fit_mm} mm", flush=True)
            else:
                apply_stage(ad, 0, cfg["stage_iters"])
            if K:
                seg2d = torch.from_numpy(np.load(cfg["schedule_path"])["seg_of"]).long().to(self.device)
                self._ma_masks = segment_masks(getattr(self, f"mask_fit_{tag}"), seg2d, ref, K)
                self._ma_gen = torch.Generator().manual_seed(int(cfg["seed"]) + 4243)
            print(f"[motion] adapter arm={cfg['arm']} channel={ch} rigid={ad.has_rigid} nonrigid={ad.has_nonrigid} "
                  f"segments={K} ref={ref} oracle={cfg['oracle']}", flush=True)
        self.motion, self._ma_it, self._ma_last, self._ma_traj = ad, 0, {}, []
        self._ma_grad = GradStats()
        if ad is not None and not cfg["oracle"]:
            attach_grad_stats(ad, self._ma_grad)
        _ACTIVE["tr"] = self
        orig["setup"](self)

    def groups(self):
        return append_groups(orig["groups"](self), getattr(self, "motion", None), cfg["lrs"], cfg.get("adam_eps"))

    def render(self, channel):
        ad = getattr(self, "motion", None)
        v = orig["render"](self, channel) if (ad is None or channel != ch) else render_with(self, channel, None, ad)
        if hasattr(self, "_ma_last"):
            self._ma_last[channel] = v
        return v

    def kloss(self, vol, csm, ksp_meas, mask, freq_weight, norm=None):
        ad = getattr(self, "motion", None)
        if ad is None or ksp_meas is not getattr(self, f"kspace_fit_{tag}") or vol is not self._ma_last.get(ch):
            return orig["kloss"](self, vol, csm, ksp_meas, mask, freq_weight, norm)
        if K:
            fn = lambda v, m: orig["kloss"](self, v, csm, ksp_meas, m, freq_weight, norm)
            loss = segmented_dc(fn, lambda s: render_with(self, ch, s, ad), vol, self._ma_masks, ref,
                                cfg["n_extra"], self._ma_gen)
        else:
            loss = orig["kloss"](self, vol, csm, ksp_meas, mask, freq_weight, norm)
        loss = loss + ad.regulariser(cfg["reg"])
        self._ma_it += 1
        if self._ma_it == 1 or self._ma_it % 200 == 0:
            st = ad.state()
            st.pop("ctrl_mm", None)
            self._ma_traj.append(dict(it=self._ma_it, grad=self._ma_grad.flush(), **st))
        if not cfg["oracle"]:
            apply_stage(ad, self._ma_it, cfg["stage_iters"])
        return loss

    def ept(vol, kappa, w_floor, eps=0.0, vector=False, guide=None, guide_kappa=1.0, guide_alpha=0.0):
        tr = _ACTIVE.get("tr")
        ad = getattr(tr, "motion", None)
        if ad is not None and ch == "contrast" and guide is not None and guide_alpha > 0 and not ad.is_identity():
            which = guide_frame(vol, guide, tr._ma_last)
            with torch.no_grad():
                if which == "plain_in_contrast_frame":
                    guide = render_with(tr, "plain", None, ad)
                elif which == "contrast_in_canonical_frame":
                    guide = orig["render"](tr, "contrast")
        return orig["ept"](vol, kappa, w_floor, eps, vector, guide=guide, guide_kappa=guide_kappa,
                           guide_alpha=guide_alpha)

    @torch.no_grad()
    def save(self):
        out = Path(cfg["out_dir"])
        ad = getattr(self, "motion", None)
        rec = dict(exp=cfg["exp"], arm=cfg["arm"], channel=ch, iterations=getattr(self, "_ma_it", 0),
                   config=dict({k: cfg[k] for k in ("rigid", "nonrigid", "segments", "n_extra", "oracle", "lrs", "reg",
                                                    "stage_iters", "native_shift_mm", "seed")},
                               adam_eps=cfg.get("adam_eps")),
                   final=ad.state() if ad is not None else None, trajectory=getattr(self, "_ma_traj", []))
        (out / f"{cfg['exp']}_motion.json").write_text(json.dumps(rec, indent=1))
        c64 = lambda t: t.detach().cpu().numpy().astype("complex64")
        x = self._render(ch)                                              # reference pose, the frame of the data
        arrays = dict(rec_raw=c64(x))
        if ad is not None:
            arrays["canon_complex"] = c64(orig["render"](self, ch))       # canonical (T1) frame
        if ad is not None and K:
            csm = getattr(self, f"csm_{tag}")
            mref = self._ma_masks[ref].unsqueeze(0)
            k_model = fft3c(csm * x.unsqueeze(0))
            k_dc = mref * getattr(self, f"kspace_dc_{tag}") + (1.0 - mref) * k_model
            arrays["rec_ref_dc"] = c64((ifft3c(k_dc) * csm.conj()).sum(0))  # hard DC with the reference segment only
        np.savez_compressed(out / f"{cfg['exp']}_canon.npz", **arrays)
        print(f"[motion] saved {cfg['exp']}_motion.json and {cfg['exp']}_canon.npz ({sorted(arrays)})", flush=True)

    def wrap_train(fn):
        def train(self, *a, **k):
            res = fn(self, *a, **k)
            save(self)
            return res
        return train

    JT._load_dataset = ST._load_dataset = load
    J.train, S.train = wrap_train(orig["jtrain"]), wrap_train(orig["strain"])
    J._setup_optimizer = setup                      # always: it registers the trainer for the saving step
    if use:
        J._optimizer_param_groups, J._render, J._kspace_loss = groups, render, kloss
        JT.edge_preserving_tv = ept
    return orig


def main():
    import runpy
    i = sys.argv.index("--")
    assert sys.argv[1] == "--motion", "usage: patch.py --motion FILE -- <cycle.py arguments>"
    cfg = validate(json.loads(Path(sys.argv[2]).read_text()))
    cyc = sys.argv[i + 1:]
    tr_root = Path.cwd()
    assert (tr_root / "arc/cycle.py").is_file(), "run from the trainer copy"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cyc[cyc.index("--gpu") + 1]       # what cycle.py sets itself
    sys.path.insert(0, str(tr_root / "paired_gs"))
    install(cfg)
    sys.argv = [str(tr_root / "arc/cycle.py"), *cyc]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
