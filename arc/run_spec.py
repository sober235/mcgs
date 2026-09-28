#!/usr/bin/env python3
"""arc/run_spec.py -- strict entry for the multi-anatomy studies (spec 2026-09-28 §3.2).

One run = one spec file; the command line carries no tuning parameter:
    python arc/run_spec.py --spec specs/<ds>/<run>.yaml --exp <prefix>_<name> --gpu <id>
    python arc/run_spec.py --spec ... --exp ... --gpu ... --audit N
The trainer sees spec["trainer"] through StrictConfig. A normal run stops at the first key that is not in
the spec. --audit N records every fallback instead, trains N iterations (warm-up shortened to N//4 so both
branches run), writes <exp>_audit.json and scores nothing; merge the result with
`python arc/spec_io.py fill-defaults`. arc/cycle.py is not used and not modified.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return {"__repr__": repr(v)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict spec-driven MC-GS run")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--audit", type=int, default=0, help="record config fallbacks over N iterations; no scoring")
    a = ap.parse_args(argv)
    if a.audit and a.audit < 3:
        ap.error("--audit N needs N >= 3 (JointTrainer requires the warm-up to stay below N/2)")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    import numpy as np
    import torch

    here = Path(__file__).resolve().parent
    root = here.parent
    for p in (root / "paired_gs", here):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from seed import set_seed
    from trainers.joint_trainer import JointTrainer
    from data.transforms import fft3c, ifft3c
    import spec_io
    import spec_scoring
    from strict_config import StrictConfig

    log = lambda *m: print(*m, flush=True)  # noqa: E731
    t0 = time.time()
    spec_path = Path(a.spec).resolve()
    spec = spec_io.load_spec(spec_path)
    spec_io.check_exp(a.exp, spec)
    spec_io.run_dir(a.exp, spec)
    res = Path(spec["spec"]["results_root"])
    res.mkdir(parents=True, exist_ok=True)
    prov = spec_io.provenance(spec_path, spec, root)
    plain_cfg = spec_io.trainer_config(spec, a.exp)
    if a.audit:
        plain_cfg["training"]["max_iterations"] = a.audit
        plain_cfg["training"]["shared_warmup_iters"] = max(1, min(int(plain_cfg["training"]["shared_warmup_iters"]),
                                                                  a.audit // 4))
    cfg = StrictConfig(plain_cfg, strict=not a.audit)
    dev = torch.device("cuda", 0)
    set_seed(int(cfg["training"]["seed"]), deterministic=bool(cfg["training"]["deterministic"]))
    log(f"[run_spec {a.exp}] spec={spec_path} sha256={prov['spec_sha256'][:12]} commit={prov['git_commit'][:10]}"
        f"{' DIRTY' if prov['git_dirty'] else ''} gpu={a.gpu} audit={a.audit}")
    tr = JointTrainer(cfg, device=dev)

    if a.audit:
        tr.train()
        seen, entries = set(), []
        for key, default in cfg.fallbacks:
            if key not in seen:
                seen.add(key)
                entries.append({"key": key, "default": _jsonable(default)})
        out = {"exp": a.exp, "spec": str(spec_path), "spec_sha256": prov["spec_sha256"], "iters": a.audit,
               "fallbacks": entries}
        (res / f"{a.exp}_audit.json").write_text(json.dumps(out, indent=2))
        log(f"[run_spec {a.exp}] audit: {len(entries)} fallback(s) -> {res / (a.exp + '_audit.json')}")
        return

    if spec["spec"]["from_ckpt"]:
        ck = torch.load(spec["spec"]["from_ckpt"], map_location="cpu", weights_only=False)
        tr.model.load_state_dict(ck["model_state"])
        log(f"[run_spec {a.exp}] loaded {spec['spec']['from_ckpt']} (iter {ck.get('iter')}); training skipped")
    else:
        tr.train()

    rel, gr = float(spec["scoring"]["anatomy_rel"]), bool(spec["scoring"]["grain_ratio"])
    names = {ch: cfg["data"][ch]["name"] for ch in ("plain", "contrast")}
    scores = {ch: spec_scoring.score_channel(tr, ch, dev, rel, gr) for ch in ("plain", "contrast")}
    combined = 0.5 * (scores["plain"]["comp"] + scores["contrast"]["comp"])
    both_beat = scores["plain"]["beat"] and scores["contrast"]["beat"]
    vram = torch.cuda.max_memory_allocated() / 1e9
    for ch in ("plain", "contrast"):
        s = scores[ch]
        n = "off" if s["grain_ratio"] is None else f"{s['grain_ratio']:.3f}"
        log(f"{names[ch]:5s} psnr={s['psnr']:.3f} ssim={s['ssim']:.4f} nmse={s['nmse']:.4f} hfen={s['hfen']:.4f} "
            f"head_psnr={s['head_psnr']:.3f} head_ssim={s['head_ssim']:.4f} head_hfen={s['head_hfen']:.4f} "
            f"N={n} sharp={s['sharp']:.3f} grain={s['grain']:.4f} head_psnr_cplx={s['head_psnr_cplx']:.3f} "
            f"comp={s['comp']:.3f} beat={s['beat']}")
    row = dict(exp=a.exp, spec=str(spec_path), **prov, argv=sys.argv[1:], elapsed_s=time.time() - t0,
               npts=int(tr.model.num_points), vram_gb=vram, plain_name=names["plain"],
               contrast_name=names["contrast"], plain=scores["plain"], contrast=scores["contrast"],
               combined=combined, both_beat=both_beat, fallbacks=[list(f) for f in cfg.fallbacks],
               resolved_config=cfg.to_plain())
    (res / f"{a.exp}_score.json").write_text(json.dumps(row, indent=2, default=repr))
    log(f"SCORE combined={combined:.3f} both_beat_zf={both_beat} vram_gb={vram:.1f} elapsed_s={time.time() - t0:.0f}")

    if spec["spec"]["save_volumes"]:
        vols = {}
        with torch.no_grad():
            for ch, tag in (("plain", "p"), ("contrast", "c")):
                csm, y = getattr(tr, f"csm_{tag}"), getattr(tr, f"kspace_{tag}")
                mask, ds = getattr(tr, f"mask_{tag}"), getattr(tr, f"ds_{tag}")
                mask_fit, y_dc = getattr(tr, f"mask_fit_{tag}"), getattr(tr, f"kspace_dc_{tag}")
                gt = ds.get_data()["ground_truth"].to(dev)
                m = (mask if mask.ndim == 3 else mask[0]).to(dev).float().unsqueeze(0)
                mf = (mask_fit if mask_fit.ndim == 3 else mask_fit[0]).to(dev).float().unsqueeze(0)
                x = tr._render(ch)
                k_model = fft3c(csm * x.unsqueeze(0))
                x_dc = (ifft3c(mf * y_dc + (1.0 - mf) * k_model) * csm.conj()).sum(0)
                x_zf = (ifft3c(m * y) * csm.conj()).sum(0)
                n = names[ch]
                for key, t in ((f"rec_{n}", x_dc), (f"gt_{n}", gt), (f"zf_{n}", x_zf)):
                    vols[key] = t.abs().cpu().numpy().astype("float32")
                    vols[key + "_complex"] = t.cpu().numpy().astype("complex64")
        np.savez_compressed(res / f"{a.exp}_volumes.npz", **vols)
        log(f"  saved volumes -> {res / (a.exp + '_volumes.npz')}")


if __name__ == "__main__":
    main()
