#!/usr/bin/env python3
"""cmp_select.py -- pick each comparison arm's winner per scan by that arm's OWN pre-registered rule (CMP_ARMS_AUDIT.md §3)
and print the make_cmp_topic.py --replace arguments. 2026-10-10.

  3D-DIP, 3D-INR : lowest held-out k-space residual (kres_heldout), no ground truth
  ZS-SSDU        : lowest loss on the registered 10 % Gamma (val_loss_gamma10; for runs at the registered rho_val 0.10 this is
                   their early-stopping val_loss), no ground truth
  L1-SPIRiT      : highest head PSNR (the registered protocol picks kernel and lambda on the truth)
  CG-SENSE       : highest head PSNR over the tik x iters sweep (informational: the registered arm is printed if it wins)
  CS-SENSE / Joint SENSE / MC-INR : registered results kept (convergence / mu / step x lr checks did not move them, §5)
Smoke runs (method key containing 'smoke') are ignored. The registered run of each arm competes too.
  python scripts/cmp_select.py            # table + the --replace line
"""
import glob, json, os
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))


def rows(pattern, scan, prefix):
    out = []
    for f in sorted(glob.glob(str(R / pattern))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for r in d.get("rows", []):
            m = r.get("method", "")
            if r.get("contrast", scan) == scan and m.startswith(prefix) and "smoke" not in m:
                out.append(r)
    return out


def zs_key(r):
    if r.get("val_loss_gamma10") is not None:
        return r["val_loss_gamma10"]
    if r.get("rho_val") in (None, 0.1):            # registered ratio: the early-stopping loss is the Gamma-10 loss
        return r["val_loss"]
    return None                                     # a larger Gamma without the common-set loss: not comparable


def main():
    rep, lines = [], []
    for scan in ("pre", "post"):
        lines.append(f"=== {scan}")
        for arm, pat, prefix, key, better, reg in [
            ("dip", f"logs/dip*_{scan}.json", "dip", lambda r: r.get("kres_heldout"), min, "dip"),
            ("inr", f"logs/inr*_{scan}.json", "inr", lambda r: r.get("kres_heldout"), min, "inr"),
            ("zsssdu", f"logs/zsssdu*_{scan}.json", "zsssdu", zs_key, min, "zsssdu"),
            ("spirit", f"logs/spirit_{scan}*.json", "spirit", lambda r: r.get("head_psnr"), max, None),
            ("cgsense", f"logs/classical_{scan}*.json", "cgsense", lambda r: r.get("head_psnr"), max, "cgsense")]:
            cands = [r for r in rows(pat, scan, prefix) if key(r) is not None]
            if not cands:
                lines.append(f"  {arm:8s} no runs"); continue
            win = better(cands, key=key)
            regrow = next((r for r in cands if r["method"] == reg), None) if reg else None
            if arm == "spirit":   # the registered arm is the summarize.py family maximum over the plain (uncompressed) grid
                plain = [r for r in cands if not r.get("vcoils")]
                regrow = max(plain, key=lambda r: r["head_psnr"]) if plain else None
            lines.append(f"  {arm:8s} winner {win['method']:34s} rule={key(win):.5f}  head {win['head_psnr']:.3f} / {win['head_ssim']:.4f} / "
                         f"{win['head_nmse']:.5f} / {win['head_hfen']:.4f}"
                         + (f"   [registered {regrow['method']}: rule={key(regrow):.5f} head {regrow['head_psnr']:.3f}]" if regrow else "")
                         + f"   ({len(cands)} runs)")
            if regrow is None or win["method"] != regrow["method"]:
                if arm != "cgsense":
                    rep.append(f"--replace {arm}:{scan}={win['method']}")
                elif win["head_psnr"] - regrow["head_psnr"] >= 0.01:
                    lines.append("           (CG-SENSE winner beats the registered run by >= 0.01 dB: install by hand, make_cmp_topic.py has no cgsense sweep rule)")
    print("\n".join(lines))
    print("\nREPLACE " + " ".join(rep))


if __name__ == "__main__":
    main()
