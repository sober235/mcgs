#!/usr/bin/env python3
"""cmp_ledger.py -- the ledger of the comparison-arm improvement loop (2026-10-09): every probe next to its registered arm,
with the arm's OWN selection metric first (CMP_ARMS_AUDIT.md §3): head PSNR for the classical arms, L1-SPIRiT and MC-INR;
held-out k-space residual for 3D-DIP / 3D-INR; the Gamma loss for ZS-SSDU. GT metrics are printed for every row but only
decide for the arms whose registered protocol uses them.
  python scripts/cmp_ledger.py            # prints the ledger, writes logs/cmp/ledger.md"""
import glob, json, os, re
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
G = ("head_psnr", "head_ssim", "head_nmse", "head_hfen")


def rows_of(pattern):
    out = []
    for f in sorted(glob.glob(str(R / pattern))):
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        for r in doc.get("rows", []):
            if r.get("head_psnr") is not None:
                out.append((Path(f).name, r))
    return out


def fmt(r, extra=()):
    s = " ".join(f"{k}={r[k]:.4f}" if k != "head_psnr" else f"{k}={r[k]:.3f}" for k in G)
    for k in extra:
        if k in r and r[k] is not None:
            v = r[k]; s += f" {k}={v:.5f}" if isinstance(v, float) else f" {k}={v}"
    return s


def section(title, pattern, scan, extra, select, reverse, prefix=None):
    rs = [(f, r) for f, r in rows_of(pattern) if r.get("contrast", scan) == scan
          and (prefix is None or any(r["method"].startswith(p) for p in prefix))]
    if not rs:
        return [f"### {title} [{scan}]: no runs yet", ""]
    rs.sort(key=lambda fr: fr[1].get(select, 0), reverse=reverse)
    lines = [f"### {title} [{scan}]  (selected by {select}, {'higher' if reverse else 'lower'} is better; best first)", ""]
    for f, r in rs:
        lines.append(f"- `{r['method']}` {fmt(r, extra)}   <{f}>")
    return lines + [""]


def main():
    out = ["# comparison-arm ledger", f"topic {R}", ""]
    for scan in ("pre", "post"):
        out += section("CG-SENSE (tik x iters)", f"logs/classical_{scan}*.json", scan, ("tik",), "head_psnr", True, prefix=("cgsense",))
        out += section("CS-SENSE (outer, mu, lambda)", f"logs/classical_{scan}*.json", scan, ("lam", "mu"), "head_psnr", True, prefix=("cssense",))
        out += section("Joint SENSE", f"logs/jcssense_*.json", scan, ("lam", "mu"), "head_psnr", True)
        out += section("L1-SPIRiT", f"logs/spirit_{scan}*.json", scan, ("lam", "ksize", "tyk", "vcoils", "vcoil_energy"), "head_psnr", True)
        out += section("ZS-SSDU", f"logs/zsssdu*_{scan}.json", scan, ("val_loss", "epochs_run", "blocks", "cg_iters_net"), "val_loss", False)
        out += section("3D-DIP", f"logs/dip*_{scan}.json", scan, ("kres_heldout", "best_iter", "iters", "lr", "support", "dip_base"), "kres_heldout", False)
        out += section("3D-INR", f"logs/inr*_{scan}.json", scan, ("kres_heldout", "best_iter", "iters", "lr", "support", "inr_nmax", "tv", "cosine"), "kres_heldout", False)
        out += section("MC-INR (full volume)", f"logs/mcinr*_{scan}.json", scan, ("n_epochs", "lr"), "head_psnr", True)
        sub = []
        for f in sorted(glob.glob(str(R / f"logs/mcinr*_subset_{scan}.json"))):
            d = json.load(open(f)); sub.append((d["head_psnr"], Path(f).name, d))
        if sub:
            out += [f"### MC-INR subset sweep [{scan}] (20 slices, head PSNR on the gauge head mask; comparable only within this list)", ""]
            for p, f, d in sorted(sub, reverse=True):
                out.append(f"- head_psnr={p:.3f} head_nmse={d['head_nmse']:.5f} n_epochs={d['n_epochs']} lr={d['lr']} ({d['seconds_loop']:.0f} s)   <{f}>")
            out.append("")
    text = "\n".join(out)
    (R / "logs/cmp").mkdir(parents=True, exist_ok=True)
    (R / "logs/cmp/ledger.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
