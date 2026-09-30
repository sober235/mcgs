#!/usr/bin/env python3
"""ar_ledger.py -- the autoresearch ledger of the vdPoisMex line: every scored ar_* run against the frozen MC-GS, the
best non-GS arm and the GS-single baselines, on the head-only gauge, with the pre-registered reading applied.

Rows come from logs/ar_<exp>_<scan>.json (written by ar_queue.py via extract_gs_volumes.py); the reference numbers
from logs/summary.json (the registered eleven arms). Reading rule (autoresearch_vdpois.md): KEEP if both scans gain
>= +0.05 dB head PSNR over the frozen MC-GS and none of SSIM / NMSE / HFEN regresses on either scan; MIXED if the PSNR
gain is one-sided; DROP otherwise. GS-single runs (ar_s_*) are listed as baselines, not read.
  python scripts/ar_ledger.py            # prints the table; writes logs/ar/ledger.json and ledger.md
"""
import glob, json, os
from pathlib import Path

R = Path(os.environ.get("RETRO_TOPIC", "/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois"))
M = ("head_psnr", "head_ssim", "head_nmse", "head_hfen")
GAIN = 0.05


def row_of(path):
    d = json.load(open(path)); r = d["rows"][0]
    return {m: float(r[m]) for m in M}


def main():
    summ = json.load(open(R / "logs/summary.json"))
    ref = {(r["contrast"], r["method"]): {m: float(r[m]) for m in M} for r in summ}
    best_non_gs = {}
    for scan in ("pre", "post"):
        cands = [(r["head_psnr"], r["method"]) for r in summ if r["contrast"] == scan and r["method"] not in ("gs_joint", "gs_single")]
        best_non_gs[scan] = max(cands)
    runs = {}
    for p in sorted(glob.glob(str(R / "logs/ar_*_pre.json")) + glob.glob(str(R / "logs/ar_*_post.json"))):
        exp, scan = Path(p).name[:-5].rsplit("_", 1)          # keep the full key, e.g. ar_p1_kh10 / ar_s_pre_e0
        runs.setdefault(exp, {})[scan] = row_of(p)
    fair = {s: runs.get(f"ar_s_{s}_e0", {}).get(s) for s in ("pre", "post")}   # rule-A GS-single at the frozen recipe
    out = {"reference": {"frozen_mcgs": {s: ref[(s, "gs_joint")] for s in ("pre", "post")},
                         "registered_single": {s: ref[(s, "gs_single")] for s in ("pre", "post")},
                         "best_non_gs": {s: {"method": best_non_gs[s][1], "head_psnr": best_non_gs[s][0]} for s in ("pre", "post")}},
           "runs": {}}
    lines = ["| run | pre PSNR (Δ) | post PSNR (Δ) | pre SSIM/NMSE/HFEN Δ | post SSIM/NMSE/HFEN Δ | reading |", "|---|---|---|---|---|---|"]
    for exp, sc in runs.items():
        is_single = exp.startswith("ar_s_") or exp.startswith("s_")
        base_key = "gs_single" if is_single else "gs_joint"
        deltas, ok_metrics, gains = {}, True, []
        for scan in ("pre", "post"):
            if scan not in sc:
                continue
            b = ref[(scan, base_key)]; r = sc[scan]
            d = {m: r[m] - b[m] for m in M}; deltas[scan] = d; gains.append(d["head_psnr"])
            if d["head_ssim"] < -1e-4 or d["head_nmse"] > 1e-5 or d["head_hfen"] > 1e-4:
                ok_metrics = False
        if is_single:
            reading = "baseline"
        elif len(gains) < 2:
            reading = "incomplete"
        elif all(g >= GAIN for g in gains) and ok_metrics:
            reading = "KEEP"
        elif any(g >= GAIN for g in gains) and not all(g < 0 for g in gains):
            reading = "MIXED"
        else:
            reading = "DROP"
        def f(scan):
            if scan not in sc: return "—", "—"
            r, d = sc[scan], deltas[scan]
            return f"{r['head_psnr']:.3f} ({d['head_psnr']:+.3f})", f"{d['head_ssim']:+.4f} / {d['head_nmse']:+.5f} / {d['head_hfen']:+.4f}"
        p1, p2 = f("pre"); q1, q2 = f("post")
        lines.append(f"| {exp} | {p1} | {q1} | {p2} | {q2} | {reading} |")
        out["runs"][exp] = {"scores": sc, "delta_vs_" + base_key: deltas, "reading": reading}
    fs = " / ".join(f"{fair[s]['head_psnr']:.3f}" if fair[s] else "—" for s in ("pre", "post"))
    fm = " / ".join(f"{ref[(s,'gs_joint')]['head_psnr'] - fair[s]['head_psnr']:+.3f}" if fair[s] else "—" for s in ("pre", "post"))
    hdr = (f"reference: frozen MC-GS pre {ref[('pre','gs_joint')]['head_psnr']:.3f} / post {ref[('post','gs_joint')]['head_psnr']:.3f}; "
           f"registered GS-single {ref[('pre','gs_single')]['head_psnr']:.3f} / {ref[('post','gs_single')]['head_psnr']:.3f}; "
           f"fair GS-single (lambda_edge 0) {fs}, frozen MC-GS margin over it {fm}; "
           f"best non-GS {best_non_gs['pre'][1]} {best_non_gs['pre'][0]:.3f} / {best_non_gs['post'][1]} {best_non_gs['post'][0]:.3f}")
    out["reference"]["fair_single"] = fair
    print(hdr); print("\n".join(lines))
    (R / "logs/ar").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(R / "logs/ar/ledger.json", "w"), indent=1)
    open(R / "logs/ar/ledger.md", "w").write(hdr + "\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
