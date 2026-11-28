#!/usr/bin/env python3
"""Gauge check for the three-pattern merge: score the variable-density Poisson AF 16 volumes of the AF study
(t1flair_af_testset_20261121) with THIS study's scoring (the uniform-density clone of each scan, collect_stats.eval_scan)
and compare with the numbers stored in the AF study's table.json. CPU only, reads only.

  gpu.py run --cpu -- @python autoresearch/diffmask_testset_20261122/gauge_check.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AFDIR = HERE.parent / "t1flair_af_testset_20261121"
KEYS = ("head_psnr", "head_ssim", "head_nmse")

LIST = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import collect_stats as CS
av = CS.availability()
out = []
for (scan, af, c, label), v in av.items():
    if af == 16:
        assert isinstance(v, dict), (scan, c, label, v)
        out.append([scan, c, label, str(v["file"]), v["member"]])
print("LIST" + json.dumps(out))
"""


def one_scan(scan, tasks_json, cache):
    sys.path.insert(0, str(HERE))
    import collect_stats as CS
    tasks = [("vdp16", c, label, f, m, None) for c, label, f, m in json.loads(tasks_json)]
    res = CS.eval_scan(scan, tasks, cache)
    print("RES" + json.dumps([[r["contrast"], r["label"], r.get("rc"), r.get("error")] for r in res]), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        return one_scan(sys.argv[2], sys.argv[3], sys.argv[4])
    o = subprocess.run([sys.executable, "-c", LIST, str(AFDIR)], capture_output=True, text=True, cwd=str(AFDIR))
    assert o.returncode == 0, o.stderr[-2000:]
    items = json.loads([l for l in o.stdout.splitlines() if l.startswith("LIST")][0][4:])
    scans = sorted({i[0] for i in items})
    assert len(items) == 6 * 2 * 11, len(items)
    table = json.load(open(AFDIR / "table.json"))
    stored = {(c["scan"], c["contrast"]): c["methods"] for c in table["cells"].values() if c["af"] == 16}
    cache = tempfile.mkdtemp(prefix="gauge_")
    procs = {s: subprocess.Popen([sys.executable, __file__, "--scan", s, json.dumps([i[1:] for i in items if i[0] == s]), cache],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for s in scans}
    rows, worst = [], {k: 0.0 for k in KEYS}
    for s, p in procs.items():
        out, err = p.communicate()
        assert p.returncode == 0, (s, err[-2000:])
        for c, label, rc, e in json.loads([l for l in out.splitlines() if l.startswith("RES")][0][3:]):
            assert e is None and rc is not None, (s, c, label, e)
            st = stored[(s, c)][label]
            d = {k: rc[k] - st[k] for k in KEYS}
            for k in KEYS:
                worst[k] = max(worst[k], abs(d[k]))
            rows.append(dict(scan=s, contrast=c, label=label, here={k: rc[k] for k in KEYS}, af_table={k: st[k] for k in KEYS}, diff=d))
    rep = dict(what="vd-Poisson AF16 volumes of the AF study scored with the diff-mask study's scoring (uniform-density clone) "
                    "minus the AF study's stored table values", n=len(rows), max_abs_diff=worst, rows=rows)
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
    (art / "gauge_check.json").write_text(json.dumps(rep, indent=1))
    print("GAUGE n", len(rows), "max |diff|:", ", ".join(f"{k} {v:.6f}" for k, v in worst.items()))


if __name__ == "__main__":
    main()
