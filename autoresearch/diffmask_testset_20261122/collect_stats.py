#!/usr/bin/env python3
"""collect_stats.py -- collection, statistics and summary figure of the different-mask study (T1 + FLAIR, AF 16) on the
six test scans. Ported from autoresearch/t1flair_af_testset_20261121/collect_stats.py: the "af" key is the sampling
pattern here ("uniform", "radial"); statistics and scoring are unchanged. The vd-Poisson AF 16 cells of the other study
are NOT read here (third pattern, to be merged from that study's own table.json once it is final). The panel-alignment
gate of the source script is not used (plain matplotlib).

One re-runnable command, run on the local Mac (it launches ONE CPU job, waits for it, fetches and stores the results):

    /usr/bin/python3 autoresearch/diffmask_testset_20261122/collect_stats.py --pipeline

What the CPU job (--server, on amax, at most 24 threads) does, whatever part of the 12 x 2 x 11 results exists:
  1. For every (scan, AF, contrast, method) decide whether the result is present: GS rows need BOTH <exp>_score.json and
     <exp>_volumes.npz; comparator arms need every done-file of their row (run_arms.row_state == done); MC-INR must also
     have slices == all in the topic meta (and no SMOKE note). Everything else is listed as missing.
  2. Re-score every present volume with ONE scoring code: the cloned topic's t1t2_common.Pair.score (the uniform-density clone of the
     scan, CPU), ground truth from that scan's own full k-space. Head NMSE = sum_head |rec-gt|^2 / sum_head |gt|^2 (the
     definition of the record's head_nmse.py); per-slice metrics follow slice_trend.py of the trend study (head = |GT| >
     0.05 P99.9, slices whose head covers >= 5 % of the slice, three planes pooled). The score each runner logged is
     compared with the recomputed one; the largest differences are reported.
  3. Statistics, table.json / stats.json / table.md / RESULTS.md and the summary figure (all written to MCGS_ARTIFACT_DIR).
A cell (scan, AF, contrast) missing any of the 11 methods is flagged incomplete and excluded from rank counts and statistics.

Volume slots: GS joint = recT1_complex (T1) / recFL_complex (FLAIR); GS-single = recT1_complex, the first channel, which is the
plain channel of the registered score (singleFLAIR has FLAIR in that slot).

Time rule: no server clock is read or printed; job ids are shown by their last 8 characters only.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

sys.dont_write_bytecode = True        # nothing but OUT/stats_cache may be written under OUT (clone scripts are imported)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ts_common import AFS, OUT, SCANS   # noqa: E402

try:
    import numpy as np
except ImportError:          # the local --pipeline needs no numpy
    np = None

CONTRASTS = ("T1", "FLAIR")
LABELS = ["ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU", "3D-DIP", "3D-INR", "MC-INR",
          "GS-single", "MC-GS"]
NONGS = [m for m in LABELS if m not in ("GS-single", "MC-GS")]
ARM_OF = {"ZF": ("classical", "zf"), "CG-SENSE": ("classical", "cgsense"), "CS-SENSE": ("cssense", "cssense_o200"),
          "Joint SENSE": ("jcssense", "jcssense"), "L1-SPIRiT": ("spirit", "spirit"), "ZS-SSDU": ("zsssdu", "zsssdu"),
          "3D-DIP": ("dip", "dip"), "3D-INR": ("inr", "inr"), "MC-INR": ("mcinr", "mcinr")}
METRICS = ("head_psnr", "head_ssim", "head_nmse")
MNAME = {"head_psnr": "PSNR", "head_ssim": "SSIM", "head_nmse": "NMSE"}
HIGHER = {"head_psnr": True, "head_ssim": True, "head_nmse": False}
COMPS = ("GS-single", "strongest")
CACHE_VERSION = "v2"
READ_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error)   # a file that is still being written
RESULTS_REL = "autoresearch/diffmask_testset_20261122"
PAT = {"uniform": "Uniform density", "radial": "Pseudo-radial"}
PATZH = {"uniform": "均匀密度", "radial": "伪径向"}
log = lambda *a: print(*a, flush=True)


# ===================================================================== pure statistics
def rank_of_mcgs(cell, higher_is_better):
    """Competition rank of MC-GS among the 11 methods of a cell (1 = first; ties with the best count as first).
    A cell that lacks any of the 11 methods is incomplete: returns None (no rank is counted)."""
    if any(m not in cell or cell[m] is None or math.isnan(cell[m]) for m in LABELS):
        return None
    mc = cell["MC-GS"]
    return 1 + sum((cell[m] > mc) if higher_is_better else (cell[m] < mc) for m in LABELS if m != "MC-GS")


def person_of(scan):
    return SCANS[scan]["person"]


def per_person(values):
    """{scan: value} -> {person: mean over that person's scans} (vol2 + vol6 and vol4 + vol7 are the same persons)."""
    acc = {}
    for scan, v in values.items():
        acc.setdefault(person_of(scan), []).append(v)
    return {p: sum(v) / len(v) for p, v in acc.items()}


def _midranks(a):
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def exact_wilcoxon_p(d):
    """Exact two-sided Wilcoxon signed-rank p (zeros dropped, mid-ranks for ties, enumeration by dynamic programming)."""
    d = [float(x) for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    r2 = [int(round(2 * r)) for r in _midranks([abs(x) for x in d])]
    total = sum(r2)
    wp = sum(r for r, x in zip(r2, d) if x > 0)
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in r2:
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    tail = sum(counts[:min(wp, total - wp) + 1]) / 2 ** n
    return min(1.0, 2 * tail)


def paired_summary(diffs):
    d = [float(x) for x in diffs]
    n = len(d)
    mean = sum(d) / n if n else float("nan")
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return dict(mean=mean, sd=sd, min=min(d) if n else float("nan"), n_positive=sum(x > 0 for x in d), n=n,
                wilcoxon_p=exact_wilcoxon_p(d) if n else float("nan"))


def improvement(metric, mc, ot):
    """positive = MC-GS better; PSNR in dB, SSIM absolute, NMSE as relative reduction in %"""
    mc, ot = np.asarray(mc, float), np.asarray(ot, float)
    return 100.0 * (ot - mc) / ot if metric == "head_nmse" else mc - ot


def slice_metrics(rec, gt, head=None):
    """Per-slice head PSNR / SSIM / NMSE of every plane of a volume (vectorised, slice_trend.py of the trend study):
    PSNR over head pixels against max|GT| of the volume, SSIM map (7x7 uniform window in the plane, 49/48 covariance,
    data range max|GT|) averaged over head pixels, NMSE over head pixels. Returns ({(ax, metric): array}, {ax: keep})."""
    from scipy.ndimage import uniform_filter
    gt = np.abs(gt).astype(np.float32)
    if head is None:
        head = gt > 0.05 * float(np.quantile(gt, 0.999))
    r = np.abs(rec).astype(np.float64)
    g64 = gt.astype(np.float64)
    gmax = float(gt.max())
    c1, c2, cn = (0.01 * gmax) ** 2, (0.03 * gmax) ** 2, 49.0 / 48.0
    other = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    out, keep = {}, {}
    for ax in (0, 1, 2):
        size = [7, 7, 7]
        size[ax] = 1
        n = head.sum(axis=other[ax])
        area = float(np.prod([gt.shape[a] for a in other[ax]]))
        keep[ax] = n >= 0.05 * area
        nn = np.maximum(n, 1)
        sse = (((r - g64) ** 2) * head).sum(axis=other[ax])
        gg = ((g64 ** 2) * head).sum(axis=other[ax])
        ux = uniform_filter(g64, size)
        vx = cn * (uniform_filter(g64 * g64, size) - ux * ux)
        uy = uniform_filter(r, size)
        vy = cn * (uniform_filter(r * r, size) - uy * uy)
        vxy = cn * (uniform_filter(g64 * r, size) - ux * uy)
        s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
        out[(ax, "head_psnr")] = 20 * np.log10(gmax) - 10 * np.log10(np.maximum(sse / nn, 1e-30))
        out[(ax, "head_ssim")] = (s * head).sum(axis=other[ax]) / nn
        out[(ax, "head_nmse")] = sse / np.maximum(gg, 1e-30)
    return out, keep


def pooled(sm, keep, metric):
    return np.concatenate([sm[(ax, metric)][keep[ax]] for ax in (0, 1, 2)])


def tests_from_metrics(sm_mc, sm_ot, keep):
    """per-slice paired comparison, MC-GS against another method, three planes pooled; two-sided Wilcoxon signed-rank"""
    from scipy.stats import wilcoxon
    res = {}
    for m in METRICS:
        d = improvement(m, pooled(sm_mc, keep, m), pooled(sm_ot, keep, m))
        p = 1.0 if not np.any(d != 0) else float(wilcoxon(d, alternative="two-sided").pvalue)
        res[m] = dict(n=int(d.size), median=float(np.median(d)), mean=float(d.mean()), win_share=float(np.mean(d > 0)),
                      wilcoxon_p=p)
    return res


def slice_tests(mcgs, other, gt, head=None):
    """mcgs, other, gt: 3-D volumes (|complex| is taken). Slice selection as in the overview figure (see slice_metrics)."""
    gt = np.abs(gt)
    sm1, keep = slice_metrics(mcgs, gt, head)
    sm2, _ = slice_metrics(other, gt, head)
    return tests_from_metrics(sm1, sm2, keep)


def selftest():
    s = paired_summary([0.1] * 6)
    assert s["n_positive"] == 6 and s["n"] == 6 and abs(s["wilcoxon_p"] - 0.03125) < 1e-12, s
    assert abs(paired_summary([0.1] * 4)["wilcoxon_p"] - 0.125) < 1e-12
    assert abs(paired_summary([0.3, 0.1, 0.2, -0.05, 0.4, 0.5])["wilcoxon_p"] - 0.0625) < 1e-12   # W- = 1 of 6 ranks
    assert paired_summary([0.0, 0.0])["wilcoxon_p"] == 1.0
    ps = paired_summary([1.0, 2.0, -1.0, 4.0])
    assert ps["n_positive"] == 3 and ps["min"] == -1.0 and abs(ps["mean"] - 1.5) < 1e-12 and ps["sd"] > 0
    full = {m: float(i) for i, m in enumerate(LABELS)}          # MC-GS last = highest
    assert rank_of_mcgs(full, True) == 1 and rank_of_mcgs(full, False) == 11
    tie = dict(full, **{"3D-DIP": full["MC-GS"]})
    assert rank_of_mcgs(tie, True) == 1                          # tie with the best counts as first
    behind = dict(full, **{"3D-DIP": full["MC-GS"] + 1})
    assert rank_of_mcgs(behind, True) == 2
    assert rank_of_mcgs({k: v for k, v in full.items() if k != "MC-INR"}, True) is None
    assert rank_of_mcgs(dict(full, **{"MC-INR": None}), True) is None
    vals = {"vol1_huangyuwei": 1.0, "vol2_jiangdian": 2.0, "vol4_tangbing": 3.0, "vol5_sunxinyi": 4.0,
            "vol6_jiangdian_790": 4.0, "vol7_tangbin_790": 5.0}
    pp = per_person(vals)
    assert len(pp) == 4 and pp["jiangdian"] == 3.0 and pp["tangbing"] == 4.0 and pp["huangyuwei"] == 1.0, pp
    assert abs(improvement("head_nmse", 0.01, 0.02) - 50.0) < 1e-9 and improvement("head_psnr", 31.0, 30.0) == 1.0
    # slice metrics against direct computation and skimage on a small synthetic volume
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, 24)] * 3, indexing="ij")
    gt = (np.exp(-(x ** 2 + y ** 2 + z ** 2) * 3) + 0.02 * rng.random((24, 24, 24))).astype(np.float32)
    rec = (gt + 0.03 * rng.standard_normal(gt.shape)).astype(np.float32)
    rec2 = (gt + 0.06 * rng.standard_normal(gt.shape)).astype(np.float32)
    head = gt > 0.05 * float(np.quantile(gt, 0.999))
    sm, keep = slice_metrics(rec, gt)
    gmax = float(gt.max())
    for ax, idx in ((0, 12), (1, 9), (2, 14)):
        sl = [slice(None)] * 3
        sl[ax] = idx
        g2, r2, h2 = gt[tuple(sl)].astype(np.float64), np.abs(rec)[tuple(sl)].astype(np.float64), head[tuple(sl)]
        e = r2[h2] - g2[h2]
        assert abs(sm[(ax, "head_psnr")][idx] - (20 * np.log10(gmax) - 10 * np.log10((e * e).mean()))) < 1e-9
        assert abs(sm[(ax, "head_nmse")][idx] - (e * e).sum() / (g2[h2] ** 2).sum()) < 1e-9
        _, smap = structural_similarity(g2, r2, data_range=gmax, full=True, win_size=7)
        assert abs(sm[(ax, "head_ssim")][idx] - smap[h2].mean()) < 1e-6, (ax, sm[(ax, "head_ssim")][idx], smap[h2].mean())
    t = slice_tests(rec, rec2, gt)                               # the less noisy one must win
    assert all(t[m]["win_share"] > 0.9 for m in METRICS) and t["head_psnr"]["n"] == int(sum(k.sum() for k in keep.values())), t
    assert slice_tests(rec, rec, gt)["head_psnr"]["wilcoxon_p"] == 1.0
    assert rank_of_mcgs(dict(full, **{"3D-INR": float("nan")}), True) is None      # NaN is missing, never ranked behind MC-GS
    selftest_assemble()
    selftest_docs()
    print("SELFTEST_OK")


def selftest_assemble():
    """completeness per (scan, AF), the MC-GS recompute check on incomplete cells, NaN and unreadable files"""
    import tempfile
    scan = "vol1_huangyuwei"

    def build(missing=(), bias=0.0, nan=None, err=None, only_t1=False):
        av = {(s, af, c, l): "arm not done" for s in SCANS for af in AFS for c in CONTRASTS for l in LABELS}
        res = []
        for c in CONTRASTS:
            for i, l in enumerate(LABELS):
                if (c, l) in missing or (only_t1 and c == "FLAIR"):
                    continue
                lg = dict(head_psnr=24.0 + i, head_ssim=0.8 + 0.01 * i, nmse=0.05 / (1 + i))
                av[(scan, AFS[0], c, l)] = dict(file="x", member=None, logged=lg)
                bi = bias if (l == "MC-GS" and c == "T1") else 0.0
                rc = dict(head_psnr=lg["head_psnr"] + bi, head_ssim=lg["head_ssim"], head_nmse=0.05 / (1 + i),
                          nmse_whole=lg["nmse"], ssim_whole=0.9, psnr_whole=30.0)
                if nan == (c, l):
                    rc["head_ssim"] = float("nan")
                r = dict(af=AFS[0], contrast=c, label=l, logged=lg, scan=scan, cache="none", rc=rc)
                if err == (c, l):
                    r = dict(af=AFS[0], contrast=c, label=l, logged=lg, scan=scan, error="unreadable (being written?): BadZipFile")
                res.append(r)
        return assemble(av, res)

    cells, chk = build()
    assert n_complete_cells(cells) == 1 and cells[f"{scan}|{AFS[0]}|T1"]["complete"] and cells[f"{scan}|{AFS[0]}|FLAIR"]["rank"]["head_psnr"] == 1
    assert chk["mcgs_mismatch"] == [] and chk["unreadable"] == []
    cells, chk = build(missing={("FLAIR", "ZS-SSDU")})           # FLAIR lacks a method: BOTH contrasts excluded
    assert n_complete_cells(cells) == 0 and not cells[f"{scan}|{AFS[0]}|T1"]["complete"] and not cells[f"{scan}|{AFS[0]}|T1"]["rank"]
    assert "diffs" not in cells[f"{scan}|{AFS[0]}|T1"] and cells[f"{scan}|{AFS[0]}|T1"]["contrast_complete"]
    cells, chk = build(only_t1=True)                              # only T1 present
    assert n_complete_cells(cells) == 0 and not cells[f"{scan}|{AFS[0]}|T1"]["complete"]
    assert group_stats(cells)[f"T1|AF{AFS[0]}|head_psnr|GS-single"]["all6"] is None
    cells, chk = build(missing={("FLAIR", "ZS-SSDU")}, bias=0.01)  # mismatched MC-GS score in an INCOMPLETE cell must be flagged
    assert n_complete_cells(cells) == 0 and len(chk["mcgs_mismatch"]) == 1, chk["mcgs_mismatch"]
    cells, chk = build(nan=("T1", "3D-INR"))
    assert cells[f"{scan}|{AFS[0]}|T1"]["missing"] == {"3D-INR": "non-finite metric"} and n_complete_cells(cells) == 0
    cells, chk = build(err=("FLAIR", "MC-INR"))
    assert "MC-INR" in cells[f"{scan}|{AFS[0]}|FLAIR"]["missing"] and len(chk["unreadable"]) == 1 and n_complete_cells(cells) == 0
    # a truncated npz / npy becomes "unreadable", not a crash
    tmp = Path(tempfile.mkdtemp(prefix="cs_trunc_"))
    try:
        np.savez_compressed(tmp / "v.npz", recT1_complex=rng_arr(), recFL_complex=rng_arr())
        raw = (tmp / "v.npz").read_bytes()
        (tmp / "t.npz").write_bytes(raw[: len(raw) // 2])
        np.save(tmp / "a.npy", rng_arr())
        raw = (tmp / "a.npy").read_bytes()
        (tmp / "t.npy").write_bytes(raw[: len(raw) // 2])
        assert guarded(crc_key, tmp / "t.npz", "recT1_complex")[1].startswith("unreadable")
        assert guarded(lambda: np.load(tmp / "t.npz")["recT1_complex"])[1].startswith("unreadable")
        assert guarded(lambda: np.load(tmp / "t.npy"))[1].startswith("unreadable")
        assert guarded(crc_key, tmp / "v.npz", "recT1_complex")[1] is None
        (tmp / "bad.json").write_text('{"a": ')
        assert guarded(lambda: json.load(open(tmp / "bad.json")))[1].startswith("unreadable")
    finally:
        shutil.rmtree(tmp)


def rng_arr():
    return (np.random.default_rng(3).random((16, 16, 16)) + 1j).astype(np.complex64)


def selftest_docs():
    """statistics, documents and figure on SYNTHETIC complete cells (random numbers, never written to the repo)"""
    import tempfile
    rng = np.random.default_rng(1)
    tmp = Path(tempfile.mkdtemp(prefix="cs_selftest_"))
    try:
        cells = {}
        for scan in SCANS:
            for af in AFS:
                for c in CONTRASTS:
                    cell = dict(scan=scan, af=af, contrast=c, methods={}, missing={}, rank={}, first={})
                    for i, l in enumerate(LABELS):
                        base = 24 + 1.1 * i + rng.normal(0, 0.2)
                        cp = tmp / f"{scan}_{af}_{c}_{i}.npz"
                        arrs = {f"{m}{ax}": rng.random(30).astype(np.float32) + (base if m == "head_psnr" else 0.5)
                                for ax in (0, 1, 2) for m in METRICS}
                        arrs.update({f"keep{ax}": np.ones(30, bool) for ax in (0, 1, 2)})
                        np.savez(cp, **arrs)
                        cell["methods"][l] = dict(head_psnr=base, head_ssim=0.8 + 0.01 * i, head_nmse=0.05 / (1 + i),
                                                  cache=str(cp), diff=dict(head_psnr=0.0, head_ssim=0.0, nmse=0.0))
                    cell["complete"] = True
                    for m in METRICS:
                        vals = {l: cell["methods"][l][m] for l in LABELS}
                        cell["rank"][m] = rank_of_mcgs(vals, HIGHER[m])
                        cell["first"][m] = cell["rank"][m] == 1
                        if cell["rank"][m] != 1:
                            best = (max if HIGHER[m] else min)(vals, key=vals.get)
                            cell.setdefault("not_first", {})[m] = dict(rank=cell["rank"][m], best=best, best_value=vals[best],
                                                                    mcgs_value=vals["MC-GS"])
                    cell["strongest_non_gs"] = max(NONGS, key=lambda l: cell["methods"][l]["head_psnr"])
                    cell["diffs"] = {comp: {m: float(improvement(m, cell["methods"]["MC-GS"][m],
                                            cell["methods"]["GS-single" if comp == "GS-single" else cell["strongest_non_gs"]][m]))
                                            for m in METRICS} for comp in COMPS}
                    cells[f"{scan}|{af}|{c}"] = cell
        g = group_stats(cells)
        e = g[f"T1|AF{AFS[0]}|head_psnr|GS-single"]
        assert e["all6"]["n"] == 6 and e["persons"]["n"] == 4 and e["aligned"]["n"] == 3 and e["offset"]["n"] == 3, e
        assert e["persons"]["wilcoxon_p"] >= 0.125 and e["all6"]["wilcoxon_p"] >= 0.03125 - 1e-12
        st = slice_stats(cells)
        assert len(st) == 6 * len(AFS) * 2 * 2
        chk = dict(worst={k: dict(abs=0.0, where="synthetic") for k in ("head_psnr", "head_ssim", "nmse")},
                   per_label={l: dict(head_psnr=0.0, head_ssim=0.0, nmse=0.0) for l in LABELS})
        write_table_md(cells, tmp)
        chk.update(mcgs_mismatch=[], unreadable=[])
        write_results_md(cells, chk, g, st, tmp, len(SCANS) * len(AFS))
        draw_figure(cells, tmp)
        for n in ("table.md", "RESULTS.md", "testset_summary.png", "testset_summary.pdf"):
            assert (tmp / n).stat().st_size > 1000, n
        # time rule: no generation date inside any produced figure file (positive control: matplotlib's default has them)
        import io
        import matplotlib.pyplot as _plt
        _f = _plt.figure(figsize=(1, 1))
        for _fmt, _pat in (("pdf", b"CreationDate"), ("svg", b"dc:date")):
            _b = io.BytesIO()
            _f.savefig(_b, format=_fmt)
            assert _pat in _b.getvalue(), f"the date check cannot detect {_fmt}"
        _plt.close(_f)
        for f in tmp.glob("testset_summary*"):
            if f.suffix in (".pdf", ".svg", ".png"):
                b = f.read_bytes()
                for pat in (b"CreationDate", b"ModDate", b"dc:date", b"<dc:date", b"tIME", b"date:create", b"date:modify"):
                    assert pat not in b, (f.name, pat)
        assert "不是第一" in (tmp / "RESULTS.md").read_text(encoding="utf-8")
        if os.environ.get("MCGS_ARTIFACT_DIR"):          # layout check only: random numbers, named so
            for ext in ("png", "pdf"):
                shutil.copy2(tmp / f"testset_summary.{ext}", Path(os.environ["MCGS_ARTIFACT_DIR"]) / f"SYNTHETIC_selftest_layout.{ext}")
    finally:
        shutil.rmtree(tmp)


# ===================================================================== availability and logged scores
def jobtail(s):
    return s[-8:]


def crc_key(path, member=None):
    """cheap content key: CRC32 stored in the npz directory, or CRC32 of the npy bytes, plus the size"""
    path = Path(path)
    if member is not None:
        with zipfile.ZipFile(path) as z:
            zi = z.getinfo(member + ".npy")
            return f"{zi.CRC:08x}-{zi.file_size}"
    crc = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            crc = zlib.crc32(b, crc)
    return f"{crc:08x}-{path.stat().st_size}"


def guarded(fn, *a):
    """(value, None), or (None, reason) when a file is unreadable (e.g. still being written); nothing is cached for it"""
    try:
        return fn(*a), None
    except READ_ERRORS as e:
        return None, f"unreadable (being written?): {type(e).__name__}"


def _availability_row(run_arms, run_gs, scan, af, label, av):
    if label in ("GS-single", "MC-GS"):
        for c in CONTRASTS:
            if label == "MC-GS":
                row, slot = f"{scan}:{af}:joint", ("recT1_complex" if c == "T1" else "recFL_complex")
            else:
                row, slot = f"{scan}:{af}:single{c}", "recT1_complex"
            p = run_gs.row_paths(OUT, row)
            vol = p["outs"][3]
            if not (p["score"].exists() and vol.exists()):
                av[(scan, af, c, label)] = "missing score or volumes"
                continue
            s = json.load(open(p["score"]))
            if s["plain_name"] == c:
                blk = s["plain"]
            else:
                assert label == "MC-GS" and s["contrast_name"] == c, (row, s["plain_name"], s["contrast_name"])
                blk = s["contrast"]
            av[(scan, af, c, label)] = dict(file=vol, member=slot, logged=dict(
                head_psnr=blk["head_psnr"], head_ssim=blk["head_ssim"], nmse=blk["nmse"]))
        return
    arm, reg = ARM_OF[label]
    row = f"{scan}:{af}:{arm}"
    if run_arms.row_state(OUT, row) != "done":
        for c in CONTRASTS:
            av[(scan, af, c, label)] = f"arm {arm} not done"
        return
    if arm == "mcinr":
        T = run_arms.topic_dir(scan, af, OUT)
        meta = json.load(open(T / "mcinr/meta.json"))
        if meta.get("slices") != "all":
            for c in CONTRASTS:
                av[(scan, af, c, label)] = f"MC-INR slices = {meta.get('slices')!r}, not all"
            return
    _, res = run_arms.arm_plan(OUT, row, "0")
    for r in res:
        if r["reg"] != reg:
            continue
        if arm == "mcinr" and "SMOKE" in str(json.load(open(r["json"])).get("note", "")):
            av[(scan, af, r["contrast"], label)] = "MC-INR record is a smoke run"
            continue
        lg = run_arms.result_row(r)
        av[(scan, af, r["contrast"], label)] = dict(file=r["recon"], member=None, logged=dict(
            head_psnr=lg["head_psnr"], head_ssim=lg["head_ssim"], nmse=lg["nmse"]))


def availability():
    """{(scan, af, contrast, label): dict(file, member, logged) | str reason of absence}"""
    import run_arms
    import run_gs
    av = {}
    for scan in SCANS:
        for af in AFS:
            for label in LABELS:
                _, err = guarded(_availability_row, run_arms, run_gs, scan, af, label, av)
                if err:
                    for c in CONTRASTS:
                        av[(scan, af, c, label)] = err
    return av


# ===================================================================== workers (one process per scan)
def score_volume(P, x, label, torch):
    """the ONE scoring step of a volume: Pair.score (head PSNR / SSIM, whole PSNR / SSIM / NMSE), head NMSE, slice metrics"""
    assert x.shape == P.gtn.shape, (x.shape, P.gtn.shape)
    row = P.score(torch.from_numpy(x).to(torch.complex64), label)
    rec = np.abs(x).astype(np.float32)
    d = float(((rec[P.head] - P.gtn[P.head]) ** 2).sum())
    nm = d / float((P.gtn[P.head] ** 2).sum())
    chk = 20 * np.log10(float(P.gtn.max())) - 10 * np.log10(d / int(P.head.sum()))
    assert abs(chk - row["head_psnr"]) < 1e-3, (chk, row["head_psnr"])   # same head error sum
    sm, keep = slice_metrics(x, P.gtn, P.head)
    rc = dict(head_psnr=row["head_psnr"], head_ssim=row["head_ssim"], head_nmse=nm, nmse_whole=row["nmse"],
              ssim_whole=row["ssim"], psnr_whole=row["psnr"])
    return rc, sm, keep


def scoring_hash(TC, T, c):
    """key of everything a cached value depends on: the scoring source (clone t1t2_common, the metric modules it imports,
    the head-mask module, this file's volume scoring and slice metrics) and the ground-truth inputs of the scan"""
    import hashlib
    import inspect
    h = hashlib.sha256()
    files = [T / "scripts/t1t2_common.py", *sorted((TC.ROOT / "paired_gs/metrics").glob("*.py")),
             *sorted((TC.ROOT / "diag_artefact").glob("*.py"))]
    for f in files:
        h.update(str(f.name).encode() + f.read_bytes())
    h.update((inspect.getsource(score_volume) + inspect.getsource(slice_metrics)).encode())
    for f in (T / f"data/ksp_{c}.h5", T / f"data/csm_espirit_{c}.h5"):
        h.update(crc_key(f).encode())
    return h.hexdigest()[:16]


def eval_scan(scan, tasks, cache_dir):
    """tasks: list of (af, contrast, label, file, member, logged). Returns list of result dicts (with "error" for a file
    that cannot be read). CPU, 4 threads."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch
    torch.set_num_threads(4)
    from run_arms import topic_dir
    T = topic_dir(scan, "uniform", OUT)                   # ONE scoring code for all cells: the uniform-density clone of the scan
    sys.path.insert(0, str(T / "scripts"))
    import t1t2_common as TC
    assert Path(TC.R) == T, TC.R
    from diag_artefact.headonly import head_mask
    dev = torch.device("cpu")
    pairs, vers, out = {}, {}, []
    for af, c, label, file, member, logged in tasks:
        base = dict(af=af, contrast=c, label=label, logged=logged)
        try:
            if c not in vers:
                vers[c] = scoring_hash(TC, T, c)
            key = crc_key(file, member)
            cpath = Path(cache_dir) / f"{CACHE_VERSION}_{scan}_af{af}_{c}_{label.replace(' ', '')}_{vers[c]}_{key}.npz"
            if cpath.exists():
                z = np.load(cpath)
                rc = {k: float(z[k]) for k in ("head_psnr", "head_ssim", "head_nmse", "nmse_whole", "ssim_whole", "psnr_whole")}
            else:
                if c not in pairs:
                    pairs[c] = TC.Pair(c, dev)
                    pairs[c].gtn = pairs[c].gt.cpu().numpy()
                    pairs[c].head = head_mask(pairs[c].gtn)
                x = np.load(file) if member is None else np.load(file)[member]
                rc, sm, keep = score_volume(pairs[c], x, label, torch)
                arrs = {f"{m}{ax}": sm[(ax, m)].astype(np.float32) for ax in (0, 1, 2) for m in METRICS}
                arrs.update({f"keep{ax}": keep[ax] for ax in (0, 1, 2)})
                tmp = cpath.with_name(cpath.name + f".tmp{os.getpid()}.npz")
                np.savez_compressed(tmp, **{k: np.float64(v) for k, v in rc.items()}, **arrs)
                os.link(tmp, cpath) if not cpath.exists() else None      # never overwrite an existing cache file
                os.unlink(tmp)
        except READ_ERRORS as e:                       # a file still being written: missing for now, nothing cached
            out.append(dict(base, error=f"unreadable (being written?): {type(e).__name__}"))
            print(f"[{scan}] AF{af} {c} {label} UNREADABLE ({type(e).__name__})", flush=True)
            continue
        out.append(dict(base, cache=str(cpath), rc=rc))
        print(f"[{scan}] AF{af} {c} {label} head PSNR {rc['head_psnr']:.3f}", flush=True)
    return out


def _worker(args):
    return eval_scan(*args)


# ===================================================================== assembly, statistics
def assemble(av, results):
    """cells[(scan|af|contrast)]; a (scan, AF) is complete only when all 11 methods exist for BOTH contrasts (ruling);
    only complete (scan, AF) cells get ranks and differences. A NaN metric counts as missing."""
    cells = {}
    byk = {(r["af"], r["contrast"], r["label"], r["scan"]): r for r in results}
    worst = {k: dict(abs=0.0, where=None) for k in ("head_psnr", "head_ssim", "nmse")}
    per_label, mcgs_bad, unreadable = {}, [], []
    for scan in SCANS:
        for af in AFS:
            for c in CONTRASTS:
                cell = dict(scan=scan, af=af, contrast=c, methods={}, missing={}, rank={}, first={})
                for label in LABELS:
                    a = av[(scan, af, c, label)]
                    if isinstance(a, str):
                        cell["missing"][label] = a
                        if a.startswith("unreadable"):
                            unreadable.append(f"{scan} AF{af} {c} {label}")
                        continue
                    r = byk.get((af, c, label, scan))
                    if r is None or "error" in r:
                        cell["missing"][label] = r["error"] if r else "no result"
                        unreadable.append(f"{scan} AF{af} {c} {label}")
                        continue
                    rc, lg = r["rc"], r["logged"]
                    if not all(math.isfinite(rc[m]) for m in METRICS):
                        cell["missing"][label] = "non-finite metric"
                        continue
                    dif = dict(head_psnr=rc["head_psnr"] - lg["head_psnr"], head_ssim=rc["head_ssim"] - lg["head_ssim"],
                               nmse=rc["nmse_whole"] - lg["nmse"])
                    cell["methods"][label] = dict(rc, logged=lg, diff=dif, cache=r["cache"])
                    if label == "MC-GS" and not abs(dif["head_psnr"]) <= 0.005:      # every MC-GS volume, complete cell or not
                        mcgs_bad.append(f"{scan} AF{af} {c}: {dif['head_psnr']:+.4g} dB")
                    for k, v in dif.items():
                        if abs(v) > worst[k]["abs"]:
                            worst[k] = dict(abs=abs(v), where=f"{scan} AF{af} {c} {label}")
                        pl = per_label.setdefault(label, dict(head_psnr=0.0, head_ssim=0.0, nmse=0.0))
                        pl[k] = max(pl[k], abs(v))
                cells[f"{scan}|{af}|{c}"] = cell
    for scan in SCANS:
        for af in AFS:
            pair = [cells[f"{scan}|{af}|{c}"] for c in CONTRASTS]
            ok = all(not x["missing"] for x in pair)
            for cell in pair:
                cell["complete"] = ok
                cell["contrast_complete"] = not cell["missing"]
                if not ok:
                    continue
                for m in METRICS:
                    vals = {l: cell["methods"][l][m] for l in LABELS}
                    cell["rank"][m] = rank_of_mcgs(vals, HIGHER[m])
                    best = (max if HIGHER[m] else min)(vals, key=vals.get)
                    cell["first"][m] = cell["rank"][m] == 1
                    if cell["rank"][m] != 1:
                        cell.setdefault("not_first", {})[m] = dict(rank=cell["rank"][m], best=best, best_value=vals[best],
                                                                mcgs_value=vals["MC-GS"])
                cell["strongest_non_gs"] = max(NONGS, key=lambda l: cell["methods"][l]["head_psnr"])
                cell["diffs"] = {}
                for comp in COMPS:
                    o = cell["strongest_non_gs"] if comp == "strongest" else "GS-single"
                    cell["diffs"][comp] = {m: float(improvement(m, cell["methods"]["MC-GS"][m], cell["methods"][o][m]))
                                           for m in METRICS}
    return cells, dict(worst=worst, per_label=per_label, mcgs_mismatch=mcgs_bad, unreadable=unreadable)


def n_complete_cells(cells):
    """number of complete (scan, AF) cells (each has a T1 and a FLAIR entry in `cells`)"""
    return sum(c["complete"] for c in cells.values()) // len(CONTRASTS)


def group_stats(cells):
    """between-case statistics per (contrast, AF, metric, comparator) and unit set"""
    groups = {"all6": list(SCANS), "aligned": [s for s in SCANS if SCANS[s]["group"] == "aligned"],
              "offset": [s for s in SCANS if SCANS[s]["group"] == "offset"]}
    out = {}
    for c in CONTRASTS:
        for af in AFS:
            ok = {s: cells[f"{s}|{af}|{c}"] for s in SCANS if cells[f"{s}|{af}|{c}"]["complete"]}
            for m in METRICS:
                for comp in COMPS:
                    vals = {s: cell["diffs"][comp][m] for s, cell in ok.items()}
                    ent = {}
                    for g, scans in groups.items():
                        v = [vals[s] for s in scans if s in vals]
                        ent[g] = paired_summary(v) if v else None
                    # persons: only persons whose scans are all complete
                    people = {}
                    for s in SCANS:
                        people.setdefault(person_of(s), []).append(s)
                    pv = per_person({s: vals[s] for p, ss in people.items() if all(x in vals for x in ss) for s in ss})
                    ent["persons"] = paired_summary(list(pv.values())) if pv else None
                    if ent["persons"]:
                        ent["persons"]["persons"] = sorted(pv)
                    out[f"{c}|AF{af}|{m}|{comp}"] = ent
    return out


def slice_stats(cells):
    out = {}
    for key, cell in cells.items():
        if not cell["complete"]:
            continue
        mc = np.load(cell["methods"]["MC-GS"]["cache"])
        keep = {ax: mc[f"keep{ax}"] for ax in (0, 1, 2)}
        sm_mc = {(ax, m): mc[f"{m}{ax}"] for ax in (0, 1, 2) for m in METRICS}
        for comp in COMPS:
            o = cell["strongest_non_gs"] if comp == "strongest" else "GS-single"
            z = np.load(cell["methods"][o]["cache"])
            sm_o = {(ax, m): z[f"{m}{ax}"] for ax in (0, 1, 2) for m in METRICS}
            res = tests_from_metrics(sm_mc, sm_o, keep)
            for m in METRICS:
                res[m]["comparator"] = o
            out[f"{key}|{comp}"] = res
    return out


# ===================================================================== documents
def fmt(v, m):
    return f"{v:.3f}" if m == "head_psnr" else f"{v:.4f}" if m == "head_ssim" else f"{v:.5f}"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_table_md(cells, art):
    L = ["# 逐病例结果表（头部掩膜内 PSNR / SSIM / NMSE，全体积）", "",
         "每个（扫描，掩膜）一张表，T1 与 FLAIR 并排；`—` = 该方法的结果尚缺（见 table.json 的 missing）。",
         "格子 = （扫描，掩膜）；只有 T1 与 FLAIR 两种对比度的 11 个方法都齐全才算完整。完整格子里粗体 = 该列最佳；",
         "不完整的格子标注“不完整”，不加粗、不给名次、不计入名次与统计。NMSE 为头部 NMSE（越小越好）。", ""]
    for scan in SCANS:
        for af in AFS:
            cs = {c: cells[f"{scan}|{af}|{c}"] for c in CONTRASTS}
            full = all(c["complete"] for c in cs.values())
            tag = "" if full else "（不完整）"
            L += [f"## {scan}，{PATZH[af]}（{SCANS[scan]['group']}组）{tag}", "",
                  "| 方法 | T1 PSNR | T1 SSIM | T1 NMSE | FLAIR PSNR | FLAIR SSIM | FLAIR NMSE |", "|---|---|---|---|---|---|---|"]
            best = {}
            if full:
                for c in CONTRASTS:
                    for m in METRICS:
                        v = {l: cs[c]["methods"][l][m] for l in LABELS}
                        best[(c, m)] = (max if HIGHER[m] else min)(v, key=v.get)
            for l in LABELS:
                row = []
                for c in CONTRASTS:
                    for m in METRICS:
                        mm = cs[c]["methods"].get(l)
                        if mm is None:
                            row.append("—")
                        else:
                            s = fmt(mm[m], m)
                            row.append(f"**{s}**" if best.get((c, m)) == l else s)
                L.append(f"| {'**MC-GS**' if l == 'MC-GS' else l} | " + " | ".join(row) + " |")
            rk = [c + " " + "/".join(str(cs[c]["rank"][m]) if full else "-" for m in METRICS) for c in CONTRASTS]
            L += ["", "MC-GS 名次（PSNR/SSIM/NMSE）：" + "；".join(rk), ""]
    (art / "table.md").write_text("\n".join(L), encoding="utf-8")


def write_results_md(cells, chk, gst, sst, art, n_complete):
    nc = len(SCANS) * len(AFS)
    L = ["# 测试集 T1 + FLAIR 不同采样掩膜（AF16，均匀密度 / 伪径向）：结果", "",
         f"> 完整格子（扫描 x 掩膜；T1 与 FLAIR 两种对比度的 11 个方法都齐全）{n_complete} / {nc}。"
         f"不完整的格子对两种对比度都不计入名次与统计。全部跑完后由 "
         f"`{RESULTS_REL}/collect_stats.py --pipeline` 重新生成本文件、table.json、stats.json、table.md 与图。", "",
         "口径：头部掩膜内（|GT| > 0.05 P99.9）的 PSNR / SSIM / NMSE；评分代码是各课题共用的 `t1t2_common.Pair.score`，"
         "真值来自该扫描自己的全采样 k 空间，所有数字都由保存的体积重新计算，再与各运行脚本记录的分数对照（见第 6 节）。"
         "NMSE 的差值写成相对降低（%），其余为绝对差；大于 0 表示 MC-GS 更好。", ""]
    # 1 first counts
    L += ["## 1. MC-GS 在完整格子里是否第一", "", "| 对比度 | 掩膜 | 完整格子（共 6） | PSNR 第一 | SSIM 第一 | NMSE 第一 |", "|---|---|---|---|---|---|"]
    for c in CONTRASTS:
        for af in AFS:
            ok = [cells[f"{s}|{af}|{c}"] for s in SCANS if cells[f"{s}|{af}|{c}"]["complete"]]
            L.append(f"| {c} | {af} | {len(ok)} | " + " | ".join(f"{sum(x['first'][m] for x in ok)} / {len(ok)}" for m in METRICS) + " |")
    tot = [cells[k] for k in cells if cells[k]["complete"]]
    L += ["", f"合计：完整格子 {n_complete} / {nc}，即 {len(tot)} 列（T1 与 FLAIR 各一列）；MC-GS 第一的列数："
          + "，".join(f"{MNAME[m]} {sum(x['first'][m] for x in tot)} / {len(tot)}" for m in METRICS) + "。"
          + "分对比度：" + "；".join(f"{c} " + "，".join(f"{MNAME[m]} {sum(x['first'][m] for x in tot if x['contrast'] == c)} / {n_complete}" for m in METRICS)
                                  for c in CONTRASTS) + "。", ""]
    nf = [(k, m, v) for k, cl in cells.items() if cl["complete"] for m, v in cl.get("not_first", {}).items()]
    L += ["### MC-GS 不是第一的列（逐个列出，不省略）", ""]
    if nf:
        L += ["| 扫描 | 掩膜 | 对比度 | 指标 | MC-GS 名次 | 第一名 | 第一名数值 | MC-GS 数值 |", "|---|---|---|---|---|---|---|---|"]
        for k, m, v in nf:
            sc, af, c = k.split("|")
            L.append(f"| {sc} | {af} | {c} | {MNAME[m]} | {v['rank']} | {v['best']} | {fmt(v['best_value'], m)} | {fmt(v['mcgs_value'], m)} |")
    else:
        L.append("（目前的完整格子里没有。）")
    L += ["", "## 2. 病例间：MC-GS 相对两个对照的差值", "",
          "对照：GS-single；该格最强的非 GS 方法（按该格头部 PSNR 选，三个指标都与同一个方法比）。"
          "“符号一致”= 差值大于 0 的单位数 / 单位总数。p 为精确双侧 Wilcoxon 符号秩检验（零差值舍去，并列用平均秩；"
          "单位数 4 时 p 最小 0.125，6 时 0.031，见第 5 节）。", ""]
    names = {"all6": "{n} 次扫描", "aligned": "对齐组（vol1、vol2、vol4）{n} 次", "offset": "偏移组（vol5、vol6、vol7）{n} 次",
             "persons": "{n} 个人（重复扫描先平均）"}
    for m in METRICS:
        unit = {"head_psnr": "dB", "head_ssim": "", "head_nmse": "%（相对降低）"}[m]
        L += [f"### {MNAME[m]}差值 {unit}", "", "| 对比度 | 掩膜 | 对照 | 单位 | 均值 | 标准差 | 最小 | 符号一致 | p |", "|---|---|---|---|---|---|---|---|---|"]
        for c in CONTRASTS:
            for af in AFS:
                for comp in COMPS:
                    ent = gst[f"{c}|AF{af}|{m}|{comp}"]
                    for g in ("all6", "aligned", "offset", "persons"):
                        s = ent[g]
                        if not s:
                            continue
                        L.append(f"| {c} | {af} | {'GS-single' if comp == 'GS-single' else '最强非 GS'} | {names[g].format(n=s['n'])} | "
                                 f"{s['mean']:.4g} | {s['sd']:.3g} | {s['min']:.4g} | {s['n_positive']} / {s['n']} | {pfmt(s['wilcoxon_p'])} |")
        L.append("")
    # 3 slice tests
    L += ["## 3. 病例内：逐层面配对检验", "",
          "每个完整格子内，三个平面里头部占层面 >= 5 % 的层面合并（规则同 `t1flair_af_record_fig_20261120/fig_af_overview.py` 所读的 "
          "slice_trend 记录），逐层面配对 Wilcoxon（双侧）。这里的 p 值用正态近似（每个病例约 600 层），零差值被舍去；"
          "第 2 节的扫描 / 人层面检验才是精确检验。下表对每个（对比度，掩膜，对照，指标）把扫描分成三类：p < 0.05 且中位差为正（显著更好）、"
          "p < 0.05 且中位差为负（显著更差）、其余（不显著），并列出后两类的扫描；最后一列是各扫描里 MC-GS 更好的层面占比的范围。"
          "层面属于同一个体积，彼此不独立，p 值只说明该病例内的一致性，不是人群推断。逐病例的全部数值在 stats.json 的 `slices`。", "",
          "| 对比度 | 掩膜 | 对照 | 指标 | 显著更好 | 显著更差（扫描） | 不显著（扫描） | 更好层面占比范围 |", "|---|---|---|---|---|---|---|---|"]
    for c in CONTRASTS:
        for af in AFS:
            for comp in COMPS:
                for m in METRICS:
                    rs = {s: sst[f"{s}|{af}|{c}|{comp}"][m] for s in SCANS if f"{s}|{af}|{c}|{comp}" in sst}
                    if not rs:
                        continue
                    better = [s for s, r in rs.items() if r["wilcoxon_p"] < 0.05 and r["median"] > 0]
                    worse = [s for s, r in rs.items() if r["wilcoxon_p"] < 0.05 and r["median"] < 0]
                    ns = [s for s in rs if s not in better and s not in worse]
                    ws = [r["win_share"] for r in rs.values()]
                    L.append(f"| {c} | {af} | {'GS-single' if comp == 'GS-single' else '最强非 GS'} | {MNAME[m]} | {len(better)} / {len(rs)} | "
                             f"{len(worse)}（{'、'.join(worse) or '无'}） | {len(ns)}（{'、'.join(ns) or '无'}） | "
                             f"{100 * min(ws):.0f}% - {100 * max(ws):.0f}% |")
    # 4 missing
    L += ["", "## 4. 缺失清单", ""]
    by = {}
    for cl in cells.values():
        for l, r in cl["missing"].items():
            by.setdefault((cl["scan"], cl["af"], cl["contrast"]), []).append(l)
    inc = sum(1 for s in SCANS for af in AFS if not cells[f"{s}|{af}|T1"]["complete"])
    L.append(f"不完整格子 {inc} / {nc}。每个（扫描，掩膜，对比度）缺的方法（原因见 table.json 的 `missing`）：")
    L.append("")
    for (s, af, c), ls in sorted(by.items()):
        L.append(f"- {s} {PATZH[af]} {c}：" + "、".join(l for l in LABELS if l in ls))
    if chk.get("unreadable"):
        L += ["", "暂时读不出的文件（可能正在写入，已按缺失处理，没有缓存）：" + "；".join(chk["unreadable"])]
    # 5 limits
    L += ["", "## 5. 必须写明的限制", "",
          "- 只有 4 个人。以人为单位的 Wilcoxon 检验即使全胜，精确双侧 p 最小为 0.125；以 6 次扫描为单位最小为 0.031，"
          "但其中两次是重复扫描（vol2 与 vol6 是同一个人，vol4 与 vol7 是同一个人），不独立。结论写成“几个里几个一致”。",
          "- 层面级检验的层面彼此不独立（同一体积、三个平面共用数据），只说明病例内的一致性。",
          "- 一个种子。",
          "- ZS-SSDU 是 5 个残差块的版本（各课题登记的实现）。",
          "- GS-single 是登记版：同一对比度送入联合训练器的两个通道（登记的 singleT1 / singleFLAIR 配置），取第一个通道的结果。",
          "- 偏移组（vol5、vol6、vol7）按采集到的样子跑，没有做偏移校正；对齐组与偏移组分开列出。",
          "- 所有设置取自调参病例（jiangyizhou）各掩膜课题的登记值，没有在测试扫描上调参；真值只用于评分。",
          "- “伪径向”按 9 月的伪径向掩膜理解，尚未得到用户确认。",
          "- 第三种掩膜（变密度泊松 AF16）在加速倍数测试集研究里，本文件不含，待该研究定稿后合并。", ""]
    # 6 recompute check
    w = chk["worst"]
    L += ["## 6. 重算分数与运行脚本记录的分数", "",
          "每个体积（含不完整格子里的）都用同一段评分代码（均匀密度克隆的 `t1t2_common.Pair.score`，CPU）重算，再与运行脚本记录的值比较。"
          "任何一个 MC-GS 体积的头部 PSNR 相差超过 0.005 dB 都会使整次运行失败。最大绝对差：", "",
          "| 量 | 最大绝对差 | 出现在 |", "|---|---|---|"]
    for k, nm in (("head_psnr", "头部 PSNR（dB）"), ("head_ssim", "头部 SSIM"), ("nmse", "整体 NMSE")):
        L.append(f"| {nm} | {w[k]['abs']:.3g} | {w[k]['where']} |")
    L += ["", "各方法的最大 |头部 PSNR 差|（dB）：" + "，".join(f"{l} {chk['per_label'][l]['head_psnr']:.2g}" for l in LABELS if l in chk["per_label"]), ""]
    L += ["## 7. 图", "", f"汇总图：`{RESULTS_REL}/figures/testset_summary.png`（PNG / PDF / SVG 与数据源 CSV 都在 figures 文件夹，随结果一起提交）。"
          "每种掩膜一栏，每次扫描一个点，纵轴是头部 PSNR 差（MC-GS 减对照）；零线为界；圆点 = 对齐组，三角 = 偏移组；只画完整格子。", ""]
    (art / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


# ===================================================================== figure (matplotlib, Nature style)
def draw_figure(cells, art):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "legend.frameon": False})
    NPG = {"aligned": "#3C5488", "offset": "#E64B35"}
    MK = {"aligned": "o", "offset": "^"}
    mm = 1 / 25.4
    fig, axs = plt.subplots(2, 2, figsize=(183 * mm, 98 * mm), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.12, wspace=0.14, hspace=0.12)
    order = list(SCANS)
    rows = [("GS-single", "MC-GS minus GS-single\nhead PSNR (dB)"),
            ("strongest", "MC-GS minus strongest non-GS\nhead PSNR (dB)")]
    for i, (comp, ylab) in enumerate(rows):
        for j, c in enumerate(CONTRASTS):
            ax = axs[i, j]
            ax.axhline(0, color="#0b0b0b", lw=0.8, zorder=1)
            ys = []
            for a, af in enumerate(AFS):
                for k, s in enumerate(order):
                    cl = cells[f"{s}|{af}|{c}"]
                    if not cl["complete"]:
                        continue
                    y = cl["diffs"][comp]["head_psnr"]
                    ys.append(y)
                    g = SCANS[s]["group"]
                    ax.scatter(a + (k - 2.5) * 0.11, y, s=14, marker=MK[g], zorder=3, linewidths=0.6,
                               facecolors=NPG[g] if g == "aligned" else "none", edgecolors=NPG[g])
            if ys:
                lo, hi = min(ys + [0]), max(ys + [0])
                pad = 0.12 * (hi - lo or 1)
                ax.set_ylim(lo - pad, hi + pad)
            else:
                ax.set_ylim(-1, 1)
            ax.set_xticks(range(len(AFS)))
            ax.set_xticklabels([PAT[af] for af in AFS])
            ax.set_xlim(-0.5, len(AFS) - 0.5)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.text(0.015, 0.97, c, transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
            ax.text(-0.17 if j == 0 else -0.06, 1.04, "abcd"[2 * i + j], transform=ax.transAxes, fontsize=8,
                    fontweight="bold", ha="left", va="bottom")
    h = [Line2D([], [], marker=MK["aligned"], ls="", mfc=NPG["aligned"], mec=NPG["aligned"], ms=4, mew=0.6, label="aligned scans (vol1, vol2, vol4)"),
         Line2D([], [], marker=MK["offset"], ls="", mfc="none", mec=NPG["offset"], ms=4, mew=0.6, label="offset scans (vol5, vol6, vol7)")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 0.995), fontsize=6, handletextpad=0.3, columnspacing=1.5)
    fig.canvas.draw()
    # time rule: no generation date inside the files (PDF CreationDate, SVG dc:date; PNG carries no time chunk)
    fig.savefig(art / "testset_summary.png", dpi=600)
    fig.savefig(art / "testset_summary.pdf", metadata={"CreationDate": None})
    fig.savefig(art / "testset_summary.svg", metadata={"Date": None})
    src = ["scan,group,pattern,contrast,vs_GS-single_dB,vs_strongest_non_GS_dB,strongest_non_GS"]
    for k, cl in cells.items():
        if cl["complete"]:
            src.append(f"{cl['scan']},{SCANS[cl['scan']]['group']},{cl['af']},{cl['contrast']},"
                       f"{cl['diffs']['GS-single']['head_psnr']:.6f},{cl['diffs']['strongest']['head_psnr']:.6f},{cl['strongest_non_gs']}")
    (art / "testset_summary_source.csv").write_text("\n".join(src) + "\n")
    (art / "testset_summary_legend.txt").write_text(
        "Head PSNR gain of MC-GS over GS-single (a, b) and over the strongest non-GS method of the same cell (c, d), T1 (a, c) and "
        "FLAIR (b, d), for the uniform-density and pseudo-radial sampling patterns at AF 16. One marker per test scan (circles: aligned scans; triangles: offset scans), "
        "complete cells only. Zero line: no difference. One seed; scans vol2 / vol6 and vol4 / vol7 are the same persons.\n")


# ===================================================================== server job
def server(args):
    import multiprocessing as mp
    art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "collect_out")).resolve()
    art.mkdir(parents=True, exist_ok=True)
    cache = OUT / "stats_cache"
    cache.mkdir(exist_ok=True)
    av = availability()
    present = {k: v for k, v in av.items() if not isinstance(v, str)}
    log(f"present {len(present)} of {len(av)} (scan, AF, contrast, method) results")
    tasks = {}
    for (scan, af, c, label), a in present.items():
        tasks.setdefault(scan, []).append((af, c, label, a["file"], a["member"], a["logged"]))
    jobs = [(s, t, str(cache)) for s, t in tasks.items()]
    results = []
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(6, len(jobs))) as pool:
            for scan_res, (scan, _, _) in zip(pool.map(_worker, jobs, chunksize=1), jobs):
                for r in scan_res:
                    r["scan"] = scan
                results += scan_res
    cells, chk = assemble(av, results)
    n_complete = n_complete_cells(cells)
    gst = group_stats(cells)
    sst = slice_stats(cells)
    table = dict(n_cells=len(SCANS) * len(AFS), n_complete=n_complete, labels=LABELS, recompute_check=chk, cells={
        k: {kk: vv for kk, vv in c.items() if kk != "methods"} | {"methods": {l: {a: b for a, b in m.items() if a != "cache"}
                                                                               for l, m in c["methods"].items()}}
        for k, c in cells.items()})
    (art / "table.json").write_text(json.dumps(table, indent=1))
    (art / "stats.json").write_text(json.dumps(dict(n_complete=n_complete, group=gst, slices=sst), indent=1))
    write_table_md(cells, art)
    write_results_md(cells, chk, gst, sst, art, n_complete)
    draw_figure(cells, art)
    w = chk["worst"]["head_psnr"]["abs"]
    log(f"cells complete {n_complete} / {len(SCANS) * len(AFS)}; largest recomputed-vs-logged head PSNR difference {w:.3g} dB "
        f"({chk['worst']['head_psnr']['where']}); SSIM {chk['worst']['head_ssim']['abs']:.3g}; NMSE {chk['worst']['nmse']['abs']:.3g}")
    if chk["unreadable"]:
        log(f"unreadable (treated as missing, not cached): {len(chk['unreadable'])}: {chk['unreadable']}")
    bad = chk["mcgs_mismatch"]          # every recomputed MC-GS volume, complete cell or not
    log(f"MC-GS recompute-vs-logged > 0.005 dB: {bad or 'none'}")
    log("SERVER_DONE" if not bad else "SERVER_MISMATCH")
    sys.exit(0 if not bad else 2)


# ===================================================================== local pipeline
def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=HERE.parents[1], **kw)


def pipeline(resume=None):
    gpu = ["/usr/bin/python3", "../gpu-tools/gpu.py"]
    if resume:
        jid = resume
    else:
        r = sh([*gpu, "run", "--cpu", "--", "@python", f"{RESULTS_REL}/collect_stats.py", "--server"])
        m = re.search(r'"id":\s*"(j-[^"]+)"', r.stdout + r.stderr)
        assert m, "could not read the job id"
        jid = m.group(1)
    log(f"job ...{jobtail(jid)} started")
    import time
    keep = re.compile(r'^(SERVER_\w+|cells complete .*|MC-GS recompute-vs-logged.*|unreadable .*|\s*"state": "\w+",?|\s*"exit_code": \w+,?)\s*$')
    clock = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|T\d{6}")      # never echo anything that could carry a server time
    while True:
        time.sleep(30)
        out = sh([*gpu, "logs", jid]).stdout
        mine = [ln for ln in out.splitlines() if keep.match(ln) and not clock.search(ln)]
        ec = re.search(r'"exit_code":\s*(\d+)', out)
        if re.search(r'"state":\s*"(completed|failed)"', out) and ec:
            break
    for ln in mine:
        log(ln)
    if int(ec.group(1)) != 0 or "SERVER_DONE" not in out:     # SERVER_MISMATCH (exit 2) or a crash: store nothing
        raise SystemExit(f"job ...{jobtail(jid)} ended with exit_code {ec.group(1)}; nothing stored in the repo")
    f = sh([*gpu, "fetch", jid])
    hits = sorted((HERE.parents[1].parent / "results").glob(f"{jid}-*/artifacts"))
    assert hits, "fetch gave no artifacts directory"
    src = hits[-1]
    (HERE / "figures").mkdir(exist_ok=True)
    for n in ("table.json", "stats.json", "table.md", "RESULTS.md"):
        shutil.copy2(src / n, HERE / n)
    for p in src.glob("testset_summary*"):
        shutil.copy2(p, HERE / "figures" / p.name)
    log(f"stored {RESULTS_REL}/{{table.json, stats.json, table.md, RESULTS.md}} and figures/testset_summary.*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--server", action="store_true", help="the CPU job on amax")
    ap.add_argument("--pipeline", action="store_true", help="local: run the job, fetch, store")
    ap.add_argument("--resume_job", default=None, help="with --pipeline: wait for / fetch this existing job instead of starting one")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.server:
        server(a)
    elif a.pipeline:
        pipeline(a.resume_job)
    else:
        print(__doc__)
