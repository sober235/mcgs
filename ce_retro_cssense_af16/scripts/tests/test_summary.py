import json


def _row(c, m, hp, **kw):
    r = dict(contrast=c, method=m, psnr=30.0, ssim=0.9, nmse=0.01, hfen=0.3, head_psnr=hp, head_ssim=0.9, head_nmse=0.01,
             head_hfen=0.3, psnr_cplx=28.0, sharp=1.0, grain=0.01, grain_ratio=1.0, seconds=1.0)
    r.update(kw); return r


def test_summarize_picks_best_lambda_per_scan_and_orders_arms(tiny_topic):
    import summarize as S, arms as A
    top = tiny_topic["root"]; L = top / "logs"
    json.dump({"contrast": "pre", "rows": [_row("pre", "zf", 20.0), _row("pre", "cgsense", 25.0)]}, open(L / "classical_pre.json", "w"))
    for lam, hp in ((0.0005, 30.0), (0.001, 31.5), (0.002, 31.0)):
        json.dump({"contrast": "pre", "rows": [_row("pre", f"cssense_o200_lam{lam:g}", hp, lam=lam)]}, open(L / f"classical_pre_o200_lam{lam:g}.json", "w"))
    json.dump({"contrasts": ["pre", "post"], "lam": 0.001, "rows": [_row("pre", "jcssense_o200_lam0.001", 30.5, lam=0.001), _row("post", "jcssense_o200_lam0.001", 29.0, lam=0.001)]},
              open(L / "jcssense_o200_lam0.001.json", "w"))
    json.dump({"contrast": "pre", "rows": [_row("pre", "spirit_ks5_lam0.001", 29.5, lam=0.001, ksize=5)]}, open(L / "spirit_pre_ks5_lam0.001.json", "w"))
    json.dump({"contrast": "pre", "rows": [_row("pre", "gs_joint", 33.0)]}, open(L / "gs_joint_pre.json", "w"))
    rows = S.summarize()
    pre = {r["method"]: r for r in rows if r["contrast"] == "pre"}
    assert pre["cssense_o200"]["lam"] == 0.001 and pre["cssense_o200"]["n_lambda"] == 3
    assert pre["jcssense"]["head_psnr"] == 30.5 and pre["spirit"]["ksize"] == 5
    order = [r["method"] for r in rows if r["contrast"] == "pre"]
    assert order == [m for m in A.ORDER if m in order]
    assert (top / "logs/summary.json").exists() and (top / "logs/summary.tsv").exists()
    files = dict((k, p) for k, _, p in A.arm_files("pre"))
    assert files["cssense_o200"].name == "cssense_o200_lam0.001_pre_complex.npy"
    assert files["spirit"].name == "spirit_ks5_lam0.001_pre_complex.npy" and files["gs_joint"].name == "gs_joint_pre_complex.npy"
