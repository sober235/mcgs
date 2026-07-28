"""G2 体素域快跑(spec §8 item 3):共享信息到底有没有用的 gate 实验。

对同一例的 T1/FLAIR(12 线圈,256x220x256,AF=12,ACS=16):
  条件 A(same-mask)     :两对比度共用一个掩膜 -> 两者的未采样位置**完全重合**
  条件 B(complementary):互补掩膜,预算不变   -> 一方采到的正是另一方漏掉的
  方法  independent      :各自 `fista_tv`
  方法  joint            :`fista_joint_tv`(vectorial 复数 TV 跨对比度耦合)
主要终点 E_unobs(未采样区相对满采 k 的误差),次要终点脑内 PSNR/SSIM/NMSE。

预注册预测(spec §2.5):same-mask 下两个对比度的留出位置重合,共享模型借不到任何信息,
只能在"简约性"上占便宜,不可能在预测上占便宜,故 joint 相对 independent 增益 ≈ 0;
complementary 下应出现实质增益。**判决压在 complementary 这一臂上。**
本脚本原样输出测到的数,不做任何朝预测方向的调参。

---- 本脚本的五条硬约束(每一条都是前两次作废的跑法换来的) ----

1. **只调用 `crisp.recon.cs_sense` 的公开入口** `fista_tv` / `fista_joint_tv`。上一版脚本
   import 了私有符号并自带一份 FISTA 循环副本,静默地最小化了另一个泛函(round 4 的幅度
   TV),数值全部作废。这里不复制任何求解器代码。
2. **两个 arm 各自扫 lam,用各自最优比(best-against-best),绝不在同一 lam 下比。**
   vectorial TV 的归一化项是 sqrt(sum_m sum_a |g|^2 + eps^2),同一 lam 下每个通道实际
   受到的正则强度只有独立版的 1/sqrt(M)(见 `fista_joint_tv` docstring)。用两个**完全
   相同**的通道(跨对比度信息为零)实测,joint 在同一 lam 下仍然高 +0.2~+1.0 dB——
   在同一 lam 下比会**凭空制造出本实验要检验的那个效应**。
3. **lam 网格必须把最优点包在内部。** 64^3 体模实测最优在 3e-5~1e-4,旧网格 [1e-3,1e-1]
   偏离近两个数量级。本脚本先扫粗网格(半个数量级步长)定位,再在最优点附近以 sqrt(2)
   步长细化——细化到 sqrt(2) 是必须的:两臂最优点的理论间距就是 sqrt(2),粗网格上两臂
   会落在同一格,best-against-best 就退化回被禁止的同 lam 比较。
4. **迭代数 >= 1500。** 平台从约 1500 步开始;600 步以内 arm 间差值会在
   -1.61 -> +1.89 -> +0.45 -> +1.39 dB 之间来回摆,更早比较比的是暂态。
5. **两个掩膜条件的实际采样率必须对齐。** round 5 起正则强度 rho = lam * 采样率,
   而不同种子的实际采样率在 0.0823~0.0911 间抖动。本实验里 same 用 (m1,m1)、
   complementary 用 (m1,m2),m1 与 m_same 是同一个掩膜(同 seed 同参数),因此 T1 一侧
   逐位相同,只有 FLAIR 一侧换掩膜;两者采样率差 <1%(JSON 里逐条记录)。
6. **每格都要过零填充的 sanity gate**(两个终点分别处理,理由见 `gate()` 的 docstring:
   PSNR 告负立即中止;E_unobs 告负记标记并继续),并且要出图
   (GT | 零填充 | 重建 | 误差图),不能只出曲线——上一次每格都比零填充低 4~14 dB,
   直到有人手读 JSON 才发现。

用法(前台分块跑,每块一个 lam;条件 x arm 四路并行到四张空闲卡):
    python -m crisp.experiments.g2_voxel --mode run --cond same          --arms independent --lams 1e-4 --gpu 2
    python -m crisp.experiments.g2_voxel --mode run --cond complementary --arms independent --lams 1e-4 --gpu 3
    python -m crisp.experiments.g2_voxel --mode run --cond same          --arms joint       --lams 1e-4 --gpu 5
    python -m crisp.experiments.g2_voxel --mode run --cond complementary --arms joint       --lams 1e-4 --gpu 7
    python -m crisp.experiments.g2_voxel --mode finalize
每个 (cond, arm) 写自己的分片(`results_crisp/diagnostics/`),所以四路并行不会互相踩;
`--mode finalize` 把分片合并成 `results_crisp/g2_voxel_huangyuwei.json` 并出图。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crisp.io import load_trainer_mat, normalize_p999
from crisp.masks.vdpoisson import align_to_dc, complementary_pair, vd_poisson_2d
from crisp.metrics.coil_domain import e_unobs, ifft3c
from crisp.metrics.image_domain import brain_gauge
from crisp.recon.cs_sense import fista_joint_tv, fista_tv

ROOT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
OUT = ROOT / "results_crisp"
SHARD_DIR = OUT / "diagnostics"
BASE = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
        "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_%s_trainer.mat")
CONTRASTS = ("T1", "FLAIR")
CONDS = ("same", "complementary")
ARMS = ("independent", "joint")
SLICE_AXIS, SLICE_IDX = 2, 128     # 读出轴切片 => 显示的正是欠采样发生的相位编码平面


def _lk(lam: float) -> str:
    return f"{lam:g}"


def _shard(cond: str, arm: str, lam: float, iters: int) -> Path:
    """分片粒度 = (条件, arm, lam, iters),这样任意多个格子可以并行到多张卡而不互相踩文件。

    `iters` 必须进文件名:第一版漏了它,于是 3000 步的收敛核验把同 lam 的 1500 步分片
    **就地覆盖**掉了,而覆盖的恰好是两臂各自的最优格,主扫网格因此少了 4 个点。
    """
    return SHARD_DIR / f"g2_cells_{cond}_{arm}_it{iters}_lam{_lk(lam)}.json"


def _slices_path(cond: str, arm: str, lam: float, iters: int) -> Path:
    return SHARD_DIR / f"g2_slices_{cond}_{arm}_it{iters}_lam{_lk(lam)}.npz"


def _stack(mask2d: np.ndarray, shape) -> torch.Tensor:
    """2D 相位编码平面掩膜 -> 3D(读出轴 = 最后一轴,全采)。"""
    return torch.from_numpy(np.repeat(mask2d[:, :, None], shape[-1], axis=2).astype(np.float32))


def _pe_slice(vol_mag: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vol_mag[:, :, SLICE_IDX])


def build_masks(pe_shape, ksp_np, af, acs, seed):
    """same 用 (m1, m1),complementary 用 (m1, m2);m_same 与 m1 同参数同种子,故逐位相同。"""
    m_same = vd_poisson_2d(pe_shape, af, acs, seed)
    m1, m2, overlap = complementary_pair(pe_shape, af, acs, seed)
    m_same, shift = align_to_dc(m_same, ksp_np)
    m1, _ = align_to_dc(m1, ksp_np)
    m2, _ = align_to_dc(m2, ksp_np)
    info = {
        "dc_shift": list(shift),
        "complementary_overlap": overlap,
        "fraction_m_same": float(m_same.mean()),
        "fraction_m1": float(m1.mean()),
        "fraction_m2": float(m2.mean()),
        "fraction_rel_diff_m2_vs_m1": float(abs(m2.mean() - m1.mean()) / m1.mean()),
        "m_same_identical_to_m1": bool(np.array_equal(m_same, m1)),
        "condition_masks": {"same": ["m_same", "m_same"], "complementary": ["m1", "m2"]},
    }
    return {"same": (m_same, m_same), "complementary": (m1, m2)}, info


def load_data(dev):
    data = {}
    for c in CONTRASTS:
        ksp, csm = load_trainer_mat(BASE % c)
        ksp, csm = ksp.to(dev), csm.to(dev)
        ksp, scale = normalize_p999(ksp, csm)
        gt = (csm.conj() * ifft3c(ksp)).sum(0)
        gt_mag = gt.abs().cpu().numpy()
        data[c] = dict(ksp=ksp, csm=csm, gt_mag=gt_mag, p999_scale=float(scale))
        del gt
    return data


def _score(x, contrast, data, mask):
    d = data[contrast]
    rec_mag = x.abs().cpu().numpy()
    m = dict(e_unobs=e_unobs(x, d["csm"], d["ksp"], mask))
    m.update(brain_gauge(rec_mag, d["gt_mag"]))
    return m, _pe_slice(rec_mag)


def zero_fill(data, masks3d):
    """零填充参考 = 伴随算子 A^H y(SENSE 合成的零填充图),每个条件/对比度各一份。"""
    out, sl = {}, {}
    for c in CONTRASTS:
        d, mask = data[c], masks3d[c]
        y = d["ksp"] * mask.unsqueeze(0)
        x = (ifft3c(mask.unsqueeze(0) * y) * d["csm"].conj()).sum(0)
        out[c], sl[c] = _score(x, c, data, mask)
        del x, y
    torch.cuda.empty_cache()
    return out, sl


def gate(cell, zf, cond, lam, contrast, arm):
    """每格都必须打得过零填充,否则停下来报告,不许把整条网格扫完再说。

    **两个终点分别处理,理由记录在此**(否则后来的人会以为是偷偷放水):
    * 脑内 PSNR 低于零填充 => **立刻中止**。这正是上一次事故的失效模式(每格都比零填充低
      4~14 dB,直到有人手读 JSON 才发现),图像域塌掉说明配置/求解器有问题,继续扫没有意义。
    * E_unobs 高于零填充 => **记 `warn_e_unobs_below_zero_fill` 标记、大声打印、继续扫**。
      零填充解 x0 = A^H y 在未采样区并不是零,它的 E_unobs 只比"输出全零"(定义上恰好 1.0)
      好一点点(实测 0.85 附近);而 TV 重建往未采样区填内容时,填错的部分会让这个数先涨后落。
      实测 30 步暂态里 FLAIR 就出现过 E_unobs 0.909 > 零填充 0.852 而脑内 PSNR 高 5.3 dB 的
      组合。若因此中止,整条 lam 曲线根本量不出来,而"主要终点在某些 lam 上打不过零填充"
      本身就是要报告的结果,不是跑崩了。两个终点同时告负才是真的崩,那时 PSNR 分支会中止。
    """
    cell["beats_zero_fill_e_unobs"] = bool(cell["e_unobs"] < zf["e_unobs"])
    cell["beats_zero_fill_psnr"] = bool(cell["psnr_brain"] > zf["psnr_brain"])
    cell["psnr_margin_vs_zero_fill_db"] = cell["psnr_brain"] - zf["psnr_brain"]
    cell["e_unobs_margin_vs_zero_fill"] = zf["e_unobs"] - cell["e_unobs"]
    if not cell["beats_zero_fill_e_unobs"]:
        print(f"  WARNING cond={cond} lam={lam:g} {contrast} {arm}: E_unobs "
              f"{cell['e_unobs']:.4f} > zero-fill {zf['e_unobs']:.4f} "
              f"(brain PSNR margin {cell['psnr_margin_vs_zero_fill_db']:+.2f} dB)", flush=True)
    if not cell["beats_zero_fill_psnr"]:
        print(f"\nSANITY GATE FAILED  cond={cond} lam={lam:g} {contrast} {arm}: brain PSNR "
              f"{cell['psnr_brain']:.2f} < zero-fill {zf['psnr_brain']:.2f}\n"
              "stopping the sweep (spec constraint 7).", flush=True)
        sys.exit(2)


def run(a):
    dev = f"cuda:{a.gpu}"
    # 必须显式 set_device:`torch.cuda.empty_cache()` 作用在**当前**设备上,而默认当前设备是
    # cuda:0——这台机器上 0 号卡属于别人且已占满,于是 empty_cache 会在 0 号卡上建 context
    # 并抛 "CUDA error: out of memory",实测已经吃掉两次跑。
    torch.cuda.set_device(a.gpu)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_data(dev)
    vol_shape = tuple(data["T1"]["ksp"].shape[1:])
    pe_shape = vol_shape[:2]
    ksp_np = data["T1"]["ksp"].cpu().numpy()
    cond_masks, mask_info = build_masks(pe_shape, ksp_np, a.af, a.acs, a.seed)
    del ksp_np
    print(f"[{a.cond}/{'+'.join(a.arms)}] loaded in {time.time()-t0:.1f}s; "
          f"vol={vol_shape} {mask_info['fraction_m1']:.6f}/{mask_info['fraction_m2']:.6f}", flush=True)

    ma, mb = cond_masks[a.cond]
    masks3d = {"T1": _stack(ma, vol_shape).to(dev), "FLAIR": _stack(mb, vol_shape).to(dev)}
    ys = {c: data[c]["ksp"] * masks3d[c].unsqueeze(0) for c in CONTRASTS}
    zf, zf_sl = zero_fill(data, masks3d)
    print(f"[{a.cond}] zero-fill: "
          + "  ".join(f"{c}: E={zf[c]['e_unobs']:.4f} PSNR={zf[c]['psnr_brain']:.2f}" for c in CONTRASTS),
          flush=True)

    for arm in a.arms:
        for lam in a.lams:
            path, spath = _shard(a.cond, arm, lam, a.iters), _slices_path(a.cond, arm, lam, a.iters)
            rec = {"meta": dict(
                af=a.af, acs=a.acs, seed=a.seed, iters=a.iters, cond=a.cond, arm=arm, lam=lam,
                vol_shape=list(vol_shape), masks=mask_info,
                p999_scale={c: data[c]["p999_scale"] for c in CONTRASTS},
                slice_axis=SLICE_AXIS, slice_idx=SLICE_IDX), "zero_fill": zf, "cells": {}}
            store = {}
            for c in CONTRASTS:
                store[f"gt|{c}"] = _pe_slice(data[c]["gt_mag"])
                store[f"zf|{c}"] = zf_sl[c]
            t = time.time()
            if arm == "independent":
                xs = [fista_tv(ys[c], data[c]["csm"], masks3d[c], lam, a.iters) for c in CONTRASTS]
            else:
                xs = fista_joint_tv([ys[c] for c in CONTRASTS], [data[c]["csm"] for c in CONTRASTS],
                                    [masks3d[c] for c in CONTRASTS], lam, a.iters)
            dt = time.time() - t
            cell = {}
            for x, c in zip(xs, CONTRASTS):
                m, s = _score(x, c, data, masks3d[c])
                m["seconds"] = dt
                cell[c] = m
                store[f"rec|{c}|{_lk(lam)}"] = s
            del xs
            torch.cuda.empty_cache()
            rec["cells"][_lk(lam)] = cell
            path.write_text(json.dumps(rec, indent=2))
            np.savez_compressed(spath, **store)
            print(f"[{a.cond}/{arm}] lam={lam:g} ({dt:.0f}s) " + "  ".join(
                f"{c}: E={cell[c]['e_unobs']:.4f} PSNR={cell[c]['psnr_brain']:.2f} "
                f"(zf {zf[c]['psnr_brain']:.2f}) SSIM={cell[c]['ssim_brain']:.4f}" for c in CONTRASTS),
                flush=True)
            for c in CONTRASTS:
                gate(cell[c], zf[c], a.cond, lam, c, arm)


# ----------------------------------------------------------------------------- finalize


def _mean_over_contrasts(cells, lam, key):
    return float(np.mean([cells[lam][c][key] for c in CONTRASTS]))


def finalize(a):
    merged = {"cells": {}, "zero_fill": {}, "meta": {}}
    for cond in CONDS:
        merged["cells"][cond] = {}
        for arm in ARMS:
            cells = {}
            for p in sorted(SHARD_DIR.glob(f"g2_cells_{cond}_{arm}_it{a.iters}_lam*.json")):
                r = json.loads(p.read_text())
                if r["meta"]["iters"] != a.iters:
                    print(f"skipping {p.name}: iters={r['meta']['iters']} != {a.iters}", flush=True)
                    continue
                cells.update(r["cells"])
                merged["meta"].setdefault(arm, {})[cond] = {
                    k: v for k, v in r["meta"].items() if k != "lam"}
                merged["zero_fill"][cond] = r["zero_fill"]
            if cells:
                merged["cells"][cond][arm] = cells
        if not merged["cells"][cond]:
            del merged["cells"][cond]

    analysis = {"sign_convention":
                "joint_minus_independent: E_unobs delta<0 = joint better (lower error); "
                "PSNR delta>0 = joint better (higher PSNR). Each arm evaluated at its OWN best lambda.",
                "selection_rule": "best lambda per (condition, arm) = argmin of E_unobs averaged over T1 and FLAIR",
                "conditions": {}}
    worst_psnr = {"margin_db": float("inf"), "cell": None}
    worst_e = {"margin": float("inf"), "cell": None}
    failed_e = []
    for cond in CONDS:
        if cond not in merged["cells"] or not merged["cells"][cond]:
            continue
        zf = merged["zero_fill"][cond]
        best, curves = {}, {}
        for arm, cells in merged["cells"][cond].items():
            lams = sorted(cells, key=float)
            means = {L: _mean_over_contrasts(cells, L, "e_unobs") for L in lams}
            b = min(lams, key=lambda L: means[L])
            best[arm] = {
                "lam": float(b),
                "interior_optimum": bool(len(lams) > 2 and b not in (lams[0], lams[-1])),
                "mean_e_unobs": means[b],
                "per_contrast": {c: cells[b][c] for c in CONTRASTS},
                "per_contrast_best_lam": {
                    c: float(min(lams, key=lambda L: cells[L][c]["e_unobs"])) for c in CONTRASTS},
            }
            curves[arm] = {"lams": [float(L) for L in lams],
                           "mean_e_unobs": [means[L] for L in lams],
                           "mean_psnr": [_mean_over_contrasts(cells, L, "psnr_brain") for L in lams]}
            for L in lams:
                for c in CONTRASTS:
                    tag = f"{cond}|{arm}|lam={L}|{c}"
                    dp = cells[L][c]["psnr_brain"] - zf[c]["psnr_brain"]
                    de = zf[c]["e_unobs"] - cells[L][c]["e_unobs"]
                    if dp < worst_psnr["margin_db"]:
                        worst_psnr = {"margin_db": dp, "cell": tag}
                    if de < worst_e["margin"]:
                        worst_e = {"margin": de, "cell": tag}
                    if de <= 0:
                        failed_e.append(tag)
        entry = {"best_per_arm": best, "curves": curves, "zero_fill": zf}
        if set(best) == set(ARMS):
            bi, bj = best["independent"], best["joint"]
            entry["best_vs_best"] = {
                "lam_independent": bi["lam"], "lam_joint": bj["lam"],
                "lam_ratio_joint_over_independent": bj["lam"] / bi["lam"],
                "delta_mean_e_unobs": bj["mean_e_unobs"] - bi["mean_e_unobs"],
                "delta_relative_e_unobs": (bj["mean_e_unobs"] - bi["mean_e_unobs"]) / bi["mean_e_unobs"],
                "per_contrast": {
                    c: {"delta_e_unobs": bj["per_contrast"][c]["e_unobs"] - bi["per_contrast"][c]["e_unobs"],
                        "delta_psnr_db": bj["per_contrast"][c]["psnr_brain"] - bi["per_contrast"][c]["psnr_brain"],
                        "delta_ssim": bj["per_contrast"][c]["ssim_brain"] - bi["per_contrast"][c]["ssim_brain"],
                        "delta_nmse": bj["per_contrast"][c]["nmse_brain"] - bi["per_contrast"][c]["nmse_brain"]}
                    for c in CONTRASTS},
            }
            entry["best_vs_best"]["delta_mean_psnr_db"] = float(np.mean(
                [entry["best_vs_best"]["per_contrast"][c]["delta_psnr_db"] for c in CONTRASTS]))
            # 交叉核对:1/sqrt(M) 说预测 joint@lam ~ independent@lam/sqrt(2)
            ci, cj = merged["cells"][cond]["independent"], merged["cells"][cond]["joint"]
            xchk = {}
            for L in sorted(cj, key=float):
                tgt = float(L) / (2 ** 0.5)
                near = min(ci, key=lambda K: abs(np.log(float(K) / tgt)))
                if abs(np.log(float(near) / tgt)) < 0.06:      # <6% 相对偏差才算命中
                    xchk[L] = {"independent_lam": float(near),
                               "delta_mean_e_unobs": _mean_over_contrasts(cj, L, "e_unobs")
                                                     - _mean_over_contrasts(ci, near, "e_unobs"),
                               "delta_mean_psnr_db": _mean_over_contrasts(cj, L, "psnr_brain")
                                                     - _mean_over_contrasts(ci, near, "psnr_brain")}
            entry["cross_check_joint_lam_vs_independent_lam_over_sqrt2"] = xchk
        analysis["conditions"][cond] = entry

    analysis["sanity_gate"] = {
        "rule": ("every cell is checked against zero-fill on both endpoints; a brain-PSNR failure "
                 "aborts the sweep, an E_unobs failure is flagged and the sweep continues "
                 "(see gate() docstring for why)"),
        "all_cells_beat_zero_fill_psnr": bool(worst_psnr["margin_db"] > 0),
        "all_cells_beat_zero_fill_e_unobs": bool(worst_e["margin"] > 0),
        "worst_psnr_margin_db": worst_psnr["margin_db"], "worst_psnr_cell": worst_psnr["cell"],
        "worst_e_unobs_margin": worst_e["margin"], "worst_e_unobs_cell": worst_e["cell"],
        "cells_not_beating_zero_fill_on_e_unobs": failed_e,
    }
    if {"same", "complementary"} <= set(analysis["conditions"]):
        bs = analysis["conditions"]["same"].get("best_vs_best")
        bc = analysis["conditions"]["complementary"].get("best_vs_best")
        if bs and bc:
            analysis["registered_prediction"] = {
                "statement": ("same-mask: joint gain over independent ~ 0 (held-out k-space "
                              "locations coincide, nothing to borrow); complementary: real gain. "
                              "The verdict rests on the complementary arm."),
                "same_delta_mean_e_unobs": bs["delta_mean_e_unobs"],
                "same_delta_mean_psnr_db": bs["delta_mean_psnr_db"],
                "complementary_delta_mean_e_unobs": bc["delta_mean_e_unobs"],
                "complementary_delta_mean_psnr_db": bc["delta_mean_psnr_db"],
                "complementary_minus_same_psnr_db": bc["delta_mean_psnr_db"] - bs["delta_mean_psnr_db"],
            }
    merged["analysis"] = analysis
    out_json = OUT / f"g2_voxel_huangyuwei{a.label}.json"
    out_json.write_text(json.dumps(merged, indent=2))

    panel = _panel(merged, analysis, a.label, a.iters)
    curves = _curves(merged, analysis, a.label)
    print(json.dumps(analysis, indent=2))
    print(f"\nwrote {out_json}\n      {panel}\n      {curves}", flush=True)


def _panel(merged, analysis, label="", iters=1500):
    """GT | 零填充 | independent@best | joint@best | 三张误差图(jet,固定 [0,0.1]*GT峰)。"""
    rows = [(cond, c) for cond in CONDS if cond in analysis["conditions"] for c in CONTRASTS]
    fig, axes = plt.subplots(len(rows), 7, figsize=(20.5, 3.2 * len(rows)), squeeze=False)
    for i, (cond, c) in enumerate(rows):
        best = analysis["conditions"][cond]["best_per_arm"]
        peak = merged["zero_fill"][cond][c]["peak_used"]
        sl = {}
        for arm in ARMS:
            z = dict(np.load(_slices_path(cond, arm, best[arm]["lam"], iters)))
            sl["gt"], sl["zf"] = z[f"gt|{c}"], z[f"zf|{c}"]
            sl[arm] = z[f"rec|{c}|{_lk(best[arm]['lam'])}"]
        gt = sl["gt"] / peak
        zfm = merged["zero_fill"][cond][c]
        cells = {arm: merged["cells"][cond][arm][_lk(best[arm]["lam"])][c] for arm in ARMS}
        imgs = [
            (gt, "gray", (0, 1.0), f"{cond} / {c}\nGT (peak={peak:.3f})"),
            (sl["zf"] / peak, "gray", (0, 1.0),
             f"zero-fill\nPSNR {zfm['psnr_brain']:.2f}  E {zfm['e_unobs']:.3f}"),
            (sl["independent"] / peak, "gray", (0, 1.0),
             f"independent  lam={best['independent']['lam']:g}\n"
             f"PSNR {cells['independent']['psnr_brain']:.2f}  E {cells['independent']['e_unobs']:.3f}"),
            (sl["joint"] / peak, "gray", (0, 1.0),
             f"joint  lam={best['joint']['lam']:g}\n"
             f"PSNR {cells['joint']['psnr_brain']:.2f}  E {cells['joint']['e_unobs']:.3f}"),
            (np.abs(sl["zf"] / peak - gt), "jet", (0, 0.1), "|zero-fill - GT|"),
            (np.abs(sl["independent"] / peak - gt), "jet", (0, 0.1), "|independent - GT|"),
            (np.abs(sl["joint"] / peak - gt), "jet", (0, 0.1), "|joint - GT|"),
        ]
        for j, (arr, cmap, (lo, hi), title) in enumerate(imgs):
            ax = axes[i][j]
            im = ax.imshow(arr.T, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if cmap == "jet":
                fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle("G2 voxel-domain gate: readout slice %d (phase-encode plane), "
                 "error maps jet, fixed [0, 0.1] of GT brain-P99.9 peak" % SLICE_IDX, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    p = OUT / f"g2_voxel_huangyuwei{label}_panel.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def _curves(merged, analysis, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, ylab in zip(axes, ("mean_e_unobs", "mean_psnr"),
                             ("E_unobs (mean over T1/FLAIR)", "brain PSNR dB (mean over T1/FLAIR)")):
        for cond, ls in zip(CONDS, ("--", "-")):
            if cond not in analysis["conditions"]:
                continue
            cur = analysis["conditions"][cond]["curves"]
            for arm, col in (("independent", "tab:blue"), ("joint", "tab:red")):
                if arm not in cur:
                    continue
                ax.plot(cur[arm]["lams"], cur[arm][key], ls, marker="o", color=col,
                        label=f"{cond} / {arm}")
                b = analysis["conditions"][cond]["best_per_arm"][arm]["lam"]
                ax.axvline(b, color=col, ls=":", lw=0.8, alpha=0.5)
        zf = merged["zero_fill"]["same"]
        ax.axhline(np.mean([zf[c]["e_unobs" if key == "mean_e_unobs" else "psnr_brain"]
                            for c in CONTRASTS]), color="k", lw=1, label="zero-fill")
        ax.set_xscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    p = OUT / f"g2_voxel_huangyuwei{label}_curves.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "finalize"), default="run")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--cond", choices=CONDS, default="same")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--af", type=float, default=12.0)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[5e-3, 7.07e-3, 1e-2, 1.41e-2, 2e-2, 2.83e-2, 4e-2, 5.66e-2, 8e-2, 1.13e-1])
    ap.add_argument("--label", default="",
                    help="finalize 用:输出文件名后缀,用来把收敛核验(--iters 3000)与主扫分开存")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.mode == "run":
        run(a)
    else:
        finalize(a)


if __name__ == "__main__":
    main()
