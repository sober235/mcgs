"""G2 体素域快跑(spec §8 item 3)。

对同一例的 T1/FLAIR:
  条件 A(same-mask)     :两对比度共用一个掩膜
  条件 B(complementary):互补掩膜,预算不变
  方法  independent      :各自 fista_tv
  方法  joint            :fista_joint_tv(vectorial TV 耦合)
主要终点 E_unobs(相对满采 k),次要终点脑内 PSNR/SSIM/NMSE。
lam 在给定网格上扫描,若最优(按 independent 的 E_unobs 选)落在网格边界,
则需扩大 --lams 并重跑,直到最优落在内部(spec §5 G1 规则)。

预注册预测(spec §2.5):same-mask 下 joint 相对 independent 的 E_unobs 增益 ≈ 0;
complementary 下应出现实质增益。本脚本输出直接对照该预测,写入 JSON 的 verdict 字段,
无论结果是否符合预测都原样记录。

第二轮修正(协调者纠偏,详见 task-7-report.md):第一轮用 e_unobs 随迭代上升误判为
求解器发散——用采样区残差(FISTA 的下降保证量)核验后,残差单调下降、FISTA 优于无
动量 GD,求解器本身没问题;e_unobs 上升是 lam 太小时欠定问题的正常表现(未采样零空间
能量增长,不受约束,不是数值不稳定)。修正:lam 网格上移到 [1e-3,1e-1] 量级、iters=300,
且每格(每次 independent/joint 重建)在若干 checkpoint 处监控 FISTA 目标函数(数据项+
lam*TV,而非 e_unobs)是否不增;若不满足,该格 step 减半重跑,并把实际使用的 step 记入
JSON——不静默换 step。输出 JSON 支持增量合并(按 --lams 传入的子集追加/覆盖对应格),
使得可以分批在前台运行,不因单次调用耗时过长或后台任务丢失而丢失已完成的格。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crisp.io import load_trainer_mat, normalize_p999
from crisp.masks.vdpoisson import align_to_dc, complementary_pair, vd_poisson_2d
from crisp.metrics.coil_domain import e_unobs, fft3c, ifft3c
from crisp.metrics.image_domain import brain_gauge
from crisp.recon.cs_sense import _EPS, _check_csm_scale, _data_grad, _grads, _tv_grad

OUT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_crisp")
BASE = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
        "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_%s_trainer.mat")


def _stack(mask2d, shape):
    """2D 相位编码平面掩膜 -> 3D(读出轴 = 最后一轴,全采)。"""
    return torch.from_numpy(np.repeat(mask2d[:, :, None], shape[-1], axis=2).astype(np.float32))


def _objective_indep(x, y, csm, mask, lam):
    resid = mask.unsqueeze(0) * (fft3c(csm * x.unsqueeze(0)) - y)
    data_term = 0.5 * float(torch.sum(resid.abs() ** 2))
    gx, gy, gz = _grads(x.abs().clamp_min(_EPS))
    tv = float(torch.sum(torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + _EPS ** 2)))
    return data_term + lam * tv


def _objective_joint(xs, ys, csms, masks, lam):
    data_term = 0.0
    accg = None
    for x, y, c, m in zip(xs, ys, csms, masks):
        resid = m.unsqueeze(0) * (fft3c(c * x.unsqueeze(0)) - y)
        data_term += 0.5 * float(torch.sum(resid.abs() ** 2))
        gx, gy, gz = _grads(x.abs().clamp_min(_EPS))
        s = gx ** 2 + gy ** 2 + gz ** 2
        accg = s if accg is None else accg + s
    tv = float(torch.sum(torch.sqrt(accg + _EPS ** 2)))
    return data_term + lam * tv


def _checkpoints(iters):
    return sorted(set(max(1, iters * k // 4) for k in (1, 2, 3, 4)))


def fista_tv_monitored(y, csm, mask, lam, iters, step0=0.25, max_halvings=6, tol=1.01):
    """fista_tv 的等价实现(复用 cs_sense.py 的私有内层函数,不改该文件),外加每
    checkpoint 的目标函数非增监控;若某次尝试目标函数回升,则 step 减半重跑整格,
    直到稳定或达到 max_halvings。返回 (x, diag),diag 记录实际使用的 step。
    """
    _check_csm_scale(csm)
    cps = _checkpoints(iters)
    step = step0
    for attempt in range(max_halvings + 1):
        x = (ifft3c(mask.unsqueeze(0) * y) * csm.conj()).sum(0)
        z, t = x.clone(), 1.0
        trace, prev, stable, t_prev = [], None, True, 0
        for cp in cps:
            for _ in range(cp - t_prev):
                g = _data_grad(z, y, csm, mask) + lam * _tv_grad(z)
                x_new = z - step * g
                t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
                z = x_new + ((t - 1) / t_new) * (x_new - x)
                x, t = x_new, t_new
            o = _objective_indep(x, y, csm, mask, lam)
            trace.append([cp, o])
            if prev is not None and o > prev * tol:
                stable = False
                break
            prev, t_prev = o, cp
        if stable or attempt == max_halvings:
            return x, dict(step_used=step, halvings=attempt, stable=stable, objective_trace=trace)
        step = step / 2


def fista_joint_tv_monitored(ys, csms, masks, lam, iters, step0=0.25, max_halvings=6, tol=1.01):
    """fista_joint_tv 的等价实现 + 每 checkpoint 目标函数非增监控 + step 减半重跑。"""
    for c in csms:
        _check_csm_scale(c)
    cps = _checkpoints(iters)
    step = step0
    for attempt in range(max_halvings + 1):
        xs = [(ifft3c(m.unsqueeze(0) * y) * c.conj()).sum(0) for y, c, m in zip(ys, csms, masks)]
        zs, t = [v.clone() for v in xs], 1.0
        trace, prev, stable, t_prev = [], None, True, 0
        for cp in cps:
            for _ in range(cp - t_prev):
                acc = None
                for z in zs:
                    gx, gy, gz = _grads(z.abs().clamp_min(_EPS))
                    s = gx ** 2 + gy ** 2 + gz ** 2
                    acc = s if acc is None else acc + s
                n = torch.sqrt(acc + _EPS ** 2)
                new_xs = []
                for z, y, c, m in zip(zs, ys, csms, masks):
                    g = _data_grad(z, y, c, m) + lam * _tv_grad(z, coupled_norm=n)
                    new_xs.append(z - step * g)
                t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
                zs = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(new_xs, xs)]
                xs, t = new_xs, t_new
            o = _objective_joint(xs, ys, csms, masks, lam)
            trace.append([cp, o])
            if prev is not None and o > prev * tol:
                stable = False
                break
            prev, t_prev = o, cp
        if stable or attempt == max_halvings:
            return xs, dict(step_used=step, halvings=attempt, stable=stable, objective_trace=trace)
        step = step / 2


def _load_or_init(out_path, a):
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if all(prev.get(k) == getattr(a, k) for k in ("af", "acs", "seed", "iters")):
                print("merging into existing compatible results file", flush=True)
                return prev
            print("existing results file has incompatible settings (af/acs/seed/iters) "
                 "-- starting fresh", flush=True)
        except Exception as e:
            print(f"existing results file unreadable ({e}) -- starting fresh", flush=True)
    return None


def _finalize(results, conds, out_path, overlap):
    lams_all = results["lams"]
    best_lam, verdict = {}, {}
    for cond in conds:
        avail = [L for L in lams_all if f"{cond}|lam={L:g}" in results["cells"]]
        if not avail:
            continue
        best_lam[cond] = min(avail, key=lambda L: np.mean(
            [results["cells"][f"{cond}|lam={L:g}"][k]["e_unobs"] for k in
             ("T1_independent", "FLAIR_independent")]))
        results.setdefault("boundary_optimum", {})[cond] = bool(best_lam[cond] in (avail[0], avail[-1]))

        gain_by_lam = {}
        for lam in avail:
            cell = results["cells"][f"{cond}|lam={lam:g}"]
            gain_by_lam[f"{lam:g}"] = float(np.mean(
                [cell[f"{c}_independent"]["e_unobs"] - cell[f"{c}_joint"]["e_unobs"] for c in ("T1", "FLAIR")]))
        cell_best = results["cells"][f"{cond}|lam={best_lam[cond]:g}"]
        gain_at_best = {c: cell_best[f"{c}_independent"]["e_unobs"] - cell_best[f"{c}_joint"]["e_unobs"]
                        for c in ("T1", "FLAIR")}
        mean_gain = float(np.mean(list(gain_at_best.values())))
        mean_indep = float(np.mean([cell_best[f"{c}_independent"]["e_unobs"] for c in ("T1", "FLAIR")]))
        signs = [g > 0 for g in gain_by_lam.values()]
        verdict[cond] = dict(
            best_lam=best_lam[cond],
            gain_e_unobs_at_best_lam=gain_at_best,
            mean_gain_e_unobs=mean_gain,
            relative_gain=mean_gain / mean_indep if mean_indep else float("nan"),
            gain_by_lam=gain_by_lam,
            sign_consistent_across_grid=bool(all(signs) or not any(signs)),
        )
    results["verdict"] = verdict
    out_path.write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cond in zip(axes, conds):
        avail = [L for L in lams_all if f"{cond}|lam={L:g}" in results["cells"]]
        for name, style in (("independent", "o--"), ("joint", "s-")):
            for c, color in (("T1", "tab:blue"), ("FLAIR", "tab:orange")):
                yv = [results["cells"][f"{cond}|lam={lam:g}"][f"{c}_{name}"]["e_unobs"] for lam in avail]
                ax.plot(avail, yv, style, color=color, label=f"{c} {name}")
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_title(f"{cond} (overlap={overlap:.3f})" if cond == "complementary" else cond)
        if cond in best_lam:
            ax.axvline(best_lam[cond], color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("E_unobs")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "g2_voxel_huangyuwei.png", dpi=110, bbox_inches="tight")

    print(json.dumps({k: results[k] for k in
                      ("dc_shift", "complementary_overlap", "sampling_fraction", "boundary_optimum", "verdict")},
                     indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--af", type=float, default=12.0)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-3, 2.5e-3, 6.3e-3, 1.6e-2, 4e-2, 1e-1])
    ap.add_argument("--step0", type=float, default=0.25)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    OUT.mkdir(exist_ok=True)
    out_path = OUT / "g2_voxel_huangyuwei.json"

    results = _load_or_init(out_path, a)

    data = {}
    for c in ("T1", "FLAIR"):
        ksp, csm = load_trainer_mat(BASE % c)
        ksp, _ = normalize_p999(ksp, csm)
        gt = (csm.conj() * ifft3c(ksp)).sum(0)
        data[c] = dict(ksp=ksp.to(dev), csm=csm.to(dev), gt=gt.to(dev),
                       gt_mag=gt.abs().cpu().numpy())

    pe_shape = tuple(data["T1"]["ksp"].shape[1:3])
    vol_shape = tuple(data["T1"]["ksp"].shape[1:])
    m_same = vd_poisson_2d(pe_shape, a.af, a.acs, a.seed)
    m1, m2, overlap = complementary_pair(pe_shape, a.af, a.acs, a.seed)
    ksp_np = data["T1"]["ksp"].cpu().numpy()
    m_same, shift = align_to_dc(m_same, ksp_np)
    m1, _ = align_to_dc(m1, ksp_np)
    m2, _ = align_to_dc(m2, ksp_np)
    del ksp_np

    conds = {"same": (m_same, m_same), "complementary": (m1, m2)}

    if results is None:
        results = {"dc_shift": list(shift), "complementary_overlap": overlap,
                   "sampling_fraction": float(m_same.mean()), "lams": [],
                   "af": a.af, "acs": a.acs, "seed": a.seed, "iters": a.iters,
                   "step0": a.step0,
                   "pe_shape": list(pe_shape), "vol_shape": list(vol_shape), "cells": {}}
    results["lams"] = sorted(set(results["lams"]) | set(a.lams))

    for cond, (ma, mb) in conds.items():
        masks = {"T1": _stack(ma, vol_shape).to(dev), "FLAIR": _stack(mb, vol_shape).to(dev)}
        ys = {c: data[c]["ksp"] * masks[c].unsqueeze(0) for c in ("T1", "FLAIR")}
        for lam in a.lams:
            ind, ind_diag = {}, {}
            for c in ("T1", "FLAIR"):
                x, diag = fista_tv_monitored(ys[c], data[c]["csm"], masks[c], lam, a.iters, step0=a.step0)
                ind[c], ind_diag[c] = x, diag
            jnt, jnt_diag = fista_joint_tv_monitored(
                [ys["T1"], ys["FLAIR"]], [data["T1"]["csm"], data["FLAIR"]["csm"]],
                [masks["T1"], masks["FLAIR"]], lam, a.iters, step0=a.step0)
            cell = {}
            for i, c in enumerate(("T1", "FLAIR")):
                cell[f"{c}_independent"] = dict(
                    e_unobs=e_unobs(ind[c], data[c]["csm"], data[c]["ksp"], masks[c]),
                    **brain_gauge(ind[c].abs().cpu().numpy(), data[c]["gt_mag"]),
                    **ind_diag[c])
                cell[f"{c}_joint"] = dict(
                    e_unobs=e_unobs(jnt[i], data[c]["csm"], data[c]["ksp"], masks[c]),
                    **brain_gauge(jnt[i].abs().cpu().numpy(), data[c]["gt_mag"]),
                    **jnt_diag)
            results["cells"][f"{cond}|lam={lam:g}"] = cell
            print(cond, lam,
                 {k: round(v["e_unobs"], 4) for k, v in cell.items()},
                 {k: (cell[k]["step_used"], cell[k]["stable"], cell[k]["halvings"])
                  for k in cell}, flush=True)
            del ind, jnt
            torch.cuda.empty_cache()
            out_path.write_text(json.dumps(results, indent=2))  # persist after every cell
        del masks, ys
        torch.cuda.empty_cache()

    _finalize(results, conds, out_path, overlap)


if __name__ == "__main__":
    main()
