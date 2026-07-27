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
from crisp.metrics.coil_domain import e_unobs, ifft3c
from crisp.metrics.image_domain import brain_gauge
from crisp.recon.cs_sense import fista_joint_tv, fista_tv

OUT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_crisp")
BASE = ("/data1/data1_congcong/data/3D/T2_csm_ksp_n4_mat/"
        "Train_3dBrain_huangyuwei_20230705-160927-1451_161015_%s_trainer.mat")


def _stack(mask2d, shape):
    """2D 相位编码平面掩膜 -> 3D(读出轴 = 最后一轴,全采)。"""
    return torch.from_numpy(np.repeat(mask2d[:, :, None], shape[-1], axis=2).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--af", type=float, default=12.0)
    ap.add_argument("--acs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lams", type=float, nargs="+", default=[3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3])
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    OUT.mkdir(exist_ok=True)

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
    results = {"dc_shift": list(shift), "complementary_overlap": overlap,
               "sampling_fraction": float(m_same.mean()), "lams": a.lams,
               "af": a.af, "acs": a.acs, "seed": a.seed, "iters": a.iters,
               "pe_shape": list(pe_shape), "vol_shape": list(vol_shape), "cells": {}}

    for cond, (ma, mb) in conds.items():
        masks = {"T1": _stack(ma, vol_shape).to(dev), "FLAIR": _stack(mb, vol_shape).to(dev)}
        ys = {c: data[c]["ksp"] * masks[c].unsqueeze(0) for c in ("T1", "FLAIR")}
        for lam in a.lams:
            ind = {c: fista_tv(ys[c], data[c]["csm"], masks[c], lam, a.iters) for c in ("T1", "FLAIR")}
            jnt = fista_joint_tv([ys["T1"], ys["FLAIR"]],
                                 [data["T1"]["csm"], data["FLAIR"]["csm"]],
                                 [masks["T1"], masks["FLAIR"]], lam, a.iters)
            cell = {}
            for i, c in enumerate(("T1", "FLAIR")):
                for name, x in (("independent", ind[c]), ("joint", jnt[i])):
                    cell[f"{c}_{name}"] = dict(
                        e_unobs=e_unobs(x, data[c]["csm"], data[c]["ksp"], masks[c]),
                        **brain_gauge(x.abs().cpu().numpy(), data[c]["gt_mag"]))
            results["cells"][f"{cond}|lam={lam:g}"] = cell
            print(cond, lam, {k: round(v["e_unobs"], 4) for k, v in cell.items()}, flush=True)
            del ind, jnt
            torch.cuda.empty_cache()
        del masks, ys
        torch.cuda.empty_cache()

    # 边界检查:最优 lam(按 independent 的 E_unobs 均值选)不得落在网格端点
    best_lam = {}
    for cond in conds:
        best_lam[cond] = min(a.lams, key=lambda L: np.mean(
            [results["cells"][f"{cond}|lam={L:g}"][k]["e_unobs"] for k in
             ("T1_independent", "FLAIR_independent")]))
        results.setdefault("boundary_optimum", {})[cond] = bool(best_lam[cond] in (a.lams[0], a.lams[-1]))

    # 判读:joint 相对 independent 的 E_unobs 增益(独立-联合;>0 = 联合更优),
    # 在 best_lam(公平操作点,由 independent 单独选出)处报告,并附全网格增益供稳健性核查。
    verdict = {}
    for cond in conds:
        gain_by_lam = {}
        for lam in a.lams:
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

    (OUT / "g2_voxel_huangyuwei.json").write_text(json.dumps(results, indent=2))

    # 图:E_unobs vs lam,independent vs joint,same 与 complementary 各一个子图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cond in zip(axes, conds):
        for name, style in (("independent", "o--"), ("joint", "s-")):
            for c, color in (("T1", "tab:blue"), ("FLAIR", "tab:orange")):
                ys_plot = [results["cells"][f"{cond}|lam={lam:g}"][f"{c}_{name}"]["e_unobs"] for lam in a.lams]
                ax.plot(a.lams, ys_plot, style, color=color, label=f"{c} {name}")
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_title(f"{cond} (overlap={overlap:.3f})" if cond == "complementary" else cond)
        ax.axvline(best_lam[cond], color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("E_unobs")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "g2_voxel_huangyuwei.png", dpi=110, bbox_inches="tight")

    print(json.dumps({k: results[k] for k in
                      ("dc_shift", "complementary_overlap", "sampling_fraction", "boundary_optimum", "verdict")},
                     indent=2))


if __name__ == "__main__":
    main()
