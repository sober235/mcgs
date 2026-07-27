"""CE pair 验证(spec §4.7 / §8 item 1)。

两个序列同 StudyInstanceUID、同协议、相隔 7 min 01 s,协议元数据无法区分平扫/增强,
只能靠图像:若为平扫-增强对,差值图应在血管/硬膜/垂体等强化结构上出现结构化正差,
且差值集中在少数体素;若为重复扫描,差值应接近噪声、无解剖结构。

判据(先于运行固定):
  ENHANCED  若 (a) 差值图脑内 P99.9 / 脑内中位差 > 5,且 (b) 正差体素占比 < 15%;
  REPEAT    若上述不成立。
两项都必须报告,不得只报其一。
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

from crisp.io import normalize_p999
from crisp.metrics.image_domain import brain_mask
from crisp.recon.cg_sense import cg_sense

OUT = Path("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_crisp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--series", nargs=2, required=True,
                    help="两个序列的 .h5 路径,顺序为 …0010 …0060")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    OUT.mkdir(exist_ok=True)

    # CE_Data 的 kspace/csm 分属同目录下两个独立 .h5 文件(见 crisp/io.py 模块 docstring
    # 的 Step 1 记录),load_pair_h5(ksp_path) 会在同目录自动找唯一的 csm*.h5,
    # 因此这里仍只需传入 ksp 路径,调用形式不必改。
    from crisp.io import load_pair_h5
    recs = []
    for p in a.series:
        ksp, csm = load_pair_h5(p)
        ksp, _ = normalize_p999(ksp, csm)
        mask = torch.ones(ksp.shape[1:], dtype=torch.float32)
        x = cg_sense(ksp, csm, mask, lam=a.lam, iters=a.iters, device=dev)
        recs.append(x.abs().cpu().numpy().astype(np.float32))
        del ksp, csm, mask, x
        torch.cuda.empty_cache()

    a_img, b_img = recs
    brain = brain_mask(a_img)
    diff = b_img - a_img
    pos = diff[brain] > 0
    ratio = float(np.quantile(np.abs(diff[brain]), 0.999) /
                  (np.median(np.abs(diff[brain])) + 1e-12))
    pos_frac = float(pos.mean())
    verdict = "ENHANCED" if (ratio > 5.0 and pos_frac < 0.15) else "REPEAT"

    res = dict(series=a.series, lam=a.lam, iters=a.iters,
               tail_to_median_ratio=ratio, positive_fraction=pos_frac, verdict=verdict)
    (OUT / "ce_check.json").write_text(json.dumps(res, indent=2))

    z = a_img.shape[2] // 2
    sl = lambda v: np.rot90(v[:, :, z])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
    vmax = float(np.quantile(a_img[brain], 0.995))
    ax[0].imshow(sl(a_img), cmap="gray", vmin=0, vmax=vmax); ax[0].set_title("series 1 (…0010)")
    ax[1].imshow(sl(b_img), cmap="gray", vmin=0, vmax=vmax); ax[1].set_title("series 2 (…0060)")
    h = ax[2].imshow(sl(diff) / vmax, cmap="jet", vmin=0, vmax=0.10)
    ax[2].set_title(f"(2)-(1)  jet [0,0.1]\nratio {ratio:.1f}  pos {pos_frac:.2f}  -> {verdict}")
    plt.colorbar(h, ax=ax[2], fraction=0.04)
    for x_ in ax: x_.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "ce_check.png", dpi=110, bbox_inches="tight")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
