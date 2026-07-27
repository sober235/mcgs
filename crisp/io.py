"""数据加载。线圈轴一律返回在最前(C, D, H, W)。

各来源的真实结构(Step 1 探明后填入此处,勿凭记忆):
  - trainer .mat(huangyuwei):h5py,'ksp' 为 (D,H,W,C) 复合 real/imag;'map' 同形
  - T1/FLAIR_csm_ksp .h5:'kspace' 与 'csm',均为 (D,C,H,W) complex64
  - CE_Data:实测输出(2026-07-27,`h5py.File(p).visit(...)`,path 见下)::

      /data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/csm.h5
          csm (42, 256, 232, 176) complex64
      /data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370010/ksp1.h5
          kspace (42, 256, 232, 176) complex64
      /data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/csm2.h5
          csm (42, 256, 232, 176) complex64
      /data1/data1_congcong/data/3D/CE_Data/t1_3D_gre_fsp_sag_iso_ACS_16887049370060/ksp2.h5
          kspace (42, 256, 232, 176) complex64

    与本文件早先假定的"单文件同时含 kspace 与 csm"不同:CE_Data 把两者存在同目录下
    两个**独立**的 .h5 文件里(序列一为 ksp1.h5 + csm.h5,序列二为 ksp2.h5 + csm2.h5,
    csm 文件命名前缀不规则,不是简单地把 'ksp' 换成 'csm')。两个 dataset 形状都是
    (42, 256, 232, 176) complex64,42 为线圈数(该维最小,与 _to_coil_first 的
    argmin 取轴逻辑一致,42*256*232*176*8B ≈ 文件实际字节数,核对无误)。
    同目录下还有 ksp.mat / csm.mat / ksp2.mat / csm2.mat 作为 .mat 兜底
    (MATLAB v7.3,h5py 可直接打开,dataset 名为 'ksp'/'csm',shape
    (42, 256, 176, 232),复合 real/imag 字段,用法同 trainer .mat);本任务里
    .h5 可正常读取,未触发该兜底路径。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def _to_coil_first(arr: np.ndarray) -> np.ndarray:
    """把长度等于线圈数的轴挪到最前。线圈数取该数组最小的那个维度。"""
    axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, axis, 0)


def load_trainer_mat(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as f:
        k = f["ksp"][()]
        m = f["map"][()]
    ksp = k["real"].astype(np.complex64) + 1j * k["imag"].astype(np.complex64)
    csm = m["real"].astype(np.complex64) + 1j * m["imag"].astype(np.complex64)
    return torch.from_numpy(_to_coil_first(ksp)), torch.from_numpy(_to_coil_first(csm))


def load_pair_h5(ksp_path: str, csm_path: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """读取一个序列的 (kspace, csm)。

    CE_Data 里 kspace 与 csm 分属同目录下两个独立 .h5 文件(见模块 docstring)。
    若不显式给出 csm_path,则在 ksp_path 同目录下寻找唯一匹配 'csm*.h5' 的文件。
    """
    ksp_path = Path(ksp_path)
    if csm_path is None:
        cands = sorted(ksp_path.parent.glob("csm*.h5"))
        if len(cands) != 1:
            raise FileNotFoundError(
                f"expected exactly one sibling csm*.h5 next to {ksp_path}, found {cands}"
            )
        csm_path = cands[0]
    with h5py.File(ksp_path, "r") as f:
        ksp = f["kspace"][()].astype(np.complex64)
    with h5py.File(csm_path, "r") as f:
        csm = f["csm"][()].astype(np.complex64)
    return torch.from_numpy(_to_coil_first(ksp)), torch.from_numpy(_to_coil_first(csm))


def normalize_p999(kspace: torch.Tensor, csm: torch.Tensor) -> tuple[torch.Tensor, float]:
    """按 GT 幅值 P99.9 归一化 k 空间(spec Global Constraints)。"""
    from crisp.metrics.coil_domain import ifft3c
    gt = (csm.conj() * ifft3c(kspace)).sum(0)
    scale = float(np.quantile(gt.abs().cpu().numpy(), 0.999))
    return kspace / scale, scale
