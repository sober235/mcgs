"""数据包 raw/ksp_*.h5 是否与厂商原始 CFL 逐点相同。
h5 是 (C,X,Y,Z) complex64 = CFL 的 (X,Y,Z,C) 第 5 维取 0 后 moveaxis(3,0)。
按读出层 x 分块比较，全部 256 层都比，不抽样。"""
import os, sys, numpy as np, h5py

SRC = "/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou"
B = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/export/af24_p3_mcinr_spirit/raw"
FILES = {"T1": "t1_fsp_train_jiangyizhou_cc", "FLAIR": "flair_train_jiangyizhou_cc"}

def cfl(base):
    with open(base + ".hdr") as f:
        dims = [int(x) for x in f.read().split("\n")[1].split()]
    n = int(np.prod(dims))
    return np.memmap(base + ".cfl", dtype=np.complex64, mode="r", shape=(n,)).reshape(dims, order="F")[..., 0]

for tag, f in FILES.items():
    k = cfl(os.path.join(SRC, f))                      # (X,Y,Z,C)
    with h5py.File(f"{B}/ksp_{tag}.h5", "r") as h:
        d = h["kspace"]
        print(f"[{tag}] cfl {k.shape} {k.dtype}   h5 {d.shape} {d.dtype}", flush=True)
        assert d.shape == (k.shape[3], k.shape[0], k.shape[1], k.shape[2])
        bad = 0
        nx = k.shape[0]
        for x0 in range(0, nx, 32):
            a = np.moveaxis(np.asarray(k[x0:x0 + 32]), 3, 0)   # (C,x,Y,Z)
            b = d[:, x0:x0 + 32]
            bad += int((a != b).sum())
        print(f"[{tag}] 逐点不同的元素数 = {bad}  (0 = 完全一致)", flush=True)
    del k
