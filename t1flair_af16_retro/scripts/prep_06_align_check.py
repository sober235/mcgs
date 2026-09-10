#!/usr/bin/env python3
"""prep_06_align_check.py -- RSS references for T1/FLAIR + a co-registration check.

The MC-GS joint model shares ONE Gaussian geometry between the two contrasts, so it assumes the
pair is co-registered. Both scans come from the same exam on the same grid, but that is an
assumption until measured. This script measures it: brain-mask Dice, centroid offset, and the
integer shift that maximises normalised cross-correlation of the two masks.
"""
import json, os
import h5py, numpy as np, torch, torch.fft as tfft

R = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_retro"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "1"))
dev = torch.device("cuda:0")


def _c(fn, x, d): return tfft.fftshift(fn(tfft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)
def ifft3c(x): return _c(tfft.ifftn, x, (-3, -2, -1))


def rss(tag):
    with h5py.File(f"{R}/data/ksp_{tag}.h5", "r") as f:
        k = f["kspace"]
        acc = torch.zeros(k.shape[1:], dtype=torch.float32, device=dev)
        for i in range(0, k.shape[0], 4):
            blk = torch.from_numpy(k[i:i + 4]).to(dev)
            acc += ifft3c(blk).abs().square().sum(0)
    return acc.sqrt()


def main():
    out = {}
    vols = {}
    for tag in ("T1", "FLAIR"):
        v = rss(tag)
        vols[tag] = v
        np.save(f"{R}/recon/rss_{tag}.npy", v.cpu().numpy().astype(np.float32))
        out[tag] = {"max": float(v.max()), "p999": float(np.quantile(v.cpu().numpy()[::4], 0.999))}
        print(f"[{tag}] RSS done, max={v.max():.4e}", flush=True)

    masks = {t: (vols[t] > 0.10 * torch.quantile(vols[t].flatten()[::37], 0.999)) for t in vols}
    a, b = masks["T1"].float(), masks["FLAIR"].float()
    inter = float((a * b).sum()); dice = 2 * inter / float(a.sum() + b.sum())
    idx = torch.stack(torch.meshgrid(*[torch.arange(s, device=dev, dtype=torch.float32)
                                       for s in a.shape], indexing="ij"))
    ca = [float((idx[i] * a).sum() / a.sum()) for i in range(3)]
    cb = [float((idx[i] * b).sum() / b.sum()) for i in range(3)]
    # integer shift maximising mask cross-correlation, via FFT
    A = tfft.fftn(a.to(torch.complex64)); B = tfft.fftn(b.to(torch.complex64))
    cc = tfft.ifftn(A * B.conj()).real
    pk = torch.argmax(cc)
    sh = np.unravel_index(int(pk), tuple(a.shape))
    sh = [int(s) - int(n) if int(s) > int(n) // 2 else int(s) for s, n in zip(sh, a.shape)]
    out["alignment"] = {"mask_dice": dice, "centroid_T1": ca, "centroid_FLAIR": cb,
                        "centroid_delta_vox": [cb[i] - ca[i] for i in range(3)],
                        "peak_xcorr_shift_vox": sh,
                        "brain_frac_T1": float(a.mean()), "brain_frac_FLAIR": float(b.mean())}
    print(json.dumps(out["alignment"], indent=2), flush=True)
    json.dump(out, open(f"{R}/logs/prep_06_align.json", "w"), indent=2)


if __name__ == "__main__":
    main()
