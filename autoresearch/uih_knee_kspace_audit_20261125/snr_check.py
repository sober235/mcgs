"""SNR of the UIH 3D knee images as reconstructed for the figures (read-only, CPU).

usage: snr_check.py SUBJECT [SUBJECT ...]      (S-numbers in sorted path order; no names printed)

Per contrast:
  sigma_k   noise s.d. of one real component of one coil, from the outermost measured k-space shell
            (readout edge 6 % on each side AND phase-encode ellipse radius 0.80-0.95): 1.4826 * MAD of
            real and imaginary parts, median over coils. With the orthonormal FFT the same value holds in
            the image. Residual signal in the shell can only make it larger (SNR lower bound).
  sigma_air the same quantity from the air background of the root-sum-of-squares (RSS) image:
            sqrt(median(RSS^2) / median of chi2 with 2*coils d.o.f.), air = more than 10 voxels away from
            the knee mask.
  SNR       RSS intensity / sigma_k inside the knee mask (PD above 10 % of its 99.5th percentile, eroded 2):
            5th / 25th / 50th / 75th percentile.
  floor     mean RSS of pure noise = sigma * sqrt(2*coils) approx., as a share of the tissue median.
  local     s.d. / mean in 5x5x5 boxes inside the mask (median over boxes): what the eye sees as grain,
            compared with 1 / SNR expected from noise alone at the same boxes.
  corr      mean absolute noise correlation between coils in the k-space shell.
"""
import glob
import os
import sys

import h5py
import numpy as np
from scipy import fft as F, ndimage as nd, stats

ROOT = "/data2/congcong/data/3D/3D_data20261112/kspace_csm/knee"


def load(si, tag):
    subs = sorted({os.path.dirname(p) for p in glob.glob(ROOT + "/**/*.mat", recursive=True)})
    p = glob.glob(subs[si - 1] + "/*_" + tag + "_mx3d*.mat")[0]
    with h5py.File(p, "r") as f:
        k = f["kspace"][()]
        sm_shape, sm_dtype = f["s_maps"].shape, f["s_maps"].dtype
    return k, sm_shape, sm_dtype


def one(si, tag, ref):
    k, sm_shape, sm_dtype = load(si, tag)
    k = np.moveaxis(k, 1, 0)  # coils, n0, n2, n3
    L, shp = k.shape[0], k.shape[1:]
    # readout axis = the spatial axis along which no k-space line is empty
    E = (np.abs(k) ** 2).sum(axis=0)
    empt = []
    for a in range(3):
        P = E.sum(axis=a)
        empt.append(float((P <= 1e-6 * np.median(P[P > 0])).mean()))
    ro = int(np.argmax(empt))  # the plane summed over the readout axis shows the unmeasured positions
    pe = [a for a in range(3) if a != ro]
    P = E.sum(axis=ro)
    del E
    measured = P > 1e-6 * np.median(P[P > 0])
    na, nb = shp[pe[0]], shp[pe[1]]
    r = np.sqrt((((np.arange(na) - na // 2) / (na / 2.0)) ** 2)[:, None]
                + (((np.arange(nb) - nb // 2) / (nb / 2.0)) ** 2)[None, :])
    shell = measured & (r > 0.80) & (r < 0.95)
    nr = shp[ro]
    e = max(4, int(round(0.06 * nr)))
    kk = np.moveaxis(k, ro + 1, 1)  # coils, readout, pe_a, pe_b
    samp = kk[:, np.r_[0:e, nr - e:nr]][:, :, shell].reshape(L, -1)
    comp = np.concatenate([samp.real, samp.imag], axis=1)
    sig_c = 1.4826 * np.median(np.abs(comp - np.median(comp, axis=1, keepdims=True)), axis=1)
    sigma_k = float(np.median(sig_c))
    c = np.corrcoef(samp)
    corr = float(np.abs(c[~np.eye(L, dtype=bool)]).mean())
    ax = (1, 2, 3)
    x = F.fftshift(F.ifftn(F.ifftshift(k, axes=ax), axes=ax, norm="ortho", workers=4), axes=ax)
    rss = np.sqrt((np.abs(x) ** 2).sum(axis=0)).astype(np.float32)
    del k, kk, x
    own = ref is None or ref[0].shape != rss.shape
    if own:  # PD comes first; a contrast on another grid gets its own mask
        mask0 = rss > 0.1 * np.percentile(rss, 99.5)
        m = (mask0, nd.binary_erosion(mask0, iterations=2), ~nd.binary_dilation(mask0, iterations=10))
        if ref is None:
            ref = m
    else:
        m = ref
    mask0, mask, air = m
    sigma_air = float(np.sqrt(np.median(rss[air].astype(np.float64) ** 2) / stats.chi2.median(2 * L)))
    q = [float(t) for t in np.percentile(rss[mask] / sigma_k, [5, 25, 50, 75])]
    m1 = nd.uniform_filter(rss, 5)
    m2 = nd.uniform_filter(rss * rss, 5)
    inner = nd.binary_erosion(mask, iterations=3)
    mu = m1[inner]
    cv = float(np.median(np.sqrt(np.maximum(m2 - m1 * m1, 0))[inner] / mu))
    cv_noise = float(np.median(sigma_k / mu))
    row = {"subject": "S%02d" % si, "contrast": tag, "grid": "x".join(str(v) for v in shp), "coils": L,
           "readout_axis": ro, "mask": "own" if own and tag != "pd" else "pd", "sigma_k": sigma_k,
           "sigma_air": sigma_air, "air_over_k": sigma_air / sigma_k, "coil_sigma_min": float(sig_c.min()),
           "coil_sigma_max": float(sig_c.max()), "coil_corr": corr, "snr_p05": q[0], "snr_p25": q[1],
           "snr_p50": q[2], "snr_p75": q[3], "noise_floor_share": float(np.sqrt(2 * L - 0.5)) / q[2],
           "local_cv": cv, "local_cv_noise_only": cv_noise, "s_maps_shape": "x".join(str(v) for v in sm_shape)}
    return row, ref


def main():
    out = os.environ.get("MCGS_ARTIFACT_DIR", ".")
    subs = [int(a) for a in sys.argv[1:]]
    rows = []
    for si in subs:
        ref = None
        for tag in ("pd", "t1", "t2"):
            try:
                row, ref = one(si, tag, ref)
            except Exception as ex:  # keep going; the message must not carry a path (names)
                print("S%02d %-2s FAILED: %s" % (si, tag, type(ex).__name__))
                continue
            rows.append(row)
            print("S%02d %-2s grid %s ro %d mask %s | sigma_k %.3e air/k %.2f corr %.2f | SNR 5/25/50/75 %%: "
                  "%.1f / %.1f / %.1f / %.1f | floor %.0f %% | local cv %.3f vs noise %.3f"
                  % (si, tag, row["grid"], row["readout_axis"], row["mask"], row["sigma_k"], row["air_over_k"],
                     row["coil_corr"], row["snr_p05"], row["snr_p25"], row["snr_p50"], row["snr_p75"],
                     100 * row["noise_floor_share"], row["local_cv"], row["local_cv_noise_only"]))
            sys.stdout.flush()
    if rows:
        cols = list(rows[0])
        with open(os.path.join(out, "snr_S%02d-S%02d.tsv" % (min(subs), max(subs))), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r_ in rows:
                f.write("\t".join(("%.6g" % r_[c]) if isinstance(r_[c], float) else str(r_[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
