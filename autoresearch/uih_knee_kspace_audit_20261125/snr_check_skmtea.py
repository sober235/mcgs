"""SNR of SKM-TEA knee qDESS (3 T, echoes E1/E2, 8 coils), same quantities as snr_check.py (read-only, CPU).

usage: snr_check_skmtea.py MTR_006 MTR_045 ...

File layout (knee/paths.py of the SKM-TEA study): kspace (x hybrid 512, ky 512, kz 160 zero-padded, echo 2,
coil 8); the acquired partitions are kz 40:120; outside the ky band 244..267 only even ky were acquired and the
odd ones were synthesised by the vendor's parallel-imaging step. Images here: centred inverse 2D FFT over
(ky, kz 40:120), root-sum-of-squares (RSS) over coils.

  sigma_k   1.4826 * MAD of real/imaginary parts on acquired (even) ky lines in the outer shell of the
            (ky, kz) ellipse (radius 0.80-0.95), all x, median over coils.
  sigma_air from the air background of the RSS image, as in snr_check.py. This one includes the noise of the
            synthesised lines, so it is the image noise proper; SNR is quoted against BOTH.
  mask      E1 RSS above 10 % of its 99.5th percentile, eroded by 2; used for both echoes.
"""
import os
import sys

import h5py
import numpy as np
from scipy import fft as F, ndimage as nd, stats

SKM_DIR = "/data2/congcong/data/FM_data/SKM-TEA/files_recon_calib-24"
KZ0, KZ1 = 40, 120


def main():
    out = os.environ.get("MCGS_ARTIFACT_DIR", ".")
    rows = []
    for sid in sys.argv[1:]:
        with h5py.File(os.path.join(SKM_DIR, sid + ".h5"), "r") as f:
            full_shape = f["kspace"].shape
            k = f["kspace"][:, :, KZ0:KZ1]  # x, ky, kz, echo, coil
        nx, ny, nz, ne, L = k.shape
        ref = None
        for e in range(ne):
            ke = np.moveaxis(k[:, :, :, e, :], -1, 0)  # coil, x, ky, kz
            P = (np.abs(ke) ** 2).sum(axis=(0, 1))
            measured = P > 1e-6 * np.median(P[P > 0])
            r = np.sqrt((((np.arange(ny) - ny // 2) / (ny / 2.0)) ** 2)[:, None]
                        + (((np.arange(nz) - nz // 2) / (nz / 2.0)) ** 2)[None, :])
            shell = measured & (r > 0.80) & (r < 0.95)
            shell[1::2, :] = False  # acquired lines only (even ky outside the centre band)
            samp = ke[:, :, shell].reshape(L, -1)
            comp = np.concatenate([samp.real, samp.imag], axis=1)
            sig_c = 1.4826 * np.median(np.abs(comp - np.median(comp, axis=1, keepdims=True)), axis=1)
            sigma_k = float(np.median(sig_c))
            ax = (2, 3)
            x = F.fftshift(F.ifftn(F.ifftshift(ke, axes=ax), axes=ax, norm="ortho", workers=4), axes=ax)
            rss = np.sqrt((np.abs(x) ** 2).sum(axis=0)).astype(np.float32)
            del x, ke
            if ref is None:
                mask0 = rss > 0.1 * np.percentile(rss, 99.5)
                ref = (nd.binary_erosion(mask0, iterations=2), ~nd.binary_dilation(mask0, iterations=10))
            mask, air = ref
            sigma_air = float(np.sqrt(np.median(rss[air].astype(np.float64) ** 2) / stats.chi2.median(2 * L)))
            sig = rss[mask]
            qk = [float(t) for t in np.percentile(sig / sigma_k, [5, 25, 50, 75])]
            qa = [float(t) for t in np.percentile(sig / sigma_air, [5, 25, 50, 75])]
            m1 = nd.uniform_filter(rss, 5)
            m2 = nd.uniform_filter(rss * rss, 5)
            inner = nd.binary_erosion(mask, iterations=3)
            mu = m1[inner]
            cv = float(np.median(np.sqrt(np.maximum(m2 - m1 * m1, 0))[inner] / mu))
            row = {"subject": sid, "echo": "E%d" % (e + 1), "grid": "%dx%dx%d" % (nx, ny, nz),
                   "file_shape": "x".join(str(v) for v in full_shape), "coils": L,
                   "measured_share": float(measured.mean()), "sigma_k": sigma_k, "sigma_air": sigma_air,
                   "air_over_k": sigma_air / sigma_k, "air_voxels": int(air.sum()), "mask_voxels": int(mask.sum()),
                   "snr_k_p05": qk[0], "snr_k_p25": qk[1], "snr_k_p50": qk[2], "snr_k_p75": qk[3],
                   "snr_air_p05": qa[0], "snr_air_p25": qa[1], "snr_air_p50": qa[2], "snr_air_p75": qa[3],
                   "local_cv": cv, "local_cv_noise_only_air": float(np.median(sigma_air / mu))}
            rows.append(row)
            print("%s %s grid %s coils %d measured %.2f | sigma_k %.3e air/k %.2f (air voxels %d) | "
                  "SNR vs sigma_k 5/25/50/75 %%: %.1f / %.1f / %.1f / %.1f | vs sigma_air: %.1f / %.1f / %.1f / %.1f"
                  " | local cv %.3f vs noise %.3f"
                  % (sid, row["echo"], row["grid"], L, row["measured_share"], sigma_k, row["air_over_k"],
                     row["air_voxels"], *qk, *qa, cv, row["local_cv_noise_only_air"]))
            sys.stdout.flush()
        del k
    if rows:
        cols = list(rows[0])
        with open(os.path.join(out, "snr_skmtea.tsv"), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r_ in rows:
                f.write("\t".join(("%.6g" % r_[c]) if isinstance(r_[c], float) else str(r_[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
