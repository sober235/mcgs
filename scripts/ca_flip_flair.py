#!/usr/bin/env python3
"""ca_flip_flair.py -- Stage 3 of the coupling-attribution study: the placebo second contrast.

Keep jiangyizhou's own FLAIR k-space, coils, SNR and vendor shutter, but flip the FLAIR volume
so that it is a real brain that does not register to T1.  Reads the two read-only symlinks in the
AF16 topic's data/ and writes the flipped pair into results_arc/placebo/ (gitignored).

Run: env PYTHONNOUSERSITE=1 PYTHONPATH= python scripts/ca_flip_flair.py
"""
import numpy as np


def fft3c(x):
    ax = (0, 1, 2)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def ifft3c(x):
    ax = (0, 1, 2)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def _reflect_about_dc(a, axis):
    """Reflect `a` about the DC sample (index N//2 in the fftshift-centred convention)."""
    return np.roll(np.flip(a, axis=axis), 1, axis=axis)


def flip_pair(ksp, csm, axis):
    """Flip the object and the coils together along `axis`, in image space. Arrays are coil-last.

    Flipping the coil images and the sensitivity maps by the same amount keeps the model
    self-consistent: the SENSE combination of the result is exactly `np.flip` of the original.

    The k-space support does NOT follow `np.flip`.  An image-space reflection reflects k-space
    about the DC sample, so the support goes S -> _reflect_about_dc(S), and no roll in image space
    can change that: `np.flip` and the DC-centred reflection differ by a one-voxel image shift,
    which multiplies k-space by a unit-modulus linear phase and therefore cannot move a zero.
    (The brief's `np.roll(..., 1)` was meant to "re-centre DC"; it only translates the object by
    one voxel and costs the `np.flip` identity above, so it is dropped.)  The FFT round trip
    leaves ~1e-16 of round-off in the shuttered samples, so the reflected zero set is re-imposed
    exactly -- otherwise the output has no zeros at all.

    Consequence: a shutter symmetric about the DC sample comes back sample-for-sample and
    `known_zero` stays valid; a shutter that is not (e.g. a one-sided partial-Fourier cut) comes
    back mirrored about DC.
    """
    ksp_f = fft3c(np.flip(ifft3c(ksp), axis=axis))
    ksp_f[_reflect_about_dc(ksp == 0, axis)] = 0
    return ksp_f, np.flip(csm, axis=axis)


if __name__ == "__main__":
    import argparse
    import hashlib
    import os
    import h5py

    SRC = "/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data"
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_arc/placebo")
    # Files are COIL-FIRST (C, x, ky, kz); flip_pair is coil-last, so the coil axis is moved at the
    # I/O boundary.  configs/t1flair_v3_af16_jyz.yaml records readout_axis 0 and phase_axes [1, 2] in
    # spatial, coil-excluded numbering, so readout x = coil-last axis 0 and ky = coil-last axis 1.
    #
    # Default axis 0 (readout x), not the spec's ky.  Both are in-plane axes of this sagittal slab and
    # both misregister grossly -- the readout flip more so, NCC(|T1|, |FLAIR_flip|) 0.349 vs 0.427 for
    # ky and 0.678 for the near-symmetric kz slab direction (original 0.801).  But FLAIR's support is
    # a DC-centred ellipse with a ONE-SIDED partial-Fourier cut at ky >= 20, so it is not symmetric
    # about DC in ky: a ky flip mirrors the cut to ky <= 236 and the known-zero estimator then finds no
    # rect cut at all and explains only 199 of 229 zero sampled lines (frac_z 0.2151 vs 0.2423), i.e.
    # the placebo's second channel would get a weaker physics prior than the anchor's.  Along the
    # readout the support is complete (34139 lines at every one of the 256 x indices), so it is
    # trivially symmetric about DC and comes back sample-for-sample.
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=int, default=0, choices=[0, 1],
                    help="coil-last spatial axis to flip: 0 = readout x (default), 1 = ky (spec's axis)")
    args = ap.parse_args()
    sfx = "" if args.axis == 0 else "_ky"
    os.makedirs(OUT, exist_ok=True)
    pairs = [("ksp_FLAIR.h5", "kspace", f"ksp_FLAIR_flip{sfx}.h5"),
             ("csm_espirit_FLAIR.h5", "map", f"csm_espirit_FLAIR_flip{sfx}.h5")]
    with h5py.File(f"{SRC}/{pairs[0][0]}", "r") as fk, h5py.File(f"{SRC}/{pairs[1][0]}", "r") as fc, \
         h5py.File(f"{OUT}/{pairs[0][2]}", "w") as gk, h5py.File(f"{OUT}/{pairs[1][2]}", "w") as gc:
        k_in, c_in = fk[pairs[0][1]], fc[pairs[1][1]]
        assert k_in.shape == c_in.shape, (k_in.shape, c_in.shape)
        C = k_in.shape[0]
        k_out = gk.create_dataset(pairs[0][1], shape=k_in.shape, dtype=k_in.dtype)
        c_out = gc.create_dataset(pairs[1][1], shape=c_in.shape, dtype=c_in.dtype)
        for c in range(C):                                    # coil by coil: 2.2 GiB per file
            kf, cf = flip_pair(np.asarray(k_in[c])[..., None], np.asarray(c_in[c])[..., None], args.axis)
            k_out[c] = kf[..., 0].astype(k_in.dtype)
            c_out[c] = np.ascontiguousarray(cf[..., 0]).astype(c_in.dtype)
            print(f"  coil {c + 1}/{C}", flush=True)
    for _, _, name in pairs:
        h = hashlib.md5()
        with open(f"{OUT}/{name}", "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        print(f"{name}  md5 {h.hexdigest()}  {os.path.getsize(f'{OUT}/{name}') / 2**30:.2f} GiB")
