# AF24 figure set of record — data bundle

Everything behind the jiangyizhou AF 24 comparison figures (`_p3_mcinr_spirit`, ten methods), in one self-contained
folder: real copies, symlinks dereferenced, so the directory can be moved or tarred on its own. Written 2026-09-15 by
`Results_20260908/T1FLAIR_AF24_v3_jiangyizhou/scripts/export_af24_bundle.py` (user request: save the AF 24 raw data
and result figures on this server). `MANIFEST.tsv` lists every file with shape, dtype, size and md5; `checksums.md5`
verifies the copy (`md5sum -c checksums.md5`) — run at export time: **72/72 OK** (`md5check.out`), 12G on disk.
Same layout as `../../../T1FLAIR_AF16_v3_jiangyizhou/export/af16_phase3/` and
`../../../T1FLAIR_AF12_v3_jiangyizhou/export/af12_p14_mcinr/`, so the three bundles can be read side by side.

## What the reconstruction is

Subject vol3_jiangyizhou, 3D T1 + FLAIR, 26-coil k-space, volumes 256 x 256 x 176 with the readout (axis 0) fully
sampled. One vd-Poisson mask for both contrasts, **1877 of 45056 phase-encode points**, chosen from the generator's pool
so that T1 is at exactly 24x. Ground truth = the conjugate-SENSE image of the FULL k-space with the same ESPIRiT maps,
normalised by P99.9 of its magnitude; every arm is in that gauge. MC-GS and GS-single run the AF16 phase-3 recipe
unchanged (`gs_joint_p3`, `gs_single_p3`; 4000 iterations); every classical lambda was re-swept on this mask and is
interior to its grid; L1-SPIRiT is the AF12/16/20 arm (kernel 5 x 5, lambda 1e-3); MC-INR is the Niessen 2025
reproduction run on this mask and re-scored through this topic's scorer (agreement 1.2e-7).

**Acceleration labels — which gauge (user decision, 2026-09-16).** The label is the whole phase-encode plane (45056
points) divided by the mask's sampled points (1877): **24.00x on both contrasts**, since both read the same mask. The
FLAIR figures state under the panels that 1766 of those points carry FLAIR data — its acquisition covers 34139 of the
45056 lines (elliptical scanning plus ky rows 0-19). Two other gauges are reported in the text where they matter and are
NOT the label: 45056 / points-with-data (the 2026-09-14 per-contrast gauge, FLAIR 25.51x) and 34139 / 1766 = 19.33x,
FLAIR against its own acquisition. The `_p3_mcinr_spirit`, mask and raw-support figures in this bundle were re-rendered
under this label on 2026-09-16 (checksums and MANIFEST updated with them); the archived nine-arm `_p3` renders keep the
older titles, and no data changed.

## Head-only PSNR of the arms in this bundle (T1 / FLAIR, dB)

MC-GS 33.767 / 33.182 (first on head PSNR, SSIM and HFEN on both contrasts); GS-single 33.588 / 33.110 — the strongest
comparison arm (GS-single counts as one, user 2026-09-13), so the quoted margin is MC-GS - GS-single = **+0.179 /
+0.071**; strongest non-GS arm CS-SENSE (TV, converged) 32.820 on T1 and Joint SENSE (JTV) 32.605 on FLAIR; ZS-SSDU
32.578 / 32.107; L1-SPIRiT 31.783 / 31.803; CG-SENSE 30.340 / 31.259; MC-INR 30.076 / 30.947; 3D-DIP 30.500 / 30.360;
3D-INR 28.444 / 29.146; ZF 25.333 / 27.301. L1-ESPIRiT (33.401 / 32.782) is in the bundle as data but has not been
shown in any table or figure since 2026-09-15 (user decision). Full rows with SSIM, HFEN and sharpness:
`logs/summary.tsv`.

## Layout

```
raw/       ksp_T1.h5, ksp_FLAIR.h5 (26-coil full k-space), csm_espirit_T1.h5, csm_espirit_FLAIR.h5  (~2.2 GiB each)
mask/      mask_af24_3d.h5 + mask_vdpois_256x176_af24.mat (the mask of record); mask_af24_{db,dc,dd}_3d.h5 (variance draws)
volumes/   T1/, FLAIR/: ground truth, ZF, every arm at its kept parameters (complex64, 256x256x176); GS arms also as magnitude
mcinr/     MC-INR normalised outputs, losses, meta, reproduction score / gauge logs
logs/      summary.json/tsv, readme_tables.md, prep_04_export.json, extra_gs_arms.json, spirit + mcinr score jsons
figures/   every figure of the topic: the _p3_mcinr_spirit set of record (PNG + PDF), the archived nine-column _p3 set,
           the 174/110/47 render, the mask and acquisition-support figures
README_topic.md   the topic README at export time;  MANIFEST.tsv, checksums.md5, md5check.out
```

The masks cannot be regenerated (the generator seeds from the clock); this bundle and the topic's `data/` are the only
copies. The k-space and maps are the AF12 topic's files, dereferenced.

## Provenance of the raw data — where `raw/` comes from, and the check that it is the raw data (2026-09-16)

Vendor source of record, untouched since acquisition and still on this server:

```
/data1/data1_congcong/data/MC_Jia/Train_SpiritDiffusion_CFL/vol3_jiangyizhou/
  t1_fsp_train_jiangyizhou_cc.cfl   2,399,141,888 B   md5 c449fdbd70cf896173dc576993e2cb47
  t1_fsp_train_jiangyizhou_cc.hdr              31 B   md5 efccd5dc6cdcae22f02b1cfee6541ad8   dims 256 256 176 26 1
  flair_train_jiangyizhou_cc.cfl    2,399,141,888 B   md5 ce3627e9c84e88ee1b8cfc557070f3c7
  flair_train_jiangyizhou_cc.hdr               31 B   md5 efccd5dc6cdcae22f02b1cfee6541ad8   dims 256 256 176 26 1
```

(`t2_train_jiangyizhou_cc.*` sits in the same folder and is NOT used by this study.)

`raw/ksp_T1.h5` and `raw/ksp_FLAIR.h5` are those two files in HDF5 form, written by the AF12 topic's
`scripts/prep_04_export.py`: read the CFL Fortran-order as `(X,Y,Z,C,1)`, drop the trailing singleton, then
`moveaxis(3, 0)` to `(C,X,Y,Z)` complex64. No cropping, no scaling, no reordering of the data itself.

Checked on 2026-09-16 by reading both representations and comparing every element (all 256 readout planes, no
sampling): **0 differing elements out of 299,892,736 per contrast** — the bundle's k-space *is* the vendor raw data,
bit for bit.

```
[T1]    cfl (256, 256, 176, 26) complex64   h5 (26, 256, 256, 176) complex64   differing elements = 0
[FLAIR] cfl (256, 256, 176, 26) complex64   h5 (26, 256, 256, 176) complex64   differing elements = 0
```

`raw/csm_espirit_*.h5` are *derived*, not raw: BART `ecalib -t 0.001 -c 0.0 -r 24 -m 1` on those same CFL files
(AF12 topic `scripts/prep_03_csm.sh`), reproducible from the raw data at any time. The masks are the opposite case —
the generator seeds from the clock, so `mask/` cannot be regenerated and this bundle plus the topic's `data/` are the
only copies.

So the raw data now lives on this server in two independent places: the vendor CFL on `/data1` and this self-contained
bundle on `/data0`. Re-verify the bundle at any time with `md5sum -c checksums.md5` (72/72 OK, re-run 2026-09-16).
