"""Header-only walk of the MTP Cartesian raw: build the (PE,SPE) sampling mask per (contrast=echo, set=flip angle)."""
import struct, collections, numpy as np, sys, time
fn = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/UID_126819142341469_mtp_cartesian_.7x.7x1.raw"
DHL = struct.Struct("<B B B B I Q I I I i i i H H I Q H H H H H H H H H H H 5H H H H H H 3f 4f")
t0 = time.time()
with open(fn, "rb") as f:
    magic = f.read(12); n = struct.unpack("<I", f.read(4))[0]; f.seek(n + 4 + 12)
    recs = []
    while True:
        hdr = f.read(192)
        if len(hdr) < 192: break
        v = DHL.unpack_from(hdr, 0)
        ucVersion, nsamp, nch, ctrl = v[0], v[12], v[13], v[15]
        if ctrl & 1: break
        rep, cph, slc, avg, con, sset, shot, seg, pe, spe = v[16:26]
        kind = "noise" if (ctrl >> 24) & 1 else "img"
        ref = (ctrl >> 9) & 1
        if kind == "img": recs.append((con, sset, pe, spe, ref, shot, seg))
        f.seek(2 * ((nsamp + 2) if ucVersion >= 2 else nsamp) * nch * 4, 1)
r = np.array(recs, dtype=np.int32)
print("walk time %.0f s, imaging lines %d" % (time.time() - t0, len(r)))
con, sset, pe, spe, ref = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]
NPE, NSPE = pe.max() + 1, spe.max() + 1
masks = {}
for c in np.unique(con):
    for s in np.unique(sset):
        m = np.zeros((NPE, NSPE), bool); sel = (con == c) & (sset == s); m[pe[sel], spe[sel]] = True; masks[(c, s)] = m
        dup = sel.sum() - m.sum()
        print(f"echo {c} set {s}: lines {sel.sum()}, unique (PE,SPE) {m.sum()} of {NPE}x{NSPE}={NPE*NSPE} -> AF {NPE*NSPE/m.sum():.2f}; duplicates {dup}")
m0 = masks[(0, 0)]
print("identical mask across echoes and sets:", all(np.array_equal(m0, m) for m in masks.values()))
# structure of the mask: PE lines used, ACS block, elliptical envelope
pe_used = np.where(m0.any(1))[0]; print("PE lines with any sample:", len(pe_used), "gaps", dict(collections.Counter(np.diff(pe_used).tolist())))
acs_pe = np.where(m0.all(1))[0]; print("PE lines fully sampled along SPE (ACS block candidates):", acs_pe[:5], "...", acs_pe[-5:], "count", len(acs_pe))
refl = np.unique(pe[ref == 1]); print("reflines flagged PE:", refl.min(), "..", refl.max(), "count", len(refl))
# elliptical envelope: fraction of the rectangular grid inside the sampled envelope
env = np.zeros_like(m0)
for i in pe_used:
    j = np.where(m0[i])[0]; env[i, j.min(): j.max() + 1] = True
print("envelope (per-PE-line SPE span) covers %.1f%% of the rectangle; samples inside envelope %.1f%%" % (100 * env.mean(), 100 * m0.sum() / env.sum()))
# readout sample count and PE/SPE centre from protocol: PE centre 152 of FT 304, SPE FT 104
print("SPE range", spe.min(), spe.max(), " PE range", pe.min(), pe.max())
np.savez("/tmp/claude-1002/-home-congcongliu--claude/fd861f29-d86a-49cf-be4f-a26d183fe383/scratchpad/mtp_cart_masks.npz", **{f"e{c}_s{s}": m for (c, s), m in masks.items()}, pe_spe_lines=r)
# ascii thumbnail of the mask (downsampled)
th = m0[::4, ::2]
print("\nmask thumbnail (rows = PE /4, cols = SPE /2):")
for row in th: print("".join("#" if x else "." for x in row))
