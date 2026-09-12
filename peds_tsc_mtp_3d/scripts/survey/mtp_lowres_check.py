"""Read only the ACS-block lines (PE 140..163, all SPE) of the MTP raw for the six populated
(set, echo) combos x u1 in {0,1}, make low-resolution RSS images, and match them against the
vendor DICOM FA4/FA16 images to identify which loop counter is the flip angle and what set 0 is."""
import struct, collections, numpy as np, time, os, glob
import pydicom
fn = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/UID_126819142341469_mtp_cartesian_.7x.7x1.raw"
L = np.load("/tmp/claude-1002/-home-congcongliu--claude/fd861f29-d86a-49cf-be4f-a26d183fe383/scratchpad/mtp_cart_loops.npz", allow_pickle=True)
recs, cols = L["recs"], list(L["cols"])
ci = {c: cols.index(c) for c in cols}
u1 = recs[:, ci["u1"]]
# run-length structure of u1 in file order
runs = [(k, len(list(g))) for k, g in __import__("itertools").groupby(u1.tolist())]
print("u1 runs in file order: n_runs", len(runs), "first runs", runs[:6])
# mask equality among populated combos
key = lambda r: (int(r[ci["set"]]), int(r[ci["con"]]), int(r[ci["u1"]]))
masks = collections.defaultdict(set)
for r in recs: masks[key(r)].add((int(r[ci["pe"]]), int(r[ci["spe"]])))
ks = sorted(masks); print("populated (set,echo,u1) combos:", ks); print("all masks identical:", all(masks[k] == masks[ks[0]] for k in ks), "size", len(masks[ks[0]]))
# ---- read ACS-block lines ----
DHL = struct.Struct("<B B B B I Q I I I i i i H H I Q H H H H H H H H H H H 5H H H H H H 3f 4f")
want = {k: {} for k in ks}
t0 = time.time()
with open(fn, "rb") as f:
    magic = f.read(12); n = struct.unpack("<I", f.read(4))[0]; f.seek(n + 4 + 12)
    while True:
        hdr = f.read(192)
        if len(hdr) < 192: break
        v = DHL.unpack_from(hdr, 0)
        ucVersion, nsamp, nch, ctrl = v[0], v[12], v[13], v[15]
        if ctrl & 1: break
        per_ch = (nsamp + 2) if ucVersion >= 2 else nsamp
        nbytes = 2 * per_ch * nch * 4
        if (ctrl >> 24) & 1: f.seek(nbytes, 1); continue
        con, sset, pe, spe, uu1 = v[20], v[21], v[24], v[25], v[28]
        if 140 <= pe <= 163:
            buf = np.frombuffer(f.read(nbytes), dtype=np.float32).reshape(nch, per_ch, 2)
            want[(sset, con, uu1)][(pe, spe)] = (buf[:, 2:, 0] + 1j * buf[:, 2:, 1]).astype(np.complex64)   # (nch, nsamp)
        else:
            f.seek(nbytes, 1)
print("read time %.0f s; lines per combo:" % (time.time() - t0), {k: len(v) for k, v in want.items()})
def lowres(d):
    k = np.zeros((32, 640, 24, 104), np.complex64)
    for (pe, spe), line in d.items(): k[:, :, pe - 140, spe] = line
    # remove RO oversampling: ifft along RO, crop centre 320, keep image domain along RO
    x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=1), axis=1), axes=1)[:, 160:480]
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=(2, 3)), axes=(2, 3)), axes=(2, 3))
    return np.sqrt((np.abs(x) ** 2).sum(0))          # (320 RO, 24 PE, 104 SPE) RSS
imgs = {k: lowres(v) for k, v in want.items()}
np.savez("/tmp/claude-1002/-home-congcongliu--claude/fd861f29-d86a-49cf-be4f-a26d183fe383/scratchpad/mtp_lowres_rss.npz", **{f"s{k[0]}_e{k[1]}_u{k[2]}": v for k, v in imgs.items()})
# ---- DICOM reference at matching low resolution: filter cols (PE) to the 24 central k-lines, keep rows (RO) and Z ----
ROOT = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/dicom/mtp_cartesian_.7x.7x1_M_802"
groups = collections.defaultdict(list)
for fpath in sorted(glob.glob(os.path.join(ROOT, "*.dcm"))):
    d = pydicom.dcmread(fpath); groups[(float(d.FlipAngle), float(d.EchoTime))].append((float(d.ImagePositionPatient[2]), d.pixel_array.astype(np.float32)))
ref = {k: np.stack([a for _, a in sorted(v, key=lambda t: t[0])], 0) for k, v in groups.items()}   # (80 Z, 320 rows=RO, 300 cols=PE)
def ref_lowres(vol):
    K = np.fft.fftshift(np.fft.fft(vol, axis=2), axes=2); c = 150
    lo = np.fft.ifft(np.fft.ifftshift(K[:, :, c - 12:c + 12], axes=2), axis=2)     # 24-line PE resolution, kept on a 24-sample grid
    return np.abs(lo)                                                              # (80, 320, 24)
refl = {k: ref_lowres(v) for k, v in ref.items()}
def corr(a, b): a = a.ravel() - a.mean(); b = b.ravel() - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))
def raw_to_dicom_axes(im):   # raw (320 RO, 24 PE, 104 SPE) -> (Z=80 central SPE, RO, PE)
    return np.transpose(im[:, :, 12:92], (2, 0, 1))
print("\nraw combo -> DICOM match (3D low-res volume correlation, best over axis flips):")
for k, im in imgs.items():
    if im.max() == 0: print("  ", k, "empty"); continue
    a = raw_to_dicom_axes(im); res = []
    for rk, b in refl.items():
        for fz in (1, -1):
            for fr in (1, -1):
                for fp in (1, -1):
                    res.append((corr(a, b[::fz, ::fr, ::fp]), rk, (fz, fr, fp)))
    res.sort(reverse=True)
    other_fa = [x for x in res if x[1][0] != res[0][1][0]][0]
    print(f"  set{k[0]} echo{k[1]} u1={k[2]}: best FA{res[0][1][0]:.0f} TE{res[0][1][1]:.2f} r={res[0][0]:.3f} flips={res[0][2]} | best of the other flip angle: FA{other_fa[1][0]:.0f} TE{other_fa[1][1]:.2f} r={other_fa[0]:.3f} | mean {im.mean():.1f}")
for u in (0, 1):
    a, b = imgs[(0, 0, u)], imgs[(1, 0, u)]
    print(f"u1={u}: corr(set0 echo0, set1 echo0) = {corr(a, b):.4f}, mean ratio set0/set1 = {a.mean() / b.mean():.3f}, max|diff|/max = {np.abs(a - b).max() / b.max():.3f}")
