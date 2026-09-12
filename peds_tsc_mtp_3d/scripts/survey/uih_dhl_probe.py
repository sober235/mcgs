"""Header-only probe of a UIH .raw file: walks every DHL (192 bytes), skips the data payload,
and reports the encoding loop counters, flags and slice geometry. Read-only."""
import sys, struct, collections
import numpy as np

def read_prot_len(f):
    magic = f.read(12)
    if magic != b"RAWDATA-V004":
        f.seek(0); magic = b""
    n = struct.unpack("<I", f.read(4))[0]
    return magic, n

DHL = struct.Struct("<B B B B I Q I I I i i i H H I Q H H H H H H H H H H H 5H H H H H H 3f 4f")  # up to slice info (126 bytes)
assert DHL.size == 126, DHL.size

def probe(fn, max_lines=None):
    with open(fn, "rb") as f:
        magic, n = read_prot_len(f)
        f.seek(n + 4 + (12 if magic else 0))
        rows = []; flags_ct = collections.Counter(); ver = None; samples_ct = collections.Counter(); ch_ct = collections.Counter()
        i = 0
        while True:
            hdr = f.read(192)
            if len(hdr) < 192: print("EOF without ACQUISITION_END"); break
            v = DHL.unpack_from(hdr, 0)
            (ucVersion, _rf, rxgain, _r, dlen, measuid, systs, scants, vsmts, tx, ty, tz, nsamp, nch, _r2, ctrl,
             rep, cph, slc, avg, con, sset, shot, seg, pe, spe, uflag, u0,u1,u2,u3,u4, cuth, cutt, roc, pec, spec,
             px,py,pz, o0,o1,o2,o3) = v
            if ctrl & 1: break
            ver = ucVersion
            noise = bool(ctrl >> 24 & 1); fb = bool(ctrl >> 10 & 1); pc = bool(ctrl >> 7 & 1); ref = bool(ctrl >> 9 & 1); rev = bool(ctrl >> 8 & 1)
            kind = "noise" if noise else "feedback" if fb else "phasecor" if pc else "img"
            flags_ct[(kind, ref)] += 1
            samples_ct[nsamp] += 1; ch_ct[nch] += 1
            if kind == "img":
                rows.append((slc, pe, spe, avg, con, rep, sset, shot, seg, roc, pec, cuth, cutt, rxgain, px, py, pz, o0, o1, o2, o3, ref))
            per_ch = (nsamp + 2) if ucVersion >= 2 else nsamp
            f.seek(2 * per_ch * nch * 4, 1)
            i += 1
            if max_lines and i >= max_lines: break
    print("=" * 90); print(fn.split("/")[-1]); print("DHL version", ver, "lines", i, "flags", dict(flags_ct), "samples", dict(samples_ct), "channels", dict(ch_ct))
    r = np.array(rows, dtype=object)
    slc = r[:, 0].astype(int); pe = r[:, 1].astype(int); spe = r[:, 2].astype(int); avg = r[:, 3].astype(int); con = r[:, 4].astype(int)
    print("imaging lines", len(r), "slices", slc.min(), "..", slc.max(), "(n=%d)" % len(np.unique(slc)), "PE", pe.min(), "..", pe.max(), "SPE", spe.min(), "..", spe.max(), "avg", np.unique(avg), "contrast", np.unique(con), "rep", np.unique(r[:,5].astype(int)), "set", np.unique(r[:,6].astype(int)), "seg", np.unique(r[:,8].astype(int)))
    print("ROCenter", np.unique(r[:, 9].astype(int)), "PECenter", np.unique(r[:, 10].astype(int)), "cutoff head/tail", np.unique(r[:, 11].astype(int)), np.unique(r[:, 12].astype(int)), "rxgain", np.unique(r[:, 13].astype(int)))
    # sampling pattern per slice
    npe_full = pe.max() + 1
    for s in np.unique(slc)[:3]:
        m = slc == s
        lines = np.sort(np.unique(pe[m]))
        ref_lines = np.sort(np.unique(pe[m & (r[:, 21] == True)]))
        print(f"slice {s}: {m.sum()} lines, unique PE {len(lines)} of {npe_full} (AF vs max+1 = {npe_full/len(lines):.2f}); reflines flagged {len(ref_lines)} {ref_lines[:3]}..{ref_lines[-3:] if len(ref_lines) else ''}; duplicates {m.sum()-len(lines)}")
        gaps = np.diff(lines); print("   PE gaps histogram", dict(collections.Counter(gaps.tolist())), " first/last lines", lines[:6], lines[-6:])
    same = all(np.array_equal(np.sort(np.unique(pe[slc == s])), np.sort(np.unique(pe[slc == np.unique(slc)[0]]))) for s in np.unique(slc))
    print("identical PE pattern across slices:", same)
    # geometry per slice
    for s in np.unique(slc)[:2].tolist() + [np.unique(slc)[-1]]:
        m = np.where(slc == s)[0][0]
        print(f"slice {s} pos(sag,cor,tra)=({r[m,14]:.2f},{r[m,15]:.2f},{r[m,16]:.2f}) normal=({r[m,17]:.3f},{r[m,18]:.3f},{r[m,19]:.3f}) inplane_rot={r[m,20]:.2f}")
    pos = np.array([[r[np.where(slc == s)[0][0], k] for k in (14, 15, 16)] for s in np.unique(slc)], dtype=float)
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1); print("slice spacing (mm) from DHL positions:", np.round(d, 2)[:5], "...", "slice order by position:", np.argsort(pos @ np.array([r[0,17], r[0,18], r[0,19]], dtype=float))[:6])

for fn in sys.argv[1:]:
    probe(fn)
