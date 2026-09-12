"""Header-only walk of the MTP Cartesian raw capturing every loop counter, to explain the doubled (PE,SPE) lines."""
import struct, collections, numpy as np, time
fn = "/data0/congcong/data/3D_data/Children_data/TSC_20220124_age16/rawdata/UID_126819142341469_mtp_cartesian_.7x.7x1.raw"
DHL = struct.Struct("<B B B B I Q I I I i i i H H I Q H H H H H H H H H H H 5H H H H H H 3f 4f")
names = ["rep","cph","slc","avg","con","set","shot","seg","pe","spe","uflag","u0","u1","u2","u3","u4"]
recs = []
with open(fn, "rb") as f:
    magic = f.read(12); n = struct.unpack("<I", f.read(4))[0]; f.seek(n + 4 + 12)
    while True:
        hdr = f.read(192)
        if len(hdr) < 192: break
        v = DHL.unpack_from(hdr, 0)
        ucVersion, nsamp, nch, ctrl = v[0], v[12], v[13], v[15]
        if ctrl & 1: break
        if not (ctrl >> 24) & 1:
            flags = dict(ref=(ctrl >> 9) & 1, first_acq=(ctrl >> 1) & 1, last_acq=(ctrl >> 2) & 1, first_slc=(ctrl >> 3) & 1, first_spe=(ctrl >> 11) & 1, user=[(ctrl >> b) & 1 for b in range(48, 54)])
            recs.append(tuple(v[16:32]) + (flags["ref"], v[32], v[33], v[34], v[35], v[36], v[8], v[9]) + tuple(flags["user"]))   # counters..., ref, cuthead, cuttail, ROC, PEC, SPEC, scants, vsmts, user bits
        f.seek(2 * ((nsamp + 2) if ucVersion >= 2 else nsamp) * nch * 4, 1)
cols = names + ["ref","cuthead","cuttail","roc","pec","spec","scan_ts","vsm_ts"] + [f"ub{i}" for i in range(6)]
r = np.array(recs, dtype=np.int64)
print("imaging lines", len(r))
for c in cols:
    u = np.unique(r[:, cols.index(c)])
    print(f"{c:8s} unique {len(u):6d}  {u[:12]}{' ...' if len(u) > 12 else ''}")
print("\nfirst 24 imaging lines in file order (con set shot seg pe spe u0..u4 ref scan_ts):")
for row in r[:24]:
    d = dict(zip(cols, row)); print(" ".join(f"{d[k]}" for k in ["con","set","shot","seg","pe","spe","u0","u1","u2","u3","u4","ref","scan_ts"]))
# how do the two copies of one (set, con, pe, spe) differ?
key = collections.defaultdict(list)
for i, row in enumerate(r):
    d = dict(zip(cols, row)); key[(d["set"], d["con"], d["pe"], d["spe"])].append(i)
diffcols = collections.Counter()
for k, idx in list(key.items()):
    if len(idx) == 2:
        a, b = r[idx[0]], r[idx[1]]
        for c in cols:
            if a[cols.index(c)] != b[cols.index(c)]: diffcols[c] += 1
print("\ncolumns that differ between the two copies of the same (set,con,pe,spe):", dict(diffcols), " pairs:", sum(1 for v in key.values() if len(v) == 2), " singles:", sum(1 for v in key.values() if len(v) == 1))
ex = [k for k, v in key.items() if len(v) == 2][:3]
for k in ex:
    print("example", k)
    for i in key[k]: print("   ", dict(zip(cols, r[i])))
# time structure: which counters change fastest?
print("\nscan timestamps: first", r[:3, cols.index('scan_ts')], "last", r[-3:, cols.index('scan_ts')], "total s", (r[-1, cols.index('scan_ts')] - r[0, cols.index('scan_ts')]) / 1000)
np.savez("/tmp/claude-1002/-home-congcongliu--claude/fd861f29-d86a-49cf-be4f-a26d183fe383/scratchpad/mtp_cart_loops.npz", recs=r, cols=np.array(cols))
