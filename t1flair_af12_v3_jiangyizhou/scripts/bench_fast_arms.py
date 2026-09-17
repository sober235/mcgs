#!/usr/bin/env python3
"""bench_fast_arms.py -- reconstruction time of the fast arms on one exclusive card, solver only.

The recorded `seconds` of these arms are not reconstruction times: recon_classical.py times solver + scoring (ZF shows
~90 s, all of it scoring), recon_joint_cssense.py times loading + solver. Here each solver is re-timed with the code of
those scripts (copied verbatim: cg / grad3 / div3 / the ADMM loops, the same iteration counts and the lambdas of record),
loading and scoring timed separately, torch.cuda.synchronize() around every block, on the AF12 data of this topic.
ZF and CG-SENSE are repeated three times (median kept); the ADMM arms run once. Any foreign process on the card during a
block is recorded (`shared`), so a contaminated timing is visible instead of silently wrong.
Output: logs/bench/fast_arms.json and stdout.  Run: bench_fast_arms.py --gpu 7
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--lam_cs", type=float, default=0.0007)       # CS-SENSE (converged TV) lambda of record, both contrasts
ap.add_argument("--lam_joint", type=float, default=0.001)     # Joint SENSE lambda of record
ap.add_argument("--admm_outer", type=int, default=200)        # the "_o200" runs
ap.add_argument("--admm_cg", type=int, default=8)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--cg_iters", type=int, default=60)
ap.add_argument("--tik", type=float, default=0.01)
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import t1t2_common as C
dev = torch.device("cuda:0")
log = lambda *a: print(*a, flush=True)
ME = os.getpid()


def foreign():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(args.gpu)],
                         capture_output=True, text=True).stdout.split()
    return sorted(int(p) for p in out if p.strip().isdigit() and int(p) != ME)


class Timer:
    def __enter__(self):
        torch.cuda.synchronize(); self.f0 = foreign(); self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        torch.cuda.synchronize(); self.s = time.perf_counter() - self.t; self.f1 = foreign()
        self.shared = sorted(set(self.f0) | set(self.f1))


def cg(applyA, b, iters, tol=1e-10):                          # verbatim recon_classical.py / recon_joint_cssense.py
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r.conj() * r).sum().real
    for _ in range(iters):
        Ap = applyA(p)
        a = rs / (p.conj() * Ap).sum().real.clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rs_new = (r.conj() * r).sum().real
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x


def grad3(x):
    return torch.stack([torch.roll(x, -1, 0) - x, torch.roll(x, -1, 1) - x, torch.roll(x, -1, 2) - x], 0)


def div3(g):
    gx, gy, gz = g[0], g[1], g[2]
    return ((torch.roll(gx, 1, 0) - gx) + (torch.roll(gy, 1, 1) - gy) + (torch.roll(gz, 1, 2) - gz))


def cssense(P, lam):                                          # recon_classical.py run("cssense") with admm_outer 200
    b0 = P.sense_adj(P.y)
    x = torch.zeros_like(b0); z = grad3(x); u = torch.zeros_like(z)
    for _ in range(args.admm_outer):
        rhs = b0 + args.mu * div3(z - u)
        x = cg(lambda v: P.AHA(v) + args.mu * div3(grad3(v)), rhs, args.admm_cg)
        v = grad3(x) + u
        nv = torch.sqrt((v.abs() ** 2).sum(0, keepdim=True) + 1e-12)
        z = v * torch.clamp(1.0 - lam / (args.mu * nv), min=0.0)
        u = v - z
    return x


def jcssense(pairs, lam):                                     # recon_joint_cssense.py main loop
    b0 = [P.sense_adj(P.y) for P in pairs]
    x = [torch.zeros_like(b) for b in b0]
    z = [grad3(xi) for xi in x]
    u = [torch.zeros_like(zi) for zi in z]
    for it in range(args.admm_outer):
        for k, P in enumerate(pairs):
            rhs = b0[k] + args.mu * div3(z[k] - u[k])
            x[k] = cg(lambda v, P=P: P.AHA(v) + args.mu * div3(grad3(v)), rhs, args.admm_cg)
        v = [grad3(x[k]) + u[k] for k in range(len(pairs))]
        nv = torch.sqrt(sum((vk.abs() ** 2).sum(0, keepdim=True) for vk in v) + 1e-12)
        shrink = torch.clamp(1.0 - lam / (args.mu * nv), min=0.0)
        for k in range(len(pairs)):
            z[k] = v[k] * shrink
            u[k] = v[k] - z[k]
    return x


res = {"gpu": args.gpu, "device": torch.cuda.get_device_name(0), "settings": vars(args), "rows": []}
def rec(arm, c, s, shared, **kw):
    row = dict(arm=arm, contrast=c, seconds=s, shared=shared, **kw); res["rows"].append(row)
    log(f"  {arm:14s} {c:6s} {s:9.2f} s" + (f"   SHARED with pids {shared}" if shared else "") + (f"  {kw}" if kw else ""))

pairs = {}
for c in ("T1", "FLAIR"):
    with Timer() as t:
        pairs[c] = C.Pair(c, dev)
    rec("load", c, t.s, t.shared, coils=pairs[c].NC, grid=list(pairs[c].G))
    P = pairs[c]
    for arm in ("zf", "cgsense"):
        ss = []
        for rep in range(3):
            with Timer() as t:
                if arm == "zf":
                    x = P.sense_adj(P.y)
                else:
                    b = P.sense_adj(P.y); x = cg(lambda v: P.AHA(v) + args.tik * v, b, args.cg_iters)
            ss.append(t.s); sh = t.shared
        rec(arm, c, statistics.median(ss), sh, reps=ss)
    with Timer() as t:
        x = cssense(P, args.lam_cs)
    rec("cssense_o200", c, t.s, t.shared, lam=args.lam_cs)
    with Timer() as t:
        row = P.score(x, "bench")
    rec("score", c, t.s, t.shared, head_psnr=row["head_psnr"])
with Timer() as t:
    xs = jcssense([pairs["T1"], pairs["FLAIR"]], args.lam_joint)
rec("jcssense_o200", "T1+FLAIR", t.s, t.shared, lam=args.lam_joint)
for c, x in zip(("T1", "FLAIR"), xs):
    rec("jcssense_check", c, 0.0, [], head_psnr=pairs[c].score(x, "bench")["head_psnr"])
p = R / "logs/bench/fast_arms.json"
json.dump(res, open(p, "w"), indent=1); log("wrote", p)
