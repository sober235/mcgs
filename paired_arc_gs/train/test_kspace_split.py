"""Standalone test for P1 frequency-selective Θ/Λ split."""
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
import torch
from paired_arc_gs.train.kspace_split import frequency_holdout_split

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ok = []
D, H, W = 64, 64, 8                      # phase axes 0,1 ; readout axis 2 fully sampled
torch.manual_seed(0)
c0, c1 = D // 2, H // 2
ii = torch.arange(D).view(D, 1) - c0
jj = torch.arange(H).view(1, H) - c1
rad = torch.sqrt((ii.float() ** 2 + jj.float() ** 2))
prob = torch.clamp(1.5 / (1.0 + rad), 0, 1)                # denser near center
plane = (torch.rand(D, H) < prob).float()
plane[rad < 8] = 1.0                                       # full ACS disc
mask = plane[:, :, None].expand(D, H, W).contiguous().to(dev)

gen = torch.Generator(device="cpu").manual_seed(123)
theta, lam = frequency_holdout_split(mask, phase_axes=(0, 1), r_acs=8.0, rho=0.3, generator=gen)

sampled = (mask > 0).float()
r3 = rad[:, :, None].expand(D, H, W).to(dev)
ok.append(("disjoint", float((theta * lam).sum()) == 0.0))
ok.append(("union == sampled", torch.equal((theta + lam), sampled)))
ok.append(("ACS never held out", float(lam[r3 < 8.0].sum()) == 0.0))
ok.append(("lambda is high-freq only", float(((lam > 0) & (r3 < 8.0)).sum()) == 0.0))
hf = sampled * (r3 >= 8.0)
frac = float(lam.sum()) / max(1.0, float(hf.sum()))
ok.append(("held-out fraction ~= rho", 0.2 < frac < 0.4))
# per-line holdout: within a phase location Λ is all-or-nothing along the readout axis (axis 2)
ok.append(("Λ holds out whole readout lines", bool((lam.amax(dim=2) == lam.amin(dim=2)).all())))
gen2 = torch.Generator(device="cpu").manual_seed(123)
_, lam2 = frequency_holdout_split(mask, (0, 1), 8.0, 0.3, gen2)
ok.append(("reproducible with same seed", torch.equal(lam, lam2)))

npass = sum(1 for _, v in ok if v)
for name, v in ok:
    print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{npass}/{len(ok)} passed")
print("ALL_PASS" if npass == len(ok) else "SOME_FAIL")
