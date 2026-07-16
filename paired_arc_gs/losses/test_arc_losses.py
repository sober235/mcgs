"""Unit tests for §12 loss terms: gradient flow + formula correctness."""
import sys, math
# import paired_arc_gs as a PACKAGE (its parent on path), and paired_gs for the shared
# transforms — order matters: paired_arc_gs.losses must not be shadowed by paired_gs/losses.
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.losses.arc_losses import (prewhiten, data_consistency, geometry_reg,
                               change_sparsity, route_exclusivity, phase_reg,
                               deformation_reg, total_loss)
from data.transforms import fft3c

torch.manual_seed(0)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
C, D, H, W = 4, 8, 8, 8
ok = []

# ---- §12.1 DC: gradient + hand-check + division by |Ω_fit| ----
x = torch.randn(D, H, W, dtype=torch.complex64, device=dev, requires_grad=True)
S = torch.randn(C, D, H, W, dtype=torch.complex64, device=dev)
y = torch.randn(C, D, H, W, dtype=torch.complex64, device=dev)
mask = (torch.rand(D, H, W, device=dev) > 0.5).float()
dc = data_consistency(x, S, y, mask, fft3c, L_inv=None)
dc.backward()
grad_finite = torch.isfinite(x.grad).all().item()
# hand reference (no whitening): residual energy over ALL coils at sampled locations,
# divided by |Ω_fit| = number of sampled LOCATIONS (spec §12.1; the mask is per-location,
# shared across coils — NOT coils×locations).
with torch.no_grad():
    km = fft3c(S * x.unsqueeze(0)); m = mask.unsqueeze(0)
    ref = ((m*(km-y)).abs()**2).sum() / mask.sum().clamp_min(1)   # |Ω_fit| = location count
dc_match = abs(float(dc) - float(ref)) < 1e-3
ok.append(("§12.1 DC grad finite", grad_finite))
ok.append(("§12.1 DC matches hand-formula (÷|Ω_fit|)", dc_match))

# ---- pre-whitening actually mixes coils ----
Linv = torch.randn(C, C, dtype=torch.complex64, device=dev)
yt, St = prewhiten(y, S, Linv)
mixes = not torch.allclose(yt, y)
# identity L_inv -> unchanged
I = torch.eye(C, dtype=torch.complex64, device=dev)
yI, _ = prewhiten(y, S, I)
identity_noop = torch.allclose(yI, y, atol=1e-5)
ok.append(("§12.1 prewhiten mixes coils", mixes))
ok.append(("§12.1 prewhiten identity=noop", identity_noop))

# ---- §12.4 geometry: zero inside bounds, positive outside, grad flows ----
cfg = {"geo": {"scale_min": 0.5, "scale_max": 20.0, "lambda_aniso": 0.1, "lambda_overlap": 0.0}}
s_in = torch.full((10, 3), 5.0, device=dev, requires_grad=True)          # inside -> R_scale=0, isotropic -> R_aniso=0
g_in = geometry_reg(s_in, cfg)
s_out = torch.full((10, 3), 100.0, device=dev, requires_grad=True)        # above s_max
g_out = geometry_reg(s_out, cfg); g_out.backward()
ok.append(("§12.4 geo=0 inside bounds+isotropic", abs(float(g_in)) < 1e-5))
ok.append(("§12.4 geo>0 above s_max", float(g_out) > 0))
ok.append(("§12.4 geo grad finite", torch.isfinite(s_out.grad).all().item()))
# anisotropy penalized
s_aniso = torch.tensor([[1.0, 1.0, 50.0]], device=dev)  # in-range but anisotropic (50>20 triggers scale too; use in-range)
s_aniso2 = torch.tensor([[2.0, 2.0, 10.0]], device=dev)
ok.append(("§12.4 aniso term >0 for anisotropic", float(geometry_reg(s_aniso2, cfg)) > 0))

# ---- §12.2 change sparsity: group-L2, drives whole primitive ----
a = torch.randn(15, dtype=torch.complex64, device=dev, requires_grad=True)
rs = change_sparsity(a); rs.backward()
ok.append(("§12.2 change sparsity grad finite", torch.isfinite(a.grad).all().item()))
ok.append(("§12.2 change sparsity = Σ|α|", abs(float(rs) - float(a.detach().abs().sum())) < 1e-3))
empty = change_sparsity(torch.zeros(0, dtype=torch.complex64, device=dev))
ok.append(("§12.2 empty change -> 0", float(empty) == 0.0))

# ---- §12.3 route exclusivity: penalizes overlap, zero when disjoint ----
aA = torch.zeros(D, H, W, device=dev); aA[:4] = 1.0
aD_overlap = torch.zeros(D, H, W, device=dev); aD_overlap[:4] = 1.0
aD_disjoint = torch.zeros(D, H, W, device=dev); aD_disjoint[4:] = 1.0
ok.append(("§12.3 route>0 on overlap", float(route_exclusivity(aA, aD_overlap)) > 0))
ok.append(("§12.3 route=0 disjoint", float(route_exclusivity(aA, aD_disjoint)) == 0.0))

# ---- §9 phase reg: wrap-aware (no blowup across ±π) ----
phi = torch.linspace(-math.pi, math.pi, D*H*W, device=dev).reshape(D, H, W)
e = torch.exp(1j*phi).detach().clone().requires_grad_(True)  # leaf
pr = phase_reg(e); pr.backward()
# compare to naive angle-TV which WOULD spike at wrap; e^{jφ} gradient stays bounded
ok.append(("§9 phase reg finite across ±π", torch.isfinite(pr).item() and float(pr) < 1e6))

# ---- §8 deformation reg ----
u = (torch.randn(D, H, W, 3, device=dev) * 0.01).detach().clone().requires_grad_(True)  # leaf
detJ = torch.ones(D, H, W, device=dev) + 0.01*torch.randn(D, H, W, device=dev)
dr = deformation_reg(u, detJ, lambda_jacobian=1.0); dr.backward()
ok.append(("§8 deformation reg grad finite", torch.isfinite(u.grad).all().item()))

# ---- aggregator: only active terms summed, weights applied ----
terms = {"dc": torch.tensor(2.0, device=dev), "geo": torch.tensor(10.0, device=dev),
         "route": torch.tensor(5.0, device=dev)}
loss, log = total_loss(terms, active=("dc", "geo"), weights={"lambda_geo": 0.1})
expect = 1.0*2.0 + 0.1*10.0   # route excluded (not active)
ok.append(("aggregator active-only + weights", abs(float(loss) - expect) < 1e-5 and "route" not in log))

# ---- P3 cross-contrast LLR: matches explicit per-block svdvals sum; rank-1 on identical ----
from paired_arc_gs.losses.arc_losses import cross_contrast_llr

def _ref_llr(a, b, blk, st):
    D, H, W = a.shape; tot = a.new_zeros(())
    for i in range(0, D - blk + 1, st):
        for j in range(0, H - blk + 1, st):
            for k in range(0, W - blk + 1, st):
                A = a[i:i+blk, j:j+blk, k:k+blk].reshape(-1)
                B = b[i:i+blk, j:j+blk, k:k+blk].reshape(-1)
                M = torch.stack([A, B], dim=1)            # (blk^3, 2)
                tot = tot + torch.linalg.svdvals(M).sum()
    return tot

ap = torch.rand(8, 8, 8, device=dev)                     # small vol, blk4/st4 -> 8 blocks
bp = torch.rand(8, 8, 8, device=dev)
fast = cross_contrast_llr(ap, bp, block=4, stride=4)
ref = _ref_llr(ap, bp, 4, 4)
ok.append(("P3 LLR matches per-block svd sum", abs(float(fast) - float(ref)) / float(ref) < 2e-3))
# identical contrasts -> every block rank-1 -> nuclear == Frobenius (== sqrt(2)*||A||)
same = cross_contrast_llr(ap, ap, block=4, stride=4)
fro = _ref_llr(ap, ap, 4, 4)
ok.append(("P3 LLR rank-1 on identical", abs(float(same) - float(fro)) / float(fro) < 2e-3))
# gradient flows and is finite
xg = torch.rand(8, 8, 8, device=dev, requires_grad=True)
yg = torch.rand(8, 8, 8, device=dev, requires_grad=True)
cross_contrast_llr(xg, yg, block=4, stride=4).backward()
ok.append(("P3 LLR grad finite", torch.isfinite(xg.grad).all().item() and torch.isfinite(yg.grad).all().item()))
# accepts complex input (uses magnitude)
xc_ = torch.randn(8, 8, 8, dtype=torch.complex64, device=dev)
yc_ = torch.randn(8, 8, 8, dtype=torch.complex64, device=dev)
ok.append(("P3 LLR accepts complex", torch.isfinite(cross_contrast_llr(xc_, yc_, block=4, stride=4)).item()))

print("="*60)
npass = sum(1 for _, v in ok if v)
for name, v in ok:
    print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{npass}/{len(ok)} passed")
print("ALL_PASS" if npass == len(ok) else "SOME_FAIL")
