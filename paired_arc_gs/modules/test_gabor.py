
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.modules.gabor_branch import GaborBranch, _quat_to_R
from data.transforms import fft3c

dev=torch.device("cuda:0"); torch.manual_seed(0)
shape=(32,32,32); ok=[]
gb=GaborBranch(2, shape, dev, max_primitives=100, init_scale_vox=2.0).to(dev)

ok.append(("empty branch renders None", gb.forward(0) is None and gb.count()==0))

# add ONE gabor at center with a HIGH carrier frequency along axis0
pos=torch.tensor([[0.0,0.0,0.0]],device=dev)
xi_hi=torch.tensor([[6.0,0.0,0.0]],device=dev)   # 6 cycles/unit along z
gb.add_primitives(pos, xi=xi_hi)
with torch.no_grad(): gb.alpha[0].copy_(torch.tensor([[1.0,0.0]],device=dev))
vb=gb.forward(0)
ok.append((f"after add: renders complex volume (count={gb.count()})", vb is not None and vb.is_complex() and tuple(vb.shape)==shape))

# §11 KEY PROPERTY: Gabor with carrier ξ shifts spectral energy AWAY from DC.
# Compare to a plain Gaussian (ξ=0) of same envelope. Measure fraction of |FFT| energy in
# the low-freq central cube vs high-freq shell.
gb0=GaborBranch(2, shape, dev, max_primitives=10, init_scale_vox=2.0).to(dev)
gb0.add_primitives(pos, xi=torch.zeros(1,3,device=dev))   # ξ=0 => plain Gaussian
with torch.no_grad(): gb0.alpha[0].copy_(torch.tensor([[1.0,0.0]],device=dev))
vg=gb0.forward(0)
def lowfreq_frac(vol):
    K=fft3c(vol.unsqueeze(0))[0].abs()   # centered FFT magnitude
    D,H,W=K.shape; cz,cy,cx=D//2,H//2,W//2; r=3
    low=K[cz-r:cz+r,cy-r:cy+r,cx-r:cx+r].sum()
    return (low/K.sum()).item()
lf_gauss=lowfreq_frac(vg); lf_gabor=lowfreq_frac(vb)
ok.append((f"§11: Gabor shifts energy OFF DC (low-freq frac: Gaussian={lf_gauss:.3f} >> Gabor={lf_gabor:.3f})",
           lf_gabor < lf_gauss))

# the Gabor spectral peak should be near the carrier freq (shifted by ~xi in cycles)
K=fft3c(vb.unsqueeze(0))[0].abs()
peak=torch.nonzero(K==K.max())[0]; center=torch.tensor([16,16,16],device=dev)
off=(peak-center).abs().float()
ok.append((f"Gabor spectral peak shifted from DC along carrier axis (peak offset={off.tolist()})",
           off[0].item() > 2))   # shifted along axis0 (the carrier axis)

# per-contrast amplitude: different alpha => different render from SAME geometry
with torch.no_grad(): gb.alpha[1].copy_(torch.tensor([[0.0,0.0]],device=dev))
vb1=gb.forward(1)
ok.append(("per-contrast amplitude (contrast1 α=0 => ~0 render)", vb1.abs().max().item() < 1e-5))

# gradient flows to gabor params
gb2=GaborBranch(2, shape, dev, max_primitives=10, init_scale_vox=2.0).to(dev)
gb2.add_primitives(pos, xi=xi_hi)
for p in [gb2.positions, gb2.xi]: p.requires_grad_(True)
gb2.alpha[0].requires_grad_(True)
v=gb2.forward(0); loss=v.abs().sum(); loss.backward()
ok.append(("grad flows to gabor (positions/xi/alpha)",
           gb2.xi.grad is not None and gb2.alpha[0].grad is not None and torch.isfinite(gb2.xi.grad).all().item()))

print("="*56)
n=sum(1 for _,v in ok if v)
for name,v in ok: print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{n}/{len(ok)} passed"); print("ALL_PASS" if n==len(ok) else "SOME_FAIL")
