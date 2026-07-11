
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from data.transforms import fft3c, ifft3c
from paired_arc_gs.train.cg_project import cg_data_consistency, _AHA, _AHy

dev=torch.device("cuda:0"); torch.manual_seed(0)
D=H=W=24; C=6; ok=[]
# synthetic true image + coils
x_true = torch.randn(D,H,W,dtype=torch.complex64,device=dev)
csm = torch.randn(C,D,H,W,dtype=torch.complex64,device=dev)
csm = csm/(csm.abs().pow(2).sum(0,keepdim=True).sqrt()+1e-8)
k_full = fft3c(csm*x_true.unsqueeze(0))
mask = (torch.rand(D,H,W,device=dev)<0.5).float()   # 50% sampled
ksp = k_full*mask.unsqueeze(0)                        # acquired

# a corrupted anchor (model recon that's off)
x_tilde = x_true + 0.5*torch.randn_like(x_true)

def dc_resid(x):  # ‖M(F S x) - y‖ on sampled set
    k=fft3c(csm*x.unsqueeze(0)); return (mask.unsqueeze(0)*(k-ksp)).abs().pow(2).sum().item()

r_tilde = dc_resid(x_tilde)
# CG with moderate eta
x_cg = cg_data_consistency(x_tilde, csm, ksp, mask, fft3c, ifft3c, eta=10.0, n_iters=20)
r_cg = dc_resid(x_cg)
ok.append((f"CG reduces DC residual ({r_tilde:.3f} -> {r_cg:.3f})", r_cg < r_tilde))

# large eta -> should nearly satisfy data on sampled set (residual small vs anchor)
x_cg_hi = cg_data_consistency(x_tilde, csm, ksp, mask, fft3c, ifft3c, eta=1e4, n_iters=40)
r_hi = dc_resid(x_cg_hi)
ok.append((f"large-η CG drives DC residual toward 0 (r={r_hi:.4g} << {r_tilde:.3f})", r_hi < 0.05*r_tilde))

# solution stays finite + same shape/dtype
ok.append(("CG output finite complex volume", torch.isfinite(x_cg.abs()).all().item() and x_cg.shape==x_true.shape and x_cg.is_complex()))

# CG should not blow up the anchor away from truth: on sampled data it's more consistent AND
# closer to x_true than the corrupted anchor (data pulls it back)
err_tilde=(x_tilde-x_true).abs().pow(2).sum().item(); err_cg=(x_cg-x_true).abs().pow(2).sum().item()
ok.append((f"CG moves anchor closer to truth ({err_tilde:.2f} -> {err_cg:.2f})", err_cg < err_tilde))

# vs hard replacement: CG (high eta) should match hard-replacement DC residual roughly (both enforce data)
k_hard = mask.unsqueeze(0)*ksp + (1-mask.unsqueeze(0))*fft3c(csm*x_tilde.unsqueeze(0))
x_hard = (ifft3c(k_hard)*csm.conj()).sum(0)
r_hard = dc_resid(x_hard)
ok.append((f"high-η CG ≈ hard-replacement on DC (CG r={r_hi:.4g}, hard r={r_hard:.4g})", r_hi <= r_hard + 1e-3))

# positive-definiteness sanity: (I+ηAHA) is PD => <v,Mv> > 0
v=torch.randn(D,H,W,dtype=torch.complex64,device=dev)
Mv=v+10.0*_AHA(v,csm,mask,fft3c,ifft3c)
quad=torch.vdot(v.flatten(),Mv.flatten()).real.item()
ok.append((f"(I+ηA^HA) positive-definite (<v,Mv>={quad:.3f}>0)", quad>0))

print("="*56)
n=sum(1 for _,ok_ in ok if ok_)
for name,v in ok: print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{n}/{len(ok)} passed"); print("ALL_PASS" if n==len(ok) else "SOME_FAIL")
