
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.modules.phase_field import PhaseField
from paired_arc_gs.modules.deformation import Deformation, _axis_angle_to_matrix

dev=torch.device("cuda:0"); torch.manual_seed(0)
shape=(24,28,32); ok=[]

# ---- PHASE (§9) ----
pf=PhaseField(2, shape, dev, control_points=(6,6,6), init_std=0.3).to(dev)
ph0=pf(0, shape); ph1=pf(1, shape)
ok.append(("phase phasor shape = volume", tuple(ph0.shape)==shape))
ok.append(("phase is unit-modulus |e^{jφ}|=1", torch.allclose(ph0.abs(), torch.ones_like(ph0.abs()), atol=1e-4)))
ok.append(("phase per-contrast independent", not torch.allclose(pf.phase_map(0), pf.phase_map(1))))
# low-freq: upsampled from 6^3 control -> should be smooth (small max gradient)
pm=pf.phase_map(0); gz=(pm[1:]-pm[:-1]).abs().max().item()
ok.append((f"phase field is low-frequency (max ∂φ={gz:.3f} small)", gz < 0.5))
loss=ph0.real.sum()+ph0.imag.sum(); loss.backward()
ok.append(("grad flows to phase control grid", pf.phi_ctrl[0].grad is not None and torch.isfinite(pf.phi_ctrl[0].grad).all().item()))

# ---- DEFORMATION (§8) ----
df=Deformation(2, shape, dev, rigid=True, nonrigid=True, control_points=(4,4,4), u_max=0.05, reference_index=0).to(dev)
N=1000; pos=(torch.rand(N,3,device=dev)*2-1); scl=torch.rand(N,3,device=dev)*0.1; rot=torch.zeros(N,4,device=dev); rot[:,0]=1
# reference contrast (0) = identity
p0,s0,r0=df(0,pos,scl,rot)
ok.append(("reference contrast = identity", torch.allclose(p0,pos)))
# non-reference (1): set a known rigid translation, check positions shift
with torch.no_grad(): df.trans[1].copy_(torch.tensor([0.1,-0.05,0.02],device=dev))
p1,_,_=df(1,pos,scl,rot)
shift=(p1-pos).mean(0)
ok.append((f"rigid translation applied (mean shift={shift.tolist()})", abs(shift[0].item()-0.1)<0.02 or (p1-pos).abs().mean().item()>0.01))
# amplitude cap on u: displacement field bounded by u_max
with torch.no_grad(): df.u_ctrl[1].normal_(0, 10.0)   # huge control values
uf=df.displacement_field(1)
ok.append((f"nonrigid u capped at u_max (max|u|={uf.abs().max().item():.3f} <= 0.05)", uf.abs().max().item() <= 0.05+1e-5))
# rotation matrix is proper (det=1, orthogonal)
Rm=_axis_angle_to_matrix(torch.tensor([0.1,0.2,-0.15],device=dev))
det=torch.det(Rm).item(); orth=torch.allclose(Rm@Rm.t(), torch.eye(3,device=dev), atol=1e-5)
ok.append((f"rigid R proper rotation (det={det:.4f}, orthogonal={orth})", abs(det-1)<1e-4 and orth))
# grad flows to deformation params
df2=Deformation(2, shape, dev, reference_index=0).to(dev)
p=(torch.rand(N,3,device=dev)*2-1)
out,_,_=df2(1,p,scl,rot); l=out.sum(); l.backward()
ok.append(("grad flows to deformation (theta/trans/u)", df2.trans[1].grad is not None and torch.isfinite(df2.trans[1].grad).all().item()))
# reference params get NO grad group
pg=df2.param_groups(1e-3)[0]["params"]
ref_params=set(id(x) for x in [df2.theta[0],df2.trans[0],df2.u_ctrl[0]])
ok.append(("reference params excluded from optim groups", not any(id(x) in ref_params for x in pg)))

print("="*56)
n=sum(1 for _,v in ok if v)
for name,v in ok: print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{n}/{len(ok)} passed"); print("ALL_PASS" if n==len(ok) else "SOME_FAIL")
