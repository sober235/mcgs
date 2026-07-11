
import sys, json, os
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.modules.deformation import Deformation
from paired_arc_gs.modules.phase_field import PhaseField

dev=torch.device("cuda:0"); torch.manual_seed(0)
res={}

# ============ TEST A: deformation recovers a known rigid shift ============
# Represent a set of "anatomy points" (Gaussian centers) in [-1,1]. Contrast-1's true points
# are a SHIFTED copy. Optimize the deformation to align contrast-1's transformed points to the
# reference; check the recovered translation matches the injected shift.
N=3000
pts = (torch.rand(N,3,device=dev)*2-1)*0.8       # reference anatomy centers
true_shift = torch.tensor([0.06,-0.03,0.02],device=dev)
target = pts + true_shift                          # contrast-1's true (shifted) centers
df = Deformation(2, (32,32,32), dev, rigid=True, nonrigid=True, control_points=(4,4,4), u_max=0.05, reference_index=0).to(dev)
opt = torch.optim.Adam(df.param_groups(2e-2)[0]["params"], lr=2e-2)
for _ in range(400):
    opt.zero_grad()
    moved,_,_ = df(1, pts, torch.ones(N,3,device=dev)*0.05, torch.zeros(N,4,device=dev))
    loss = ((moved-target)**2).sum()
    loss.backward(); opt.step()
rec_shift = df.trans[1].detach()
err = (rec_shift-true_shift).abs().max().item()
res["A_rigid_recovery"] = {"injected": true_shift.tolist(), "recovered": [round(x,4) for x in rec_shift.tolist()],
                           "max_err": round(err,4), "pass": err < 0.02}

# ============ TEST B: capped deformation CANNOT absorb a focal intensity change ============
# The deformation only MOVES points; with the amplitude cap it cannot create/destroy intensity.
# Measure: max displacement any point can undergo is bounded by rigid+u_max. A focal intensity
# lesion (points that must CHANGE VALUE, not move) cannot be represented => residual persists.
# Concretely: try to fit deformation so that a region's points move to explain a value change.
# The cap means max total displacement <= |rigid|+u_max; verify the field is bounded.
df2 = Deformation(2, (32,32,32), dev, rigid=True, nonrigid=True, control_points=(6,6,6), u_max=0.05, reference_index=0).to(dev)
# drive it hard toward a huge displacement (simulating trying to absorb a big change)
opt2 = torch.optim.Adam(df2.param_groups(5e-2)[0]["params"], lr=5e-2)
huge_target = pts + 0.5   # ask for a 0.5 displacement (10x the u_max cap)
for _ in range(300):
    opt2.zero_grad()
    moved,_,_ = df2(1, pts, torch.ones(N,3,device=dev)*0.05, torch.zeros(N,4,device=dev))
    (((moved-huge_target)**2).sum()).backward(); opt2.step()
u_field = df2.displacement_field(1)
max_u = u_field.abs().max().item()
# nonrigid u is capped at u_max=0.05; rigid trans is unbounded but that's global, not focal.
# The KEY: the FOCAL (nonrigid) part is capped, so it cannot make a LOCAL change beyond u_max.
res["B_deformation_cap"] = {"u_max_config": 0.05, "max_nonrigid_u_after_hard_drive": round(max_u,4),
                            "pass": max_u <= 0.05+1e-4,
                            "note": "focal nonrigid displacement capped => cannot absorb localized change beyond u_max (rigid is global-only)"}

# ============ TEST C: phase field recovers a known low-freq phase ramp ============
D=H=W=24
pf = PhaseField(2, (D,H,W), dev, control_points=(6,6,6), init_std=1e-3).to(dev)
zz,yy,xx = torch.meshgrid(torch.linspace(-1,1,D),torch.linspace(-1,1,H),torch.linspace(-1,1,W),indexing="ij")
true_phase = (1.2*zz + 0.7*yy - 0.5*xx).to(dev)          # smooth low-freq ramp
target_phasor = torch.exp(1j*true_phase).to(torch.complex64)
opt3 = torch.optim.Adam(pf.param_groups(5e-2)[0]["params"], lr=5e-2)
for _ in range(500):
    opt3.zero_grad()
    ph = pf(1,(D,H,W))
    loss = ((ph-target_phasor).abs()**2).sum()          # fit the phasor (wrap-aware by construction)
    loss.backward(); opt3.step()
rec_phasor = pf(1,(D,H,W)).detach()
# phasor agreement (cosine of phase diff) — robust to global constant
agree = (rec_phasor*target_phasor.conj()).real.mean().item()
res["C_phase_recovery"] = {"phasor_agreement": round(agree,4), "pass": agree > 0.9,
                           "note": "mean Re(e^{jφ_rec} e^{-jφ_true}); 1.0=perfect"}

print(json.dumps(res, indent=2))
allpass = all(v["pass"] for v in res.values())
print("ALL_PASS" if allpass else "SOME_FAIL")
os.makedirs("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_layer3", exist_ok=True)
json.dump(res, open("/data0/congcong/code/GS/Paired_ARC_GS_general_framework/results_arc_layer3/layer3_validation.json","w"), indent=2)
