
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from data.transforms import fft3c, ifft3c
from paired_arc_gs.train import routing as R

dev = torch.device("cuda:0")
torch.manual_seed(0)
D=H=W=32; C=4
ok=[]

# --- build a synthetic scene: a smooth "anatomy" + a localized "change" blob ---
def gauss_blob(center, sigma, amp=1.0):
    zz,yy,xx = torch.meshgrid(torch.arange(D),torch.arange(H),torch.arange(W),indexing="ij")
    r2=((zz-center[0])**2+(yy-center[1])**2+(xx-center[2])**2).float()
    return (amp*torch.exp(-r2/(2*sigma**2))).to(dev).to(torch.complex64)

anat = gauss_blob((16,16,16), 6, 1.0)                 # shared anatomy
change = gauss_blob((8,22,10), 2.0, 0.5)              # localized contrast-specific change
csm = torch.randn(C,D,H,W, dtype=torch.complex64, device=dev)
csm = csm / (csm.abs().pow(2).sum(0,keepdim=True).sqrt()+1e-8)   # normalize SoS=1

x_true = anat + change                                 # the TRUE image (anat+change)
k_full = fft3c(csm * x_true.unsqueeze(0))              # fully-sampled multicoil k-space

# --- undersample + split into fit / audit (disjoint) ---
gseed=torch.Generator().manual_seed(1)
samp = (torch.rand(D,H,W,generator=gseed) < 0.4).to(dev).float()   # 40% sampled
r = torch.rand(D,H,W,generator=gseed).to(dev)
mask_fit   = (samp>0)*(r<0.7); mask_fit=mask_fit.float()
mask_audit = (samp>0)*(r>=0.7); mask_audit=mask_audit.float()
assert float((mask_fit*mask_audit).sum())==0, "fit/audit must be disjoint"
ok.append(("fit/audit disjoint", True))
ksp = k_full * samp.unsqueeze(0)                       # acquired (undersampled) k-space

# --- anatomy-only reconstruction MISSES the change (x_anat = anat, no change) ---
x_anat = anat

# §10.2: evidence from fit residual should light up at the CHANGE location
e = R.fit_residual_evidence(x_anat, csm, ksp, mask_fit, fft3c, ifft3c)
# peak of evidence near the change center (8,22,10)?
peak = torch.nonzero(e==e.max())[0].tolist()
dist_to_change = ((peak[0]-8)**2+(peak[1]-22)**2+(peak[2]-10)**2)**0.5
dist_to_anat   = ((peak[0]-16)**2+(peak[1]-16)**2+(peak[2]-16)**2)**0.5
ok.append((f"evidence peak nearer CHANGE than anatomy (peak={peak}, d_chg={dist_to_change:.1f} vs d_anat={dist_to_anat:.1f})",
           dist_to_change < dist_to_anat))

# §10.3 TRUE-CHANGE test: adding the real change should REDUCE audit error (ΔE>0)
x_before = x_anat
x_after_good = x_anat + change                          # add the true change
dE_good = R.audit_gain(x_before, x_after_good, csm, ksp, mask_audit, fft3c)
ok.append((f"TRUE change: audit gain ΔE>0 (ΔE={dE_good.item():.4g})", dE_good.item() > 0))

# §10.3 ALIASING test: a "change" that only fits FIT samples but is wrong elsewhere.
# construct an artifact = inverse-FFT of residual restricted to FIT set (classic aliasing fit)
r_fit = mask_fit.unsqueeze(0)*(fft3c(csm*x_anat.unsqueeze(0)) - ksp)
alias = (ifft3c(-r_fit)*csm.conj()).sum(0)              # backprojected fit residual as "image"
x_after_alias = x_anat + 0.3*alias
dE_alias = R.audit_gain(x_before, x_after_alias, csm, ksp, mask_audit, fft3c)
# the aliasing fit should help LESS on audit than the true change (ideally <=0 or << true)
ok.append((f"ALIASING change: audit gain << true change (ΔE_alias={dE_alias.item():.4g} vs ΔE_true={dE_good.item():.4g})",
           dE_alias.item() < dE_good.item()))

# proposal returns valid positions in [-1,1]
pos = R.propose_from_evidence(e, 50)
ok.append((f"propose returns positions in [-1,1] (k={pos.shape[0]})",
           pos.shape[0]>0 and float(pos.min())>=-1 and float(pos.max())<=1))

print("="*56)
npass=sum(1 for _,v in ok if v)
for n,v in ok: print(f"  [{'PASS' if v else 'FAIL'}] {n}")
print(f"{npass}/{len(ok)} passed"); print("ALL_PASS" if npass==len(ok) else "SOME_FAIL")
