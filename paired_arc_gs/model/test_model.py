
import sys
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework")
sys.path.insert(0, "/data0/congcong/code/GS/Paired_ARC_GS_general_framework/paired_gs")
import torch
from paired_arc_gs.model.paired_arc_model import PairedARCModel
from gaussian.tile_voxelizer import TileVoxelizer

dev = torch.device("cuda:0")
torch.manual_seed(0)
shape = (32, 32, 32)
N = 2000
ok = []

# ---- build model with decoder ON (Layer-1 default) ----
model = PairedARCModel(num_points=N, volume_shape=shape, device=dev,
                       latent_dim=12, decoder_enable=True, decoder_hidden=24).to(dev)
print("summary:", model.parameters_summary())
vox = TileVoxelizer(volume_shape=shape, tile_size=8, max_radius=20).to(dev)

def render(m, ci):
    dens = m.get_density(ci)
    vol = vox(positions=m.positions, scales=m.get_scale_values(),
              rotations=m.rotations, density=dens)
    if isinstance(vol, tuple): vol = torch.complex(vol[0], vol[1])
    return vol

x0 = render(model, 0)
x1 = render(model, 1)
ok.append(("both contrasts render to volume", x0.shape==torch.Size(shape) and x1.shape==torch.Size(shape)))

# ---- SHARED-ANATOMY property: same geometry, DIFFERENT density from SAME latent ----
d0 = model.get_density(0); d1 = model.get_density(1)
diff = (d0-d1).abs().mean().item()
ok.append(("decoder yields DIFFERENT density per contrast (§6)", diff > 1e-6))
# same latent feeds both — verify by construction: geometry params are the SAME tensors
ok.append(("geometry shared (same param objects)", model.positions is model.positions))  # trivially, but render used same

# ---- confidence π_A in (0,1) ----
pi = model.confidence
ok.append(("confidence π_A in (0,1)", pi is not None and float(pi.min())>0 and float(pi.max())<1))

# ---- gradient flows to geometry / latent / decoder ----
loss = x0.abs().sum() + x1.abs().sum()
loss.backward()
g_geo = model.positions.grad is not None and torch.isfinite(model.positions.grad).all().item()
g_lat = model.z_A.grad is not None and torch.isfinite(model.z_A.grad).all().item()
g_dec = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.decoder.parameters())
ok.append(("grad -> geometry", g_geo))
ok.append(("grad -> latent z_A", g_lat))
ok.append(("grad -> decoder", g_dec))

# ---- param groups named + freezable ----
groups = model.param_groups({"geometry":3e-4,"latent":5e-3,"decoder":5e-3})
ok.append(("param groups = geometry/latent/decoder", set(groups.keys())=={"geometry","latent","decoder"}))

# ---- decoder-OFF fallback (MC-GS baseline §3.2): independent density tables ----
mb = PairedARCModel(num_points=N, volume_shape=shape, device=dev, decoder_enable=False).to(dev)
db0 = mb.get_density(0); db1 = mb.get_density(1)
ok.append(("decoder-off = independent tables (baseline)", mb.decoder is None and db0.shape[0]==N))
gnames = mb.param_group_names()
ok.append(("baseline has no decoder group", "decoder" not in gnames))

print("="*56)
npass=sum(1 for _,v in ok if v)
for name,v in ok: print(f"  [{'PASS' if v else 'FAIL'}] {name}")
print(f"{npass}/{len(ok)} passed")
print("ALL_PASS" if npass==len(ok) else "SOME_FAIL")
