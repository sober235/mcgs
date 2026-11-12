#!/usr/bin/env python3
"""make_af12.py -- AF12 sampling mask and trainer config for the plain-scan / enhanced pair (2026-11-12).

User 2026-11-12: pause the AF16 plain-scan / enhanced experiments, change the acceleration to 12 and run them
again. The September topic (Results_20260908/CE_retro_cssense_af16) keeps the noisy k-space of the whole grid
(sim/gs/*_kspace.h5); the trainer masks it itself with an external mask. So AF12 needs one new mask and one new
config, nothing else: truth, coil maps, noise draw and scorer stay those of the September topic.

Mask: the same generator as AF16, used unchanged -- the functions of the topic's scripts/make_mask_af16.py are
imported from the topic (read only) and only the point target is replaced. BART `poisson`, seed 0, 24 x 24 ACS, no
elliptical shutter, coarse sweep + bisection on the density parameter, trimmed from the outside in to the exact
count. Target = round(232 * 176 / 12) = 3403 points (40832 / 12 is not an integer; 3403 gives AF 11.999, 3402
would give 12.002).

Writes only below ROOT12 (a new directory; an existing file is never overwritten) and a figure + the json to
MCGS_ARTIFACT_DIR. CPU only:

  gpu.py run --cpu -- @python autoresearch/ce_af12_rerun_20261112/make_af12.py
"""
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np

SEPT = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16")
ROOT12 = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_af12_20261112")
ART = Path(os.environ["MCGS_ARTIFACT_DIR"])
AF = 12

spec = importlib.util.spec_from_file_location("mask16", SEPT / "scripts/make_mask_af16.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)
NX, NY, NZ, ACS = mk.NX, mk.NY, mk.NZ, mk.ACS
assert (NX, NY, NZ, ACS) == (256, 232, 176, 24) and mk.TARGET == NY * NZ // 16
TARGET = round(NY * NZ / AF)
assert TARGET == 3403
mk.TARGET = TARGET                      # the only change to the AF16 generator

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(path, writer):
    assert not path.exists(), f"refusing to overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    writer(tmp)
    os.replace(tmp, path)


m, info = mk.find_mask()
rn = mk.radius()
assert m.shape == (NY, NZ) and int(m.sum()) == TARGET and mk.acs_block(m).all() and m[rn > 1.0].sum() > 0
m16 = np.load(SEPT / "sim/mask/mask_af16_pe.npy").astype(bool)
assert m16.shape == m.shape and int(m16.sum()) == NY * NZ // 16

mdir = ROOT12 / "sim/mask"
write_new(mdir / "mask_af12_pe.npy", lambda t: np.save(open(t, "wb"), m.astype(np.uint8)))


def w3d(t):
    with h5py.File(t, "w") as f:
        f.create_dataset("mask", data=np.broadcast_to(m[None], (NX, NY, NZ)).astype(np.uint8))


write_new(mdir / "mask_af12_3d.h5", w3d)
with h5py.File(mdir / "mask_af12_3d.h5", "r") as f, h5py.File(SEPT / "sim/mask/mask_af16_3d.h5", "r") as g:
    assert list(f.keys()) == list(g.keys()) == ["mask"]
    assert f["mask"].shape == g["mask"].shape and f["mask"].dtype == g["mask"].dtype
    a = f["mask"][...]
    assert (a == m[None].astype(np.uint8)).all()

ver = subprocess.run([mk.BART, "version"], capture_output=True, text=True).stdout.strip()
info.update(plane=[NY, NZ], target=TARGET, acs=ACS, dc=[NY // 2, NZ // 2], bart=ver, bart_path=mk.BART,
            cmd=f"{mk.BART} poisson -Y {NY} -Z {NZ} -y {info['acc']:.3f} -z {info['acc']:.3f} -C {ACS} -v -s {info['seed']}",
            generator=str(SEPT / "scripts/make_mask_af16.py"), generator_sha256=sha(SEPT / "scripts/make_mask_af16.py"),
            shared_with_af16=int((m & m16).sum()), af16_points=int(m16.sum()),
            mask_pe_sha256=sha(mdir / "mask_af12_pe.npy"), mask_3d_sha256=sha(mdir / "mask_af12_3d.h5"))
write_new(mdir / "mask_af12.json", lambda t: json.dump(info, open(t, "w"), indent=1))

# config: the September joint config, text for text, with the mask, the declared factor and the names replaced
src = (SEPT / "configs/ce_retro_af16.yaml").read_text()
old_mask = str(SEPT / "sim/mask/mask_af16_3d.h5")
assert src.count(old_mask) == 2 and src.count("acceleration_factor: 16") == 2 and src.count("ce_retro_af16_joint") == 2
head = ("# ce_retro_af12.yaml -- the September plain-scan / enhanced joint config (ce_retro_af16.yaml) with the AF12 mask\n"
        "# (mcgs_ce_af12_20261112/sim/mask, make_af12.py). Only the mask path, the declared factor and the names differ.\n")
dst = head + (src.replace(old_mask, str(mdir / "mask_af12_3d.h5")).replace("acceleration_factor: 16", "acceleration_factor: 12")
              .replace("ce_retro_af16_joint", "ce_retro_af12_joint"))
write_new(ROOT12 / "configs/ce_retro_af12.yaml", lambda t: Path(t).write_text(dst))
import yaml
c16, c12 = yaml.safe_load(src), yaml.safe_load(dst)
for slot in ("plain", "contrast"):
    b = c12["data"][slot]
    assert b["mask_path"] == str(mdir / "mask_af12_3d.h5") and b["acceleration_factor"] == 12
    for k in ("mask_path", "acceleration_factor"):
        b.pop(k); c16["data"][slot].pop(k)
c12["output"] = c16["output"]
assert c12 == c16, "the AF12 config differs from the AF16 config in more than mask, factor and names"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="compressed")
for a, mm, t in ((ax[0], m16, f"AF16 (September): {int(m16.sum())} points"), (ax[1], m, f"AF12 (new): {TARGET} points, AF {NY * NZ / TARGET:.3f}")):
    a.imshow(mm.T, cmap="gray", origin="lower", interpolation="nearest")
    a.set_title(t); a.set_xlabel("ky (A-P)"); a.set_ylabel("kz (L-R)")
for mm, lab in ((m16, "AF16"), (m, "AF12")):
    e, p = mk.density_profile(mm, rn)
    ax[2].step(e[:-1], p, where="post", label=lab)
ax[2].set_xlabel("normalised radius"); ax[2].set_ylabel("sampled fraction"); ax[2].set_title("density profile"); ax[2].legend()
fig.savefig(ART / "mask_af12_vs_af16.png", dpi=200)
json.dump(info, open(ART / "mask_af12.json", "w"), indent=1)
(ART / "ce_retro_af12.yaml").write_text(dst)

show = {k: v for k, v in info.items() if k != "visited"}
print(json.dumps(show, indent=1))
print(f"wrote {mdir} and {ROOT12 / 'configs/ce_retro_af12.yaml'}")
print("MAKE_AF12 OK")
