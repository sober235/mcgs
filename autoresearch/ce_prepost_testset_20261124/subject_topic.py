#!/usr/bin/env python3
"""subject_topic.py -- the AF12 topic of one test subject: mask, simulated k-space, truth, scripts, trainer configs.

The layout and every rule are those of the record subject's topic (Results_20260908/CE_retro_cssense_af12, made from
CE_retro_cssense_af16); the code that defines them is imported or copied from there, not rewritten:
  mask    scripts/make_mask_af16.py of the September topic, imported; only the plane (this subject's ky x kz) and the
          point target round(ky * kz / 12) are set. BART poisson, seed 0, ACS 24 x 24, no shutter, trimmed from the
          outside in to the exact count. On a 232 x 176 plane the result must be the record's AF12 mask, bit for bit.
  k-space scripts/sim_cssense_af16.py of the September topic, its functions imported: sigma from the residual of the
          truth on the prospective mask, y = F(S x_truth) + sigma (N + iN), noise seeds 100 (plain) / 200 (enhanced).
          Gate as there: the best-fit scale of the truth must be 1 (1e-3) and the residual must reproduce the kres
          that the truth run itself reported (1e-3), or nothing is written.
  truth   <subject>/truth*/truth_selection.json (truth_cssense.py): CS-SENSE of all acquired lines at the held-out
          lambda; maps = the maps that run used.
  scripts byte copies of the AF12 topic's scripts (they take the topic from RETRO_TOPIC); sha256 recorded.
  configs the AF12 topic's three trainer configs with the data, map and mask paths and the experiment names replaced.
Writes <subject>/topic_af12 once; an existing topic is only verified. Needs a GPU for the simulation.

  gpu.py run --gpu ID -- @python autoresearch/ce_prepost_testset_20261124/subject_topic.py --subject S --csm_crop 0.8
(the card gpu.py exposes is used as cuda:0)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

RES = Path("/data0/congcong/code/GS/Results_20260908")
S16, A12 = RES / "CE_retro_cssense_af16", RES / "CE_retro_cssense_af12"
OUT = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts/mcgs_ce_prepost_testset_20261124")
ART = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
AF, ACS = 12, 24
SCANS = ("pre", "post")
SCRIPTS = ["t1t2_common.py", "arms.py", "recon_classical.py", "recon_joint_cssense.py", "recon_spirit.py", "recon_zsssdu.py",
           "recon_dip_inr.py", "run_mcinr.py", "extract_gs_volumes.py", "summarize.py", "score_volume.py",
           "diag_artefact/__init__.py", "diag_artefact/headonly.py", "diag_artefact/volumes.py", "diag_artefact/SOURCE.txt"]
CONFIGS = ["ce_retro_af12.yaml", "ce_retro_af12_single_pre.yaml", "ce_retro_af12_single_post.yaml"]
log = lambda *a: print(*a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ap = argparse.ArgumentParser()
ap.add_argument("--subject", required=True)
ap.add_argument("--csm_crop", type=float, default=None)
a = ap.parse_args()
SUB = OUT / "subjects" / a.subject
TRUTH = SUB / ("truth" if a.csm_crop is None else f"truth_crop{a.csm_crop:g}")
T = SUB / "topic_af12"
EXP = f"ce_ts{a.subject}_af12"
sel = json.load(open(TRUTH / "truth_selection.json"))
assert sel["subject"] == a.subject and set(sel["scans"]) == set(SCANS)
if (T / "logs/setup.json").is_file():
    log(f"{T} exists: nothing is written")
    log(json.dumps(json.load(open(T / "logs/setup.json"))["summary"], indent=1))
    raise SystemExit(0)
assert not T.exists(), f"{T} exists without logs/setup.json (incomplete); decide by hand"
assert os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit(), "needs exactly one card from gpu.py --gpu"
dev = torch.device("cuda:0")

# ---- shapes ----
W = {s: Path(sel["scans"][s]["work"]) for s in SCANS}
shape = np.load(W["pre"] / "ksp_full.npy", mmap_mode="r").shape
NC, NX, NY, NZ = shape
assert np.load(W["post"] / "ksp_full.npy", mmap_mode="r").shape == shape
TARGET = round(NY * NZ / AF)

# ---- mask (in memory first; nothing is written before the simulation gate has passed) ----
mk = load(S16 / "scripts/make_mask_af16.py", "mask16")
assert (mk.NX, mk.ACS) == (NX, ACS)
mk.NY, mk.NZ, mk.TARGET = NY, NZ, TARGET
m, info = mk.find_mask()
rn = mk.radius()
assert m.shape == (NY, NZ) and int(m.sum()) == TARGET and mk.acs_block(m).all()
same_as_record = None
if (NY, NZ) == (232, 176):
    same_as_record = bool((np.load(A12 / "sim/mask/mask_af12_pe.npy").astype(bool) == m).all())
    assert same_as_record, "the generator did not reproduce the record's AF12 mask on the same plane"
log(f"mask: plane {NY} x {NZ}, {TARGET} points, AF {NY * NZ / TARGET:.4f}, density {info['acc']:.3f}, trimmed {info['trimmed']}, "
    f"same as the record's mask: {same_as_record}")

# ---- simulation gate, then files ----
sim = load(S16 / "scripts/sim_cssense_af16.py", "sim16")
gate, data = {}, {}
for s in SCANS:
    x = torch.from_numpy(np.load(sel["scans"][s]["truth"])).to(torch.complex64)
    S = torch.from_numpy(np.load(W[s] / "csm.npy")).to(torch.complex64)
    y_meas = torch.from_numpy(np.load(W[s] / "ksp_full.npy")).to(torch.complex64)
    m0 = torch.from_numpy(np.load(W[s] / "mask2d.npy")[0] > 0.5)
    assert tuple(x.shape) == (NX, NY, NZ) and S.shape == y_meas.shape == shape
    alpha, kres, sigma = sim.sigma_from_residual(S, x, y_meas, m0, dev)
    ref = float(sel["scans"][s]["kres_full"])
    log(f"[{s}] alpha {alpha:.5f}, kres {kres:.5f} (truth run {ref:.5f}), sigma {sigma:.5e}")
    assert abs(alpha - 1) < 1e-3, f"[{s}] alpha {alpha} is not 1: the truth is not on the measured scale"
    assert abs(kres - ref) < 1e-3, f"[{s}] kres {kres:.5f} does not reproduce the truth run's {ref:.5f}"
    gate[s] = dict(alpha_re=alpha.real, alpha_im=alpha.imag, kres_on_prospective_mask=kres, kres_ref=ref, sigma=sigma,
                   noise_seed=sim.NOISE_SEED[s], csm=str(W[s] / "csm.npy"), truth=sel["scans"][s]["truth"],
                   lam=sel["scans"][s]["lam"], coils=NC)
    del y_meas
    data[s] = (x, S)

for d in ("sim/gs", "sim/mask", "configs", "scripts/diag_artefact", "logs/claims", "logs/run", "recon", "figures"):
    (T / d).mkdir(parents=True)
np.save(T / "sim/mask/mask_af12_pe.npy", m.astype(np.uint8))
with h5py.File(T / "sim/mask/mask_af12_3d.h5", "w") as f:
    f.create_dataset("mask", data=np.broadcast_to(m[None], (NX, NY, NZ)).astype(np.uint8))
info.update(plane=[NY, NZ], target=TARGET, acs=ACS, generator=str(S16 / "scripts/make_mask_af16.py"),
            generator_sha256=sha(S16 / "scripts/make_mask_af16.py"), same_as_record_af12_mask=same_as_record,
            mask_pe_sha256=sha(T / "sim/mask/mask_af12_pe.npy"), mask_3d_sha256=sha(T / "sim/mask/mask_af12_3d.h5"))
json.dump(info, open(T / "sim/mask/mask_af12.json", "w"), indent=1)

gt_out = {"mask2d": m}
for s in SCANS:
    x, S = data[s]
    y = sim.synth_kspace(S, x, gate[s]["sigma"], np.random.default_rng(sim.NOISE_SEED[s]), dev)
    clean = sim.fft3c(S[:2].to(dev) * x.to(dev)).cpu().numpy()
    sm = float((y[:2] - clean).real.std())
    assert abs(sm / gate[s]["sigma"] - 1) < 0.02, (sm, gate[s]["sigma"])
    with h5py.File(T / f"sim/gs/{s}_kspace.h5", "w") as f:
        f.create_dataset("kspace", data=y)
    with h5py.File(T / f"sim/gs/{s}_csm.h5", "w") as f:
        f.create_dataset("map", data=S.numpy())
    xa = x.abs().numpy()
    gt_out[s] = x.numpy()
    gt_out[f"scale_{s}"] = float(np.quantile(xa, 0.999))
    gate[s].update(sigma_measured_first_two_coils=sm, scale_p999=gt_out[f"scale_{s}"])
    log(f"[{s}] wrote sim/gs/{s}_kspace.h5 and {s}_csm.h5; truth P99.9 {gt_out[f'scale_{s}']:.4f}")
    del y
    data[s] = None
np.savez(T / "sim/gt.npz", **gt_out)
af = NY * NZ / TARGET
json.dump(dict(grid=[NX, NY, NZ], pe_points=TARGET, pe_total=NY * NZ, af=af, scans=gate,
               fft="centred orthonormal (same as t1t2_common.fft3c)", noise_chunk=6, noise_device=str(dev),
               simulator=str(S16 / "scripts/sim_cssense_af16.py"), simulator_sha256=sha(S16 / "scripts/sim_cssense_af16.py")),
          open(T / "sim/meta.json", "w"), indent=1)
json.dump(dict(mask2d_shape=[NY, NZ], mask2d_samples=TARGET, mask2d_AF_rect=af,
               contrasts={s: dict(AF_vs_own_support=af, AF_vs_full_rect=af) for s in SCANS}),
          open(T / "logs/prep_04_export.json", "w"), indent=1)

# ---- scripts: byte copies ----
copied = {}
for n in SCRIPTS:
    shutil.copyfile(A12 / "scripts" / n, T / "scripts" / n)
    shutil.copymode(A12 / "scripts" / n, T / "scripts" / n)
    copied[n] = sha(T / "scripts" / n)
    assert copied[n] == sha(A12 / "scripts" / n)

# ---- trainer configs ----
import yaml  # noqa: E402

old_mask = str(A12 / "sim/mask/mask_af12_3d.h5")
for n in CONFIGS:
    text = (A12 / "configs" / n).read_text()
    body = text
    cnt = {}
    for s in SCANS:
        for kind in ("kspace", "csm"):
            old = str(S16 / f"sim/gs/{s}_{kind}.h5")
            cnt[f"{s}_{kind}"] = body.count(old)
            body = body.replace(old, str(T / f"sim/gs/{s}_{kind}.h5"))
    cnt["mask"] = body.count(old_mask)
    body = body.replace(old_mask, str(T / "sim/mask/mask_af12_3d.h5"))
    cnt["names"] = body.count("ce_retro_af12")
    body = body.replace("ce_retro_af12", EXP)
    assert cnt["mask"] == 2 and sum(cnt[f"{s}_{k}"] for s in SCANS for k in ("kspace", "csm")) == 4, (n, cnt)
    dst = n.replace("ce_retro_af12", EXP)
    head = (f"# {dst} -- test subject {a.subject}, AF12 (subject_topic.py). The record topic's configs/{n} with the k-space,\n"
            f"# map and mask paths and the names replaced; {NC} coils, plane {NY} x {NZ}, {TARGET} points.\n")
    (T / "configs" / dst).write_text(head + body)
    c0, c1 = yaml.safe_load(text), yaml.safe_load(head + body)
    for slot in ("plain", "contrast"):
        for k in ("data_path", "csm_path", "mask_path"):
            assert str(T) in c1["data"][slot][k]
            c0["data"][slot].pop(k), c1["data"][slot].pop(k)
    assert json.loads(json.dumps(c0).replace("ce_retro_af12", EXP)) == c1, f"{n}: differs beyond paths and names"

summary = dict(subject=a.subject, topic=str(T), exp_prefix=EXP, grid=[NX, NY, NZ], coils=NC, points=TARGET, af=af,
               mask_same_as_record=same_as_record, density=info["acc"], trimmed=info["trimmed"],
               lam={s: gate[s]["lam"] for s in SCANS}, sigma={s: gate[s]["sigma"] for s in SCANS},
               kres={s: gate[s]["kres_on_prospective_mask"] for s in SCANS})
json.dump(dict(summary=summary, scripts_sha256=copied, source_scripts=str(A12 / "scripts"), truth_selection=sel,
               job=os.environ.get("MCGS_JOB_ID")), open(T / "logs/setup.json", "w"), indent=1)
(ART / f"setup_{a.subject}.json").write_text(json.dumps(summary, indent=1))
log(json.dumps(summary, indent=1))
log("SETUP OK")
