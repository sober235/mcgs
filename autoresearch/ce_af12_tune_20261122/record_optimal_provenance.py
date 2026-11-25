#!/usr/bin/env python3
"""record_optimal_provenance.py -- data, code and parameters of the optimal plain-scan / enhanced result, kept beside
the optimal figure on the server.

User, 2026-11-24 (with the [0, 0.08], text-clear clinical-verification figure): "Yes, remember to record the
corresponding data, codes, and parameters here as the optimal values for both plain scan and enhanced scan."

Writes DIR/provenance/ (DIR = the figure's server folder, v3_err008_textclear):

  data/    mcgs_optimal_volumes.npz      byte copy of the tuned MC-GS output the figure draws: recT1 / recFL = plain scan /
                                         enhanced (magnitude, drawn), *_complex, the truth gt* and zero-filled zf* the
                                         trainer saved beside them; 256 x 232 x 176 each
           mcgs_checkpoint_latest.pth    the trained Gaussians at the end of training (the volumes are rendered from
                                         this state), and mcgs_checkpoint_best.pth, both byte copies
           topic_sim_files.json          path, size and sha256 of every input of the AF12 topic (sim/: truth volumes,
                                         42-coil maps, noisy k-space, AF12 mask)
  code/    trainer_source.tar.gz         source files of the trainer that trained MC-GS (py, yaml, cu, cpp, h, cuh,
                                         sh, md; no build products, no results); trainer_source_files.json lists them
           tune12.py                     the runner, as it ran (job source snapshot of the volumes run)
           fig_af12_prepost.py, fig_lesion_plane_prepost.py, run_af12.py, af12_cmp_support.py, finalize_optimal_fig.py
                                         as they ran (job source snapshots of the figure / finalising jobs)
           comparator_scripts.tar.gz     scripts/ and configs/ of the AF12 topic (ZF, CG-SENSE, CS-SENSE, Joint SENSE,
                                         L1-SPIRiT, ZS-SSDU, MC-INR, GS-single) and scripts/ of the 3D-DIP / 3D-INR
                                         support topic
  params/  params.json                   one file with every setting: MC-GS argv (registered recipe P3 + the five
                                         autoresearch overrides, seed 42), comparator settings, figure command, scores
           mcgs_resolved_config.yaml, mcgs_run_info.json, mcgs_train_score.json, gs_single_*_score.json,
           tune_experiments.jsonl, topic_summary.json, topic_setup.json
  README.md                              what is where

Time rule of the project: no file here carries a clock value or a measured duration. Keys whose name says time /
date / elapsed / duration / epoch are removed from JSON, string values that look like a date with a time are
replaced, and text files with such a value are written in a cleaned form; for every cleaned file the sha256 of the
untouched source is recorded. Tar members get mtime 0 and no owner; the gzip header has no time. The run's training
log and TensorBoard events (clock values) are not copied.

Nothing is overwritten: a file that exists must be byte-identical to what would be written. MANIFEST.json of DIR gets
a "provenance" section. Small files (params.json, README.md, trainer_source_files.json, MANIFEST.json) are copied to
MCGS_ARTIFACT_DIR.

--replace REL (repeatable) allows one generated file to be replaced (old and new sha256 are printed); used once,
for params/params.json, whose first version miscounted the tuning runs.

  gpu.py run --cpu -- @python autoresearch/ce_af12_tune_20261122/record_optimal_provenance.py [--replace params/params.json]
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--replace", action="append", default=[], choices=["params/params.json"])
REPLACE = set(ap.parse_args().replace)
A = Path("/data0/congcong/code/GS/3DGS/outputs/codex_artifacts")
DIR =A / "mcgs_ce_af12_prepost_tuned_20261123/v3_err008_textclear"
PROV = DIR / "provenance"
TR = A / "mcgs_framework_ablation_20261110/trainer"
RUN = TR / "results_arc/ta12_x1combo3_vol"
TUNE = A / "mcgs_ce_af12_tune_20261122"
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af12")
SUPPORT = A / "mcgs_ce_af12_cmp_20261113"
SEPT = Path("/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16/trainer/_trainer_e778bf2/results_arc")
CFG = TOPIC / "configs/ce_retro_af12.yaml"
JOBS = Path("/home/congcongliu/.cache/mcgs-vscode/jobs")
JOB_VOL, JOB_FIG, JOB_FIN = "j-20261124T011133-8fbcb2a0", "j-20261125T022030-76f94229", "j-20261125T024048-863d3d40"
FIG_REC = "record_tuned_MC_GS_AF12_autoresearch_seed_42_three.json"
SRC_EXT = {".py", ".yaml", ".yml", ".cu", ".cpp", ".h", ".cuh", ".sh", ".md"}
TIMEKEY = re.compile(r"(elapsed|time|date|duration|clock|epoch|_utc$|started|finished|created)", re.I)
TIMEVAL = re.compile(r"\d{4}-\d{2}-\d{2}[T _]\d{2}:?\d{2}")
for d in ("data", "code", "params"):
    (PROV / d).mkdir(parents=True, exist_ok=True)
entries = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def note(rel, source, cleaned=False, source_sha=None, what=""):
    p = PROV / rel
    e = dict(file=rel, bytes=p.stat().st_size, sha256=sha(p), source=str(source) if source else None, content=what)
    if cleaned:
        e.update(cleaned=True, source_sha256=source_sha)
    entries.append(e)
    print(f"  {rel:42s} {'cleaned ' if cleaned else ''}<- {source}")


def put_bytes(rel, data):
    p = PROV / rel
    if p.exists() and rel in REPLACE and p.read_bytes() != data:
        print(f"  REPLACE {rel}: old sha256 {sha(p)}, new {sha_b(data)}")
        p.write_bytes(data)
    if p.exists():
        assert p.read_bytes() == data, f"{p} exists with other content; not overwritten"
    else:
        p.write_bytes(data)


def copy(src, rel, what=""):
    src, dst = Path(src), PROV / rel
    assert src.is_file(), src
    if dst.exists():
        assert sha(dst) == sha(src), f"{dst} exists with other content; not overwritten"
    else:
        shutil.copyfile(src, dst)
        assert sha(dst) == sha(src)
    note(rel, src, what=what)


def notime(x, dropped):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if TIMEKEY.search(str(k)):
                dropped.add(str(k))
            else:
                out[k] = notime(v, dropped)
        return out
    if isinstance(x, list):
        return [notime(v, dropped) for v in x]
    if isinstance(x, str) and TIMEVAL.search(x):
        dropped.add("<string value with a date and time>")
        return TIMEVAL.sub("<removed>", x)
    return x


def json_clean(src, rel, what=""):
    """JSON (or JSON lines) without clock values; byte copy when nothing had to be removed"""
    src = Path(src)
    raw = src.read_bytes()
    dropped = set()
    if src.suffix == ".jsonl":
        objs = [notime(json.loads(l), dropped) for l in raw.decode().splitlines() if l.strip()]
        out = "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs).encode()
    else:
        out = json.dumps(notime(json.loads(raw), dropped), indent=2, ensure_ascii=False).encode()
    if not dropped:
        copy(src, rel, what)
        return json.loads(raw) if src.suffix != ".jsonl" else None
    put_bytes(rel, out)
    note(rel, src, cleaned=True, source_sha=sha_b(raw), what=what + f" (removed: {', '.join(sorted(dropped))})")
    return json.loads(out) if src.suffix != ".jsonl" else None


def text_clean(src, rel, what=""):
    src = Path(src)
    raw = src.read_bytes()
    t = raw.decode("utf-8", errors="surrogateescape")
    if not TIMEVAL.search(t):
        copy(src, rel, what)
        return
    put_bytes(rel, TIMEVAL.sub("<removed>", t).encode("utf-8", errors="surrogateescape"))
    note(rel, src, cleaned=True, source_sha=sha_b(raw), what=what + " (date-time strings replaced)")


def det_tar(rel, members, what=""):
    """members: list of (arcname, path); deterministic tar.gz (mtime 0, no owner, mode 644); text with a date-time
    string is cleaned as in text_clean"""
    buf = io.BytesIO()
    listing, cleaned = [], []
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for arc, path in sorted(members, key=lambda m: m[0]):
            data = Path(path).read_bytes()
            t = data.decode("utf-8", errors="surrogateescape")
            if TIMEVAL.search(t):
                cleaned.append(dict(file=arc, source_sha256=sha_b(data)))
                data = TIMEVAL.sub("<removed>", t).encode("utf-8", errors="surrogateescape")
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
            listing.append(dict(file=arc, bytes=len(data), sha256=sha_b(data)))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, filename="") as g:
        g.write(buf.getvalue())
    put_bytes(rel, gz.getvalue())
    note(rel, None, what=what + f" ({len(listing)} files"
         + (f", {len(cleaned)} with date-time strings replaced)" if cleaned else ")"))
    return listing, cleaned


def snap(job, rel_path):
    p = JOBS / job / "code" / rel_path
    assert p.is_file(), p
    return p


def job_argv(job):
    return json.loads((JOBS / job / "job.json").read_text())["argv"]


# ---------------------------------------------------------------- checks: this is the run the figure draws
rec = json.loads((DIR / FIG_REC).read_text())
mc_src = Path(rec["mc_npz"]["path"])
assert mc_src == RUN.parent / "ta12_x1combo3_vol_volumes.npz", mc_src
assert rec["volumes"]["pre"]["MC-GS"]["sha256"] == rec["volumes"]["post"]["MC-GS"]["sha256"] == sha(mc_src), \
    "MC-GS volumes changed since the figure was drawn"
score_src = RUN.parent / "ta12_x1combo3_vol_score.json"
score = json.loads(score_src.read_text())
P3 = ["--iters", "4000", "--lambda_geo", "0.10", "--warmup", "500", "--max_radius", "20", "--kspace_high", "2.2",
      "--edge_tv", "--edge_tv_kappa_q", "0.85", "--edge_tv_wfloor", "0.10", "--tv_eps", "0.04",
      "--set", "gaussian.init_mode=head", "--set", "gaussian.init_scale_mode=3nn",
      "--set", "loss.edge_tv_kappa_source=zf", "--set", "gaussian.initial_num_points=3600000",
      "--set", "loss.wav_name=db4", "--set", "loss.wav_levels=1",
      "--set", "loss.lambda_wav_plain=0.12", "--set", "loss.lambda_wav=0.10",
      "--lambda_edge", "0.001", "--lambda_ptv", "0.03"]
OV = ["--lambda_edge", "0", "--set", "loss.edge_tv_guide_alpha=0.75", "--set", "loss.ptv_guide_alpha=0.5",
      "--set", "gaussian.rotation_lr=0.008", "--set", "gaussian.scale_lr=0.004"]
argv = score["argv"]
core = [a for i, a in enumerate(argv) if not (a in ("--exp", "--gpu", "--base_cfg")
                                              or (i and argv[i - 1] in ("--exp", "--gpu", "--base_cfg")))]
assert core == P3 + ["--save_volumes"] + OV, f"unexpected argv: {argv}"
assert argv[argv.index("--base_cfg") + 1] == str(CFG)
run_info_src = RUN / "ta12_x1combo3_vol/run_info.json"
assert json.loads(run_info_src.read_text())["seed"] == 42

# ---------------------------------------------------------------- data
print("data")
copy(mc_src, "data/mcgs_optimal_volumes.npz",
     "tuned MC-GS output: recT1 = plain scan, recFL = enhanced (float32 magnitude, drawn in the figure), "
     "recT1_complex / recFL_complex, truth gt*, zero-filled zf* (complex64); 256 x 232 x 176")
ck = RUN / "ta12_x1combo3_vol/checkpoints"
copy(ck / "latest.pth", "data/mcgs_checkpoint_latest.pth",
     "trained Gaussians (both scans) at the end of training; the volumes are rendered from this state")
copy(ck / "best.pth", "data/mcgs_checkpoint_best.pth", "the trainer's best.pth of the same run")
sim = []
for p in sorted((TOPIC / "sim").rglob("*")):
    if p.is_file():
        q = p.resolve()
        sim.append(dict(path=str(p.relative_to(TOPIC)), resolves_to=str(q), bytes=q.stat().st_size, sha256=sha(q)))
put_bytes("data/topic_sim_files.json", json.dumps(dict(topic=str(TOPIC), files=sim), indent=2).encode())
note("data/topic_sim_files.json", TOPIC / "sim",
     what=f"inputs of the AF12 topic, {len(sim)} files hashed (truth, coil maps, noisy k-space, mask)")

# ---------------------------------------------------------------- code
print("code")
tr_members = [(str(p.relative_to(TR)), p) for p in TR.rglob("*")
              if p.is_file() and p.suffix in SRC_EXT and "results_arc" not in p.parts
              and "__pycache__" not in p.parts and "build" not in p.parts]
tr_list, tr_clean = det_tar("code/trainer_source.tar.gz", tr_members,
                            f"source of the trainer that trained MC-GS ({TR}; copy of source 54f8a2a)")
put_bytes("code/trainer_source_files.json",
          json.dumps(dict(trainer=str(TR), files=tr_list, cleaned=tr_clean), indent=2).encode())
note("code/trainer_source_files.json", TR, what="file list of trainer_source.tar.gz with sha256")
text_clean(snap(JOB_VOL, "autoresearch/ce_af12_tune_20261122/tune12.py"), "code/tune12.py",
           f"runner of the volumes run, job {JOB_VOL} source snapshot")
for rp in ("autoresearch/ce_af12_rerun_20261112/fig_af12_prepost.py",
           "autoresearch/ce_af16_lesion_figs_20261107/fig_lesion_plane_prepost.py",
           "autoresearch/ce_af12_rerun_20261112/run_af12.py",
           "autoresearch/ce_af12_rerun_20261112/af12_cmp_support.py"):
    text_clean(snap(JOB_FIG, rp), "code/" + Path(rp).name, f"as run in figure job {JOB_FIG}")
text_clean(snap(JOB_FIN, "autoresearch/ce_af12_tune_20261122/finalize_optimal_fig.py"),
           "code/finalize_optimal_fig.py", f"as run in job {JOB_FIN}")
cmp_members = [(f"af12_topic/{p.relative_to(TOPIC)}", p) for d in ("scripts", "configs")
               for p in (TOPIC / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
cmp_members += [(f"dip_inr_support/{p.relative_to(SUPPORT)}", p) for p in (SUPPORT / "scripts").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts]
det_tar("code/comparator_scripts.tar.gz", cmp_members,
        "scripts + configs of the AF12 topic and scripts of the 3D-DIP / 3D-INR support topic")

# ---------------------------------------------------------------- parameters
print("params")
text_clean(RUN / "ta12_x1combo3_vol/config.yaml", "params/mcgs_resolved_config.yaml",
           "the trainer config as resolved for this run")
json_clean(run_info_src, "params/mcgs_run_info.json", "run info (seed, determinism, config snapshot)")
json_clean(score_src, "params/mcgs_train_score.json", "trainer score file of the volumes run")
json_clean(TUNE / "experiments.jsonl", "params/tune_experiments.jsonl", "ledger of the tuning loop")
gs = {}
for p in sorted(SEPT.glob("ce_retro_af12_single_*_score.json")):
    s = p.name.split("_single_")[1].split("_score")[0]
    gs[s] = json_clean(p, f"params/gs_single_{s}_score.json", "GS-single score file (September recipe)")
json_clean(TOPIC / "logs/summary.json", "params/topic_summary.json", "whole-head scores of the 11 methods")
json_clean(TOPIC / "logs/setup.json", "params/topic_setup.json", "topic setup (script port, hashes)")

keys = ("head_psnr", "head_ssim", "nmse", "head_hfen")
params = dict(
    request="user 2026-11-24: record the corresponding data, codes and parameters as the optimal values for both "
            "plain scan and enhanced scan",
    figure_dir=str(DIR), figure_pdf="ce_af12_prepost_clinical_verification_600dpi.pdf",
    data=dict(
        scans="UIH 3D T1 GRE, one subject: plain scan (pre-contrast, series 16887049370010) and enhanced "
              "(post-contrast, 16887049370060); 42 coils; 256 x 232 x 176, 1 mm isotropic, sagittal",
        truth="CS-SENSE reconstruction of the acquired data (prospective AF 5.33) used as ground truth; k-space "
              "re-simulated from it with the 42-coil maps and fitted noise (one draw, the same for every method)",
        mask="BART poisson, seed 0, ACS 24 x 24, no shutter; density 3.000, 13 points trimmed; 3403 of 232 x 176 = "
             "40832 points (nominal AF 12, actual 11.999); one mask for both scans",
        topic=str(TOPIC), topic_inputs="data/topic_sim_files.json", gt_sha256=rec.get("gt_sha256"),
        optimal_volumes="data/mcgs_optimal_volumes.npz (recT1 = plain scan, recFL = enhanced)",
        drawn_volumes={s: {m: dict(path=v["path"], sha256=v["sha256"]) for m, v in d.items()}
                       for s, d in rec["volumes"].items()}),
    mcgs=dict(
        model="MC-GS (joint multi-contrast Gaussian splatting), 3.6 M initial Gaussians, 4000 iterations, seed 42",
        trainer=str(TR), trainer_source="code/trainer_source.tar.gz (copy of source 54f8a2a)",
        entry="arc/cycle.py", config=str(CFG), resolved_config="params/mcgs_resolved_config.yaml",
        recipe_P3_registered=P3, autoresearch_overrides=OV, argv=argv,
        final_values={"lambda_edge": 0.0, "loss.edge_tv_guide_alpha": 0.75, "loss.ptv_guide_alpha": 0.5,
                      "gaussian.rotation_lr": 0.008, "gaussian.scale_lr": 0.004, "lambda_ptv": 0.03,
                      "lambda_geo": 0.10, "loss.lambda_wav_plain": 0.12, "loss.lambda_wav": 0.10,
                      "gaussian.initial_num_points": 3600000, "iters": 4000, "warmup": 500, "seed": 42},
        runner_argv=job_argv(JOB_VOL)[1:], runner_job=JOB_VOL,
        trainer_scores={"plain_scan": {k: score["plain"][k] for k in keys},
                        "enhanced": {k: score["contrast"][k] for k in keys}},
        tuning="9 runs on this scan (ledger params/tune_experiments.jsonl): control, 6 tuning trainings, seed-7 "
               "check of the final config (36.022 / 38.465 dB) and this volumes run; the hyperparameters were "
               "selected by head PSNR against the truth of this same scan"),
    comparators=dict(
        settings={"ZF": "zero-filled, coil-combined",
                  "CG-SENSE": "topic recon_classical.py",
                  "CS-SENSE": "ADMM, 200 outer iterations, lambda 1e-3 (plain scan) / 7e-4 (enhanced), grid choice on the truth",
                  "Joint SENSE": "ADMM, 200 outer iterations, lambda 1e-3, grid choice on the truth",
                  "L1-SPIRiT": "kernel 5, lambda 1e-3, grid choice on the truth",
                  "ZS-SSDU": "topic recon_zsssdu.py, September setting",
                  "3D-DIP": "output support 6, base 96, TV 0.03, 12000 iterations, checkpoint by held-out k-space",
                  "3D-INR": "output support 6, n_max 512, TV 0.03, 1500 iterations, checkpoint by held-out k-space",
                  "MC-INR": "topic run_mcinr.py, September setting",
                  "GS-single": "September registered recipe P3, one scan in both channels of the joint trainer "
                               "(trainer e778bf2)"},
        gs_single_argv={s: g.get("argv") for s, g in gs.items() if g},
        command_lines="code/run_af12.py (queue of the AF12 rerun) and code/af12_cmp_support.py (3D-DIP / 3D-INR)",
        scripts="code/comparator_scripts.tar.gz", whole_head_scores="params/topic_summary.json"),
    figure=dict(argv=job_argv(JOB_FIG)[1:], job=JOB_FIG, finalise_job=JOB_FIN, record=FIG_REC,
                planes={"transverse": 201, "coronal": 131, "sagittal": 48},
                error_display=rec["figures"][0]["error_display"], text_corner=rec["figures"][0].get("text_corner"),
                pdf_image_dpi=rec["figures"][0].get("pdf_image_dpi"),
                panel_numbers=rec.get("numbers"),
                margins_mcgs_over_best_other={str(p["index"]): p["margins"] for p in rec["figures"][0]["planes"]}))
put_bytes("params/params.json", json.dumps(params, indent=2, ensure_ascii=False).encode())
note("params/params.json", None, what="all settings in one file")

readme = f"""# Optimal plain-scan / enhanced result: data, code and parameters

User, 2026-11-24: "record the corresponding data, codes, and parameters here as the optimal values for both plain
scan and enhanced scan". Figure: ../ce_af12_prepost_clinical_verification_600dpi.pdf (MANIFEST.json, section
"provenance", lists the sha256 of every file below).

Optimal values = tuned MC-GS, AF12, seed 42. Trainer scores (head): plain scan PSNR {score['plain']['head_psnr']:.3f} dB,
SSIM {score['plain']['head_ssim']:.4f}; enhanced PSNR {score['contrast']['head_psnr']:.3f} dB, SSIM {score['contrast']['head_ssim']:.4f}.

| folder | file | content |
|---|---|---|
""" + "".join(f"| {Path(e['file']).parent} | {Path(e['file']).name} | {e['content']} |\n" for e in entries) + """
MC-GS = registered recipe P3 plus `--lambda_edge 0 --set loss.edge_tv_guide_alpha=0.75 --set loss.ptv_guide_alpha=0.5
--set gaussian.rotation_lr=0.008 --set gaussian.scale_lr=0.004`, seed 42 (full argv in params/params.json).
Load the volumes: `np.load('data/mcgs_optimal_volumes.npz')['recT1']` (plain scan), `['recFL']` (enhanced).

Caveat: the MC-GS hyperparameters were selected on this scan against its truth; no other subject was tested.
No file here carries a clock value or a measured duration (project rule); cleaned files record the sha256 of the
untouched source.
"""
put_bytes("README.md", readme.encode())
note("README.md", None, what="this overview")

man_p = DIR / "MANIFEST.json"
man = json.loads(man_p.read_text())
man["provenance"] = dict(note="data, code and parameters of the optimal result (user, 2026-11-24)", dir="provenance",
                         files={e["file"]: {k: v for k, v in e.items() if k != "file"} for e in entries})
man_p.write_text(json.dumps(man, indent=2, ensure_ascii=False))
art = Path(os.environ.get("MCGS_ARTIFACT_DIR", "."))
for rel in ("params/params.json", "README.md", "code/trainer_source_files.json"):
    shutil.copyfile(PROV / rel, art / Path(rel).name)
shutil.copyfile(man_p, art / "MANIFEST.json")
tot = sum(e["bytes"] for e in entries)
print(f"PROVENANCE OK: {len(entries)} files, {tot / 1e9:.2f} GB in {PROV}; MANIFEST.json updated")
