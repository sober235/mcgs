"""export_bundle.py copies every registered volume plus the simulation/log/figure inputs; a real run touches tens of
GB, so this exercises the copy/manifest/md5 logic (arms.py's arm_files, put(), --no_raw) against a byte-sized fake
topic instead. Run as a subprocess for the same reason as test_readme_tables.py: no __main__ guard, and a fresh
process shares no sys.modules with the rest of the suite (arms.py's module-level R would otherwise stick to
whichever RETRO_TOPIC an earlier import saw)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
ORDER = ["zf", "cgsense", "cssense_o200", "jcssense", "spirit", "zsssdu", "dip", "inr", "mcinr", "gs_single", "gs_joint"]


def _build_fake_topic(root):
    """A byte-sized topic with one file per place export_bundle.py reads from (summary.json = [] so arm_files
    falls back to the plain <key>_<scan>_complex.npy name for every arm, including the three swept families)."""
    for d in ("recon", "sim/mask", "logs", "figures"):
        (root / d).mkdir(parents=True)
    for scan in ("pre", "post"):
        for key in ORDER:
            np.save(root / f"recon/{key}_{scan}_complex.npy", np.zeros((2, 2, 2), np.complex64))
        np.save(root / f"recon/gt_{scan}_complex.npy", np.zeros((2, 2, 2), np.complex64))
    np.savez(root / "sim/gt.npz", pre=np.zeros((2, 2, 2), np.complex64), post=np.zeros((2, 2, 2), np.complex64))
    json.dump({"note": "fake"}, open(root / "sim/meta.json", "w"))
    np.save(root / "sim/mask/mask_af16_pe.npy", np.zeros((2, 2), np.uint8))
    json.dump({"seed": 0}, open(root / "sim/mask/mask_af16.json", "w"))
    json.dump([], open(root / "logs/summary.json", "w"))
    json.dump({"ok": True}, open(root / "logs/check_sim.json", "w"))
    (root / "figures/mask_af16.png").write_bytes(b"not a real png, just bytes")


def test_help_exits_before_touching_the_topic(tmp_path):
    env = dict(os.environ, RETRO_TOPIC=str(tmp_path))
    p = subprocess.run([sys.executable, str(SCRIPTS / "export_bundle.py"), "--help"], cwd=SCRIPTS, env=env, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and "usage" in p.stdout
    assert list(tmp_path.iterdir()) == []


def test_bundle_copies_every_arm_writes_a_verifiable_manifest_and_respects_no_raw(tmp_path):
    _build_fake_topic(tmp_path)
    expected = sum(1 for p in tmp_path.rglob("*") if p.is_file())
    (tmp_path / "sim/gs").mkdir()
    (tmp_path / "sim/gs/pre_kspace.h5").write_bytes(b"raw k-space --no_raw must not copy this")
    env = dict(os.environ, RETRO_TOPIC=str(tmp_path))
    p = subprocess.run([sys.executable, str(SCRIPTS / "export_bundle.py"), "--no_raw"], cwd=SCRIPTS, env=env, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr

    out = next((tmp_path / "export").iterdir())
    assert not (out / "raw").exists()
    manifest = (out / "MANIFEST.tsv").read_text().splitlines()
    md5s = (out / "md5sums.txt").read_text().splitlines()
    assert len(manifest) - 1 == expected == len(md5s)

    check = subprocess.run(["md5sum", "-c", "md5sums.txt"], cwd=out, capture_output=True, text=True)
    assert check.returncode == 0 and "FAILED" not in check.stdout

    got = {row.split("\t")[0] for row in manifest[1:]}
    assert "volumes/pre/gs_joint_pre_complex.npy" in got and "volumes/post/gs_single_post_complex.npy" in got
    assert "sim/gt.npz" in got and "sim/meta.json" in got and "sim/mask/mask_af16.json" in got
    assert "logs/summary.json" in got and "figures/mask_af16.png" in got
