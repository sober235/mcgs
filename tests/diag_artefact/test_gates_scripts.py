import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


make_gates = _load("make_gates")
eval_gates = _load("eval_gates")


def _head(run, psnr, ssim, hfen, n=0.95):
    d = {"run": run}
    for key in ("plain", "contrast"):
        d[key] = {"head_psnr": psnr, "head_ssim": ssim, "head_hfen": hfen, "grain_ratio": n,
                  "zf": {"head_psnr": psnr - 8, "head_ssim": ssim - 0.1, "head_hfen": hfen + 0.2, "grain_ratio": 1.5}}
    return d


def test_tolerance_is_twice_delta_with_floor():
    ref = _head("ref", 34.605, 0.9000, 0.5000)
    rep = _head("rep", 34.604, 0.9000, 0.5000)          # delta psnr 0.001 -> 2x = 0.002 < floor 0.005
    base = _head("base", 33.9, 0.89, 0.55)
    base_k = {"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}}
    g = make_gates.build_gates(ref, rep, base, base_k)
    p = g["contrasts"]["plain"]
    assert p["name"] == "T1" and g["contrasts"]["contrast"]["name"] == "FLAIR"
    assert p["head_psnr"]["tol"] == pytest.approx(0.005)
    assert p["head_ssim"]["tol"] == pytest.approx(0.0005)
    assert p["head_hfen"]["tol"] == pytest.approx(0.001)
    assert p["head_psnr"]["direction"] == "max" and p["head_hfen"]["direction"] == "min"
    assert p["e_unobs_baseline"] == 0.64
    assert g["grain_ratio_max"] == 1.0


def test_tolerance_grows_with_a_large_repeat_delta():
    ref = _head("ref", 34.60, 0.90, 0.50)
    rep = _head("rep", 34.50, 0.90, 0.50)               # delta 0.1 -> tol 0.2
    g = make_gates.build_gates(ref, rep, _head("b", 33, 0.8, 0.6),
                               {"modalities": {"plain": {"e_unobs": 1}, "contrast": {"e_unobs": 1}}})
    assert g["contrasts"]["plain"]["head_psnr"]["tol"] == pytest.approx(0.2)


def _score(exp, psnr, ssim, hfen, e_unobs, n):
    d = {"exp": exp}
    for key in ("plain", "contrast"):
        d[key] = {"head_psnr": psnr, "head_ssim": ssim, "head_hfen": hfen, "e_unobs": e_unobs, "grain_ratio": n}
    return d


def test_evaluate_passes_and_fails_per_gate():
    g = make_gates.build_gates(_head("ref", 34.6, 0.90, 0.50), _head("rep", 34.6, 0.90, 0.50),
                               _head("b", 33, 0.8, 0.6),
                               {"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}})
    v = eval_gates.evaluate(g, _score("cand", 34.597, 0.9001, 0.5009, 0.60, 0.99))   # inside every tolerance
    c = v["contrasts"]["plain"]
    assert c["head_psnr"]["pass"] and c["head_ssim"]["pass"] and c["head_hfen"]["pass"]
    assert c["e_unobs"]["pass"] and c["grain_ratio"]["pass"]
    assert c["n_pass"] == 5 and c["all_pass"]
    v2 = eval_gates.evaluate(g, _score("bad", 34.59, 0.90, 0.5011, 0.65, 1.01))      # psnr, hfen, e_unobs, N fail
    c2 = v2["contrasts"]["contrast"]
    assert not c2["head_psnr"]["pass"] and c2["head_ssim"]["pass"] and not c2["head_hfen"]["pass"]
    assert not c2["e_unobs"]["pass"] and not c2["grain_ratio"]["pass"]
    assert c2["n_pass"] == 1 and not c2["all_pass"]
    assert v2["run"] == "bad"


def test_cli_roundtrip_and_append(tmp_path):
    ref, rep, base = (tmp_path / n for n in ("ref.json", "rep.json", "base.json"))
    ref.write_text(json.dumps(_head("ref", 34.6, 0.9, 0.5)))
    rep.write_text(json.dumps(_head("rep", 34.6, 0.9, 0.5)))
    base.write_text(json.dumps(_head("base", 33, 0.8, 0.6)))
    bk = tmp_path / "bk.json"
    bk.write_text(json.dumps({"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}}))
    gates = tmp_path / "gates.json"
    make_gates.main(["--ref", str(ref), "--repeat", str(rep), "--baseline", str(base),
                     "--baseline-kspace", str(bk), "--out", str(gates)])
    score = tmp_path / "cand_score.json"
    score.write_text(json.dumps(_score("cand", 34.6, 0.9, 0.5, 0.6, 0.9)))
    verdict = tmp_path / "v.json"
    dec = tmp_path / "decisions.jsonl"
    eval_gates.main(["--gates", str(gates), "--score", str(score), "--out", str(verdict),
                     "--append", str(dec), "--phase", "1", "--note", "toy"])
    v = json.loads(verdict.read_text())
    assert v["contrasts"]["plain"]["all_pass"]
    lines = dec.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["run"] == "cand" and rec["phase"] == "1" and rec["note"] == "toy" and "ts" in rec


def test_make_gates_refuses_to_overwrite_a_pre_registered_file(tmp_path):
    ref, rep, base = (tmp_path / n for n in ("ref.json", "rep.json", "base.json"))
    ref.write_text(json.dumps(_head("ref", 34.6, 0.9, 0.5)))
    rep.write_text(json.dumps(_head("rep", 34.6, 0.9, 0.5)))
    base.write_text(json.dumps(_head("base", 33, 0.8, 0.6)))
    bk = tmp_path / "bk.json"
    bk.write_text(json.dumps({"modalities": {"plain": {"e_unobs": 0.64}, "contrast": {"e_unobs": 0.63}}}))
    gates = tmp_path / "gates.json"
    argv = ["--ref", str(ref), "--repeat", str(rep), "--baseline", str(base), "--baseline-kspace", str(bk), "--out", str(gates)]
    make_gates.main(argv)
    before = gates.read_text()
    with pytest.raises(FileExistsError):
        make_gates.main(argv)
    assert gates.read_text() == before          # the pre-registered file is untouched
