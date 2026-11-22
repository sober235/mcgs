import numpy as np

import mfigures


def _inputs():
    cont = np.zeros(110, dtype=bool)
    cont[40:47] = True
    rng = np.random.default_rng(0)
    z = rng.normal(size=110)
    z[cont] += 9.0
    feas = dict(auc=1.0, auc_history=[dict(it=600, auc=0.71), dict(it=700, auc=0.93), dict(it=4000, auc=1.0)],
                loss=1.2, recover=0.62, b0_c0=36.3, b0=35.1, ow=35.844, n_contaminated=7, n_trains=110,
                pass_auc=True, pass_recover=True, decision="go")
    return feas, dict(m2=dict(z=z.tolist(), settings=dict(z0=4.0, z1=8.0))), dict(contaminated=cont.tolist())


def test_feas_figure_writes_png(tmp_path):
    feas, mm, truth = _inputs()
    out = mfigures.feas_figure(feas, mm, truth, tmp_path / "feas.png")
    assert out.exists() and out.stat().st_size > 20_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_feas_figure_handles_a_failed_gate(tmp_path):
    feas, mm, truth = _inputs()
    feas.update(recover=None, loss=0.04, decision="stop: motion too small to measure", pass_recover=False, ow=None)
    assert mfigures.feas_figure(feas, mm, truth, tmp_path / "f.png").exists()
