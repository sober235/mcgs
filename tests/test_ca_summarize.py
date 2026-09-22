import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ca_summarize import margins, dose_rows


def test_margins_split_the_published_gain():
    rows = {
        ("joint", 0.001): {"T1": 35.706, "FLAIR": 34.928},
        ("single", 0.001): {"T1": 35.558, "FLAIR": 34.886},
        ("joint", 0.0): {"T1": 35.600, "FLAIR": 34.900},
        ("single", 0.0): {"T1": 35.550, "FLAIR": 34.890},
    }
    m = margins(rows)
    assert abs(m["T1"]["M1"] - 0.148) < 1e-9
    assert abs(m["T1"]["M0"] - 0.050) < 1e-9
    assert abs(m["T1"]["delta"] - 0.098) < 1e-9
    assert abs(m["FLAIR"]["M1"] - 0.042) < 1e-9
    assert abs(m["FLAIR"]["M0"] - 0.010) < 1e-9


def test_margins_reports_a_negative_shared_geometry_margin():
    rows = {
        ("joint", 0.001): {"T1": 35.0, "FLAIR": 34.0},
        ("single", 0.001): {"T1": 34.9, "FLAIR": 34.0},
        ("joint", 0.0): {"T1": 34.8, "FLAIR": 33.9},
        ("single", 0.0): {"T1": 34.9, "FLAIR": 34.0},
    }
    m = margins(rows)
    assert m["T1"]["M0"] < 0
    assert abs(m["T1"]["M0"] + 0.1) < 1e-9


def test_dose_rows_places_margin_only_where_both_arms_ran():
    def cell(psnr):
        return {"head_psnr": psnr, "head_ssim": 0.9, "head_hfen": 0.1, "head_nmse": 0.01}

    dose_cells = {
        ("joint", 0.0): {"T1": cell(35.645), "FLAIR": cell(34.876)},
        ("joint", 0.0003): {"T1": cell(35.670), "FLAIR": cell(34.896)},
        ("joint", 0.001): {"T1": cell(35.706), "FLAIR": cell(34.928)},
        ("single", 0.0): {"T1": cell(35.599), "FLAIR": cell(34.904)},
        ("single", 0.001): {"T1": cell(35.558), "FLAIR": cell(34.886)},
    }
    rows = dose_rows(dose_cells)

    margin_lambdas = sorted({r["lambda_edge"] for r in rows if r["arm"] == "margin"})
    assert margin_lambdas == [0.0, 0.001]  # not 0.0003 -- the single arm was never run there

    m = next(r for r in rows if r["arm"] == "margin" and r["lambda_edge"] == 0.001 and r["contrast"] == "T1")
    assert abs(m["head_psnr"] - 0.148) < 1e-9
