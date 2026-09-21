import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ca_summarize import margins


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
