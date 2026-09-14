"""The bars a candidate MC-GS run has to clear (uih_mtp.bars)."""
import sys

import pytest

sys.path.insert(0, "/data0/congcong/code/GS/mcgs_peds_wt")


def _rows():
    """A cut-down logs/rescore.json 'rows' map: one classical arm, the single GS arm, the joint GS arm."""
    def r(method, c, psnr, ssim, hfen):
        return {"method": method, "contrast": c, "head_psnr": psnr, "head_ssim": ssim, "head_hfen": hfen}
    return {"l1espirit|T1w": r("l1espirit", "T1w", 38.830, 0.9674, 0.0756),
            "gs_single_ar|T1w": r("gs_single_ar", "T1w", 38.086, 0.9629, 0.0910),
            "gs_joint_ar|T1w": r("gs_joint_ar", "T1w", 37.855, 0.9618, 0.0922),
            "l1espirit|T2s": r("l1espirit", "T2s", 33.860, 0.9050, 0.1882),
            "gs_single_ar|T2s": r("gs_single_ar", "T2s", 34.175, 0.9052, 0.1855),
            "gs_joint_ar|T2s": r("gs_joint_ar", "T2s", 34.160, 0.9047, 0.1858)}


def test_the_single_gs_arm_counts_as_a_competitor_and_the_joint_one_does_not():
    from uih_mtp.bars import competitor_bars
    bars = competitor_bars(_rows(), ("T1w", "T2s"))
    assert bars["T2s"]["head_psnr"] == (34.175, "gs_single_ar")
    assert bars["T1w"]["head_psnr"] == (38.830, "l1espirit")


def test_the_hfen_bar_is_the_smallest_value_not_the_largest():
    from uih_mtp.bars import competitor_bars
    bars = competitor_bars(_rows(), ("T1w",))
    assert bars["T1w"]["head_hfen"] == (0.0756, "l1espirit")


def test_a_tuned_joint_arm_added_later_is_still_not_a_competitor():
    from uih_mtp.bars import competitor_bars
    rows = _rows()
    rows["gs_joint_p3|T1w"] = {"method": "gs_joint_p3", "contrast": "T1w",
                               "head_psnr": 99.0, "head_ssim": 0.99, "head_hfen": 0.001}
    bars = competitor_bars(rows, ("T1w",))
    assert bars["T1w"]["head_psnr"] == (38.830, "l1espirit")


def test_margins_are_positive_when_the_candidate_wins_the_cell():
    from uih_mtp.bars import competitor_bars, margins
    bars = competitor_bars(_rows(), ("T1w",))
    m = margins({"head_psnr": 39.0, "head_ssim": 0.97, "head_hfen": 0.070}, bars["T1w"])
    assert m["head_psnr"] == pytest.approx(0.170)
    assert m["head_hfen"] == pytest.approx(0.0056)          # lower is better, so bar - value
    assert min(m.values()) > 0
