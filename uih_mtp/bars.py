"""What a candidate MC-GS run has to clear to lead this topic's table.

The bar in a cell is the best value any OTHER arm reaches there -- the single-contrast GS arm
included (user decision 2026-09-14: GS-single is a comparison method), every joint GS arm excluded,
since those are the candidate's own family. Lower is better for HFEN, higher for PSNR and SSIM.
"""
CELLS = {"head_psnr": max, "head_ssim": max, "head_hfen": min}
JOINT_PREFIX = "gs_joint"


def competitor_bars(rows, contrasts=("T1w", "T2s"), joint_prefix=JOINT_PREFIX):
    """rows: the {key: row} map of logs/rescore.json -> {contrast: {metric: (bar, arm)}}"""
    out = {}
    for c in contrasts:
        sub = [r for r in rows.values()
               if r.get("contrast") == c and not str(r.get("method", "")).startswith(joint_prefix)]
        if not sub:
            raise ValueError(f"no competitor rows for {c}")
        cell = {}
        for m, pick in CELLS.items():
            best = pick(sub, key=lambda r: r[m])
            cell[m] = (best[m], best["method"])
        out[c] = cell
    return out


def margins(row, bars_of_one_contrast):
    """Signed margins of one row against one contrast's bars; positive means the row leads the cell."""
    out = {}
    for m, (bar, _) in bars_of_one_contrast.items():
        out[m] = row[m] - bar if CELLS[m] is max else bar - row[m]
    return out
