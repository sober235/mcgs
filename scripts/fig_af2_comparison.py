#!/usr/bin/env python3
"""Render the AF=2 control-A comparison against the AF=12 working point.

The error panels preserve the project convention: ``jet`` with a fixed
``[0, 0.1]`` range, normalized by the whole-volume ground-truth peak.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paired-arc-gs")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_arc"
OUTPUT_DIR = RESULTS / "diag_artefact"
PNG_PATH = OUTPUT_DIR / "LOOK_AF2_vs_AF12_T1_axial_polished.png"
PDF_PATH = OUTPUT_DIR / "LOOK_AF2_vs_AF12_T1_axial_polished.pdf"
SLICE_INDEX = 160

sys.path.insert(0, str(ROOT))

from diag_artefact.metrics import selectivity  # noqa: E402
from diag_artefact.volumes import brain_mask, load_volumes, wm_mask  # noqa: E402


RUNS = (
    ("tvs4k_e040", r"AF=12  $\lambda=0.10$"),
    ("ctrlA_af2_lam010", r"AF=2  $\lambda=0.10$"),
    ("ctrlA_af2_lam060", r"AF=2  $\lambda=0.60$"),
)

ACADEMIC_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D8D1C7",
    "axes.labelcolor": "#4B5563",
    "axes.linewidth": 0.8,
    "axes.titlelocation": "left",
    "axes.titlesize": 11,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _load_run(tag: str, gt: np.ndarray, wm: np.ndarray) -> dict:
    volumes = load_volumes(str(RESULTS / f"{tag}_volumes.npz"), "T1")
    run_gt = np.abs(volumes["gt"]).astype(np.float64)
    if not np.allclose(run_gt, gt, rtol=0.0, atol=1e-5):
        raise ValueError(f"{tag}: ground truth differs from the working-point volume")

    rec = np.abs(volumes["rec"]).astype(np.float64)
    with (RESULTS / f"{tag}_score.json").open() as handle:
        psnr = float(json.load(handle)["plain"]["psnr"])
    return {
        "rec": rec,
        "psnr": psnr,
        "selectivity": selectivity(rec, gt, wm),
    }


def render() -> tuple[Path, Path]:
    plt.rcParams.update(ACADEMIC_STYLE)

    baseline = load_volumes(str(RESULTS / "tvs4k_e040_volumes.npz"), "T1")
    gt = np.abs(baseline["gt"]).astype(np.float64)
    zf = np.abs(baseline["zf"]).astype(np.float64)
    peak = float(gt.max())
    wm = wm_mask(gt, brain_mask(gt))
    runs = [(label, _load_run(tag, gt, wm)) for tag, label in RUNS]

    def axial(volume: np.ndarray) -> np.ndarray:
        return np.rot90(volume[:, :, SLICE_INDEX])

    fig = plt.figure(figsize=(13.6, 6.7), facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=(1, 1, 1, 1, 0.045),
        left=0.025,
        right=0.965,
        bottom=0.085,
        top=0.84,
        wspace=0.07,
        hspace=0.18,
    )
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)])
    colorbar_axis = fig.add_subplot(grid[1, 4])
    fig.add_subplot(grid[0, 4]).axis("off")

    axes[0, 0].imshow(axial(gt), cmap="gray", vmin=0, vmax=peak)
    axes[0, 0].set_title("Ground truth", fontweight="semibold", pad=7)
    axes[1, 0].imshow(axial(zf), cmap="gray", vmin=0, vmax=peak)
    axes[1, 0].set_title("Zero-filled  ·  AF=12", pad=7)

    error_image = None
    for column, (label, run) in enumerate(runs, start=1):
        axes[0, column].imshow(axial(run["rec"]), cmap="gray", vmin=0, vmax=peak)
        axes[0, column].set_title(
            f"{label}\nPSNR {run['psnr']:.2f} dB",
            fontweight="semibold" if column == 2 else "normal",
            pad=5,
        )
        error_image = axes[1, column].imshow(
            axial(np.abs(run["rec"] - gt)) / peak,
            cmap="jet",
            vmin=0,
            vmax=0.1,
        )
        axes[1, column].set_title(
            rf"$|\mathrm{{recon}}-\mathrm{{GT}}|$  ·  $S={run['selectivity']:.3f}$",
            fontweight="semibold" if column == 2 else "normal",
            pad=7,
        )

    for axis in axes.flat:
        axis.grid(False)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#D8D1C7")
            spine.set_linewidth(0.7)

    assert error_image is not None
    colorbar = fig.colorbar(error_image, cax=colorbar_axis, ticks=np.linspace(0, 0.1, 6))
    colorbar.set_label(r"Normalized error  $|\Delta|/\max(\mathrm{GT})$", labelpad=8)
    colorbar.ax.tick_params(length=2.5)

    fig.suptitle(
        f"T1 reconstruction at AF=12 and AF=2  ·  axial z={SLICE_INDEX}",
        x=0.495,
        y=0.965,
        fontsize=14.5,
        fontweight="semibold",
    )
    fig.text(
        0.495,
        0.905,
        r"AF=2 control A versus the shipped working point; $S$: selectivity (GT = 1.00)",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.495,
        0.025,
        r"Fixed error scale: jet [0, 0.10], normalized by the whole-volume GT peak. "
        r"The $\lambda=0.60$ arm is dilution-scaled.",
        ha="center",
        fontsize=9.2,
        color="#6B7280",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=220)
    fig.savefig(PDF_PATH)
    plt.close(fig)
    return PNG_PATH, PDF_PATH


if __name__ == "__main__":
    for output_path in render():
        print(output_path)
