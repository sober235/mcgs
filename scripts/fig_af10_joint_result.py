#!/usr/bin/env python3
"""Render the nominal-AF=10 T1W + FLAIR joint-reconstruction result.

Columns are zero-filled input, hard-DC joint reconstruction, ground truth, and
normalized absolute reconstruction error.  The display window is shared within
each contrast; the error scale is shared across both contrasts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paired-arc-gs")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "results_arc/af10_joint_edgetv_e040_volumes.npz"
DEFAULT_SCORE = ROOT / "results_arc/af10_joint_edgetv_e040_score.json"
DEFAULT_OUT = ROOT / "results_arc/AF10_joint_T1W_FLAIR_result.png"
STYLE = Path(
    "/home/congcongliu/.codex/skills/deepscientist-figure-polish/"
    "assets/deepscientist-academic.mplstyle"
)


def _apply_academic_style() -> None:
    """Matplotlib-compatible mirror of the bundled figure-polish style.

    Matplotlib 3.10 treats the asset's bare hex color values as inline
    comments. Apply the same restrained style through rcParams so rendering is
    warning-free while preserving the skill's visual contract.
    """
    if not STYLE.exists():
        raise FileNotFoundError(f"figure-polish style not found: {STYLE}")
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.edgecolor": "#D8D1C7",
            "axes.labelcolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "xtick.color": "#6B7280",
            "ytick.color": "#6B7280",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _axial(volume: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(np.abs(volume[:, :, index]))


def render(npz_path: Path, score_path: Path, out_path: Path, slice_index: int) -> None:
    _apply_academic_style()

    volumes = np.load(npz_path)
    score = json.loads(score_path.read_text())
    rows = (
        ("T1W", volumes["zfT1"], volumes["recT1"], volumes["gtT1"], score["plain"]),
        ("FLAIR", volumes["zfFL"], volumes["recFL"], volumes["gtFL"], score["contrast"]),
    )

    fig = plt.figure(figsize=(12.8, 6.45), facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=(1, 1, 1, 1, 0.045),
        left=0.045,
        right=0.955,
        bottom=0.105,
        top=0.80,
        wspace=0.07,
        hspace=0.24,
    )
    axes = np.asarray([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(2)])
    colorbar_axis = fig.add_subplot(grid[:, 4])
    column_titles = ("Zero-filled", "AF=10 joint recon", "Ground truth", "|Recon − GT|")
    error_image = None

    for row_index, (name, zf, rec, gt, metrics) in enumerate(rows):
        zf_slice = _axial(zf, slice_index)
        rec_slice = _axial(rec, slice_index)
        gt_slice = _axial(gt, slice_index)
        peak = float(np.max(np.abs(gt)))
        foreground = np.abs(gt) > 0.01 * peak
        display_max = float(np.quantile(np.abs(gt)[foreground], 0.995))
        error = np.abs(rec_slice - gt_slice) / peak
        zf_psnr, zf_ssim = float(metrics["zf"][0]), float(metrics["zf"][1])
        rec_psnr, rec_ssim = float(metrics["psnr"]), float(metrics["ssim"])

        axes[row_index, 0].imshow(zf_slice, cmap="gray", vmin=0, vmax=display_max)
        axes[row_index, 1].imshow(rec_slice, cmap="gray", vmin=0, vmax=display_max)
        axes[row_index, 2].imshow(gt_slice, cmap="gray", vmin=0, vmax=display_max)
        error_image = axes[row_index, 3].imshow(
            error,
            cmap="magma",
            vmin=0,
            vmax=0.10,
        )

        axes[row_index, 0].set_xlabel(f"PSNR {zf_psnr:.2f} dB  ·  SSIM {zf_ssim:.3f}")
        axes[row_index, 1].set_xlabel(
            f"PSNR {rec_psnr:.2f} dB  ·  SSIM {rec_ssim:.3f}\n"
            f"ΔPSNR +{rec_psnr - zf_psnr:.2f} dB"
        )
        axes[row_index, 2].set_xlabel("Reference")
        axes[row_index, 3].set_xlabel(
            f"NMSE {float(metrics['nmse']):.4f}  ·  HFEN {float(metrics['hfen']):.3f}"
        )
        axes[row_index, 0].set_ylabel(name, fontsize=12, fontweight="semibold", labelpad=8)

        for column_index, axis in enumerate(axes[row_index]):
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(
                    column_titles[column_index],
                    fontsize=11.2,
                    fontweight="semibold" if column_index == 1 else "normal",
                    pad=7,
                )
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#D8D1C7")
                spine.set_linewidth(0.7)

    if error_image is None:
        raise RuntimeError("no error image rendered")
    colorbar = fig.colorbar(error_image, cax=colorbar_axis, ticks=np.linspace(0, 0.1, 6))
    colorbar.set_label(r"Normalized error  $|x-\mathrm{GT}|/\max(\mathrm{GT})$", labelpad=8)
    colorbar.ax.tick_params(length=2.5)

    fig.suptitle(
        f"Nominal AF=10 joint reconstruction · T1W + FLAIR · axial z={slice_index}",
        x=0.50,
        y=0.965,
        fontsize=14.2,
        fontweight="semibold",
    )
    fig.text(
        0.50,
        0.895,
        "MC-GS edge-TV · actual net AF 9.87 · 4000 iterations · shared geometry",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.50,
        0.025,
        "Within each contrast: identical 99.5th-percentile GT display window. "
        "Error: shared [0, 0.10] scale normalized by the whole-volume GT peak.",
        ha="center",
        fontsize=9.2,
        color="#6B7280",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--slice", type=int, default=160, dest="slice_index")
    args = parser.parse_args()
    render(args.npz, args.score, args.out, args.slice_index)


if __name__ == "__main__":
    main()
