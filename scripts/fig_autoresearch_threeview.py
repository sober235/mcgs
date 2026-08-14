#!/usr/bin/env python3
"""Fixed-window three-view visual gate for one autoresearch candidate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paired-arc-gs")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTRASTS = {
    "T1W": ("zfT1", "recT1", "gtT1", "plain"),
    "FLAIR": ("zfFL", "recFL", "gtFL", "contrast"),
}


def _load_score(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"score root must be an object: {path}")
    return value


def _slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume, got {volume.shape}")
    if not 0 <= index < volume.shape[axis]:
        raise ValueError(f"slice {index} outside axis {axis} with size {volume.shape[axis]}")
    return np.rot90(np.take(np.abs(volume), index, axis=axis))


def _crop_from_gt(gt_slice: np.ndarray, peak: float, margin: int = 10) -> tuple[slice, slice]:
    points = np.argwhere(gt_slice > 0.02 * peak)
    if not len(points):
        return slice(None), slice(None)
    lo = np.maximum(points.min(axis=0) - margin, 0)
    hi = np.minimum(points.max(axis=0) + margin + 1, gt_slice.shape)
    return slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1]))


def _validate_npz(volumes: np.lib.npyio.NpzFile, label: str) -> None:
    expected = {key for values in CONTRASTS.values() for key in values[:3]}
    if set(volumes.files) != expected:
        raise ValueError(f"{label} unexpected NPZ keys: {volumes.files}")
    for key in expected:
        array = volumes[key]
        if array.shape != (256, 220, 256) or not np.isfinite(array).all():
            raise ValueError(f"{label}.{key} invalid shape/values: {array.shape}")


def _metric_line(score: dict, score_key: str) -> str:
    channel = score[score_key]
    extra = f" · e_unobs {channel['e_unobs']:.4f}" if "e_unobs" in channel else ""
    return (
        f"PSNR {channel['psnr']:.2f} · SSIM {channel['ssim']:.3f} · "
        f"HFEN {channel['hfen']:.3f}{extra}"
    )


def render_contrast(
    name: str,
    baseline: np.lib.npyio.NpzFile,
    candidate: np.lib.npyio.NpzFile,
    baseline_score: dict,
    candidate_score: dict,
    views: tuple[tuple[str, int, int], ...],
    output: Path,
) -> None:
    zf_key, rec_key, gt_key, score_key = CONTRASTS[name]
    zf = candidate[zf_key]
    base = baseline[rec_key]
    cand = candidate[rec_key]
    gt = candidate[gt_key]
    if not np.array_equal(baseline[gt_key], gt):
        raise ValueError(f"{name}: baseline and candidate GT arrays differ")
    if not np.array_equal(baseline[zf_key], zf):
        raise ValueError(f"{name}: baseline and candidate ZF arrays differ")

    peak = float(np.max(gt))
    foreground = gt > 0.01 * peak
    display_max = float(np.quantile(gt[foreground], 0.995))
    columns = (
        "Zero-filled",
        "Same-budget baseline",
        "Candidate",
        "Ground truth",
        "|Baseline − GT|",
        "|Candidate − GT|",
    )

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "black",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(len(views), len(columns), figsize=(18.0, 9.4), facecolor="white")
    error_image = None
    for row, (view_name, axis, index) in enumerate(views):
        slices = [_slice(v, axis, index) for v in (zf, base, cand, gt)]
        crop = _crop_from_gt(slices[-1], peak)
        base_error = np.abs(slices[1] - slices[3]) / max(peak, 1e-12)
        candidate_error = np.abs(slices[2] - slices[3]) / max(peak, 1e-12)
        for col, image in enumerate(slices):
            axes[row, col].imshow(image[crop], cmap="gray", vmin=0, vmax=display_max)
        axes[row, 4].imshow(base_error[crop], cmap="magma", vmin=0, vmax=0.08)
        error_image = axes[row, 5].imshow(candidate_error[crop], cmap="magma", vmin=0, vmax=0.08)
        axes[row, 0].set_ylabel(f"{view_name}\nindex {index}", fontsize=10, color="black")
        for col, axis_obj in enumerate(axes[row]):
            axis_obj.set_xticks([])
            axis_obj.set_yticks([])
            for spine in axis_obj.spines.values():
                spine.set_color("#D8D1C7")
                spine.set_linewidth(0.7)
            if row == 0:
                axis_obj.set_title(columns[col], fontsize=10.5, pad=6, color="black")

    if error_image is None:
        raise RuntimeError("no slices rendered")
    colorbar_axis = fig.add_axes((0.962, 0.33, 0.008, 0.34))
    colorbar = fig.colorbar(error_image, cax=colorbar_axis)
    colorbar.set_label("|reconstruction − GT| / max(GT)")
    base_exp = baseline_score["exp"]
    cand_exp = candidate_score["exp"]
    fig.suptitle(
        f"{name} fixed-window three-view artifact gate · {cand_exp} vs {base_exp}",
        fontsize=13.5,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        f"Baseline: {_metric_line(baseline_score, score_key)}    |    "
        f"Candidate: {_metric_line(candidate_score, score_key)}",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.text(
        0.5,
        0.012,
        "All grayscale panels share one GT-derived whole-volume window; crops are GT-derived. "
        "Error panels share [0, 0.08]. No per-image normalization.",
        ha="center",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.04, right=0.95, top=0.91, bottom=0.055, wspace=0.035, hspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, dpi=220, format=output.suffix.lstrip("."))
        os.link(temporary, output)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--candidate-score", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--sagittal", type=int, default=110)
    parser.add_argument("--coronal", type=int, default=128)
    parser.add_argument("--axial", type=int, default=160)
    args = parser.parse_args()

    outputs = {name: Path(f"{args.out_prefix}_{name}.png") for name in CONTRASTS}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing partial/overwrite render; targets already exist: {existing}")

    baseline = np.load(args.baseline_npz)
    candidate = np.load(args.candidate_npz)
    _validate_npz(baseline, "baseline")
    _validate_npz(candidate, "candidate")
    baseline_score = _load_score(args.baseline_score)
    candidate_score = _load_score(args.candidate_score)
    views = (
        ("Sagittal", 1, args.sagittal),
        ("Coronal", 0, args.coronal),
        ("Axial", 2, args.axial),
    )
    for name in CONTRASTS:
        render_contrast(
            name,
            baseline,
            candidate,
            baseline_score,
            candidate_score,
            views,
            outputs[name],
        )


if __name__ == "__main__":
    main()
