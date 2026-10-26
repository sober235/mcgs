#!/usr/bin/env python3
"""Frozen, strict eight-condition / 59-plane objective; CPU-only and no clipping."""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASELINE_SHA256 = "eb351e158b8029b846b37a45d85bf0b6a99825b326ad5c411f59095d9db0df22"
LABELS = ("ZF", "CG-SENSE", "CS-SENSE", "Joint SENSE", "L1-SPIRiT", "ZS-SSDU", "3D-DIP", "3D-INR", "MC-INR", "GS-single", "MC-GS")
SCANS = ("pre", "post")
METRICS = ("psnr", "ssim", "nmse", "hfen")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value} in {path}")
    with Path(path).open() as stream:
        return json.load(stream, parse_constant=invalid)


def load_baseline(path):
    if sha256_file(path) != BASELINE_SHA256:
        raise ValueError("baseline hash differs from the audited frozen comparison")
    rows = read_json(path)
    validate(rows)
    return rows


def validate(rows):
    if not isinstance(rows, list) or len(rows) != 59:
        raise ValueError("exactly 59 frozen lesion-containing planes required")
    planes = set()
    for row in rows:
        key = (row["axis"], row["index"], row["core"])
        if key in planes or row["axis"] not in (0, 1, 2) or row["core"] < 150:
            raise ValueError(f"invalid/duplicate plane {key}")
        planes.add(key)
        if set(row["scans"]) != set(SCANS):
            raise ValueError("both and only pre/post scans required")
        for scan in SCANS:
            metrics = row["scans"][scan]["metrics"]
            if set(metrics) != set(LABELS):
                raise ValueError("all eleven audited labels required; no omissions or additions")
            for label, values in metrics.items():
                if set(values) != set(METRICS):
                    raise ValueError(f"all four metrics required for {label}")
                for metric, value in values.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"invalid {scan}/{label}/{metric}: {value}")
                    if metric in ("nmse", "hfen") and value <= 0:
                        raise ValueError(f"{metric} must be finite and strictly positive")
                    if metric == "ssim" and not -1 <= value < 1:
                        raise ValueError("SSIM must be in [-1,1) for finite log-error margins")


def _log_ratio(numerator, denominator):
    delta = numerator - denominator
    # log1p preserves strict tiny differences. No epsilon clipping is permitted.
    if abs(delta) < denominator / 2:
        return math.log1p(delta / denominator)
    return math.log(numerator) - math.log(denominator)


def score(candidate, baseline, head_metrics=None):
    validate(baseline)
    validate(candidate)
    planes = []
    for row, frozen in zip(candidate, baseline):
        key = (row["axis"], row["index"], row["core"])
        if key != (frozen["axis"], frozen["index"], frozen["core"]):
            raise ValueError("frozen plane order/index/core changed")
        margins, comparisons, blockers = {}, {}, []
        for scan in SCANS:
            m = row["scans"][scan]["metrics"]
            original = frozen["scans"][scan]["metrics"]
            for label in LABELS[:-1]:
                if m[label] != original[label]:
                    raise ValueError(f"frozen opponent changed at {key}/{scan}/{label}")
            for metric in METRICS:
                vals = {label: m[label][metric] for label in LABELS[:-1]}
                best = (max if metric in ("psnr", "ssim") else min)(vals.values())
                labels = [label for label, value in vals.items() if value == best]
                mc = m["MC-GS"][metric]
                raw_win = mc > best if metric in ("psnr", "ssim") else mc < best
                if metric == "psnr":
                    margin = (mc - best) * math.log(10) / 10
                elif metric == "ssim":
                    margin = _log_ratio(1 - best, 1 - mc)
                else:
                    margin = _log_ratio(best, mc)
                if not math.isfinite(margin) or (margin > 0) != raw_win:
                    raise ValueError("margin numerical precision disagrees with authoritative strict raw comparison")
                name = f"{scan}:{metric}"
                margins[name] = margin
                comparisons[name] = {"candidate": mc, "best_opponent": best, "labels": labels, "strict_win": raw_win}
                if not raw_win:
                    blockers.append({"condition": name, **comparisons[name], "margin": margin})
        planes.append({"axis": key[0], "index": key[1], "core": key[2], "score": min(margins.values()),
                       "strict_success": not blockers, "margins": margins, "comparisons": comparisons, "blockers": blockers})
    best = max(planes, key=lambda row: row["score"])
    success = any(row["strict_success"] for row in planes)
    if (best["score"] > 0) != success:
        raise ValueError("objective/raw-comparison disagreement")
    return {"objective_version": 1, "baseline_sha256": BASELINE_SHA256,
            "score": best["score"], "strict_success": success,
            "witness": best if success else None, "best_plane": best, "planes": planes,
            "whole_head_metrics": head_metrics,
            "interpretation": "S=max_59 min_(pre/post x four metrics) dimensionless margin; raw strict comparisons authoritative. PSNR and NMSE duplicate ordering on each fixed plane. Head-masked lesion-containing planes, not lesion-ROI metrics.",
            "claim_status": "provisional development result; requires improved fair GS-single and fresh repeat"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--baseline", type=Path, default=Path(__file__).parent.parent / "baseline_lesion_allmetrics.json")
    ap.add_argument("--head-metrics", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    baseline = load_baseline(args.baseline)
    head = read_json(args.head_metrics) if args.head_metrics else None
    result = score(read_json(args.candidate), baseline, head)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("score", "strict_success", "baseline_sha256")}))


if __name__ == "__main__":
    main()
