"""Bounded scan-internal patch low-rank magnitude denoising.

There is no learned prior or cross-scan projection. Matching streams one spatial
SSD map at a time; only selected patches for one batch are materialized. Torch
and its CUDA context are initialized on the first nontrivial invocation, never
when importing this module or constructing the denoiser.
"""
from itertools import product
import math
import time

import numpy as np


class PatchLowRankDenoiser:
    """Mean-preserving local patch grouping and singular-value soft shrinkage.

    Distances are mean squared differences of scalar-mean-centered patches.
    Ties retain the self match first, then offsets in lexicographic order.
    CUDA scatter-add may have ordinary floating-point reduction-order variation;
    match selection itself uses deterministic insertion rather than tied top-k.
    """

    def __init__(self, patch_size=5, search_radius=8, search_stride=2,
                 anchor_stride=3, group_size=16, min_group_size=4,
                 distance_factor=3.0, threshold_scale=1.0, batch_size=256,
                 device="cuda:0"):
        for name, value, minimum in (
            ("patch_size", patch_size, 1), ("search_radius", search_radius, 0),
            ("search_stride", search_stride, 1), ("anchor_stride", anchor_stride, 1),
            ("group_size", group_size, 1), ("min_group_size", min_group_size, 1),
            ("batch_size", batch_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            setattr(self, name, int(value))
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        if self.min_group_size > self.group_size:
            raise ValueError("min_group_size must not exceed group_size")
        for name, value in (("distance_factor", distance_factor), ("threshold_scale", threshold_scale)):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, value)
        self.device = str(device)
        self.last_stats = {}

    @staticmethod
    def _box_mean(image, patch_size):
        # Separable valid pooling costs 3*p rather than p**3 operations per voxel.
        import torch.nn.functional as functional
        result = image[None, None]
        for axis in range(3):
            kernel = [1, 1, 1]
            kernel[axis] = patch_size
            result = functional.avg_pool3d(result, tuple(kernel), stride=1)
        return result[0, 0]

    def _match_groups(self, image, sigma):
        """Return anchor top-left coordinates and sorted compact match tables."""
        import torch
        p = self.patch_size
        valid_shape = tuple(n - p + 1 for n in image.shape)
        axes = []
        for n in valid_shape:
            coordinates = list(range(0, n, self.anchor_stride))
            if coordinates[-1] != n - 1:
                coordinates.append(n - 1)
            axes.append(torch.tensor(coordinates, dtype=torch.long, device=image.device))
        anchors = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        count = anchors.shape[0]
        best_distances = torch.full((count, self.group_size), float("inf"), device=image.device)
        best_indices = torch.full((count, self.group_size), -1, dtype=torch.long, device=image.device)
        best_distances[:, 0] = 0
        strides = torch.tensor((valid_shape[1] * valid_shape[2], valid_shape[2], 1),
                               device=image.device, dtype=torch.long)
        anchor_indices = (anchors * strides).sum(dim=1)
        best_indices[:, 0] = anchor_indices
        if self.group_size == 1:
            return anchors, best_indices, best_distances
        patch_means = self._box_mean(image, p)
        column = torch.arange(self.group_size, device=image.device)[None, :]
        gate = self.distance_factor * sigma * sigma
        offsets = range(-self.search_radius, self.search_radius + 1, self.search_stride)
        for delta in product(offsets, repeat=3):
            if delta == (0, 0, 0):
                continue
            # Restrict the two volumes to their overlap before forming one SSD map.
            lower = tuple(max(0, -d) for d in delta)
            upper = tuple(min(n, n - d) for n, d in zip(valid_shape, delta))
            if any(lo >= hi for lo, hi in zip(lower, upper)):
                continue
            selected = torch.ones(count, dtype=torch.bool, device=image.device)
            for axis, (lo, hi) in enumerate(zip(lower, upper)):
                selected &= (anchors[:, axis] >= lo) & (anchors[:, axis] < hi)
            rows = torch.nonzero(selected, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            first = tuple(slice(lo, hi + p - 1) for lo, hi in zip(lower, upper))
            second = tuple(slice(lo + d, hi + d + p - 1) for lo, hi, d in zip(lower, upper, delta))
            difference = image[first] - image[second]
            difference.square_()
            ssd = self._box_mean(difference, p)
            local = anchors[rows]
            sample = tuple(local[:, axis] - lower[axis] for axis in range(3))
            shifted = tuple(local[:, axis] + delta[axis] for axis in range(3))
            here = tuple(local[:, axis] for axis in range(3))
            distances = (ssd[sample] - (patch_means[here] - patch_means[shifted]).square()).clamp_min_(0)
            previous_distances = best_distances[rows]
            distances = torch.where(distances <= gate, distances, float("inf"))
            # right=True retains earlier equal-distance matches, including self.
            insertion = torch.searchsorted(previous_distances.contiguous(), distances[:, None], right=True)
            source = (column - (column > insertion).long()).clamp_min_(0)
            new_distances = previous_distances.gather(1, source)
            new_indices = best_indices[rows].gather(1, source)
            at_insertion = column == insertion
            candidate_indices = anchor_indices[rows] + sum(d * s for d, s in zip(delta, strides))
            best_distances[rows] = torch.where(at_insertion, distances[:, None], new_distances)
            best_indices[rows] = torch.where(at_insertion, candidate_indices[:, None], new_indices)
        return anchors, best_indices, best_distances

    def _shrink_groups(self, groups, sigma):
        """Soft shrink B x P x K matrices without removing their two means."""
        import torch
        means = groups.mean(dim=1, keepdim=True)
        centered = groups - means
        shared = centered.mean(dim=2, keepdim=True)
        residual = centered - shared
        gram = residual.transpose(1, 2) @ residual
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        singular = eigenvalues.clamp_min(0).sqrt()
        threshold = self.threshold_scale * sigma * (math.sqrt(groups.shape[1]) + math.sqrt(groups.shape[2]))
        # Zero singular values must receive zero weight even when threshold is zero.
        weight = torch.where(singular > 0, (1 - threshold / singular.clamp_min(torch.finfo(groups.dtype).tiny)).clamp_min(0), 0)
        shrunk = (residual @ eigenvectors * weight[:, None, :]) @ eigenvectors.transpose(1, 2)
        result = shrunk + shared + means
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite patch low-rank shrinkage result")
        return result

    @staticmethod
    def _cuda_memory(torch, device):
        if device.type != "cuda":
            return {"allocated": None, "reserved": None, "global_peak": None}
        return {"allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "global_peak": int(torch.cuda.max_memory_allocated(device))}

    def __call__(self, crop, sigma):
        started = time.perf_counter()
        if not isinstance(crop, np.ndarray) or crop.dtype != np.float32:
            raise TypeError("crop must be a float32 NumPy array")
        if crop.ndim != 3 or not all(crop.shape):
            raise ValueError("crop must have three nonempty dimensions")
        if not np.isfinite(crop).all() or np.any(crop < 0):
            raise ValueError("crop must contain finite nonnegative magnitudes")
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        stats = {"device": self.device, "sigma": sigma, "shape": list(crop.shape),
                 "anchor_count": 0, "valid_group_count": 0, "fallback_group_count": 0,
                 "insufficient_group_count": 0, "matched_patch_count": 0,
                 "group_size_histogram": {}, "covered_voxels": 0, "uncovered_voxels": crop.size,
                 "max_materialized_groups": 0,
                 "target_rms_difference": 0.0, "target_relative_norm": 0.0,
                 "memory": {f"cuda_{name}_{when}": None for name in ("allocated", "reserved", "global_peak")
                            for when in ("before", "after")}}
        self.last_stats = stats
        if any(n < self.patch_size for n in crop.shape):
            reason = "no_valid_patches"
        elif sigma == 0:
            reason = "zero_sigma"
        elif self.threshold_scale == 0:
            reason = "zero_threshold"
        elif float(crop.min()) == float(crop.max()):
            reason = "constant_crop"
        else:
            reason = None
        if reason is not None:
            stats.update(reason=reason, elapsed_seconds=time.perf_counter() - started)
            return crop.copy()

        import torch
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = self._cuda_memory(torch, device)
        stats["memory"].update({f"cuda_{name}_before": value for name, value in before.items()})
        try:
            with torch.no_grad():
                # Copy CPU input as well: neither views nor in-place work can mutate callers.
                image = torch.tensor(crop, device=device)
                anchors, matches, distances = self._match_groups(image, sigma)
                sizes = torch.isfinite(distances).sum(dim=1)
                histogram = torch.bincount(sizes, minlength=self.group_size + 1).cpu().tolist()
                stats["anchor_count"] = int(anchors.shape[0])
                stats["group_size_histogram"] = {k: int(n) for k, n in enumerate(histogram) if n}
                stats["valid_group_count"] = int((sizes >= self.min_group_size).sum().item())
                stats["fallback_group_count"] = int((sizes < self.min_group_size).sum().item())
                stats["insufficient_group_count"] = stats["fallback_group_count"]
                stats["matched_patch_count"] = int(sizes[sizes >= self.min_group_size].sum().item()) + stats["fallback_group_count"]
                del distances
                p = self.patch_size
                valid_shape = tuple(n - p + 1 for n in crop.shape)
                patch_view = image.unfold(0, p, 1).unfold(1, p, 1).unfold(2, p, 1)
                correction = torch.zeros_like(image).flatten()
                counts = torch.zeros_like(image).flatten()
                patch_axes = torch.arange(p, device=device)
                patch_xyz = torch.stack(torch.meshgrid(patch_axes, patch_axes, patch_axes, indexing="ij"), dim=-1).reshape(-1, 3)
                volume_strides = torch.tensor((crop.shape[1] * crop.shape[2], crop.shape[2], 1), device=device)
                patch_offsets = (patch_xyz * volume_strides).sum(dim=1)

                def add_batch(coordinates, values):
                    base = (coordinates * volume_strides).sum(dim=-1)
                    indices = (base[..., None] + patch_offsets).reshape(-1)
                    counts.scatter_add_(0, indices, torch.ones(indices.shape, device=device, dtype=image.dtype))
                    if values is not None:
                        correction.scatter_add_(0, indices, values.transpose(1, 2).contiguous().reshape(-1))

                for k in range(1, self.group_size + 1):
                    rows = torch.nonzero(sizes == k, as_tuple=False).flatten()
                    for start in range(0, rows.numel(), self.batch_size):
                        batch_rows = rows[start:start + self.batch_size]
                        stats["max_materialized_groups"] = max(stats["max_materialized_groups"], int(batch_rows.numel()))
                        if k < self.min_group_size:
                            add_batch(anchors[batch_rows, None, :], None)
                            continue
                        indices = matches[batch_rows, :k]
                        x = indices // (valid_shape[1] * valid_shape[2])
                        y = (indices // valid_shape[2]) % valid_shape[1]
                        z = indices % valid_shape[2]
                        coordinates = torch.stack((x, y, z), dim=-1)
                        # This is the only patch materialization, bounded by B*K*P.
                        groups = patch_view[x, y, z].reshape(-1, k, p ** 3).transpose(1, 2).contiguous()
                        shrunk = self._shrink_groups(groups, sigma)
                        add_batch(coordinates, shrunk - groups)
                covered = counts > 0
                stats["covered_voxels"] = int(covered.sum().item())
                stats["uncovered_voxels"] = int(crop.size - stats["covered_voxels"])
                output = (image.flatten() + correction / counts.clamp_min(1)).clamp_min(0).reshape(crop.shape)
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError("Nonfinite patch low-rank output")
                result = output.cpu().numpy().astype(np.float32, copy=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            after = self._cuda_memory(torch, device)
            stats["memory"].update({f"cuda_{name}_after": value for name, value in after.items()})
            # Float64 CPU reductions avoid overflow in norm logging for valid float32 input.
            difference = result.astype(np.float64) - crop
            norm = np.linalg.norm(crop.astype(np.float64).ravel())
            stats["target_rms_difference"] = float(np.sqrt(np.mean(difference ** 2)))
            stats["target_relative_norm"] = float(np.linalg.norm(difference.ravel()) / norm) if norm else 0.0
            stats.update(reason="denoised", elapsed_seconds=time.perf_counter() - started)
            return result
        except Exception as error:
            stats.update(reason="error", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
            raise
