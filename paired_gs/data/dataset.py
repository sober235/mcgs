"""MRI dataset and paper-faithful mask generation for 3DGSMR."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from .transforms import fft3c, ifft3c
from .mask_three_way_split import split_acquired_mask, save_split_indices

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


class MRIDataset(Dataset):
    # ----------------------------------------------------------------- #
    # M^te isolation flag (Phase 10-02 EA-07 runtime layer)             #
    # ----------------------------------------------------------------- #
    # Class-level (NOT instance-level) flag gating ``get_mask_split('te')``
    # access. Default ``False`` denies all reads. Phase 13 ``final_test``
    # entry must call ``MRIDataset.allow_te_for_final_test()`` before
    # reading mask_split['te'] (and ``disable_te_access()`` in ``finally``
    # to restore isolation). See ``scripts/check_mte_isolation.py`` for
    # the static-grep counterpart.
    _allow_te_test_phase: bool = False

    @classmethod
    def allow_te_for_final_test(cls) -> None:
        """Enable ``get_mask_split('te')`` reads. Phase 13 final_test entry only."""
        cls._allow_te_test_phase = True

    @classmethod
    def disable_te_access(cls) -> None:
        """Restore the default M^te isolation. Pair with ``allow_te_for_final_test``."""
        cls._allow_te_test_phase = False

    def get_mask_split(self, key: str) -> torch.Tensor:
        """Safe accessor for ``self.mask_split[key]`` with M^te isolation.

        Parameters
        ----------
        key : str
            One of ``'tr'`` / ``'fb'`` / ``'te'``.

        Returns
        -------
        torch.Tensor
            ``self.mask_split[key]``.

        Raises
        ------
        KeyError
            If ``key`` is not in ``{'tr', 'fb', 'te'}``.
        AssertionError
            If ``key == 'te'`` and the class-level flag is False (default).
            Phase 13 final_test must call ``allow_te_for_final_test()`` first.
        """
        if key not in {"tr", "fb", "te"}:
            raise KeyError(
                f"unknown split key {key!r}; expected one of tr/fb/te"
            )
        if key == "te" and not type(self)._allow_te_test_phase:
            raise AssertionError(
                "M^te access denied outside final_test phase (EA-07 isolation). "
                "Phase 13 final_test entry must call "
                "MRIDataset.allow_te_for_final_test() before this access."
            )
        return self.mask_split[key]

    def __init__(
        self,
        data_path: str,
        acceleration_factor: int = 4,
        mask_type: str = "stacked_2d_gaussian",
        use_acs: bool = True,
        acs_lines: int = 24,
        csm_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        readout_axis: Optional[int] = None,
        phase_axes: Optional[Tuple[int, int]] = None,
        normalize_kspace: bool = True,
        normalization_percentile: float = 99.9,
        device: str = "cuda:0",
        presampled: bool = False,
        skip_readout_check: bool = False,
        center_kspace_dc: bool = False,   # roll ksp+mask so k-space DC -> array center (algorithm-code alignment)
        # === v2 INF-01/INF-02 additions (additive, default-off, backward compat) ===
        enable_mask_split: bool = False,
        split_ratio: Tuple[float, float, float] = (0.60, 0.25, 0.15),
        mask_split_seed: int = 42,                # 独立 field,与 global --seed 解耦 (CONTEXT 决策 A)
        mask_split_out_dir: Optional[str] = None, # 若提供则 dump split_indices.npz
        measurement_path: Optional[str] = None,   # noise study 2026-09-16: fit THIS k-space; GT/scale stay on data_path
    ):
        super().__init__()
        self.data_path = data_path
        self.acceleration_factor = acceleration_factor
        self.mask_type = mask_type
        self.use_acs = use_acs
        self.acs_lines = acs_lines
        self.csm_path = csm_path
        self.mask_path = mask_path
        self.readout_axis = readout_axis
        self.phase_axes = tuple(phase_axes) if phase_axes is not None else None
        self.normalize_kspace = normalize_kspace
        self.normalization_percentile = normalization_percentile
        self.device = device
        self.normalization_scale = 1.0
        self.presampled = presampled
        self.skip_readout_check = skip_readout_check
        self.center_kspace_dc = bool(center_kspace_dc)
        self._dc_roll: Tuple[int, int, int] = (0, 0, 0)
        # v2 INF-01/INF-02 config (保存 before _load_data so subclasses can introspect)
        self.enable_mask_split = bool(enable_mask_split)
        self.split_ratio: Tuple[float, float, float] = tuple(split_ratio)  # type: ignore[assignment]
        self.mask_split_seed = int(mask_split_seed)
        self.mask_split_out_dir = mask_split_out_dir
        self.mask_split: Dict[str, torch.Tensor] = {}
        self.measurement_path = measurement_path if measurement_path not in (None, "", "None") else None
        self._load_data()
        if self.enable_mask_split:
            self._build_mask_split()

    def _load_data(self):
        print(f"Loading data from {self.data_path}")
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == ".mat":
            kspace_data = self._load_mat_kspace()
        else:
            kspace_data = self._load_h5_kspace()

        kspace_data = self._normalize_kspace_layout(kspace_data)
        self.kspace_multicoil = torch.from_numpy(kspace_data).to(torch.complex64)
        self.num_coils = self.kspace_multicoil.shape[0]
        self.volume_shape = tuple(self.kspace_multicoil.shape[1:])
        self._normalize_axis_config()
        if self.center_kspace_dc:
            self._center_kspace_dc_to_array_center()

        print(f"Loaded k-space data with shape: {self.kspace_multicoil.shape}")
        print(f"Number of coils: {self.num_coils}")
        print(f"Volume shape: {self.volume_shape}")
        print(f"Readout axis: {self.readout_axis}, Phase axes: {self.phase_axes}")

        self._process_data()

    def _center_kspace_dc_to_array_center(self):
        """Roll the multicoil k-space so its DC (energy peak) lands on the array
        geometric center, and record the shift so the external mask is rolled to
        match (see _generate_mask). The shipped data's DC is offset (~+22px on a PE
        axis); the winner compensates by rolling only the *mask* to the DC, leaving
        the array-centered frequency-adaptive loss weight miscentered and an
        image-domain phase ramp for the Gaussians to absorb. Centering the data once
        removes both. Physics is unchanged: retained energy is invariant (mask is
        rolled by the same shift) and |GT| is invariant (a k-space roll is a pure
        image-domain phase modulation); only the phase ramp is removed."""
        energy = (self.kspace_multicoil.real.pow(2) + self.kspace_multicoil.imag.pow(2)).sum(dim=0)
        dc = np.unravel_index(int(torch.argmax(energy).item()), tuple(energy.shape))
        center = tuple(s // 2 for s in energy.shape)
        shift = tuple(int(center[a] - int(dc[a])) for a in range(3))
        self.kspace_multicoil = torch.roll(self.kspace_multicoil, shifts=shift, dims=(1, 2, 3)).contiguous()
        self._dc_roll = shift
        print(f"[center_kspace_dc] k-space DC {tuple(int(x) for x in dc)} -> center {center}; rolled ksp by {shift}")

    def _normalize_axis_config(self):
        n_dims = len(self.volume_shape)
        if self.readout_axis is None:
            self.readout_axis = int(np.argmax(self.volume_shape))
            print(f"Inferred readout axis from spatial shape: {self.readout_axis}")
        if not 0 <= self.readout_axis < n_dims:
            raise ValueError(f"readout_axis must be within [0, {n_dims - 1}], got {self.readout_axis}")
        if self.phase_axes is None:
            self.phase_axes = tuple(axis for axis in range(n_dims) if axis != self.readout_axis)
        if len(self.phase_axes) != 2:
            raise ValueError(f"phase_axes must contain exactly two axes, got {self.phase_axes}")
        if self.readout_axis in self.phase_axes:
            raise ValueError("readout_axis must be distinct from phase_axes")
        if sorted((self.readout_axis, *self.phase_axes)) != list(range(n_dims)):
            raise ValueError(
                f"readout_axis and phase_axes must cover all spatial axes exactly once, got {self.readout_axis}, {self.phase_axes}"
            )

    def _load_h5_kspace(self, path: Optional[str] = None) -> np.ndarray:
        path = path or self.data_path
        if h5py is None:
            raise ImportError("Reading .h5 k-space files requires h5py to be installed.")
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
            preferred_keys = ["kspace", "ksp"]
            key = next((name for name in preferred_keys if name in handle), None)
            if key is None:
                key = keys[0]
            print(f"Using H5 key: {key}")
            return handle[key][:]

    def _load_mat_kspace(self, path: Optional[str] = None) -> np.ndarray:
        path = path or self.data_path
        try:
            variables = scipy.io.whosmat(path)
        except NotImplementedError:
            print("MATLAB v7.3 file detected, falling back to h5py reader.")
            return self._load_h5_kspace(path)
        if len(variables) == 0:
            raise ValueError(f"No variables found in MATLAB file: {path}")
        preferred_keys = ["kspace", "ksp", "raw_kspace", "rawdata", "data"]
        available_names = [name for name, _, _ in variables]
        key = None
        for preferred in preferred_keys:
            match = next((name for name in available_names if name.lower() == preferred), None)
            if match is not None:
                key = match
                break
        if key is None:
            key = max(variables, key=lambda item: np.prod(item[1]))[0]
        print(f"Using MATLAB variable: {key}")
        mat = scipy.io.loadmat(path, variable_names=[key])
        return mat[key]

    def _normalize_kspace_layout(self, kspace_data: np.ndarray) -> np.ndarray:
        arr = np.asarray(kspace_data)
        arr = np.squeeze(arr)
        # Handle HDF5 structured dtype like [('real', '<f4'), ('imag', '<f4')]
        if arr.dtype.names is not None and 'real' in arr.dtype.names and 'imag' in arr.dtype.names:
            arr = arr['real'] + 1j * arr['imag']
            print(f"Converted structured real/imag dtype to complex: {arr.shape}")
        if not np.iscomplexobj(arr):
            arr = self._maybe_convert_real_imag_to_complex(arr)
        if arr.ndim != 4:
            raise ValueError(f"Expected a 4D k-space array after normalization, got shape {arr.shape}")
        coil_axis = self._infer_coil_axis(arr.shape)
        if coil_axis != 0:
            old_shape = arr.shape
            arr = np.moveaxis(arr, coil_axis, 0)
            print(f"Moved coil axis from dim {coil_axis} to dim 0: {old_shape} -> {arr.shape}")
        else:
            print(f"Detected coil axis at dim 0: {arr.shape}")
        if not np.iscomplexobj(arr):
            raise ValueError(f"K-space data must be complex-valued, but got dtype {arr.dtype}")
        return np.ascontiguousarray(arr)

    def _maybe_convert_real_imag_to_complex(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 5:
            complex_axis = None
            if arr.shape[-1] == 2:
                complex_axis = arr.ndim - 1
            elif arr.shape[0] == 2:
                complex_axis = 0
            else:
                candidates = [axis for axis, size in enumerate(arr.shape) if size == 2]
                if len(candidates) == 1:
                    complex_axis = candidates[0]
            if complex_axis is not None:
                arr = np.moveaxis(arr, complex_axis, -1)
                print(f"Merged real/imag channels from dim {complex_axis}: {arr.shape[:-1]}")
                return arr[..., 0] + 1j * arr[..., 1]
        return arr

    def _infer_coil_axis(self, shape: Tuple[int, ...]) -> int:
        candidates = [axis for axis, size in enumerate(shape) if 1 <= size <= 64]
        if len(candidates) == 0:
            return 0
        if len(candidates) == 1:
            return candidates[0]
        if shape[-1] <= 64 and shape[0] > 64:
            return len(shape) - 1
        if shape[0] <= 64 and shape[-1] > 64:
            return 0
        return min(candidates, key=lambda axis: shape[axis])

    def _process_data(self):
        if self.presampled:
            self._process_data_presampled()
        else:
            self._process_data_simulated()

    def _process_data_presampled(self):
        """Process real clinical undersampled k-space data.

        The input kspace is already undersampled — no GT is available.
        Mask is inferred from the zero/non-zero pattern of the kspace data.
        """
        print("[Presampled] Input k-space is already undersampled (real clinical data).")

        # Load CSM
        if self.csm_path and self.csm_path != "None":
            self.sensitivity_maps = self._load_external_csm(self.csm_path)
            print("[Presampled] Using external CSM for reconstruction.")
        else:
            self.sensitivity_maps = self._estimate_csm_from_acs()
            print("[Presampled] Using ACS-estimated CSM for reconstruction.")

        # Infer mask from kspace: a voxel is sampled if any coil has non-zero value
        ksp_energy = torch.sum(torch.abs(self.kspace_multicoil), dim=0)
        self.mask = (ksp_energy > 0).float()
        actual_acc = self.mask.numel() / max(self.mask.sum().item(), 1.0)
        print(f"[Presampled] Inferred mask from k-space. Actual acceleration: {actual_acc:.2f}x")
        self._warn_if_presampled_kspace_not_centered(ksp_energy)

        # Normalize intensity using zero-filled image magnitude
        images_multicoil = ifft3c(self.kspace_multicoil)
        zero_filled_complex = torch.sum(torch.conj(self.sensitivity_maps) * images_multicoil, dim=0)
        if self.normalize_kspace:
            mag = torch.abs(zero_filled_complex).reshape(-1)
            if self.normalization_percentile >= 100.0:
                scale = mag.max()
            elif mag.numel() <= 16_000_000:
                scale = torch.quantile(mag, self.normalization_percentile / 100.0)
            else:
                # torch.quantile caps at ~16M elements; use numpy for larger tensors
                scale = torch.tensor(
                    float(np.quantile(mag.cpu().numpy(), self.normalization_percentile / 100.0)),
                    dtype=mag.dtype, device=mag.device,
                )
            scale_value = float(torch.clamp(scale, min=1e-8).item())
            self.normalization_scale = scale_value
            self.kspace_multicoil = self.kspace_multicoil / scale_value
            zero_filled_complex = zero_filled_complex / scale_value
            print(f"[Presampled] Normalized by P{self.normalization_percentile:g} magnitude = {self.normalization_scale:.6f}")

        # No ground truth available — use zero-filled as placeholder
        self.ground_truth_image = zero_filled_complex
        self.kspace_full = fft3c(zero_filled_complex)

        # Build undersampled outputs (kspace is already undersampled)
        self.kspace_undersampled_complex = self.kspace_multicoil.contiguous()
        self.zero_filled_complex = zero_filled_complex.contiguous()
        self.zero_filled_image = torch.stack([zero_filled_complex.real, zero_filled_complex.imag], dim=0)
        self.kspace_undersampled = torch.cat([self.kspace_multicoil.real, self.kspace_multicoil.imag], dim=0)
        print(f"[Presampled] Zero-filled shape: {self.zero_filled_image.shape}")
        print(f"[Presampled] Undersampled k-space shape: {self.kspace_undersampled.shape}")

    def _process_data_simulated(self):
        """Original pipeline: full kspace -> generate GT -> apply mask -> undersample."""
        images_multicoil = ifft3c(self.kspace_multicoil)
        csm_gt = self._get_best_available_csm(images_multicoil=images_multicoil)
        self.ground_truth_image = torch.sum(torch.conj(csm_gt) * images_multicoil, dim=0)
        self._normalize_intensity_scale()
        self.kspace_full = fft3c(self.ground_truth_image)
        print(f"Ground truth generated using High-Res CSM. Shape: {self.ground_truth_image.shape}")

        self._generate_mask()

        if self.csm_path and self.csm_path != "None":
            self.sensitivity_maps = csm_gt
            print("Using external CSM for reconstruction input.")
        else:
            self.sensitivity_maps = self._estimate_csm_from_acs()
            print("Using ACS-estimated CSM for reconstruction input.")

        self._apply_undersampling()

    def _get_best_available_csm(self, images_multicoil: Optional[torch.Tensor] = None):
        if self.csm_path and self.csm_path != "None":
            # D-09: No try/except fallback — hard fail if external CSM fails to load
            return self._load_external_csm(self.csm_path)
        # No csm_path provided — estimate from full k-space
        print("No csm_path configured. Estimating high-resolution CSM from full k-space for GT generation...")
        if images_multicoil is None:
            images_multicoil = ifft3c(self.kspace_multicoil)
        rss = torch.sqrt(torch.sum(torch.abs(images_multicoil) ** 2, dim=0)) + 1e-12
        return images_multicoil / rss.unsqueeze(0)

    def _normalize_intensity_scale(self):
        if not self.normalize_kspace:
            self.normalization_scale = 1.0
            return

        ground_truth_mag = torch.abs(self.ground_truth_image).reshape(-1)
        if self.normalization_percentile >= 100.0:
            scale = ground_truth_mag.max()
        else:
            scale = torch.quantile(ground_truth_mag, self.normalization_percentile / 100.0)
        scale_value = float(torch.clamp(scale, min=1e-8).item())
        self.normalization_scale = scale_value
        self.kspace_multicoil = self.kspace_multicoil / scale_value
        self.ground_truth_image = self.ground_truth_image / scale_value
        print(
            f"Normalized MRI intensity scale by P{self.normalization_percentile:g} magnitude = "
            f"{self.normalization_scale:.6f}"
        )

    def _estimate_csm_from_acs(self):
        print(f"Estimating low-resolution CSM from ACS region (center {self.acs_lines} lines)...")
        acs_mask = torch.zeros_like(self.kspace_multicoil, dtype=torch.float32)
        index = [slice(None)] * self.kspace_multicoil.ndim
        for axis in range(len(self.volume_shape)):
            tensor_axis = axis + 1
            if axis == self.readout_axis:
                index[tensor_axis] = slice(None)
            else:
                size = self.volume_shape[axis]
                center = size // 2
                half_acs = self.acs_lines // 2
                start = max(0, center - half_acs)
                end = min(size, center + half_acs)
                index[tensor_axis] = slice(start, end)
        acs_mask[tuple(index)] = 1.0
        kspace_acs = self.kspace_multicoil * acs_mask
        images_acs = ifft3c(kspace_acs)
        rss = torch.sqrt(torch.sum(torch.abs(images_acs) ** 2, dim=0)) + 1e-12
        return images_acs / rss.unsqueeze(0)

    def _load_external_csm(self, path: str):
        print(f"Loading external CSM from {path}")
        possible_keys = ["map", "csm", "sensitivity_maps", "maps", "S"]
        try:
            mat = scipy.io.loadmat(path)
            csm_key = next((key for key in possible_keys if key in mat), None)
            if csm_key is None:
                keys = [key for key in mat.keys() if not key.startswith("__")]
                if keys:
                    csm_key = max(keys, key=lambda key: mat[key].size)
            if csm_key is None:
                raise ValueError("Could not find CSM variable in file")
            csm_data = mat[csm_key]
        except (NotImplementedError, ValueError):
            if h5py is None:
                raise ImportError("Reading v7.3 .mat / .h5 CSM files requires h5py.")
            print("MATLAB v7.3 / .h5 CSM file detected, using h5py reader.")
            with h5py.File(path, "r") as handle:
                csm_key = next((key for key in possible_keys if key in handle), None)
                if csm_key is None:
                    keys = list(handle.keys())
                    if keys:
                        csm_key = max(keys, key=lambda k: np.prod(handle[k].shape))
                if csm_key is None:
                    raise ValueError("Could not find CSM variable in file")
                print(f"Using H5 CSM key: {csm_key}")
                csm_data = handle[csm_key][:]
        csm_data = np.asarray(csm_data)
        # Handle structured dtype like [('real', '<f4'), ('imag', '<f4')]
        if csm_data.dtype.names is not None and 'real' in csm_data.dtype.names and 'imag' in csm_data.dtype.names:
            csm_data = csm_data['real'] + 1j * csm_data['imag']
        elif csm_data.dtype == np.void or csm_data.dtype.names is not None:
            # Try interpreting as interleaved real/imag float32
            csm_data = csm_data.view(np.float32).reshape(csm_data.shape + (2,))
            csm_data = csm_data[..., 0] + 1j * csm_data[..., 1]
        csm = torch.from_numpy(csm_data)
        if not torch.is_complex(csm):
            if csm.ndim >= 1 and csm.shape[-1] == 2:
                csm = torch.complex(csm[..., 0], csm[..., 1])
            else:
                csm = csm.to(torch.complex64)
        csm = csm.squeeze()
        if csm.shape != self.kspace_multicoil.shape and csm.ndim == 4:
            if csm.shape[-1] == self.num_coils:
                csm = csm.permute(3, 0, 1, 2)
            elif csm.shape[1] == self.num_coils and csm.shape[0] != self.num_coils:
                csm = csm.permute(1, 0, 2, 3)

        # D-17: Auto-permute CSM spatial dims to match volume_shape if needed
        if csm.ndim == 4:
            csm_spatial = csm.shape[1:]  # (D, H, W) assuming coils at dim 0
            if csm_spatial != torch.Size(self.volume_shape):
                # Try to find a permutation of spatial dims that matches volume_shape
                import itertools
                spatial_dims = list(csm_spatial)
                target_dims = list(self.volume_shape)
                for perm in itertools.permutations(range(3)):
                    if [spatial_dims[p] for p in perm] == target_dims:
                        # Apply permutation: dim 0 is coils, dims 1-3 are spatial
                        new_order = [0] + [p + 1 for p in perm]
                        csm = csm.permute(*new_order)
                        print(f"[CSM] Auto-permuted spatial dims to match volume_shape: {csm.shape}")
                        break

        # D-17: Assert final shape matches expected (coils, *volume_shape)
        expected_shape = (self.num_coils,) + tuple(self.volume_shape)
        assert csm.shape == torch.Size(expected_shape), (
            f"CSM shape {csm.shape} does not match expected {expected_shape} "
            f"(num_coils={self.num_coils}, volume_shape={self.volume_shape})"
        )
        csm_abs_max = float(torch.abs(csm).max().item())
        if csm_abs_max < 1e-8:
            raise ValueError(
                f"External CSM loaded from {path} is all-zero or numerically empty "
                f"(|CSM| max={csm_abs_max:.3e}). Refusing to train with invalid coil maps."
            )

        # D-08: Prominent startup log line
        print(f"[CSM] Loaded from: {path}, key: {csm_key}, shape: {csm.shape}")
        return csm

    def _warn_if_presampled_kspace_not_centered(self, ksp_energy: torch.Tensor):
        """Warn when prospectively sampled k-space is not aligned for centered FFT."""
        flat_idx = int(torch.argmax(ksp_energy).item())
        dc_index = tuple(int(v) for v in np.unravel_index(flat_idx, self.volume_shape))
        center = tuple(size // 2 for size in self.volume_shape)
        offset = tuple(dc - ctr for dc, ctr in zip(dc_index, center))
        center_sampled = bool(self.mask[center].item() > 0)
        max_offset = max(abs(value) for value in offset)
        if max_offset > 2 or not center_sampled:
            print(
                "[Presampled][WARNING] Brightest k-space bin / inferred DC is not centered: "
                f"dc={dc_index}, center={center}, offset={offset}, "
                f"center_sampled={center_sampled}. This pipeline uses centered FFT; "
                "recenter the raw k-space/mask in preprocessing before training."
            )

    def _conform_mask_shape(self, mask: torch.Tensor) -> torch.Tensor:
        """Ensure mask matches volume_shape; auto-stack 2D masks along the inferred readout axis."""
        vol = self.volume_shape  # e.g. (256, 174, 232)
        if mask.shape == torch.Size(vol):
            return mask
        if mask.ndim == 2:
            h, w = mask.shape
            # Find which volume axis is missing from the 2D mask
            stack_axis = None
            for ax in range(3):
                remaining = [vol[i] for i in range(3) if i != ax]
                if sorted([h, w]) == sorted(remaining):
                    stack_axis = ax
                    break
            if stack_axis is None:
                raise ValueError(
                    f"2D mask shape {(h, w)} cannot be matched to any two dimensions of volume {vol}"
                )
            # Determine if we need to transpose the 2D mask to match dimension order
            remaining_axes = [i for i in range(3) if i != stack_axis]
            expected = (vol[remaining_axes[0]], vol[remaining_axes[1]])
            if (h, w) != expected:
                mask = mask.T
            # Stack along the missing axis
            n_repeat = vol[stack_axis]
            mask_3d = mask.unsqueeze(stack_axis).expand(*vol).clone()
            print(f"2D mask {(h, w)} stacked along axis {stack_axis} (size {n_repeat}) -> {tuple(mask_3d.shape)}")
            return mask_3d
        raise ValueError(
            f"External mask shape {tuple(mask.shape)} does not match volume shape {vol}"
        )

    def _generate_mask(self):
        shape = self.volume_shape
        mask = None
        external = False
        if self.mask_path and self.mask_path != "None":
            mask = self._load_external_mask(self.mask_path)
            external = True
        if mask is None:
            if self.mask_type in {"stacked_2d_gaussian", "gaussian"}:
                mask = self._generate_stacked_2d_gaussian_mask(shape)
            elif self.mask_type == "poisson":
                mask = self._generate_poisson_mask(shape)
            elif self.mask_type == "random":
                mask = self._generate_random_mask(shape)
            else:
                raise ValueError(f"Unsupported mask_type: {self.mask_type}")
            if self.use_acs:
                mask = self._add_acs_region(mask)
        self._validate_mask_geometry(mask, skip_acs_check=external)
        if self.center_kspace_dc and any(self._dc_roll):
            mask = torch.roll(mask, shifts=self._dc_roll, dims=(0, 1, 2)).contiguous()
            print(f"[center_kspace_dc] rolled external mask by {self._dc_roll} to track centered DC")
        self.mask = mask
        actual_acc = mask.numel() / max(mask.sum().item(), 1.0)
        source = "external" if external else self.mask_type
        print(f"Using {source} mask with actual acceleration: {actual_acc:.2f}x")

    def _load_external_mask(self, path: str) -> torch.Tensor:
        print(f"Loading external mask from {path}")
        try:
            mat = scipy.io.loadmat(path)
        except (NotImplementedError, ValueError):
            if h5py is None:
                raise ImportError("Reading v7.3 .mat / .h5 mask files requires h5py.")
            print("MATLAB v7.3 / .h5 mask file detected, using h5py reader.")
            with h5py.File(path, "r") as handle:
                possible_keys = ["mask", "sampling_mask", "pattern"]
                mask_key = next((key for key in possible_keys if key in handle), None)
                if mask_key is None:
                    keys = list(handle.keys())
                    if keys:
                        mask_key = max(keys, key=lambda k: np.prod(handle[k].shape))
                if mask_key is None:
                    raise ValueError("Could not find mask variable in file")
                print(f"Using H5 mask key: {mask_key}")
                mask = torch.from_numpy(handle[mask_key][:]).float()
                if mask.ndim == 4 and mask.shape[-1] == self.num_coils:
                    mask = mask[..., 0]
                elif mask.ndim == 4 and mask.shape[0] == self.num_coils:
                    mask = mask[0]
                mask = self._conform_mask_shape(mask)
                return (mask > 0.5).float()
        possible_keys = ["mask", "sampling_mask", "pattern"]
        mask_key = next((key for key in possible_keys if key in mat), None)
        if mask_key is None:
            keys = [key for key in mat.keys() if not key.startswith("__")]
            if keys:
                mask_key = max(keys, key=lambda key: mat[key].size)
        if mask_key is None:
            raise ValueError("Could not find mask variable in file")
        mask = torch.from_numpy(mat[mask_key]).float()
        # If mask has coil dimension, take first coil (mask is same for all coils)
        if mask.ndim == 4 and mask.shape[-1] == self.num_coils:
            mask = mask[..., 0]
        elif mask.ndim == 4 and mask.shape[0] == self.num_coils:
            mask = mask[0]
        mask = self._conform_mask_shape(mask)
        mask = (mask > 0.5).float()
        return mask

    def _phase_shape(self, shape: Tuple[int, int, int]) -> Tuple[int, int]:
        return tuple(shape[axis] for axis in self.phase_axes)

    def _expand_phase_mask(self, mask_2d: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        view_shape = [1] * len(shape)
        view_shape[self.phase_axes[0]] = mask_2d.shape[0]
        view_shape[self.phase_axes[1]] = mask_2d.shape[1]
        return mask_2d.view(view_shape).expand(*shape).clone()

    def _generate_stacked_2d_gaussian_mask(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        phase_shape = self._phase_shape(shape)
        total_phase_points = phase_shape[0] * phase_shape[1]
        target_phase_points = max(int(round(total_phase_points / float(self.acceleration_factor))), 1)
        phase_centers = [size // 2 for size in phase_shape]
        coords = [torch.arange(size, dtype=torch.float32) - center for size, center in zip(phase_shape, phase_centers)]
        grid_a, grid_b = torch.meshgrid(coords[0], coords[1], indexing="ij")
        sigma = min(phase_shape) / 4.0
        weights = torch.exp(-(grid_a ** 2 + grid_b ** 2) / (2 * sigma ** 2))

        acs_mask = torch.zeros(phase_shape, dtype=torch.bool)
        if self.use_acs:
            center_a, center_b = phase_centers
            half_acs = self.acs_lines // 2
            a_start = max(0, center_a - half_acs)
            a_end = min(phase_shape[0], center_a + half_acs)
            b_start = max(0, center_b - half_acs)
            b_end = min(phase_shape[1], center_b + half_acs)
            acs_mask[a_start:a_end, b_start:b_end] = True

        candidate_mask = ~acs_mask
        num_acs_points = int(acs_mask.sum().item())
        remaining_points = min(
            max(target_phase_points - num_acs_points, 0),
            int(candidate_mask.sum().item()),
        )

        mask_2d = torch.zeros(phase_shape, dtype=torch.float32)
        if remaining_points > 0:
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False)
            candidate_weights = weights[candidate_mask]
            if float(candidate_weights.sum().item()) <= 0:
                candidate_weights = torch.ones_like(candidate_weights)
            chosen = torch.multinomial(candidate_weights, remaining_points, replacement=False)
            sampled = candidate_indices[chosen]
            mask_2d[sampled[:, 0], sampled[:, 1]] = 1.0
        return self._expand_phase_mask(mask_2d, shape)

    def _generate_poisson_mask(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        from .mask_generator import gen_poisson_mask

        phase_shape = self._phase_shape(shape)
        mask_2d_np = gen_poisson_mask(phase_shape, self.acceleration_factor, self.acs_lines)
        mask_2d = torch.from_numpy(mask_2d_np).float()
        return self._expand_phase_mask(mask_2d, shape)

    def _generate_random_mask(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        phase_shape = self._phase_shape(shape)
        total_phase_points = phase_shape[0] * phase_shape[1]
        target_phase_points = max(int(round(total_phase_points / float(self.acceleration_factor))), 1)
        mask_2d = torch.zeros(phase_shape, dtype=torch.float32)
        chosen = torch.randperm(total_phase_points)[:target_phase_points]
        mask_2d.view(-1)[chosen] = 1.0
        return self._expand_phase_mask(mask_2d, shape)

    def _add_acs_region(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clone()
        index = [slice(None)] * mask.ndim
        index[self.readout_axis] = slice(None)
        for axis in self.phase_axes:
            size = self.volume_shape[axis]
            center = size // 2
            half_acs = self.acs_lines // 2
            start = max(0, center - half_acs)
            end = min(size, center + half_acs)
            index[axis] = slice(start, end)
        mask[tuple(index)] = 1.0
        return mask

    def _validate_mask_geometry(self, mask: torch.Tensor, skip_acs_check: bool = False):
        if not self.skip_readout_check:
            readout_size = self.volume_shape[self.readout_axis]
            collapsed = mask.sum(dim=self.readout_axis)
            if not torch.all((collapsed == 0) | (collapsed == readout_size)):
                raise ValueError("Mask violates readout fully-sampled geometry.")
        if not skip_acs_check:
            acs_index = [slice(None)] * mask.ndim
            acs_index[self.readout_axis] = slice(None)
            for axis in self.phase_axes:
                size = self.volume_shape[axis]
                center = size // 2
                half_acs = self.acs_lines // 2
                start = max(0, center - half_acs)
                end = min(size, center + half_acs)
                acs_index[axis] = slice(start, end)
            if self.use_acs and not torch.all(mask[tuple(acs_index)] == 1):
                raise ValueError("ACS region is not fully sampled.")
        if not self.skip_readout_check and self.acceleration_factor > 1:
            readout_size = self.volume_shape[self.readout_axis]
            collapsed = mask.sum(dim=self.readout_axis)
            if torch.all(collapsed == readout_size):
                raise ValueError("Mask does not undersample the phase-encoding plane.")

    def _build_mask_split(self) -> None:
        """Build M^tr / M^fb / M^te (INF-01) from ``self.mask`` and store in
        ``self.mask_split: Dict[str, Tensor]``. Optionally dump split_indices.npz.

        Per CONTEXT.md decision C: ACS = ``acs_lines × acs_lines`` phase plane ×
        full readout (NOT a 24-cube). ``phase_axes`` / ``readout_axis`` are read
        from existing self fields (set by ``_normalize_axis_config``).

        Called by ``__init__`` only when ``enable_mask_split=True``; callers
        without the new flag get an empty ``self.mask_split = {}`` (backward
        compat — v1 path untouched).
        """
        if not isinstance(self.mask, torch.Tensor):
            raise RuntimeError(
                "MRIDataset._build_mask_split called before self.mask was set; "
                "must run after _process_data()."
            )
        splits = split_acquired_mask(
            mask=self.mask,
            ratios=self.split_ratio,
            seed=self.mask_split_seed,
            preserve_acs=True,
            acs_size=(int(self.acs_lines), int(self.acs_lines)),
            phase_axes=tuple(int(a) for a in self.phase_axes),
            readout_axis=int(self.readout_axis),
            device=self.mask.device,
        )
        self.mask_split = splits
        n_tr = float(splits["tr"].sum().item())
        n_fb = float(splits["fb"].sum().item())
        n_te = float(splits["te"].sum().item())
        n_total = float(self.mask.sum().item())
        print(
            f"[MaskSplit INF-01] seed={self.mask_split_seed} "
            f"ratios={self.split_ratio} -> "
            f"tr={n_tr / max(n_total, 1.0):.4f} "
            f"fb={n_fb / max(n_total, 1.0):.4f} "
            f"te={n_te / max(n_total, 1.0):.4f} (acquired={int(n_total)})"
        )
        if self.mask_split_out_dir is not None:
            dumped = save_split_indices(
                out_dir=self.mask_split_out_dir,
                splits=splits,
                seed=self.mask_split_seed,
                ratios=self.split_ratio,
                acquired_mask=self.mask,
            )
            print(f"[MaskSplit INF-01] dumped split indices -> {dumped}")

    def _load_measurement(self) -> torch.Tensor:
        """The k-space the reconstruction fits when it is not the reference that defines the ground truth
        (data.<channel>.measurement_path; noise-robustness study, spec 2026-09-16 section 4). Read and laid out like
        data_path, rolled like it, and divided by the SAME normalization_scale (the clean ground truth's), so a
        measurement file identical to data_path reproduces the default path bit for bit."""
        path = self.measurement_path
        print(f"[Measurement] fitting to {path} (ground truth and scale from {self.data_path})")
        ext = os.path.splitext(path)[1].lower()
        arr = self._load_mat_kspace(path) if ext == ".mat" else self._load_h5_kspace(path)
        meas = torch.from_numpy(self._normalize_kspace_layout(arr)).to(torch.complex64)
        if tuple(meas.shape) != tuple(self.kspace_multicoil.shape):
            raise ValueError(f"measurement k-space {tuple(meas.shape)} from {path} does not match the reference "
                             f"{tuple(self.kspace_multicoil.shape)} from {self.data_path}")
        if self.center_kspace_dc and any(self._dc_roll):
            meas = torch.roll(meas, shifts=self._dc_roll, dims=(1, 2, 3)).contiguous()
        return meas / self.normalization_scale

    def _apply_undersampling(self):
        measured = self._load_measurement() if self.measurement_path else self.kspace_multicoil
        kspace_undersampled_complex = measured * self.mask.unsqueeze(0)
        image_multicoil = ifft3c(kspace_undersampled_complex)
        zero_filled_complex = torch.sum(torch.conj(self.sensitivity_maps) * image_multicoil, dim=0)
        self.kspace_undersampled_complex = kspace_undersampled_complex.contiguous()
        self.zero_filled_complex = zero_filled_complex.contiguous()
        self.zero_filled_image = torch.stack([zero_filled_complex.real, zero_filled_complex.imag], dim=0)
        self.kspace_undersampled = torch.cat([kspace_undersampled_complex.real, kspace_undersampled_complex.imag], dim=0)
        print(f"Applied undersampling. Non-zero ratio: {self.mask.sum() / self.mask.numel():.4f}")
        print(f"Zero-filled shape: {self.zero_filled_image.shape}")
        print(f"Undersampled k-space shape: {self.kspace_undersampled.shape}")

    def get_data(self) -> Dict[str, torch.Tensor]:
        return {
            "kspace_full": self.kspace_full,
            "kspace_undersampled": self.kspace_undersampled,
            "kspace_undersampled_complex": self.kspace_undersampled_complex,
            "mask": self.mask,
            "ground_truth": self.ground_truth_image,
            "zero_filled": self.zero_filled_image,
            "zero_filled_complex": self.zero_filled_complex,
            "sensitivity_maps": self.sensitivity_maps,
            "volume_shape": self.volume_shape,
            "normalization_scale": self.normalization_scale,
            # === v2 INF-02 addition (additive; empty dict when enable_mask_split=False) ===
            "mask_split": self.mask_split,
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.get_data()
