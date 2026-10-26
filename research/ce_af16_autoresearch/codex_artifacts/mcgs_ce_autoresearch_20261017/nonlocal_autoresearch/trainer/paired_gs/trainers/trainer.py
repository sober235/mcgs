"""Training loop for paper-faithful and legacy 3DGSMR modes."""

import datetime
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None

from data import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian import GaussianModel3D, TileVoxelizer, Voxelizer
from losses import CombinedLoss
from losses.losses import BackgroundSuppressionLoss, WaveletL1Loss
from metrics import MetricPlateauStopper, evaluate_reconstruction
from trainers.voxelizer_config import apply_deterministic_voxelizer_reference


class GaussianTrainer:
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = self.config.setdefault("mode", {})
        voxelizer_note = apply_deterministic_voxelizer_reference(self.config)
        if voxelizer_note:
            print(f"[DeterministicVoxelizer] {voxelizer_note}")
        self.paper_faithful = self.mode.get("paper_faithful", False)
        self.experimental_reproduction = self.mode.get("experimental_reproduction", True)
        self.self_supervised_deploy = self.mode.get("self_supervised_deploy", False)
        self.strict_cuda = self.mode.get("strict_cuda", False)
        self.require_lpips = self.mode.get("require_lpips", False)
        self.metric_config = self.config.get("metrics", {})
        print(f"Using device: {self.device}")
        print(
            f"Modes: paper_faithful={self.paper_faithful}, experimental_reproduction={self.experimental_reproduction}, "
            f"self_supervised_deploy={self.self_supervised_deploy}, strict_cuda={self.strict_cuda}"
        )

        self.current_iteration = 0
        self._position_grads_for_densify = None
        self.best_metrics: Dict[str, float] = {}
        self.best_psnr = float("-inf")
        self.best_ssim = float("-inf")
        self.best_lpips = float("inf")
        self.metric_plateau_stopper: Optional[MetricPlateauStopper] = None
        self.loss_plateau_best = float("inf")
        self.loss_plateau_bad_count = 0
        self._skip_scheduler_step_once = False

        self._resolve_strategy()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()
        self._setup_stopping()

    def _resolve_strategy(self):
        adaptive_config = self.config.setdefault("adaptive_control", {})
        gaussian_config = self.config.setdefault("gaussian", {})
        acc = self.config["data"]["acceleration_factor"]
        strategy = adaptive_config.get("strategy", "auto")
        threshold = adaptive_config.get("high_acceleration_threshold", 10)

        if strategy == "auto":
            if acc >= threshold:
                self._resolved_use_long_axis = True
                self._resolved_use_cloning = False
                if gaussian_config.get("initial_num_points", 500) > 1000:
                    gaussian_config["initial_num_points"] = 500
            else:
                self._resolved_use_long_axis = False
                self._resolved_use_cloning = True
        elif strategy == "long_axis":
            self._resolved_use_long_axis = True
            self._resolved_use_cloning = False
        else:
            self._resolved_use_long_axis = False
            self._resolved_use_cloning = True

        if self.paper_faithful and acc < threshold and gaussian_config.get("initial_num_points") != 200000:
            print("[Strategy] Paper-faithful low-mid mode allows override, but the paper default remains M=200k + original split/clone.")
        print(
            f"[Strategy] resolved long_axis={self._resolved_use_long_axis}, cloning={self._resolved_use_cloning}, "
            f"M={gaussian_config.get('initial_num_points')}"
        )

    def _setup_data(self):
        data_config = self.config["data"]
        self.dataset = MRIDataset(
            data_path=data_config["data_path"],
            acceleration_factor=data_config["acceleration_factor"],
            mask_type=data_config.get("mask_type", "stacked_2d_gaussian"),
            use_acs=data_config.get("use_acs", True),
            acs_lines=data_config.get("acs_lines", 24),
            csm_path=data_config.get("csm_path"),
            mask_path=data_config.get("mask_path"),
            readout_axis=data_config.get("readout_axis"),
            phase_axes=tuple(data_config["phase_axes"]) if data_config.get("phase_axes") is not None else None,
            normalize_kspace=data_config.get("normalize", data_config.get("normalize_kspace", True)),
            normalization_percentile=data_config.get("normalization_percentile", 99.9),
            device=str(self.device),
            presampled=data_config.get("presampled", False),
            skip_readout_check=data_config.get("skip_readout_check", False),
        )
        data = self.dataset.get_data()
        self.kspace_full_cpu = data["kspace_full"]
        self.kspace_undersampled_complex = data.get("kspace_undersampled_complex")
        if self.kspace_undersampled_complex is None:
            kspace_ri = data["kspace_undersampled"]
            num_coils = kspace_ri.shape[0] // 2
            self.kspace_undersampled_complex = torch.complex(kspace_ri[:num_coils], kspace_ri[num_coils:])
        self.kspace_undersampled_complex = self.kspace_undersampled_complex.to(self.device)
        self.mask = data["mask"].to(self.device)
        self.mask_coils = self.mask.unsqueeze(0)
        self.volume_shape = tuple(data["volume_shape"])
        self.target_image = data["ground_truth"].to(self.device)
        self._zero_filled_for_init = data.get("zero_filled_complex", data["zero_filled"]).to(self.device)
        self.background_mask = None
        self.sensitivity_maps = data["sensitivity_maps"].to(self.device)
        self.normalization_scale = float(data.get("normalization_scale", 1.0))
        self.num_coils = self.sensitivity_maps.shape[0]
        background_cfg = self.config.get("loss", {}).get("background_reg", {})
        if background_cfg.get("enable", False):
            self.background_mask = self._build_background_mask(self._zero_filled_for_init, background_cfg)
        print(
            f"Volume shape: {self.volume_shape}, coils: {self.num_coils}, acc: {data_config['acceleration_factor']}x, "
            f"norm_scale: {self.normalization_scale:.6f}"
        )
        del self.dataset

    def _setup_model(self):
        gaussian_config = self.config["gaussian"]
        init_method = gaussian_config.get("init_method", "from_image")
        init_mode = gaussian_config.get("init_mode", "random" if self.paper_faithful else "importance")

        # Support from_volume method for CS-warm-start
        if init_method == "from_volume":
            cs_volume_path = gaussian_config.get("cs_volume_path")
            if not cs_volume_path:
                raise ValueError("init_method='from_volume' requires cs_volume_path in config")

            self.gaussian_model = GaussianModel3D.from_volume(
                volume_path=cs_volume_path,
                num_points=gaussian_config["initial_num_points"],
                density_scale_k=gaussian_config.get("density_scale_k", 0.2),
                init_mode=init_mode,
                noise_level=gaussian_config.get("warmstart_noise_level", 0.05),
                device=str(self.device),
            )
        elif init_method == "from_image":
            # Legacy CS-Warm-Start support (deprecated, use from_volume instead)
            init_image = self._zero_filled_for_init
            if gaussian_config.get("cs_warmstart", False):
                cs_volume_path = gaussian_config.get("cs_volume_path")
                if cs_volume_path:
                    import numpy as np
                    cs_volume = np.load(cs_volume_path)
                    cs_volume_tensor = torch.from_numpy(cs_volume).to(self.device, dtype=torch.float32)

                    # Add noise to avoid local minimum
                    noise_std = gaussian_config.get("warmstart_noise_std", 0.05)
                    noise = torch.randn_like(cs_volume_tensor) * noise_std * cs_volume_tensor.std()
                    cs_volume_tensor = cs_volume_tensor + noise

                    # Convert to complex for initialization
                    init_image = torch.stack([cs_volume_tensor, torch.zeros_like(cs_volume_tensor)], dim=0)
                    print(f"[CS-Warm-Start] Loaded from {cs_volume_path}, added noise std={noise_std}")

            self.gaussian_model = GaussianModel3D.from_image(
                image=init_image,
                num_points=gaussian_config["initial_num_points"],
                density_scale_k=gaussian_config.get("density_scale_k", 0.2),
                init_mode=init_mode,
                device=str(self.device),
            )
        else:
            self.gaussian_model = GaussianModel3D(
                num_points=gaussian_config["initial_num_points"],
                volume_shape=tuple(self.volume_shape),
                device=str(self.device),
            )

        if hasattr(self, '_zero_filled_for_init'):
            del self._zero_filled_for_init
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vox_config = self.config.get("voxelizer", {})
        vox_type = vox_config.get("type", "chunk")
        if vox_type in {"tile_cuda", "tile"}:
            self.voxelizer = TileVoxelizer(
                tuple(self.volume_shape),
                tile_size=vox_config.get("tile_size", 8),
                max_radius=vox_config.get("max_radius", 20),
                use_cuda=(vox_type == "tile_cuda"),
                strict_cuda=self.strict_cuda and vox_type == "tile_cuda",
                device=str(self.device),
            )
        else:
            self.voxelizer = Voxelizer(
                volume_shape=tuple(self.volume_shape),
                device=str(self.device),
                max_radius=vox_config.get("max_radius", 20),
            )
        print(f"Initialized with {self.gaussian_model.num_points} Gaussians")

    def _setup_loss(self):
        loss_config = self.config["loss"]
        image_weight = loss_config.get("image_weight", 0.0)
        if self.paper_faithful and image_weight > 0:
            raise ValueError("Paper-faithful mode does not allow image-domain auxiliary loss.")
        self.tv_type = loss_config.get("tv_type", "anisotropic")
        self.criterion = CombinedLoss(
            kspace_weight=loss_config.get("kspace_weight", 1.0),
            image_weight=image_weight,
            tv_weight=loss_config.get("tv_weight", 0.0),
            loss_type=loss_config.get("loss_type", "l2"),
            tv_type=self.tv_type,
            smooth_tv=loss_config.get("smooth_tv", False),
            # NEW: Frequency-adaptive k-space
            kspace_freq_adaptive=loss_config.get("kspace_freq_adaptive", False),
            kspace_center_weight=loss_config.get("kspace_center_weight", 0.5),
            kspace_mid_weight=loss_config.get("kspace_mid_weight", 1.0),
            kspace_high_weight=loss_config.get("kspace_high_weight", 2.0),
            kspace_center_fraction=loss_config.get("kspace_center_fraction", 0.08),
            # NEW: Multi-scale image loss
            multiscale_image_weight=loss_config.get("multiscale_image_weight", 0.0),
            multiscale_levels=loss_config.get("multiscale_levels", [1, 2, 4]),
            multiscale_loss_type=loss_config.get("multiscale_loss_type", "l1"),
        ).to(self.device)
        print(
            f"TV regularization: type={self.tv_type}, "
            f"smooth={loss_config.get('smooth_tv', False)}, "
            f"weight={loss_config.get('tv_weight', 0.0)}"
        )
        if loss_config.get("kspace_freq_adaptive", False):
            print(
                f"Frequency-adaptive k-space: center={loss_config.get('kspace_center_weight', 0.5)}, "
                f"mid={loss_config.get('kspace_mid_weight', 1.0)}, "
                f"high={loss_config.get('kspace_high_weight', 2.0)}"
            )
        if loss_config.get("multiscale_image_weight", 0.0) > 0:
            print(
                f"Multi-scale loss: weight={loss_config.get('multiscale_image_weight', 0.0)}, "
                f"levels={loss_config.get('multiscale_levels', [1, 2, 4])}"
            )
        background_cfg = loss_config.get("background_reg", {})
        self.background_criterion = None
        if background_cfg.get("enable", False):
            self.background_criterion = BackgroundSuppressionLoss(
                loss_type=background_cfg.get("loss_type", "l2")
            ).to(self.device)
            print(
                "Background suppression: "
                f"weight={background_cfg.get('weight', 0.0)}, "
                f"threshold_ratio={background_cfg.get('threshold_ratio', 0.06)}, "
                f"dilation_kernel_size={background_cfg.get('dilation_kernel_size', 5)}"
            )

    def _setup_optimizer(self):
        train_config = self.config["training"]
        gaussian_config = self.config["gaussian"]
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get("position_lr", 1e-4),
            lr_density=gaussian_config.get("density_lr", 1e-3),
            lr_scale=gaussian_config.get("scale_lr", 5e-4),
            lr_rotation=gaussian_config.get("rotation_lr", 1e-4),
        )
        self.optimizer = optim.Adam(params)
        scheduler_config = train_config.get("lr_scheduler", {})
        if scheduler_config.get("type", "exponential") == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_config.get("gamma", 0.999))
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_config["max_iterations"])

    def _setup_output(self):
        output_config = self.config["output"]
        self.output_dir = output_config["output_dir"]
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.result_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "logs")) if output_config.get("use_tensorboard", True) else None
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as handle:
            yaml.dump(self.config, handle, default_flow_style=False)
        self._emit_run_info()

    def _emit_run_info(self) -> None:
        """Write <output_dir>/run_info.json (D-13, CB-03 reproducibility provenance).

        Records seed, deterministic flag, GPU PCI_BUS_ID, CUDA_VISIBLE_DEVICES,
        config-snapshot path, gaussian_model.py SHA-256, and UTC timestamp.
        See GS_TMI/REPRODUCIBILITY.md for the schema and key semantics.
        """
        import hashlib
        import json
        gm_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "gaussian", "gaussian_model.py")
        )
        try:
            with open(gm_path, "rb") as _f:
                gm_sha = hashlib.sha256(_f.read()).hexdigest()
        except OSError:
            gm_sha = ""
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        try:
            pci_idx = int(cvd.split(",")[0]) if cvd else -1
        except ValueError:
            pci_idx = -1
        train_cfg = self.config.get("training", {})
        run_info = {
            "seed": int(train_cfg.get("seed", 42)),
            "deterministic": bool(train_cfg.get("deterministic", False)),
            "gpu_pci_index": pci_idx,
            "cuda_visible_devices": cvd,
            "config_snapshot": os.path.join(self.output_dir, "config.yaml"),
            "gaussian_model_sha256": gm_sha,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(self.output_dir, "run_info.json"), "w") as _f:
            json.dump(run_info, _f, indent=2)

    def _setup_stopping(self):
        stopping_config = self.config.get("stopping", {})
        if self.experimental_reproduction and stopping_config.get("enable_metric_plateau", False):
            self.metric_plateau_stopper = MetricPlateauStopper(
                patience_evals=stopping_config.get("patience_evals", 4),
                min_iterations_before_stop=stopping_config.get("min_iterations_before_stop", 0),
                metric_min_delta=stopping_config.get("metric_min_delta", {}),
                best_metric=stopping_config.get("best_metric", "psnr"),
            )
        self.enable_loss_plateau = stopping_config.get("enable_loss_plateau", False) and not self.experimental_reproduction
        self.loss_min_delta = stopping_config.get("loss_min_delta", 0.0)
        self.loss_patience = stopping_config.get("patience_evals", 0)
        self.loss_min_iteration = stopping_config.get("min_iterations_before_stop", 0)
        self.best_metric_name = stopping_config.get("best_metric", "psnr")

    def _get_volume(self):
        return self.voxelizer(
            positions=self.gaussian_model.positions,
            scales=self.gaussian_model.get_scale_values(),
            rotations=self.gaussian_model.rotations,
            density=self.gaussian_model.density,
        )

    @staticmethod
    def _as_complex_volume(vol_result):
        if isinstance(vol_result, tuple):
            return torch.complex(vol_result[0], vol_result[1])
        return vol_result

    def _apply_hard_data_consistency(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.mode.get("apply_hard_data_consistency", True):
            return volume
        pred_kspace = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        kspace_dc = pred_kspace * (1.0 - self.mask_coils) + self.kspace_undersampled_complex
        coil_images_dc = ifft3c(kspace_dc)
        return torch.sum(torch.conj(self.sensitivity_maps) * coil_images_dc, dim=0)

    def _apply_soft_data_consistency(self, volume: torch.Tensor, lambda_dc: float) -> torch.Tensor:
        """
        Soft data consistency: blend predicted and measured k-space.
        dc_kspace = (1 - lambda) * pred_kspace + lambda * measured_kspace
        """
        pred_kspace = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        kspace_dc = (1.0 - lambda_dc) * pred_kspace + lambda_dc * (
            pred_kspace * (1.0 - self.mask_coils) + self.kspace_undersampled_complex
        )
        coil_images_dc = ifft3c(kspace_dc)
        return torch.sum(torch.conj(self.sensitivity_maps) * coil_images_dc, dim=0)

    def _get_annealed_weight(self, iteration: int, base_weight: float, min_weight: float, max_iter: int) -> float:
        """
        Cosine annealing schedule for regularization weights.
        Starts at base_weight, decays to min_weight following cosine curve.
        """
        import math
        if iteration >= max_iter:
            return min_weight
        progress = iteration / max_iter
        return min_weight + 0.5 * (base_weight - min_weight) * (1 + math.cos(math.pi * progress))

    def _build_background_mask(self, reference_volume: torch.Tensor, background_cfg: Dict[str, Any]) -> torch.Tensor:
        mag = torch.abs(reference_volume).float()
        percentile = float(background_cfg.get("reference_percentile", 99.5))
        threshold_ratio = float(background_cfg.get("threshold_ratio", 0.06))
        if percentile >= 100.0:
            reference_scale = mag.max()
        else:
            reference_scale = torch.quantile(mag.reshape(-1), percentile / 100.0)
        threshold = max(float(reference_scale.item()) * threshold_ratio, 1e-8)

        object_mask = (mag > threshold).float().unsqueeze(0).unsqueeze(0)
        dilation_kernel_size = max(int(background_cfg.get("dilation_kernel_size", 5)), 1)
        if dilation_kernel_size % 2 == 0:
            dilation_kernel_size += 1
        dilated_object = F.max_pool3d(
            object_mask,
            kernel_size=dilation_kernel_size,
            stride=1,
            padding=dilation_kernel_size // 2,
        )
        background_mask = (1.0 - dilated_object).squeeze(0).squeeze(0)

        background_fraction = float(background_mask.mean().item())
        min_fraction = float(background_cfg.get("min_background_fraction", 0.05))
        max_fraction = float(background_cfg.get("max_background_fraction", 0.95))
        if background_fraction < min_fraction or background_fraction > max_fraction:
            fallback_percentile = float(background_cfg.get("fallback_percentile", 20.0))
            fallback_threshold = torch.quantile(mag.reshape(-1), fallback_percentile / 100.0)
            background_mask = (mag <= fallback_threshold).float()
            background_fraction = float(background_mask.mean().item())
            print(
                "[BackgroundMask] Fallback percentile mask used: "
                f"p={fallback_percentile}, fraction={background_fraction:.3f}"
            )
        else:
            print(
                "[BackgroundMask] Built from zero-filled reference: "
                f"threshold={threshold:.6f}, fraction={background_fraction:.3f}"
            )

        return background_mask.to(self.device)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self._as_complex_volume(self._get_volume())
        kspace_pred = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        return volume, kspace_pred

    def _kspace_loss(self, pred_kspace: torch.Tensor, loss_type: str) -> torch.Tensor:
        # Route through the configured KSpaceConsistencyLoss so frequency-adaptive
        # weights in YAML are actually applied.
        return self.criterion.kspace_loss(
            pred_kspace,
            self.kspace_undersampled_complex,
            self.mask_coils,
        )

    def forward_with_loss(self, iteration: int = 0):
        vol_result = self._get_volume()
        volume = self._as_complex_volume(vol_result)

        # Soft data consistency during training (REQ-02)
        soft_dc_config = self.config["loss"].get("soft_dc", {})
        if soft_dc_config.get("enable", False):
            lambda_start = soft_dc_config.get("lambda_start", 0.5)
            lambda_end = soft_dc_config.get("lambda_end", 1.0)
            max_iter = self.config["training"]["max_iterations"]
            # Linear schedule: lambda increases from start to end
            lambda_dc = lambda_start + (lambda_end - lambda_start) * min(1.0, iteration / max_iter)
            volume = self._apply_soft_data_consistency(volume, lambda_dc)

        volume_for_eval = volume.detach()

        # Annealed regularization (REQ-03)
        annealed_reg = self.config["loss"].get("annealed_reg", {})
        if annealed_reg.get("enable", False):
            max_iter = self.config["training"]["max_iterations"]
            tv_base = annealed_reg.get("tv_base", 0.05)
            tv_min = annealed_reg.get("tv_min", 0.01)
            tv_weight = self._get_annealed_weight(iteration, tv_base, tv_min, max_iter)

            wavelet_base = annealed_reg.get("wavelet_base", 1e-4)
            wavelet_min = annealed_reg.get("wavelet_min", 1e-5)
            wavelet_weight = self._get_annealed_weight(iteration, wavelet_base, wavelet_min, max_iter)
        else:
            tv_weight = self.config["loss"].get("tv_weight", 0.0)
            wavelet_weight = self.config["loss"].get("wavelet_weight", 0.0)

        background_cfg = self.config["loss"].get("background_reg", {})
        background_weight = float(background_cfg.get("weight", 0.0)) if background_cfg.get("enable", False) else 0.0
        ramp_start = int(background_cfg.get("ramp_start_iter", 0))
        ramp_end = int(background_cfg.get("ramp_end_iter", ramp_start))
        if background_weight > 0 and ramp_end > ramp_start:
            if iteration <= ramp_start:
                background_weight = 0.0
            elif iteration < ramp_end:
                progress = (iteration - ramp_start) / float(ramp_end - ramp_start)
                background_weight = background_weight * progress

        tv_loss = None
        if tv_weight > 0:
            tv_loss = self.criterion.tv_loss(volume)

        loss_type = self.config["loss"].get("loss_type", "l2")
        pred_kspace = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        kspace_loss = self._kspace_loss(pred_kspace, loss_type)

        losses = {"kspace_loss": kspace_loss}
        total_loss = self.config["loss"].get("kspace_weight", 1.0) * kspace_loss
        if tv_weight > 0 and tv_loss is not None:
            losses["tv_loss"] = tv_loss
            total_loss = total_loss + tv_weight * tv_loss

        image_weight = self.config["loss"].get("image_weight", 0.0)
        if image_weight > 0 and self.target_image is not None:
            image_loss = self.criterion.image_loss(volume, self.target_image)
            losses["image_loss"] = image_loss
            total_loss = total_loss + image_weight * image_loss

        # Wavelet regularization (WAVE-03, per D-17)
        if wavelet_weight > 0:
            wavelet_loss = WaveletL1Loss()(volume)
            losses["wavelet_loss"] = wavelet_loss
            total_loss = total_loss + wavelet_weight * wavelet_loss

        if (
            background_weight > 0
            and self.background_mask is not None
            and self.background_criterion is not None
        ):
            background_loss = self.background_criterion(volume, self.background_mask)
            losses["background_loss"] = background_loss
            total_loss = total_loss + background_weight * background_loss

        losses["total_loss"] = total_loss
        return volume_for_eval, losses

    def compute_gradient_stats(self):
        grads = self._position_grads_for_densify
        if grads is None:
            if self.gaussian_model.positions.grad is None:
                return {}
            grads = self.gaussian_model.positions.grad
        grad_norm = torch.norm(grads, dim=-1)
        return {"grad_norm": grad_norm, "mean_grad": grad_norm.mean(), "max_grad": grad_norm.max()}

    def adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        adaptive_config = self.config["adaptive_control"]
        if not adaptive_config.get("enable", True):
            return {"split": 0, "clone": 0, "prune": 0}
        start_iter = adaptive_config.get("densify_from_iter", 100)
        end_iter = adaptive_config.get("densify_until_iter", 2500)
        interval = adaptive_config.get("densify_every", 100)
        max_num_points = self.config["gaussian"].get("max_num_points", 400000)
        if self.gaussian_model.num_points >= max_num_points:
            return {"split": 0, "clone": 0, "prune": 0}
        if iteration < start_iter or iteration > end_iter or iteration % interval != 0:
            return {"split": 0, "clone": 0, "prune": 0}

        stats = {"split": 0, "clone": 0, "prune": 0}
        n_before = self.gaussian_model.num_points

        # D-04: Non-overlapping intervals — split/clone on even densify steps,
        # prune on odd. This guarantees at most one structural change per event
        # and therefore at most one _rebuild_optimizer call per event.
        densify_step_index = (iteration - start_iter) // interval
        is_split_iter = (densify_step_index % 2 == 0)
        is_prune_iter = (densify_step_index % 2 == 1)
        did_modify = False

        if is_split_iter:
            grad_stats = self.compute_gradient_stats()
            if grad_stats:
                grad_norm = grad_stats["grad_norm"]
                grad_threshold = adaptive_config.get("grad_threshold", 0.01)
                scale_threshold = adaptive_config.get("scale_threshold", 0.01)
                long_axis_offset_factor = self.config["gaussian"].get("long_axis_offset_factor", 1.0)

                high_grad_mask = grad_norm > grad_threshold
                scales = self.gaussian_model.get_scale_values()
                max_scale = scales.max(dim=-1)[0]
                split_mask = high_grad_mask & (max_scale > scale_threshold)

                if split_mask.sum() > 0 and self.gaussian_model.num_points + int(split_mask.sum().item()) <= max_num_points:
                    stats["split"] = self.gaussian_model.densify_and_split(
                        grads=grad_norm,
                        grad_threshold=grad_threshold,
                        scale_threshold=scale_threshold,
                        use_long_axis_splitting=self._resolved_use_long_axis,
                        long_axis_offset_factor=long_axis_offset_factor,
                    )
                    if stats["split"] > 0:
                        did_modify = True

                # Cloning (if enabled) also runs on split iters — but only when
                # no split happened, so a single densification event touches
                # exactly one structural op.
                if self._resolved_use_cloning and not did_modify:
                    clone_mask = high_grad_mask & (max_scale <= scale_threshold)
                    if clone_mask.sum() > 0 and self.gaussian_model.num_points + int(clone_mask.sum().item()) <= max_num_points:
                        stats["clone"] = self.gaussian_model.densify_and_clone(grad_norm, grad_threshold, scale_threshold)
                        if stats["clone"] > 0:
                            did_modify = True

        elif is_prune_iter:
            opacity_threshold = adaptive_config.get("opacity_threshold", 0.001)
            max_scale_limit = adaptive_config.get("max_scale", 0.5)
            scales = self.gaussian_model.get_scale_values()
            max_scale = scales.max(dim=-1)[0]
            densities = torch.abs(self.gaussian_model.density)
            prune_mask = (densities < opacity_threshold) | (max_scale > max_scale_limit)
            keep_mask = ~prune_mask
            if keep_mask.sum() >= 100 and prune_mask.sum() > 0:
                self.gaussian_model._update_params(keep_mask, is_prune=True)
                stats["prune"] = int(prune_mask.sum().item())
                did_modify = True

        # D-04: Single _rebuild_optimizer call per densification event.
        if did_modify:
            self._rebuild_optimizer()

        n_after = self.gaussian_model.num_points
        if did_modify:
            # D-06: Log Gaussian count and Adam state tensor count.
            n_state = sum(
                1
                for g in self.optimizer.param_groups
                for p in g["params"]
                if p in self.optimizer.state and "exp_avg" in self.optimizer.state[p]
            )
            print(
                f"[Densify iter={iteration}] Gaussians: {n_before} -> {n_after}, "
                f"Adam state tensors: {n_state}"
            )

        return stats

    # ===== INF-06 (Phase 7 v2 infrastructure) =====
    # CONTRACT: `_transplant_adam_state` iterates `new_optimizer.param_groups`
    # zipped with `old_optimizer.param_groups`, then within each group iterates
    # `params` by index. This is NAME-AGNOSTIC — it does NOT rely on parameter
    # identifiers like 'density_real' or 'positions'. Therefore, when v2
    # BayesianJointGaussianModel3D adds 6 extra posterior parameters
    # (log_s2_plain, log_s2_contrast, log_sigma2_mu, log_sigma2_ell,
    #  eta_plain, eta_contrast) via `get_optimizable_params`, this method
    # transplants their Adam state TRANSPARENTLY — provided the v2 subclass
    # appends them to the param-groups list in a stable order and overrides
    # `_update_params` to keep their `.shape[0]` consistent with positions.
    #
    # Do NOT replace this iteration with hardcoded parameter names — that
    # would break Phase 8 BP-01 (BayesianJointGaussianModel3D) silently.
    # See: .planning/phases/07-v2-v2-infrastructure/07-04-*-PLAN.md,
    #      PITFALLS.md A-06 + D-02, REQUIREMENTS.md INF-06.
    # ================================================
    @staticmethod
    def _transplant_adam_state(
        old_optimizer: optim.Adam,
        old_params_map: dict,
        keep_mask: torch.Tensor,
        is_prune: bool,
        new_optimizer: optim.Adam,
    ):
        """Copy Adam exp_avg / exp_avg_sq from old optimizer to new one,
        selecting only the kept rows (via *keep_mask*) and zero-padding any
        newly appended rows (from split / clone).

        INF-06 (Phase 7): This method is **v2-transparent** by design — it
        iterates `param_groups` and `params` by index rather than parameter
        name, so it covers any number of additional `nn.Parameter` fields
        added by future subclasses (e.g. BayesianJointGaussianModel3D's
        log_s2_*, log_sigma2_*, eta_* posterior parameters) without
        modification."""
        for new_group, old_group in zip(
            new_optimizer.param_groups, old_optimizer.param_groups
        ):
            for new_p, old_p in zip(new_group["params"], old_group["params"]):
                if old_p not in old_optimizer.state:
                    continue
                old_s = old_optimizer.state[old_p]
                if "exp_avg" not in old_s:
                    continue
                new_s = {}
                for key in ("exp_avg", "exp_avg_sq"):
                    old_buf = old_s[key]  # shape (N_old, ...)
                    kept = old_buf[keep_mask]  # shape (N_kept, ...)
                    if is_prune or kept.shape[0] == new_p.shape[0]:
                        new_s[key] = kept
                    else:
                        # split / clone appended rows; pad with zeros
                        pad_shape = list(kept.shape)
                        pad_shape[0] = new_p.shape[0] - kept.shape[0]
                        new_s[key] = torch.cat(
                            [kept, torch.zeros(pad_shape, device=kept.device, dtype=kept.dtype)],
                            dim=0,
                        )
                new_s["step"] = old_s.get("step", torch.tensor(0.0))
                new_optimizer.state[new_p] = new_s

    def _rebuild_optimizer(self):
        gaussian_config = self.config["gaussian"]
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get("position_lr", 1e-4),
            lr_density=gaussian_config.get("density_lr", 1e-3),
            lr_scale=gaussian_config.get("scale_lr", 5e-4),
            lr_rotation=gaussian_config.get("rotation_lr", 1e-4),
        )
        # Capture current LR so the new optimizer continues at the decayed LR.
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        old_optimizer = self.optimizer
        self.optimizer = optim.Adam(params)
        for group, lr in zip(self.optimizer.param_groups, current_lrs):
            group["lr"] = lr
            group["initial_lr"] = lr

        # Transplant Adam momentum from surviving Gaussian rows.
        old_params_map = getattr(self.gaussian_model, "_last_densify_old_params", None)
        keep_mask = getattr(self.gaussian_model, "_last_densify_keep_mask", None)
        is_prune = getattr(self.gaussian_model, "_last_densify_is_prune", False)
        if old_params_map is not None and keep_mask is not None:
            # D-15: Snapshot exp_avg for kept rows so we can verify the
            # transplant preserves momentum for retained Gaussians.
            old_exp_avg_snapshots = {}
            for gi, group in enumerate(old_optimizer.param_groups):
                for pi, p in enumerate(group["params"]):
                    if p in old_optimizer.state and "exp_avg" in old_optimizer.state[p]:
                        old_exp_avg_snapshots[(gi, pi)] = (
                            old_optimizer.state[p]["exp_avg"][keep_mask].clone()
                        )

            self._transplant_adam_state(
                old_optimizer, old_params_map, keep_mask, is_prune, self.optimizer
            )

            # D-15: Verify exp_avg values for retained Gaussians are unchanged.
            for gi, group in enumerate(self.optimizer.param_groups):
                for pi, p in enumerate(group["params"]):
                    if (gi, pi) in old_exp_avg_snapshots and p in self.optimizer.state:
                        expected = old_exp_avg_snapshots[(gi, pi)]
                        new_kept = self.optimizer.state[p]["exp_avg"][: expected.shape[0]]
                        if not torch.equal(new_kept, expected):
                            print(
                                f"[WARNING] Adam exp_avg value changed for retained Gaussians "
                                f"in param group {gi}, param {pi}"
                            )

            # D-05: Shape assertions after transplant — catch any mismatch
            # between Adam state tensors and their parameter shapes early.
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state and "exp_avg" in self.optimizer.state[p]:
                        assert self.optimizer.state[p]["exp_avg"].shape == p.shape, (
                            f"Adam state shape mismatch after rebuild: "
                            f"exp_avg {self.optimizer.state[p]['exp_avg'].shape} != param {p.shape}"
                        )
                        assert self.optimizer.state[p]["exp_avg_sq"].shape == p.shape, (
                            f"Adam state shape mismatch after rebuild: "
                            f"exp_avg_sq {self.optimizer.state[p]['exp_avg_sq'].shape} != param {p.shape}"
                        )

            # === INF-06 (Phase 7): v2-aware total-state-count check ===
            # When v2 BayesianJointGaussianModel3D adds 6 extra posterior
            # parameters, this assertion makes the param-vs-state count
            # divergence FAIL LOUDLY instead of silently working with stale
            # state. (Adam.state dict may legitimately be smaller than
            # parameters list for params that haven't received a gradient
            # update yet; here we only assert state <= params, not equality.)
            n_params_total = sum(
                1
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p.requires_grad
            )
            n_state_total = sum(
                1
                for group in self.optimizer.param_groups
                for p in group["params"]
                if p in self.optimizer.state and "exp_avg" in self.optimizer.state[p]
            )
            assert n_state_total <= n_params_total, (
                f"[INF-06] Adam state contains {n_state_total} entries for "
                f"{n_params_total} trainable params; state>params indicates "
                f"orphan Adam state (densify cleanup bug)."
            )

            self.gaussian_model._last_densify_old_params = None
            self.gaussian_model._last_densify_keep_mask = None

        scheduler_config = self.config["training"].get("lr_scheduler", {})
        if scheduler_config.get("type", "exponential") == "exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=scheduler_config.get("gamma", 0.999))
        else:
            remaining = max(self.config["training"]["max_iterations"] - self.current_iteration, 1)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining)
        self._skip_scheduler_step_once = True

    def train_step(self):
        self.gaussian_model.train()
        self.optimizer.zero_grad()
        _, loss_dict = self.forward_with_loss(iteration=self.current_iteration)
        loss_dict["total_loss"].backward()
        if self.gaussian_model.positions.grad is not None:
            self._position_grads_for_densify = self.gaussian_model.positions.grad.detach().clone()
        else:
            self._position_grads_for_densify = None
        max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.gaussian_model.parameters(), max_grad_norm)
        self.optimizer.step()
        return {key: float(value.item()) for key, value in loss_dict.items()}

    def evaluate(self) -> Dict[str, float]:
        if self.self_supervised_deploy or not self.experimental_reproduction:
            return {}
        self.gaussian_model.eval()
        with torch.no_grad():
            volume = self._apply_hard_data_consistency(self._as_complex_volume(self._get_volume()))
            return evaluate_reconstruction(
                pred=volume,
                target=self.target_image,
                compute_3d_ssim=True,
                compute_lpips_metric=self.metric_config.get("compute_lpips_during_train", False),
                lpips_device=self.device,
                require_lpips=self.require_lpips,
            )

    def _metric_value_for_best(self, metrics: Dict[str, float]) -> float:
        value = metrics.get(self.best_metric_name)
        if value is None:
            return float("-inf")
        if self.best_metric_name == "lpips":
            return -float(value)
        return float(value)

    def _update_best_metrics(self, metrics: Dict[str, float]) -> bool:
        if not metrics:
            return False
        score = self._metric_value_for_best(metrics)
        best_score = self._metric_value_for_best(self.best_metrics) if self.best_metrics else float("-inf")
        is_best = score > best_score
        if is_best:
            self.best_metrics = dict(metrics)
            self.best_psnr = metrics.get("psnr", self.best_psnr)
            self.best_ssim = metrics.get("ssim", self.best_ssim)
            self.best_lpips = metrics.get("lpips", self.best_lpips)
        return is_best

    def _check_loss_plateau(self, iteration: int, loss_value: float) -> bool:
        if not self.enable_loss_plateau:
            return False
        if loss_value < self.loss_plateau_best - self.loss_min_delta:
            self.loss_plateau_best = loss_value
            self.loss_plateau_bad_count = 0
        else:
            self.loss_plateau_bad_count += 1
        return iteration >= self.loss_min_iteration and self.loss_plateau_bad_count >= self.loss_patience

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        checkpoint = {
            "iteration": iteration,
            "gaussian_state": self.gaussian_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_metrics": self.best_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, "best.pth"))
        save_every = self.config["training"].get("save_every", 500)
        if iteration % save_every == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:06d}.pth"))

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gaussian_model.load_state_dict(checkpoint["gaussian_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.current_iteration = checkpoint["iteration"]
        self.best_metrics = checkpoint.get("best_metrics", {})
        self.best_psnr = self.best_metrics.get("psnr", self.best_psnr)
        self.best_ssim = self.best_metrics.get("ssim", self.best_ssim)
        self.best_lpips = self.best_metrics.get("lpips", self.best_lpips)
        print(f"Loaded checkpoint from iteration {self.current_iteration}")

    def save_reconstruction(self, iteration: int):
        if savemat is None:
            raise ImportError("Saving .mat requires scipy")
        self.gaussian_model.eval()
        with torch.no_grad():
            volume_raw = self._as_complex_volume(self._get_volume())
            volume = self._apply_hard_data_consistency(volume_raw)
            savemat(os.path.join(self.result_dir, f"reconstruction_raw_{iteration:06d}.mat"), {"reconstruction_raw": volume_raw.detach().cpu().numpy()}, do_compression=True)
            volume_np = volume.detach().cpu().numpy()
            savemat(os.path.join(self.result_dir, f"reconstruction_{iteration:06d}.mat"), {"reconstruction": volume_np}, do_compression=True)
            savemat(os.path.join(self.result_dir, "reconstruction_final.mat"), {"reconstruction": volume_np}, do_compression=True)

    def train(self, resume_from: Optional[str] = None):
        train_config = self.config["training"]
        max_iterations = train_config["max_iterations"]
        eval_every = train_config.get("eval_every", 100)
        log_every = train_config.get("log_every", 50)
        save_every = train_config.get("save_every", 500)

        if resume_from is not None:
            self.load_checkpoint(resume_from)

        print(f"\nStarting training from iteration {self.current_iteration}")
        print(f"Total iterations: {max_iterations}")
        print("-" * 60)

        should_stop = False
        stop_reason = None
        last_iteration = self.current_iteration

        # Smoke test tracking (lightweight — runs for all configs)
        smoke_losses: Dict[int, float] = {}   # {iteration: loss_value}
        smoke_grads: Dict[int, float] = {}    # {iteration: grad_norm_value}
        nan_inf_detected = False
        nan_inf_iteration: Optional[int] = None

        progress = tqdm(range(self.current_iteration, max_iterations), desc="Training", dynamic_ncols=True)
        for iteration in progress:
            self.current_iteration = iteration
            last_iteration = iteration
            loss_dict = self.train_step()

            # Track loss for smoke test report
            loss_val = loss_dict["total_loss"]
            smoke_losses[iteration] = loss_val
            if not nan_inf_detected and (
                loss_val != loss_val or abs(loss_val) == float("inf")  # NaN or Inf check
            ):
                nan_inf_detected = True
                nan_inf_iteration = iteration

            # Track grad norms every 10 iterations for smoke test report (D-11)
            if iteration % 10 == 0:
                grad_stats_smoke = self.compute_gradient_stats()
                if grad_stats_smoke:
                    smoke_grads[iteration] = float(
                        grad_stats_smoke.get("mean_grad", torch.tensor(0.0)).item()
                    )

            adaptive_stats = self.adaptive_density_control(iteration)
            if self._skip_scheduler_step_once:
                self._skip_scheduler_step_once = False
            else:
                self.scheduler.step()

            if iteration % log_every == 0:
                grad_stats = self.compute_gradient_stats()
                mean_grad = grad_stats.get("mean_grad", torch.tensor(0.0)).item() if grad_stats else 0.0
                progress.set_postfix({"loss": f"{loss_dict['total_loss']:.2e}", "grad": f"{mean_grad:.2e}", "n_pts": self.gaussian_model.num_points})
                if self.writer:
                    self.writer.add_scalar("Loss/total_loss", loss_dict["total_loss"], iteration)
                    if "kspace_loss" in loss_dict:
                        self.writer.add_scalar("Loss/kspace_loss", loss_dict["kspace_loss"], iteration)
                    if "tv_loss" in loss_dict:
                        self.writer.add_scalar("Loss/tv_loss", loss_dict["tv_loss"], iteration)
                    if "wavelet_loss" in loss_dict:
                        self.writer.add_scalar("Loss/wavelet_loss", loss_dict["wavelet_loss"], iteration)
                    self.writer.add_scalar("Stats/num_points", self.gaussian_model.num_points, iteration)
                    self.writer.add_scalar("Stats/mean_grad", mean_grad, iteration)

            metrics = {}
            if iteration % eval_every == 0 or iteration == max_iterations - 1:
                if self.experimental_reproduction and not self.self_supervised_deploy:
                    metrics = self.evaluate()
                    is_best = self._update_best_metrics(metrics)
                    if self.writer:
                        for key, value in metrics.items():
                            self.writer.add_scalar(f"Metrics/{key.upper()}", value, iteration)
                    print(f"\n[Iter {iteration}] metrics: {metrics}")
                    if any(value > 0 for value in adaptive_stats.values()):
                        print(f"  Density control: {adaptive_stats}")
                    print(f"  Num Gaussians: {self.gaussian_model.num_points}")
                    self.save_checkpoint(iteration, is_best=is_best)
                    if self.metric_plateau_stopper is not None:
                        should_stop = self.metric_plateau_stopper.update(
                            iteration,
                            {name: metrics[name] for name in ("psnr", "ssim", "lpips") if name in metrics},
                        )
                        if should_stop:
                            stop_reason = "GT metrics plateau"
                else:
                    if self._check_loss_plateau(iteration, loss_dict["total_loss"]):
                        should_stop = True
                        stop_reason = "loss plateau"
                    if iteration % save_every == 0 or iteration == max_iterations - 1:
                        self.save_checkpoint(iteration, is_best=False)

            if should_stop:
                print(f"\nEarly stop at iteration {iteration}: {stop_reason}")
                break

        print("\n" + "=" * 60)
        print("Training completed")
        if self.best_metrics:
            print(f"Best metrics: {self.best_metrics}")
        self.save_reconstruction(last_iteration)

        # Write smoke_test_report.txt (D-10, D-11)
        report_path = os.path.join(self.result_dir, "smoke_test_report.txt")
        try:
            # D-10: Compute PASS/FAIL verdict
            loss_50 = smoke_losses.get(49, smoke_losses.get(50, None))
            loss_final = smoke_losses.get(last_iteration, None)

            # Find the closest iter to 50 for warmup reference if exact not present
            if loss_50 is None:
                candidates = [i for i in smoke_losses if 40 <= i <= 60]
                if candidates:
                    loss_50 = smoke_losses[min(candidates, key=lambda x: abs(x - 50))]

            convergence_pass = (
                loss_50 is not None
                and loss_final is not None
                and loss_final < loss_50
            )

            verdict = "PASS" if (not nan_inf_detected and convergence_pass) else "FAIL"
            failure_reasons = []
            if nan_inf_detected:
                failure_reasons.append(f"NaN/Inf detected at iteration {nan_inf_iteration}")
            if not convergence_pass:
                if loss_50 is not None and loss_final is not None:
                    failure_reasons.append(
                        f"Loss did not decrease: loss@50={loss_50:.6e}, loss@final={loss_final:.6e}"
                    )
                else:
                    failure_reasons.append("Insufficient iterations to check convergence")

            max_grad = max(smoke_grads.values()) if smoke_grads else 0.0

            with open(report_path, "w") as rf:
                rf.write("3DGSMR Smoke Test Report\n")
                rf.write("=" * 40 + "\n")
                rf.write(f"Config: {self.config.get('output', {}).get('exp_name', 'unknown')}\n")
                rf.write(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                rf.write(f"Output dir: {self.result_dir}\n")
                rf.write(f"Total iterations: {last_iteration + 1}\n\n")
                rf.write(f"VERDICT: {verdict}\n\n")
                if failure_reasons:
                    rf.write("Failure reasons:\n")
                    for reason in failure_reasons:
                        rf.write(f"  - {reason}\n")
                    rf.write("\n")
                rf.write("Loss trajectory:\n")
                for it in sorted(smoke_losses.keys()):
                    if it % 10 == 0 or it == last_iteration or it in (0, 49, 50):
                        rf.write(f"  iter {it:>4d}: {smoke_losses[it]:.6e}\n")
                rf.write("\nGradient norms (every 10 iters):\n")
                for it in sorted(smoke_grads.keys()):
                    rf.write(f"  iter {it:>4d}: {smoke_grads[it]:.6f}\n")
                rf.write(
                    "\nNaN/Inf detected: "
                    + ("Yes (iter " + str(nan_inf_iteration) + ")" if nan_inf_detected else "No")
                    + "\n"
                )
                rf.write(f"Max grad norm observed: {max_grad:.6f}\n")
                if loss_50 is not None:
                    rf.write(f"Loss at ~iter 50 (post-warmup ref): {loss_50:.6e}\n")
                if loss_final is not None:
                    rf.write(f"Loss at iter {last_iteration} (final): {loss_final:.6e}\n")
                if loss_50 is not None and loss_final is not None:
                    rf.write(
                        f"Convergence check (loss_final < loss_50): "
                        f"{'PASS' if convergence_pass else 'FAIL'}\n"
                    )

            print(f"\n[Smoke Test] Report written to: {report_path}")
            print(f"[Smoke Test] VERDICT: {verdict}")
        except Exception as exc:
            print(f"[Smoke Test] Warning: failed to write report: {exc}")

        return self.best_metrics
