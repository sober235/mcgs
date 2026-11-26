"""M4: the first version's motion adapter with a control-grid deformation field for every state of the T2W scan.

A Gaussian centre r (mm, canonical frame) of state k is moved to  pose_k(inter(r + u_k(r))):  the state's field
first, then the inter-scan pose and the state's rigid pose (both the second version's). u_k is trilinear on a
G x G x G grid, limited to u_max by a tanh and with its mean removed (a mean displacement is the rigid pose's job).
The reference state has no field. The Gaussians' shapes are not changed by the field: its gradient is a few per cent
(3 mm over 30 mm or more), far below the shape's own uncertainty. With every field at zero the transform is the
parent's, bit for bit.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import MotionAdapter  # first version


def field_adapter(grid, u_max_mm):
    """The adapter class mpatch builds (same constructor as MotionAdapter), with the state fields added."""

    class StateFieldAdapter(MotionAdapter):
        GROUPS = MotionAdapter.GROUPS + (("seg_ctrl", "seg_ctrl"),)

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            assert self.n_segments > 0, "state fields need states"
            self.field_grid, self.field_max_mm = int(grid), float(u_max_mm)
            self.seg_ctrl = nn.Parameter(torch.zeros(self.n_segments, grid, grid, grid, 3))

        def field_ctrl_mm(self, seg):
            c = self.field_max_mm * torch.tanh(self.seg_ctrl[int(seg)])
            return c - c.mean(dim=(0, 1, 2), keepdim=True)

        def field(self, p, seg):
            """u_seg at normalised positions p (N, 3), mm."""
            vol = self.field_ctrl_mm(seg).permute(3, 0, 1, 2).unsqueeze(0)
            g = p.flip(-1).reshape(1, -1, 1, 1, 3).to(vol.dtype)
            out = F.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True)
            return out.reshape(3, -1).t().to(p.dtype)

        def transform(self, p, seg=None):
            if self._seg_active(seg) and (self.seg_ctrl.requires_grad or bool(torch.any(self.seg_ctrl[int(seg)] != 0))):
                half = self.half_mm.to(p.dtype)
                p = (p * half + self.field(p, seg)) / half
            return super().transform(p, seg)

        def field_regulariser(self, l2, smooth, states):
            out = torch.zeros((), device=self.half_mm.device)
            for k in states:
                u = self.field_ctrl_mm(k)
                out = out + l2 * u.pow(2).mean() + smooth * sum(u.diff(dim=d).pow(2).mean() for d in range(3))
            return out

        def state(self):
            st = super().state()
            st["field"] = dict(grid=self.field_grid, u_max_mm=self.field_max_mm,
                               ctrl_mm=[self.field_ctrl_mm(k).detach().cpu().tolist() for k in range(self.n_segments)])
            return st

        @torch.no_grad()
        def set_field(self, seg, dense_u):
            """Fix state seg's field to the least-squares fit of dense_u (*shape, 3), mm. The mean of the control
            values, which the field cannot carry, goes into the state's rigid pose as a translation (after the
            inter-scan rotation). Returns (largest control value's norm, the mean), mm."""
            saved, self.grid = self.grid, self.field_grid
            try:
                c = self._fit_ctrl(np.asarray(dense_u, dtype=np.float64))
            finally:
                self.grid = saved
            mean = c.reshape(-1, 3).mean(0)
            c = c - mean
            x = torch.tensor(np.clip(c / self.field_max_mm, -0.999, 0.999), dtype=torch.float32)
            self.seg_ctrl[int(seg)].copy_(torch.atanh(x).to(self.seg_ctrl.device))
            import geom
            from adapter import _rot
            R = _rot(self.theta.detach().double().cpu()).numpy() if self.has_rigid else np.eye(3)
            self.set_segment_pose(seg, geom.Rigid(np.eye(3), R @ mean))
            return float(np.linalg.norm(c, axis=-1).max()), mean.tolist()

    return StateFieldAdapter


def dense_field(ctrl_mm, shape):
    """Trilinear interpolant of control values (G, G, G, 3) on the voxel grid (numpy, mm): what field() gives at
    the voxel centres."""
    c = np.asarray(ctrl_mm, dtype=np.float64)
    G = c.shape[0]
    out = c
    for ax, n in enumerate(shape):
        x = (np.arange(n) + 0.5) / n * (G - 1)
        w = np.stack([np.clip(1.0 - np.abs(x - g), 0.0, None) for g in range(G)], axis=1)      # (n, G)
        out = np.moveaxis(np.tensordot(w, np.moveaxis(out, ax, 0), axes=(1, 0)), 0, ax)
    return out
