"""Motion adapter for MC-GS (design section 6): a small transform applied to the SHARED Gaussians before one
contrast is rendered.

    r' = R_s ( R (r + u(r)) + t ) + t_s

r = mm relative to the FOV centre (r = p * N/2 for the trainer's normalised position p); u = coarse non-rigid field
(5x5x5x3 control grid, trilinear, tanh-bounded, mean removed); (R, t) = inter-scan rigid pose; (R_s, t_s) = extra pose
of acquisition segment s, zero for the reference segment. Rotations are axis-angle vectors about the FOV centre.
Covariances follow the rigid part exactly (Sigma_n' = A Sigma_n A^T, A = D^-1 R D, D = diag(N/2)); the non-rigid part
moves the means only. With every parameter zero the transform is the identity.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROT_ARM_MM = 80.0          # converts a rotation (rad) into a displacement scale for the segment regulariser


def _rot(theta):
    z = torch.zeros((), dtype=theta.dtype, device=theta.device)
    K = torch.stack([torch.stack([z, -theta[2], theta[1]]),
                     torch.stack([theta[2], z, -theta[0]]),
                     torch.stack([-theta[1], theta[0], z])])
    return torch.linalg.matrix_exp(K)


class MotionAdapter(nn.Module):
    def __init__(self, volume_shape, rigid, nonrigid, n_segments, ref_segment, u_max_mm=3.0, grid=5):
        super().__init__()
        self.volume_shape = tuple(int(n) for n in volume_shape)
        self.register_buffer("half_mm", torch.tensor(self.volume_shape, dtype=torch.float32) / 2.0)
        self.has_rigid, self.has_nonrigid = bool(rigid), bool(nonrigid)
        self.n_segments, self.ref_segment = int(n_segments), int(ref_segment)
        self.u_max_mm, self.grid = float(u_max_mm), int(grid)
        self.oracle_fit_mm = None
        if self.has_rigid:
            self.theta = nn.Parameter(torch.zeros(3))
            self.t_mm = nn.Parameter(torch.zeros(3))
        if self.has_nonrigid:
            self.ctrl = nn.Parameter(torch.zeros(grid, grid, grid, 3))
        if self.n_segments > 0:
            self.seg_theta = nn.Parameter(torch.zeros(self.n_segments, 3))
            self.seg_t_mm = nn.Parameter(torch.zeros(self.n_segments, 3))

    # ---- pieces
    def _ctrl_mm(self):
        c = self.u_max_mm * torch.tanh(self.ctrl)
        return c - c.mean(dim=(0, 1, 2), keepdim=True)

    def _u(self, p):
        vol = self._ctrl_mm().permute(3, 0, 1, 2).unsqueeze(0)                  # (1, 3, D, H, W)
        g = p.flip(-1).reshape(1, -1, 1, 1, 3).to(vol.dtype)                     # grid_sample wants (x, y, z) = (W, H, D)
        out = F.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(3, -1).t().to(p.dtype)

    def _seg_active(self, seg):
        return seg is not None and self.n_segments > 0 and int(seg) != self.ref_segment

    def transform(self, p, seg=None):
        """Normalised positions (N, 3) -> (transformed normalised positions, A (3, 3))."""
        half = self.half_mm.to(p.dtype)
        r = p * half
        R = torch.eye(3, dtype=p.dtype, device=p.device)
        if self.has_nonrigid:
            r = r + self._u(p)
        if self.has_rigid:
            R = _rot(self.theta.to(p.dtype))
            r = r @ R.t() + self.t_mm.to(p.dtype)
        if self._seg_active(seg):
            Rs = _rot(self.seg_theta[int(seg)].to(p.dtype))
            r = r @ Rs.t() + self.seg_t_mm[int(seg)].to(p.dtype)
            R = Rs @ R
        A = R * half.unsqueeze(0) / half.unsqueeze(1)                            # A_ij = R_ij half_j / half_i
        return r / half, A

    def is_identity(self):
        return all(not bool(torch.any(p != 0)) for p in self.parameters())

    def trainable(self):
        return any(p.requires_grad for p in self.parameters())

    # ---- optimiser, regulariser, record
    GROUPS = (("rigid_theta", "theta"), ("rigid_t", "t_mm"), ("nonrigid", "ctrl"),
              ("seg_theta", "seg_theta"), ("seg_t", "seg_t_mm"))

    def param_groups(self, lrs, eps=None):
        """eps: Adam's eps for these groups only (None = the optimiser's default). The pose gradients are orders of
        magnitude below the default 1e-8, which turns Adam into plain gradient descent with a vanishing step."""
        extra = {} if eps is None else dict(eps=float(eps))
        return [dict(params=[getattr(self, attr)], lr=float(lrs[name]), name=name, **extra)
                for name, attr in self.GROUPS if hasattr(self, attr)]

    def regulariser(self, w):
        out = torch.zeros((), device=self.half_mm.device)
        if self.has_nonrigid:
            u = self.u_max_mm * torch.tanh(self.ctrl)
            out = out + w["ctrl_l2"] * u.pow(2).mean()
            out = out + w["ctrl_smooth"] * sum((u.diff(dim=d)).pow(2).mean() for d in range(3))
        if self.n_segments > 0:
            s = torch.cat([self.seg_t_mm, ROT_ARM_MM * self.seg_theta], dim=1)
            out = out + w["seg_l2"] * s.pow(2).mean()
            if self.n_segments > 1:
                out = out + w["seg_smooth"] * s.diff(dim=0).pow(2).mean()
        return out

    def state(self):
        st = dict(rigid=self.has_rigid, nonrigid=self.has_nonrigid, n_segments=self.n_segments,
                  ref_segment=self.ref_segment, u_max_mm=self.u_max_mm, oracle_fit_mm=self.oracle_fit_mm,
                  theta_deg=None, t_mm=None, ctrl_mm=None, segments=[])
        if self.has_rigid:
            st["theta_deg"] = torch.rad2deg(self.theta.detach()).cpu().tolist()
            st["t_mm"] = self.t_mm.detach().cpu().tolist()
        if self.has_nonrigid:
            st["ctrl_mm"] = self._ctrl_mm().detach().cpu().tolist()
        for k in range(self.n_segments):
            st["segments"].append(dict(theta_deg=torch.rad2deg(self.seg_theta[k].detach()).cpu().tolist(),
                                       t_mm=self.seg_t_mm[k].detach().cpu().tolist()))
        return st

    # ---- setting poses (requires_grad is left as it is)
    @torch.no_grad()
    def set_inter(self, inter):
        """Inter-scan rigid pose: canonical point r -> inter(r) (mm, FOV-centre origin)."""
        import geom
        assert self.has_rigid, "an inter-scan pose needs an adapter with the rigid part"
        self.theta.copy_(torch.tensor(geom.axis_angle(inter.R), dtype=torch.float32))
        self.t_mm.copy_(torch.tensor(np.asarray(inter.t, dtype=np.float64), dtype=torch.float32))

    @torch.no_grad()
    def set_segment_pose(self, seg, pose):
        """Pose of one segment, applied after the inter-scan transform."""
        import geom
        assert 0 <= int(seg) < self.n_segments, f"segment {seg} of {self.n_segments}"
        self.seg_theta[int(seg)].copy_(torch.tensor(geom.axis_angle(pose.R), dtype=torch.float32))
        self.seg_t_mm[int(seg)].copy_(torch.tensor(np.asarray(pose.t, dtype=np.float64), dtype=torch.float32))

    # ---- oracle
    @torch.no_grad()
    def set_truth(self, inter=None, dense_u=None, segments=None, native_shift_mm=(0.0, 0.0, 0.0), brain=None):
        """Fix the transform to the truth: (additive simulated motion) o (native shift s),
        T(r) = g(r + s + u(r + s)), plus the segment poses. geom.Rigid inputs; dense_u (*shape, 3) in mm.
        The reference segment's mean pose is folded into the inter-scan pose, so every segment keeps its exact
        mean pose and the reference segment stays the identity."""
        import geom
        import warp
        dev = self.half_mm.device
        g = inter if inter is not None else geom.Rigid.identity()
        s = np.asarray(native_shift_mm, dtype=np.float64)
        if segments is not None and self.n_segments > 0:
            ref = segments[self.ref_segment]
            g = ref.compose(g)
            for k, p in enumerate(segments):
                self.set_segment_pose(k, p.compose(ref.inverse()))
        mean_c = s.copy()
        want_u = None
        if dense_u is not None:
            assert self.has_nonrigid, "a non-rigid truth needs an adapter with the non-rigid part"
            r = geom.grid_mm(self.volume_shape)
            want_u = s + warp._interp_field(np.asarray(dense_u, dtype=np.float64), r + s)     # u'(r) = s + u(r + s)
            c = self._fit_ctrl(want_u)
            mean_c = c.reshape(-1, 3).mean(0)
            self.ctrl.copy_(torch.atanh(torch.tensor(np.clip(c / self.u_max_mm, -0.999, 0.999), dtype=torch.float32)))
        if self.has_rigid:
            self.set_inter(geom.Rigid(g.R, g.t + g.R @ mean_c))
        else:
            assert inter is None and not np.any(s), "a rigid truth needs an adapter with the rigid part"
        self.to(dev)
        for p in self.parameters():
            p.requires_grad_(False)
        if want_u is not None:
            r = geom.grid_mm(self.volume_shape)
            saved, self.n_segments = self.n_segments, 0                        # inter-scan part only
            got = dense_displacement(self, self.volume_shape)
            self.n_segments = saved
            err = np.linalg.norm(got - (g.apply(r + want_u) - r), axis=-1)
            self.oracle_fit_mm = float(err[brain].max() if brain is not None else err.max())
        else:
            self.oracle_fit_mm = 0.0

    def _fit_ctrl(self, want_u, step=2):
        """Least-squares control values (G, G, G, 3) whose trilinear interpolant matches want_u on the voxel grid."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        G, shape = self.grid, self.volume_shape
        sl = tuple(slice(None, None, step) for _ in range(3))
        idx = np.stack(np.meshgrid(*[np.arange(n)[::step] for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
        x = (idx + 0.5) / np.asarray(shape) * (G - 1)                  # p in [-1, 1] -> control index in [0, G-1]
        i0 = np.clip(np.floor(x).astype(int), 0, G - 2)
        f = x - i0
        rows, cols, vals = [], [], []
        for da in (0, 1):
            for db in (0, 1):
                for dc in (0, 1):
                    wgt = (f[:, 0] if da else 1 - f[:, 0]) * (f[:, 1] if db else 1 - f[:, 1]) * \
                          (f[:, 2] if dc else 1 - f[:, 2])
                    rows.append(np.arange(len(idx)))
                    cols.append(((i0[:, 0] + da) * G + (i0[:, 1] + db)) * G + (i0[:, 2] + dc))
                    vals.append(wgt)
        W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(idx), G ** 3))
        lhs = (W.T @ W + 1e-6 * sp.identity(G ** 3)).tocsc()
        rhs = W.T @ want_u[sl].reshape(-1, 3)
        return spl.splu(lhs).solve(rhs).reshape(G, G, G, 3)


def render_transformed(voxelizer, positions, scales, rotations, density, adapter, seg=None):
    """Render the Gaussians after the adapter's transform (complex volume). A frozen identity adapter (or None)
    returns the voxelizer's own output, bit for bit."""
    def as_complex(v):
        return torch.complex(v[0], v[1]) if isinstance(v, tuple) else v

    if adapter is None or (adapter.is_identity() and not adapter.trainable()):
        return as_complex(voxelizer(positions=positions, scales=scales, rotations=rotations, density=density))
    from gaussian.tile_voxelizer import (_TileVoxelizeCUDA, _TileVoxelizeCuPy, _quaternion_to_rotation_matrix,
                                         _tile_forward_pytorch, build_tile_pairs)
    d_size, h_size, w_size = voxelizer.volume_shape
    shape_tensor = torch.tensor([d_size, h_size, w_size], device=positions.device, dtype=torch.float32)
    half_shape = shape_tensor * 0.5
    p2, A = adapter.transform(positions, seg)
    center_vox = (p2 + 1.0) * 0.5 * shape_tensor - 0.5
    Rq = _quaternion_to_rotation_matrix(rotations)
    s_inv = torch.reciprocal(scales + 1e-8)
    M = Rq * s_inv.unsqueeze(1)
    A_inv = torch.linalg.inv(A)
    cov_inv = A_inv.t().unsqueeze(0) @ (M @ M.transpose(1, 2)) @ A_inv.unsqueeze(0)
    B = A.unsqueeze(0) @ Rq                                                    # Sigma' = B diag(s^2) B^T
    radius_vox = 3.0 * (B ** 2 * (scales ** 2).unsqueeze(1)).sum(-1).sqrt() * half_shape
    with torch.no_grad():
        tile_ranges, pair_ids = build_tile_pairs(center_vox.detach(), radius_vox.detach(), voxelizer.volume_shape,
                                                 voxelizer.tile_size)
    if voxelizer.use_cuda or voxelizer.use_cupy:
        fn = _TileVoxelizeCUDA if voxelizer.use_cuda else _TileVoxelizeCuPy
        vr, vi = fn.apply(center_vox, cov_inv, density.real.contiguous(), density.imag.contiguous(), tile_ranges,
                          pair_ids, half_shape, voxelizer.tile_size, d_size, h_size, w_size)
        return torch.complex(vr.view(d_size, h_size, w_size), vi.view(d_size, h_size, w_size))
    return as_complex(_tile_forward_pytorch(center_vox, cov_inv, density, half_shape, tile_ranges, pair_ids,
                                            d_size, h_size, w_size, voxelizer.tile_size))


@torch.no_grad()
def dense_displacement(adapter, shape, seg=None):
    """T(r) - r on the voxel grid, mm, numpy (*shape, 3)."""
    dev = adapter.half_mm.device
    axes = [(torch.arange(n, dtype=torch.float32, device=dev) + 0.5) / n * 2.0 - 1.0 for n in shape]
    p = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    p2, _ = adapter.transform(p, seg)
    return ((p2 - p) * adapter.half_mm).reshape(*shape, 3).cpu().numpy()
