"""Pieces of M1, the registration-driven inter-scan alignment (torch, pure functions).

canonical_zf: the T2W zero-filled image resampled into the canonical (T1) frame, so that the trainer's density
initialisation, which reads the image at the untransformed Gaussian positions, reads it where those Gaussians land
in the T2W frame. ngf_loss: edge-direction mismatch of two images of different contrast (normalised gradient field).
Convention as everywhere: T maps a canonical point r (mm, FOV-centre origin) to its position in the T2W image."""
import numpy as np
import torch
import torch.nn.functional as F


def canonical_zf(zf, inter):
    """zf: complex (N0, N1, N2) image in the T2W frame; inter: geom.Rigid. Returns the image whose value at the
    canonical voxel r is zf(T r) (trilinear; zero outside the field of view)."""
    R, t = np.asarray(inter.R, dtype=np.float64), np.asarray(inter.t, dtype=np.float64)
    if np.array_equal(R, np.eye(3)) and not t.any():
        return zf
    shape = tuple(zf.shape)
    dev = zf.device
    cen = torch.tensor([(n - 1) / 2.0 for n in shape], dtype=torch.float64, device=dev)
    idx = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64, device=dev) for n in shape],
                                     indexing="ij"), dim=-1)
    q = (idx - cen) @ torch.as_tensor(R, device=dev).T + torch.as_tensor(t, device=dev) + cen      # voxel index of T r
    norm = torch.stack([2.0 * q[..., a] / (shape[a] - 1) - 1.0 for a in (2, 1, 0)], dim=-1)        # grid_sample: x, y, z
    vr = torch.view_as_real(zf).permute(3, 0, 1, 2)[None]
    out = F.grid_sample(vr, norm[None].to(vr.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    return torch.view_as_complex(out.permute(1, 2, 3, 0).contiguous())


def ngf_loss(a, b, mask, eps_q=0.1):
    """1 - mean over the mask of (grad a . grad b)^2 / ((|grad a|^2 + ea^2)(|grad b|^2 + eb^2)); a, b: real
    (N0, N1, N2) magnitudes; central differences; e = eps_q x the median gradient norm inside the mask (constant
    for the gradient). Smallest when the edges are parallel, whatever the contrast or its sign; it does not reach 0,
    because voxels whose gradient is below e contribute less than 1."""
    ga, gb = torch.stack(torch.gradient(a)), torch.stack(torch.gradient(b))
    na, nb = (ga * ga).sum(0), (gb * gb).sum(0)
    ea = eps_q * na[mask].detach().sqrt().median()
    eb = eps_q * nb[mask].detach().sqrt().median()
    ngf = (ga * gb).sum(0).pow(2) / ((na + ea * ea) * (nb + eb * eb))
    return 1.0 - ngf[mask].mean()


class Aligner:
    """The alignment step of M1: between `start` and `end` the coarse control grid of the adapter takes one Adam step
    per iteration on  weight * ngf_loss(|moving|, |reference|, mask) + adapter.regulariser(reg).

    moving = the T1W density rendered through the adapter with the Gaussian parameters detached; reference = the
    current T2W render, detached. Only the control grid gets a gradient, from this term only, through an optimiser
    of its own: outside step() the grid has requires_grad False, so the reconstruction loss never reaches it, and
    the rigid part is never touched. After `end` the grid stays as it is."""

    def __init__(self, adapter, start, end, lr, weight, reg):
        assert adapter.has_nonrigid, "the alignment step needs an adapter with the control grid"
        self.ad, self.start, self.end, self.weight, self.reg = adapter, int(start), int(end), float(weight), reg
        self.opt = torch.optim.Adam([adapter.ctrl], lr=float(lr))
        self.n_steps, self.last = 0, None

    def active(self, it):
        return self.start <= int(it) < self.end

    def step(self, it, render_moving, reference, mask):
        """render_moving() -> complex or real volume that depends on the control grid. Returns the loss, or None
        outside [start, end)."""
        if not self.active(it):
            return None
        ctrl = self.ad.ctrl
        ctrl.requires_grad_(True)
        try:
            self.opt.zero_grad(set_to_none=True)
            loss = self.weight * ngf_loss(render_moving().abs(), reference.detach().abs(), mask) \
                + self.ad.regulariser(self.reg)
            loss.backward()
            self.opt.step()
        finally:
            ctrl.requires_grad_(False)
            ctrl.grad = None
        self.n_steps += 1
        self.last = float(loss.detach())
        return self.last
