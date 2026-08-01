#!/usr/bin/env python3
"""The tiled voxelizer's OUTPUT must not depend on its tiling.

tile_size only groups voxels into CUDA blocks. The value written to a voxel is
sum_g density_g * exp(-mahal/2) over mahal <= 9, which contains no reference to the tiling.
So two things must hold, and neither was tested before:

  1. render(tile_size=4) == render(tile_size=8) == render(tile_size=16)
  2. the tiled render equals a brute-force reference that uses no tiles at all

Both fail if build_tile_pairs' bounding box is smaller than the mahal<=9 ellipsoid, because a
gaussian is then silently cut off at the edge of whichever tiles its bbox happens to touch.
The fixture below contains one gaussian for each way that used to happen.

Run: PYTHONNOUSERSITE=1 /home/congcongliu/anaconda3/envs/GSTMI/bin/python tests/test_tile_voxelizer_invariance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paired_gs"))
import torch
from gaussian.tile_voxelizer import TileVoxelizer, _quaternion_to_rotation_matrix

# Anisotropic on purpose: half_shape = (48, 40, 48), so a scale sitting on the short axis
# index needs a LARGER voxel radius on the long axes than an isotropic max() bound gives.
# The volume must also be much wider than the old clamp (20 voxels), otherwise every bbox is
# clipped to the volume bounds, the tile union covers everything and the defect cannot show.
SHAPE = (96, 80, 96)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def fixture():
    """positions, scales, rotations, density — one gaussian per failure mode.

    Centres stay near the middle so that every support region is strictly interior; a bbox
    clipped by the volume bounds would mask the defect rather than expose it.
    """
    pos, sca, rot = [], [], []

    def add(p, s, q):
        pos.append(p)
        sca.append(s)
        rot.append(q)

    q90 = [0.7071068, 0.0, 0.0, 0.7071068]          # 90 deg about axis 2, swaps axes 0 and 1
    ident = [1.0, 0.0, 0.0, 0.0]

    # (a) controls: comfortably inside any bbox, clamp inactive
    add([0.0, 0.0, 0.0], [0.05, 0.05, 0.05], ident)
    add([-0.08, 0.06, 0.05], [0.04, 0.06, 0.05], ident)
    # (b) 3*sigma = 28.8 voxels > max_radius=20 -> the clamp truncates the support
    add([0.05, -0.05, 0.0], [0.20, 0.20, 0.20], ident)
    add([0.0, 0.05, -0.06], [0.30, 0.12, 0.12], ident)
    # (c) needle: long scale on the SHORT axis index, rotated onto voxel axis 0. The isotropic
    #     bound 3*max(scale*half_shape) = 19.2 stays under the clamp, yet the true half-extent
    #     on axis 0 is 23.04 -> under-covered by anisotropy ALONE, with no clamp involved.
    add([-0.06, -0.05, 0.04], [0.005, 0.16, 0.005], q90)
    add([0.06, 0.03, -0.04], [0.005, 0.20, 0.005], q90)

    g = torch.Generator().manual_seed(0)
    n = len(pos)
    dens = torch.complex(torch.randn(n, generator=g), torch.randn(n, generator=g))
    return (torch.tensor(pos), torch.tensor(sca), torch.tensor(rot), dens)


def reference_render(positions, scales, rotations, density):
    """Brute force, no tiles: exactly the quantity the kernel is specified to compute."""
    shape_t = torch.tensor(SHAPE, dtype=torch.float32, device=positions.device)
    half = shape_t * 0.5
    center = (positions + 1.0) * 0.5 * shape_t - 0.5
    R = _quaternion_to_rotation_matrix(rotations)
    transform = R * torch.reciprocal(scales + 1e-8).unsqueeze(1)
    cov_inv = transform @ transform.transpose(1, 2)

    grid = torch.stack(torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32, device=positions.device) for s in SHAPE],
        indexing="ij"), dim=-1).view(-1, 3)
    out = torch.zeros(grid.shape[0], dtype=torch.complex64, device=positions.device)
    for i in range(positions.shape[0]):
        d = (grid - center[i]) / half
        mahal = ((d @ cov_inv[i]) * d).sum(-1)
        w = torch.where(mahal <= 9.0, torch.exp(-0.5 * mahal), torch.zeros_like(mahal))
        out = out + w.to(torch.complex64) * density[i]
    return out.view(*SHAPE)


def render(tile_size, args):
    vox = TileVoxelizer(SHAPE, tile_size=tile_size, max_radius=20, device=DEV)
    with torch.no_grad():
        v = vox(positions=args[0], scales=args[1], rotations=args[2], density=args[3])
    return torch.complex(v[0], v[1]) if isinstance(v, tuple) else v


def _args():
    return tuple(t.to(DEV) for t in fixture())


def test_render_is_invariant_to_tile_size():
    a = _args()
    ref = render(8, a)
    scale = ref.abs().max().item()
    for ts in (4, 16, 32):
        other = render(ts, a)
        err = (other - ref).abs().max().item() / scale
        assert err < 1e-5, f"tile_size={ts} changed the render by {err:.3e} of peak"
    print("PASS test_render_is_invariant_to_tile_size")


def test_matches_brute_force_reference():
    a = _args()
    ref = reference_render(*a)
    scale = ref.abs().max().item()
    for ts in (4, 8, 16):
        got = render(ts, a)
        err = (got - ref).abs().max().item() / scale
        assert err < 1e-5, f"tile_size={ts} differs from the untiled reference by {err:.3e} of peak"
    print("PASS test_matches_brute_force_reference")


def test_large_gaussians_are_present_in_the_fixture():
    """Guards the fixture itself: if these stop triggering, the tests above go blind."""
    pos, sca, rot, _ = fixture()
    half = torch.tensor(SHAPE, dtype=torch.float32) * 0.5
    iso_clamped = ((sca * half).max(dim=-1)[0] * 3.0).clamp(max=20.0)
    R = _quaternion_to_rotation_matrix(rot)
    exact = 3.0 * (R ** 2 * (sca ** 2).unsqueeze(1)).sum(-1).sqrt() * half.unsqueeze(0)
    uncovered = (exact > iso_clamped.unsqueeze(1) + 1e-6).any(dim=1)
    clamp_active = (sca * half).max(dim=-1)[0] * 3.0 > 20.0
    aniso_only = uncovered & ~clamp_active
    interior = (exact < half.unsqueeze(0)).all(dim=1)          # bbox not clipped by the volume
    assert int(uncovered.sum()) >= 3, (
        f"fixture no longer exercises the defect: only {int(uncovered.sum())} gaussians "
        f"exceed the old bbox")
    assert int(aniso_only.sum()) >= 1, (
        "fixture lost its clamp-free case, so it can no longer tell the anisotropy defect "
        "apart from the max_radius one")
    assert bool(interior.all()), (
        "some support region reaches the volume bounds; bbox clipping would hide the defect")
    print(f"PASS test_large_gaussians_are_present_in_the_fixture "
          f"({int(uncovered.sum())}/{len(uncovered)} exceed the old bbox, "
          f"{int(aniso_only.sum())} by anisotropy alone)")


if __name__ == "__main__":
    test_large_gaussians_are_present_in_the_fixture()
    test_render_is_invariant_to_tile_size()
    test_matches_brute_force_reference()
    print("ALL PASS")
