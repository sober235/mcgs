"""Geometry conventions of the motion study (numpy).

Voxel index i on axes 0/1/2 = S-I / A-P / L-R, 1 mm. FOV centre c = (N-1)/2. Position in mm relative to the centre:
r = i - c. The trainer's normalised coordinate is p = r / (N/2).

A transform T sends a material point at r in the canonical (T1) frame to its place in the T2W image,
r' = R (r + u(r)) + t, so the moved image is m_T2 = m o T^-1. Rotations are axis-angle vectors (radians, components
on axes 0/1/2): turn = about axis 0, tilt = about axis 1, nod = about axis 2.
"""
from dataclasses import dataclass

import numpy as np

from common import N, PIVOT_OFFSET_MM


def rotmat(theta):
    """Rodrigues: axis-angle vector (3,) -> rotation matrix (3, 3)."""
    theta = np.asarray(theta, dtype=np.float64)
    a = float(np.linalg.norm(theta))
    if a < 1e-15:
        return np.eye(3)
    k = theta / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def axis_angle(R):
    """Rotation matrix -> axis-angle vector; valid for angles below pi."""
    R = np.asarray(R, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(w)) / 2.0
    c = (np.trace(R) - 1.0) / 2.0
    a = float(np.arctan2(s, c))
    if s < 1e-15:
        return np.zeros(3)
    return w / (2.0 * s) * a


@dataclass
class Rigid:
    """r' = R r + t, origin at the FOV centre, mm."""
    R: np.ndarray
    t: np.ndarray

    def apply(self, r):
        return np.asarray(r, dtype=np.float64) @ self.R.T + self.t

    def inverse(self):
        return Rigid(self.R.T, -self.R.T @ self.t)

    def compose(self, other):
        """self o other: other is applied first."""
        return Rigid(self.R @ other.R, self.R @ other.t + self.t)

    def to_json(self):
        return dict(R=np.asarray(self.R).tolist(), t=np.asarray(self.t).tolist(),
                    theta_deg=np.rad2deg(axis_angle(self.R)).tolist())

    @staticmethod
    def from_json(d):
        return Rigid(np.array(d["R"], dtype=np.float64), np.array(d["t"], dtype=np.float64))

    @staticmethod
    def identity():
        return Rigid(np.eye(3), np.zeros(3))


def rigid_from_pivot(theta_deg, t_sim_mm, pivot_offset_mm=PIVOT_OFFSET_MM):
    """r' = R (r - q) + q + t_sim with the pivot q given relative to the FOV centre, returned about the centre."""
    R = rotmat(np.deg2rad(np.asarray(theta_deg, dtype=np.float64)))
    q = np.asarray(pivot_offset_mm, dtype=np.float64)
    return Rigid(R, np.asarray(t_sim_mm, dtype=np.float64) + (np.eye(3) - R) @ q)


def grid_mm(shape=N):
    """(*shape, 3) array of r = i - c."""
    axes = [np.arange(n, dtype=np.float64) - (n - 1) / 2.0 for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def tre(est_disp, true_disp, mask):
    """Target registration error between two displacement fields (*shape, 3) over a boolean mask, mm."""
    d = np.linalg.norm(np.asarray(est_disp)[mask] - np.asarray(true_disp)[mask], axis=-1)
    return dict(mean_mm=float(d.mean()), max_mm=float(d.max()))
