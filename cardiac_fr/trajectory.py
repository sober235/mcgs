"""3D radial spiral-phyllotaxis trajectory (Piccini et al., MRM 2011), ported line by line from the
authors' phyllotaxis3D.m and computePhyllotaxis.m (QIS-MRI/ReadDataAndTrajectory_FreeRunningFatFractionHeart
@ a78f6f2). Index = acquisition line Lin = shot * n_seg + seg; seg 0 of every shot is the SI spoke."""
from __future__ import annotations

import numpy as np

GOLDEN = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * np.pi - 2 * np.pi / GOLDEN


def phyllotaxis3d(n_shot: int, n_seg: int, self_nav: bool = True) -> tuple[np.ndarray, np.ndarray]:
    n_proj = n_shot * n_seg
    kost = np.pi / (2 * np.sqrt(n_proj - n_shot)) if self_nav else np.pi / (2 * np.sqrt(n_proj))
    polar = np.zeros(n_proj)
    azim = np.zeros(n_proj)
    count = 1
    for lk in range(1, n_seg + 1):              # MATLAB: for lk = 1:m_lProjectionsPerFrame
        for frame in range(1, n_shot + 1):      # MATLAB: for lFrame = 1:m_lNumberOfFrames
            i = (lk - 1) + (frame - 1) * n_seg
            if self_nav and lk == 1:
                continue                        # SI spoke: polar = azimuth = 0, count not advanced
            polar[i] = kost * np.sqrt(count)
            azim[i] = np.mod(count * GOLDEN_ANGLE, 2 * np.pi)
            count += 1
    return polar, azim


def radial_positions(n_samp: int) -> np.ndarray:
    """computePhyllotaxis.m: r = -0.5 : 1/N : 0.5 - 1/N (sample n_samp/2 is the k-space centre)."""
    return -0.5 + np.arange(n_samp) / n_samp


def spoke_coordinates(n_shot: int = 1000, n_seg: int = 22, n_samp: int = 224,
                      self_nav: bool = True) -> np.ndarray:
    """(3, n_samp, n_shot*n_seg) in r units; rows x, y, z = MATLAB sph2cart(azimuth, pi/2 - polar, r)."""
    polar, azim = phyllotaxis3d(n_shot, n_seg, self_nav)
    r = radial_positions(n_samp)[:, None]
    elev = np.pi / 2 - polar
    return np.stack([r * np.cos(elev) * np.cos(azim),
                     r * np.cos(elev) * np.sin(azim),
                     r * np.sin(elev) + 0 * azim])
