"""Assumed acquisition order of the T2W scan (design section 7.4) and the head pose of every echo train (numpy).

Phase-encode points inside the elliptical support are acquired in linear raster order, kz (array axis 2) as the
outer loop and ky (array axis 1) as the inner loop; every ETL points form one echo train, train n at time n * TR.
The model groups the trains into K segments. The reference train is the one that holds the k-space centre."""
import numpy as np

import geom
from common import ETL, K_SEG


def build(support, k_seg=K_SEG, etl=ETL):
    """support: (ky, kz) boolean array of acquired phase-encode points."""
    support = np.asarray(support, dtype=bool)
    n_pts = int(support.sum())
    n_trains = -(-n_pts // etl)
    order = np.cumsum(support.T.reshape(-1)) - 1                 # kz outer, ky inner
    train_flat = np.where(support.T.reshape(-1), order // etl, -1)
    train_of = train_flat.reshape(support.T.shape).T.astype(np.int32)
    seg_of = np.where(train_of >= 0, train_of * k_seg // n_trains, -1).astype(np.int32)
    centre = (support.shape[0] // 2, support.shape[1] // 2)
    assert support[centre], "the k-space centre is not inside the support"
    frac = np.arange(n_trains, dtype=np.float64) / max(n_trains - 1, 1)
    return dict(train_of=train_of, seg_of=seg_of, n_trains=n_trains, ref_train=int(train_of[centre]),
                ref_seg=int(seg_of[centre]), train_time_frac=frac,
                seg_of_train=(np.arange(n_trains) * k_seg // n_trains).astype(np.int32))


def _raw_pose(events, f):
    theta, t = np.zeros(3), np.zeros(3)
    for e in events:
        if e["kind"] == "drift":
            w = f
        elif e["kind"] == "step":
            w = 1.0 if f >= e["at"] else 0.0
        else:
            raise ValueError(f"unknown intra-scan event {e['kind']!r}")
        theta += w * np.asarray(e["theta_deg"], dtype=np.float64)
        t += w * np.asarray(e["t_mm"], dtype=np.float64)
    return geom.rigid_from_pivot(theta, t)


def poses(events, train_time_frac, ref_train):
    """Pose of every echo train relative to the reference train: Delta_n = P_n o P_ref^-1, so the reference train is
    the identity and the inter-scan pose is the pose at the time the k-space centre was acquired."""
    raw = [_raw_pose(events, float(f)) for f in train_time_frac]
    inv = raw[ref_train].inverse()
    return [p.compose(inv) for p in raw]


def segment_mean_poses(train_poses, seg_of_train, k_seg=K_SEG):
    """Mean pose of each segment (axis-angle and translation averaged over its trains): what a K-segment model can
    represent at best; used by the oracle arm."""
    out = []
    for s in range(k_seg):
        idx = [i for i in range(len(train_poses)) if seg_of_train[i] == s]
        th = np.mean([geom.axis_angle(train_poses[i].R) for i in idx], axis=0)
        t = np.mean([train_poses[i].t for i in idx], axis=0)
        out.append(geom.Rigid(geom.rotmat(th), t))
    return out
