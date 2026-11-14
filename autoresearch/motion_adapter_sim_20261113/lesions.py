"""Contrast-specific synthetic lesions (design section 7.3) and the brain mask used for placement and evaluation."""
import numpy as np
import scipy.ndimage as ndi
from scipy.special import erfc

DIAMETERS_T2 = (4.0, 6.0, 8.0, 12.0)
GAINS_T2 = (1.0, 0.9, 0.75, 0.6)          # signal increase relative to local tissue, smallest lesion brightest
T1_DIAMETER, T1_GAIN = 8.0, -0.4
EDGE_SIGMA = 1.0
MIN_SEP = 25.0


def head_mask(v):
    """Rule of the alignment probe (results/motion_adapter_hyw_align/scripts/hyw_align_probe.py)."""
    m = v > 0.08 * np.quantile(v, 0.999)
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    m = ndi.binary_closing(m, iterations=3)
    return np.stack([ndi.binary_fill_holes(s) for s in m])


def brain_mask(mag, erode=12, cut_below=35):
    """Head mask eroded by `erode` voxels, cut `cut_below` voxels below the head's centre of mass (axis 0 runs
    superior -> inferior), largest component. Same rule as the alignment probe."""
    head = head_mask(mag)
    b = ndi.binary_erosion(head, iterations=erode)
    c0 = int(round(ndi.center_of_mass(head)[0]))
    b[c0 + cut_below:] = False
    lab, n = ndi.label(b)
    if n > 1:
        b = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return b


def pick_sites(t1_mag, brain, rng_seed=3):
    """Five lesion sites in white matter (brain voxels brighter than the brain's 60th percentile on T1): four T2-only
    (periventricular = deepest, deep white matter, juxtacortical, posterior-inferior) and one T1-only. Every centre is
    at least radius + 3 mm from the brain boundary and the centres are at least MIN_SEP mm apart."""
    rng = np.random.default_rng(rng_seed)
    depth = ndi.distance_transform_edt(brain)
    wm = brain & (t1_mag > np.quantile(t1_mag[brain], 0.60))
    dq = lambda q: np.quantile(depth[brain], q)
    ax0, ax1 = np.indices(brain.shape)[0], np.indices(brain.shape)[1]
    b0, b1 = ax0[brain], ax1[brain]
    zones = [
        ("T2", DIAMETERS_T2[0], GAINS_T2[0], depth >= dq(0.85)),
        ("T2", DIAMETERS_T2[1], GAINS_T2[1], (depth >= dq(0.55)) & (depth <= dq(0.80))),
        ("T2", DIAMETERS_T2[2], GAINS_T2[2], depth <= DIAMETERS_T2[2] / 2 + 7),
        ("T2", DIAMETERS_T2[3], GAINS_T2[3], (ax0 >= np.quantile(b0, 0.70)) & (ax1 >= np.median(b1))),
        ("T1", T1_DIAMETER, T1_GAIN, depth >= dq(0.40)),
    ]
    sites, centres = [], []
    for contrast, diam, gain, zone in zones:
        ok = wm & (depth >= diam / 2 + 3)
        for cand in (ok & zone, ok):                 # relax the zone if it cannot be honoured
            pts = np.argwhere(cand)
            if centres and len(pts):
                d = np.linalg.norm(pts[:, None, :] - np.asarray(centres)[None], axis=-1).min(1)
                pts = pts[d >= MIN_SEP]
            if len(pts):
                break
        assert len(pts), f"no admissible site for the {diam} mm {contrast} lesion"
        cen = pts[rng.integers(len(pts))]
        centres.append(cen)
        sites.append(dict(center_vox=[int(x) for x in cen], diameter_mm=float(diam), contrast=contrast,
                          gain=float(gain)))
    return sites


def _weight(shape, site):
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij"), axis=-1)
    d = np.linalg.norm(g - np.asarray(site["center_vox"], dtype=np.float64), axis=-1)
    return 0.5 * erfc((d - site["diameter_mm"] / 2.0) / (np.sqrt(2.0) * EDGE_SIGMA))


def insert(vol, sites, contrast):
    """vol * (1 + gain * w) for the sites of this contrast (phase untouched). The label mask (1..len(sites), w > 0.5)
    covers ALL sites, so the same mask marks where the other contrast must stay unchanged."""
    out = np.array(vol, dtype=np.complex128, copy=True)
    mask = np.zeros(vol.shape, dtype=np.uint8)
    for k, s in enumerate(sites, start=1):
        w = _weight(vol.shape, s)
        mask[w > 0.5] = k
        if s["contrast"] == contrast:
            out *= 1.0 + s["gain"] * np.where(w > 1e-9, w, 0.0)
    return out, mask
