"""Minimal reader for Siemens VD/VE multi-raid twix files (what the free-running data needs).

Layout (verified on the four subjects, 2026-09-10): an 8-byte raid header (0, count) followed by
152-byte entries (measId, fileId, offset, length, patient, protocol). Each measurement starts with a
header (length, n_buffers, then name/length/bytes buffers such as MeasYaps) followed by ADC packets:
a 192-byte scan header plus, per channel, a 32-byte channel header and complex64 samples. SYNCDATA
packets are skipped by their DMA length; ACQEND ends the measurement.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCAN_HDR = 192
CHAN_HDR = 32
MASK_ACQEND = 1 << 0
MASK_SYNCDATA = 1 << 5
MASK_NOISEADJ = 1 << 25


@dataclass
class Measurement:
    meas_id: int
    offset: int
    length: int
    protocol: str


@dataclass
class ScanIndex:
    pos: np.ndarray       # file offset of each non-sync ADC packet
    mask: np.ndarray      # evalInfoMask (uint64)
    ns: np.ndarray
    nch: np.ndarray
    lin: np.ndarray
    eco: np.ndarray
    seg: np.ndarray
    ts: np.ndarray        # timestamp ticks (2.5 ms)
    n_sync: int
    acqend: bool


@dataclass
class FreeRunningData:
    ksp: np.ndarray       # (n_echo, n_spoke, n_coil, n_samp) complex64
    noise: np.ndarray     # (n_coil, n_samp) complex64
    lin: np.ndarray       # (n_spoke,)
    seg: np.ndarray       # (n_spoke,) Seg counter (= interleave for this protocol)
    mask: np.ndarray      # (n_spoke*n_echo,) uint64, acquisition order
    eco: np.ndarray       # (n_spoke*n_echo,)
    ts_ticks: np.ndarray  # (n_spoke*n_echo,)
    protocol: str
    ascconv: dict = field(default_factory=dict)


def read_raid_header(f) -> list[Measurement]:
    f.seek(0)
    hd_size, count = struct.unpack("<II", f.read(8))
    if hd_size != 0:
        raise ValueError("not a VD/VE multi-raid twix file")
    out = []
    for _ in range(count):
        raw = f.read(152)
        meas_id, _file_id, off, length = struct.unpack("<IIQQ", raw[:24])
        out.append(Measurement(meas_id, off, length, raw[88:152].split(b"\0")[0].decode("latin1")))
    return out


def read_meas_header(f, offset: int) -> tuple[int, dict[str, bytes]]:
    f.seek(offset)
    hdr_len, n_buf = struct.unpack("<II", f.read(8))
    bufs = {}
    for _ in range(n_buf):
        name = bytearray()
        while (c := f.read(1)) != b"\0":
            name += c
        (blen,) = struct.unpack("<I", f.read(4))
        bufs[name.decode("latin1")] = f.read(blen)
    return hdr_len, bufs


def parse_ascconv(meas_yaps: bytes) -> dict[str, str]:
    text = meas_yaps.decode("latin1")
    m = re.search(r"### ASCCONV BEGIN.*?\n(.*?)### ASCCONV END", text, re.S)
    kv = {}
    for line in (m.group(1) if m else text).splitlines():
        mm = re.match(r"\s*(\S+)\s*=\s*(.+?)\s*(#.*)?$", line)
        if mm:
            kv[mm.group(1)] = mm.group(2).strip().strip('"')
    return kv


def index_scans(f, meas: Measurement, hdr_len: int) -> ScanIndex:
    pos, end = meas.offset + hdr_len, meas.offset + meas.length
    rows, n_sync, acqend = [], 0, False
    while pos + SCAN_HDR <= end:
        f.seek(pos)
        h = f.read(SCAN_HDR)
        dma = struct.unpack_from("<I", h, 0)[0] & 0x1FFFFFF
        mask = struct.unpack_from("<Q", h, 40)[0]
        if mask & MASK_ACQEND:
            acqend = True
            break
        if mask & MASK_SYNCDATA:
            n_sync += 1
            pos += dma
            continue
        ns, nch = struct.unpack_from("<HH", h, 48)
        lc = struct.unpack_from("<14H", h, 52)
        rows.append((pos, mask, ns, nch, lc[0], lc[4], lc[8], struct.unpack_from("<I", h, 12)[0]))
        pos += SCAN_HDR + nch * (CHAN_HDR + 8 * ns)
    a = np.array(rows, dtype=np.int64).reshape(-1, 8)
    return ScanIndex(pos=a[:, 0], mask=a[:, 1].astype(np.uint64), ns=a[:, 2], nch=a[:, 3],
                     lin=a[:, 4], eco=a[:, 5], seg=a[:, 6], ts=a[:, 7], n_sync=n_sync, acqend=acqend)


def read_adc(f, pos: int, ns: int, nch: int) -> np.ndarray:
    """(nch, ns) complex64 samples of the ADC packet whose scan header starts at `pos`."""
    f.seek(pos + SCAN_HDR)
    dt = np.dtype([("hdr", "V32"), ("data", "<c8", (ns,))])
    return np.frombuffer(f.read(nch * dt.itemsize), dtype=dt)["data"].copy()


def load_freerunning(path: str | Path, n_echo: int) -> FreeRunningData:
    """Last measurement of `path`: one leading noise ADC, then n_spoke x n_echo imaging ADCs in
    acquisition order (all echoes of one spoke consecutively)."""
    with open(path, "rb") as f:
        meas = read_raid_header(f)[-1]
        hdr_len, bufs = read_meas_header(f, meas.offset)
        idx = index_scans(f, meas, hdr_len)
        if not idx.acqend:
            raise ValueError("no ACQEND before the end of the measurement (truncated file?)")
        noise_rows = np.flatnonzero(idx.mask & np.uint64(MASK_NOISEADJ))
        if noise_rows.tolist() != [0]:
            raise ValueError(f"expected one noise ADC at row 0, got rows {noise_rows[:5].tolist()}")
        img = np.arange(1, idx.pos.size)
        ns, nch = int(idx.ns[1]), int(idx.nch[1])
        if not ((idx.ns[img] == ns).all() and (idx.nch[img] == nch).all()):
            raise ValueError("imaging ADCs do not all have the same size")
        if img.size % n_echo:
            raise ValueError(f"{img.size} imaging ADCs is not a multiple of {n_echo} echoes")
        n_spoke = img.size // n_echo
        eco = idx.eco[img]
        if not (eco.reshape(n_spoke, n_echo) == np.arange(n_echo)).all():
            raise ValueError("echo counter does not cycle 0..n_echo-1")
        lin = idx.lin[img].reshape(n_spoke, n_echo)
        if not (lin == lin[:, :1]).all():
            raise ValueError("Lin changes between the echoes of one spoke")
        ksp = np.empty((n_echo, n_spoke, nch, ns), np.complex64)
        for k, r in enumerate(img):
            ksp[k % n_echo, k // n_echo] = read_adc(f, int(idx.pos[r]), ns, nch)
        noise = read_adc(f, int(idx.pos[0]), int(idx.ns[0]), int(idx.nch[0]))
    return FreeRunningData(ksp=ksp, noise=noise, lin=lin[:, 0].copy(),
                           seg=idx.seg[img].reshape(n_spoke, n_echo)[:, 0].copy(),
                           mask=idx.mask[img], eco=eco, ts_ticks=idx.ts[img],
                           protocol=meas.protocol, ascconv=parse_ascconv(bufs.get("MeasYaps", b"")))
