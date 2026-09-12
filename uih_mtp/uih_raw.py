"""Reader for UIH uMR ``.raw`` files (``RAWDATA-V004``, DHL versions 2-4), Cartesian scans.

File layout (from the vendor MATLAB reader ``Read_UIH_Raw_v5_2_jiasen_2021.m`` and ``Read_UIH_DHL.m``):

    12 bytes  magic ``RAWDATA-V004`` (older files have no magic)
    4 bytes   uint32 protocol length
    N bytes   protocol XML (``<UProtocol>…</UProtocol>``, zero padded)
    lines     each = 192-byte DHL header + payload; payload = channels x (samples + 2) complex float32
              for DHL version >= 2 (the first two complex words of every channel are a per-channel header),
              channels x samples complex float32 for version 1. A header with the ACQUISITION_END flag ends
              the file.

Only what the MTP line needs is implemented: an index of every line (counters, flags, payload offset) and
the reading of one (set, contrast, u1) combination into a centred k-space array.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"RAWDATA-V004"
DHL_SIZE = 192
# The first 126 bytes of the DHL; the rest (IRIP parameters, reserved, CRC) is skipped.
_DHL = struct.Struct("<B B B B I Q I I I i i i H H I Q H H H H H H H H H H H 5H H H H H H 3f 4f")
assert _DHL.size == 126

FLAG_ACQ_END = 1 << 0
FLAG_PHASECOR = 1 << 7
FLAG_PPA_REFLINE = 1 << 9
FLAG_FEEDBACK = 1 << 10
FLAG_NOISE = 1 << 24

_COUNTERS = ("rep", "cph", "slc", "avg", "con", "set_", "shot", "seg", "pe", "spe",
             "uctrl", "u0", "u1", "u2", "u3", "u4")


def read_protocol(path: str) -> tuple[bytes, int, bytes]:
    """Return (magic, protocol length, protocol XML bytes)."""
    with open(path, "rb") as f:
        magic = f.read(12)
        if magic != MAGIC:
            f.seek(0)
            magic = b""
        n = struct.unpack("<I", f.read(4))[0]
        xml = f.read(n)
    end = xml.find(b"</UProtocol>")
    if end < 0:
        raise ValueError(f"{path}: protocol XML has no </UProtocol>")
    return magic, n, xml[: end + len(b"</UProtocol>")]


def data_offset(magic: bytes, prot_len: int) -> int:
    return prot_len + 4 + (12 if magic else 0)


def protocol_int(xml: bytes, path: str) -> int:
    """Integer value of a protocol node, e.g. ``Root/IRIP/FromSeq/KSpace/FTLengthPE``."""
    node = ET.fromstring(xml).find(path + "/Value")
    if node is None or node.text is None:
        raise KeyError(path)
    return int(float(node.text))


def payload_bytes(version: int, samples: int, channels: int) -> int:
    per_channel = samples + 2 if version >= 2 else samples
    return 2 * per_channel * channels * 4


@dataclass
class LineIndex:
    """Every line of one raw file: counters, flags and payload offsets (imaging lines only in ``offsets``)."""
    path: str
    samples: int
    channels: int
    dhl_version: int
    ro_center: int
    pe_center: int
    spe_center: int
    pe_len: int
    spe_len: int
    offsets: np.ndarray                      # int64, payload offset of each imaging line
    counters: dict = field(default_factory=dict)   # name -> int32 array (imaging lines)
    ref: np.ndarray = None                   # bool, PPA reference-line flag (imaging lines)
    rxgain: np.ndarray = None                # uint8 receive-gain code (imaging lines)
    noise_offsets: np.ndarray = None         # int64 payload offsets of the noise lines
    n_phasecor: int = 0
    n_feedback: int = 0

    @property
    def n_imaging(self) -> int:
        return int(len(self.offsets))

    @property
    def n_noise(self) -> int:
        return int(len(self.noise_offsets))

    def select(self, set_: int, con: int, u1: int) -> np.ndarray:
        c = self.counters
        return (c["set_"] == set_) & (c["con"] == con) & (c["u1"] == u1)

    def mask(self, set_: int, con: int, u1: int) -> np.ndarray:
        sel = self.select(set_, con, u1)
        m = np.zeros((self.pe_len, self.spe_len), dtype=bool)
        m[self.counters["pe"][sel], self.counters["spe"][sel]] = True
        return m


def scan_index(path: str) -> LineIndex:
    """Walk every DHL header (payloads are skipped with seek) and index the file."""
    magic, n, xml = read_protocol(path)
    pe_len = protocol_int(xml, "Root/IRIP/FromSeq/KSpace/FTLengthPE")
    spe_len = protocol_int(xml, "Root/IRIP/FromSeq/KSpace/FTLengthSPE")
    offsets, noise_offsets = [], []
    counters = {k: [] for k in _COUNTERS}
    ref, rxgain = [], []
    samples = channels = version = None
    ro_c = pe_c = spe_c = None
    n_phasecor = n_feedback = 0
    with open(path, "rb") as f:
        f.seek(data_offset(magic, n))
        while True:
            hdr = f.read(DHL_SIZE)
            if len(hdr) < DHL_SIZE:
                raise ValueError(f"{path}: file ended without ACQUISITION_END")
            v = _DHL.unpack_from(hdr, 0)
            ctrl = v[15]
            if ctrl & FLAG_ACQ_END:
                break
            ver, gain, nsamp, nch = v[0], v[2], v[12], v[13]
            nbytes = payload_bytes(ver, nsamp, nch)
            here = f.tell()
            if ctrl & FLAG_NOISE:
                noise_offsets.append(here)
            elif ctrl & FLAG_PHASECOR:
                n_phasecor += 1
            elif ctrl & FLAG_FEEDBACK:
                n_feedback += 1
            else:
                if samples is None:
                    samples, channels, version = nsamp, nch, ver
                    ro_c, pe_c, spe_c = v[34], v[35], v[36]
                elif (nsamp, nch) != (samples, channels):
                    raise ValueError(f"{path}: imaging line size changed to {nsamp}x{nch}")
                offsets.append(here)
                for name, val in zip(_COUNTERS, v[16:32]):
                    counters[name].append(val)
                ref.append(bool(ctrl & FLAG_PPA_REFLINE))
                rxgain.append(gain)
            f.seek(nbytes, 1)
    return LineIndex(
        path=path, samples=samples, channels=channels, dhl_version=version,
        ro_center=ro_c, pe_center=pe_c, spe_center=spe_c, pe_len=pe_len, spe_len=spe_len,
        offsets=np.asarray(offsets, dtype=np.int64),
        counters={k: np.asarray(vals, dtype=np.int32) for k, vals in counters.items()},
        ref=np.asarray(ref, dtype=bool), rxgain=np.asarray(rxgain, dtype=np.uint8),
        noise_offsets=np.asarray(noise_offsets, dtype=np.int64),
        n_phasecor=n_phasecor, n_feedback=n_feedback,
    )


def _gain_factor(code: int) -> float:
    return 1.99 if code == 1 else 1.0          # vendor reader: 0 = high gain, 1 = low gain


def _read_payload(f, offset: int, version: int, samples: int, channels: int) -> np.ndarray:
    """One line as (channels, samples) complex64."""
    f.seek(offset)
    per_channel = samples + 2 if version >= 2 else samples
    buf = np.frombuffer(f.read(2 * per_channel * channels * 4), dtype=np.float32).reshape(channels, per_channel, 2)
    if version >= 2:
        buf = buf[:, 2:]
    return (buf[..., 0] + 1j * buf[..., 1]).astype(np.complex64)


def read_lines(path: str, index: LineIndex, set_: int, con: int, u1: int,
               ro_crop: int | None = None) -> np.ndarray:
    """k-space of one (set, contrast, u1) as (channels, RO, PE, SPE) complex64, DC-centred on every axis.

    ``ro_crop``: if given, readout oversampling is removed line by line (see
    :func:`remove_readout_oversampling`) so the returned readout length is ``ro_crop``.
    """
    sel = np.where(index.select(set_, con, u1))[0]
    if len(sel) == 0:
        raise ValueError(f"no imaging lines for set={set_} con={con} u1={u1}")
    n_ro = ro_crop if ro_crop is not None else index.samples
    k = np.zeros((index.channels, n_ro, index.pe_len, index.spe_len), dtype=np.complex64)
    pe, spe = index.counters["pe"], index.counters["spe"]
    order = sel[np.argsort(index.offsets[sel])]              # sequential reads
    with open(path, "rb") as f:
        for i in order:
            line = _read_payload(f, int(index.offsets[i]), index.dhl_version, index.samples, index.channels)
            line *= _gain_factor(int(index.rxgain[i]))
            if ro_crop is not None:
                line = remove_readout_oversampling(line[:, :, None, None], ro_crop)[:, :, 0, 0]
            k[:, :, pe[i], spe[i]] = line
    return k


def remove_readout_oversampling(k: np.ndarray, n_out: int) -> np.ndarray:
    """Drop readout oversampling of a centred k-space (axis 1 = readout).

    Vendor recipe: inverse FFT along the readout, keep the central ``n_out`` samples of the (doubled)
    readout field of view, FFT back. Unnormalised transforms cancel exactly.
    """
    n_in = k.shape[1]
    if n_out > n_in:
        raise ValueError(f"cannot crop readout from {n_in} to {n_out}")
    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=1), axis=1), axes=1)
    start = (n_in - n_out) // 2
    img = img[:, start:start + n_out]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=1), axis=1), axes=1).astype(np.complex64)
