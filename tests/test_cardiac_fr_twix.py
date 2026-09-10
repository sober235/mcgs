import struct

import numpy as np
import pytest

from cardiac_fr.twix import (MASK_ACQEND, MASK_NOISEADJ, MASK_SYNCDATA, index_scans,
                             load_freerunning, read_meas_header, read_raid_header)

ONLINE = 1 << 3
N_SPOKE, N_ECHO, NCH, NS = 5, 3, 2, 6
YAPS = (b"junk\n### ASCCONV BEGIN ###\nsKSpace.lBaseResolution = 112\n"
        b"tProtocolName = \"FRFF_test\"\n### ASCCONV END ###\n")


def _scan_header(mask, ns, nch, lin=0, eco=0, seg=0, ts=0, dma=None):
    h = bytearray(192)
    struct.pack_into("<I", h, 0, (192 + nch * (32 + 8 * ns) if dma is None else dma) & 0x1FFFFFF)
    struct.pack_into("<I", h, 12, ts)
    struct.pack_into("<Q", h, 40, mask)
    struct.pack_into("<HH", h, 48, ns, nch)
    lc = [0] * 14
    lc[0], lc[4], lc[8] = lin, eco, seg
    struct.pack_into("<14H", h, 52, *lc)
    return bytes(h)


def _adc(mask, data, **kw):
    nch, ns = data.shape
    out = bytearray(_scan_header(mask, ns, nch, **kw))
    for c in range(nch):
        out += bytes(32) + data[c].astype("<c8").tobytes()
    return bytes(out)


def _sync(payload=100):
    return _scan_header(MASK_SYNCDATA, 0, 0, dma=192 + payload) + bytes(payload)


def _acqend():
    return _scan_header(MASK_ACQEND, 16, 0, dma=192)


def _meas(buffers, packets):
    body = bytearray()
    for name, b in buffers.items():
        body += name.encode() + b"\0" + struct.pack("<I", len(b)) + b
    return struct.pack("<II", 8 + len(body), len(buffers)) + bytes(body) + b"".join(packets)


def _write_twix(path, measurements):
    off = 8 + 152 * len(measurements)
    entries, blobs = b"", b""
    for i, (prot, payload) in enumerate(measurements):
        entries += (struct.pack("<IIQQ", i + 1, i + 1, off + len(blobs), len(payload))
                    + bytes(64) + prot.encode().ljust(64, b"\0"))
        blobs += payload
    path.write_bytes(struct.pack("<II", 0, len(measurements)) + entries + blobs)


def _value(s, e, c, j):
    return s * 1000 + e * 100 + c * 10 + j + 1j * (s + 0.5)


def _fake_file(tmp_path, echo_order=None, with_acqend=True):
    adj = _meas({"MeasYaps": b""}, [_adc(ONLINE, np.ones((NCH, NS), np.complex64)), _acqend()])
    packets = [_adc(MASK_NOISEADJ | ONLINE, np.full((NCH, NS), 7 - 7j, np.complex64))]
    t = 100
    for s in range(N_SPOKE):
        for e in (echo_order or range(N_ECHO)):
            d = np.array([[_value(s, e, c, j) for j in range(NS)] for c in range(NCH)], np.complex64)
            packets.append(_adc(ONLINE, d, lin=s, eco=e, seg=s // 2, ts=t))
            t += 1
        if s == 2:
            packets.append(_sync())
    if with_acqend:
        packets.append(_acqend())
    p = tmp_path / "fake.dat"
    _write_twix(p, [("AdjCoilSens", adj),
                    ("FRFF_test", _meas({"Config": b"x", "MeasYaps": YAPS}, packets))])
    return p


def test_raid_header_lists_both_measurements(tmp_path):
    with open(_fake_file(tmp_path), "rb") as f:
        assert [m.protocol for m in read_raid_header(f)] == ["AdjCoilSens", "FRFF_test"]


def test_index_skips_syncdata_and_stops_at_acqend(tmp_path):
    with open(_fake_file(tmp_path), "rb") as f:
        meas = read_raid_header(f)[-1]
        hdr_len, bufs = read_meas_header(f, meas.offset)
        idx = index_scans(f, meas, hdr_len)
    assert set(bufs) == {"Config", "MeasYaps"}
    assert idx.n_sync == 1 and idx.acqend
    assert idx.pos.size == 1 + N_SPOKE * N_ECHO


def test_load_places_every_sample(tmp_path):
    d = load_freerunning(_fake_file(tmp_path), n_echo=N_ECHO)
    assert d.ksp.shape == (N_ECHO, N_SPOKE, NCH, NS) and d.ksp.dtype == np.complex64
    for s in range(N_SPOKE):
        for e in range(N_ECHO):
            for c in range(NCH):
                np.testing.assert_array_equal(d.ksp[e, s, c], [_value(s, e, c, j) for j in range(NS)])
    np.testing.assert_array_equal(d.noise, np.full((NCH, NS), 7 - 7j, np.complex64))
    np.testing.assert_array_equal(d.lin, np.arange(N_SPOKE))
    np.testing.assert_array_equal(d.seg, np.arange(N_SPOKE) // 2)
    np.testing.assert_array_equal(d.ts_ticks, 100 + np.arange(N_SPOKE * N_ECHO))
    assert d.protocol == "FRFF_test" and d.ascconv["sKSpace.lBaseResolution"] == "112"


def test_rejects_truncated_measurement(tmp_path):
    with pytest.raises(ValueError, match="ACQEND"):
        load_freerunning(_fake_file(tmp_path, with_acqend=False), n_echo=N_ECHO)


def test_rejects_wrong_echo_cycle(tmp_path):
    with pytest.raises(ValueError, match="echo counter"):
        load_freerunning(_fake_file(tmp_path, echo_order=[0, 2, 1]), n_echo=N_ECHO)
