import numpy as np


def test_parse_scales_in_channel_order():
    import extract_gs_volumes as X
    log = "x\nNormalized MRI intensity scale by P99.9 magnitude = 1.2500e+02 (before)\nfoo\nNormalized MRI intensity scale by P99.9 magnitude = 130.0\n"
    assert X.parse_scales(log) == [125.0, 130.0]


def test_convert_scales_complex_volume():
    import extract_gs_volumes as X
    v = np.array([[1 + 1j]], np.complex64)
    out = X.convert(v, 2.0)
    assert out.dtype == np.complex64 and out[0, 0] == 2 + 2j
