"""How the joint objective weighs the two contrasts' data terms."""


def data_term(l_dc_plain, l_dc_contrast, w_contrast: float = 1.0):
    """Return (the data term added to the loss, the contrast part as it was added).

    w_contrast = 1 keeps the historical l_dc_plain + l_dc_contrast exactly. Any other value scales the
    contrast channel's data term only. The shared geometry sees both gradients, while each channel's
    own densities are insensitive to a constant factor under Adam, so the weight mainly decides which
    contrast drives the shared geometry.
    """
    if w_contrast == 1.0:
        return l_dc_plain + l_dc_contrast, l_dc_contrast
    weighted = w_contrast * l_dc_contrast
    return l_dc_plain + weighted, weighted
