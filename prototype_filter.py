import numpy as np
from scipy.signal import cheb1ord, cheby1, butter, buttord, zpk2tf


def compute_min_order(filter_type, wp_hz, ws_hz, gpass_db, gstop_db):
    """The smallest order that satisfies the spec exactly (scipy's automatic
    choice). This is what generate_prototype() uses by default, and sits
    exactly on the ripple/attenuation boundary with no headroom -- see the
    `order` argument there."""
    wp = 2 * np.pi * wp_hz
    ws = 2 * np.pi * ws_hz
    if filter_type == 'chebyshev':
        N, _ = cheb1ord(wp, ws, gpass_db, gstop_db, analog=True)
    elif filter_type == 'butterworth':
        N, _ = buttord(wp, ws, gpass_db, gstop_db, analog=True)
    else:
        raise ValueError("filter_type must be 'chebyshev' or 'butterworth'")
    return int(N)


def generate_prototype(filter_type, wp_hz, ws_hz, gpass_db, gstop_db, order=None):
    """Build the analog low-pass prototype transfer function for a spec.

    By default this uses the smallest order that satisfies the spec exactly
    (compute_min_order() above) -- which is why a fresh design tends to sit
    right on the ripple/attenuation boundary with no room to spare once
    realised in real parts.

    Passing `order` higher than that forces more poles at the same passband
    edge and ripple. On its own this does NOT make the design easier to
    build: for a fixed ripple and edge, more poles means a *steeper*
    required transition, which pushes every pole *closer* to the imaginary
    axis (higher Q), not further. The benefit only appears when the extra
    order is then handed to stagger_tuning.retune_poles(), which can spread
    the required selectivity across more, gentler stages instead of a few
    sharp ones -- confirmed empirically: going from the automatic minimum to
    a couple of orders higher, combined with retuning, took a stage's
    required Q from ~9 (right at the realisable ceiling) down to ~2-3.
    """
    wp = 2 * np.pi * wp_hz
    ws = 2 * np.pi * ws_hz

    if filter_type == 'chebyshev':
        N = int(order) if order else compute_min_order(filter_type, wp_hz, ws_hz, gpass_db, gstop_db)
        z, p, k = cheby1(N, gpass_db, wp, btype='low', analog=True, output='zpk')
        Wn = wp
    elif filter_type == 'butterworth':
        if order:
            N = int(order)
            # Recompute the -x dB corner for this order so the passband
            # edge still lands exactly on gpass_db, same as buttord would.
            Wn = wp / (10 ** (gpass_db / 10.0) - 1) ** (1.0 / (2 * N))
        else:
            N, Wn = buttord(wp, ws, gpass_db, gstop_db, analog=True)
        z, p, k = butter(N, Wn, btype='low', analog=True, output='zpk')
    else:
        raise ValueError("filter_type must be 'chebyshev' or 'butterworth'")

    b, a = zpk2tf(z, p, k)
    return b, a, N, Wn
