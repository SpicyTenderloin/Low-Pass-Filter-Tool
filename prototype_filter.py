import numpy as np
from scipy.signal import cheb1ord, cheby1, butter, buttord, zpk2tf


def generate_prototype(filter_type, wp_hz, ws_hz, gpass_db, gstop_db):
    """Build the analog low-pass prototype transfer function for a spec.

    All arguments are explicit (no reliance on module-level constants) so
    this can be called with whatever the user entered interactively.
    """
    wp = 2 * np.pi * wp_hz
    ws = 2 * np.pi * ws_hz

    if filter_type == 'chebyshev':
        N, Wn = cheb1ord(wp, ws, gpass_db, gstop_db, analog=True)
        z, p, k = cheby1(N, gpass_db, Wn, btype='low', analog=True, output='zpk')
    elif filter_type == 'butterworth':
        N, Wn = buttord(wp, ws, gpass_db, gstop_db, analog=True)
        z, p, k = butter(N, Wn, btype='low', analog=True, output='zpk')
    else:
        raise ValueError("filter_type must be 'chebyshev' or 'butterworth'")

    b, a = zpk2tf(z, p, k)
    return b, a, N, Wn
