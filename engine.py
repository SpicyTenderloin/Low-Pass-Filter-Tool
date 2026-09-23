"""Shared, framework-agnostic filter design pipeline.

Runs the full sequence -- prototype synthesis, pole retuning, per-stage
part realisation, gain compensation, and spec verification -- as a single
function, so the CLI (main.py) and the interactive app (app.py) both drive
the exact same validated logic instead of maintaining two copies of it.
Nothing in this module prints or plots; callers own presentation.
"""
import numpy as np
from scipy.signal import freqs

import parameters as P
from prototype_filter import generate_prototype, compute_min_order
from sallen_key_tf import build_lpf_sections_from_poles, compute_cascade_tf, compute_ideal_cascade_tf
from sk_realisation import pick_sk_parts_for_biquad, pick_sk_parts_for_first_order, pick_attenuator
from stagger_tuning import retune_poles


def realise_sections(sections, r_lib, c_lib, show_progress=True):
    """Realise every ideal (biquad/first-order) section into real parts."""
    realised, failures = [], []
    for i, s in enumerate(sections, 1):
        if s['kind'] == 'biquad':
            r = pick_sk_parts_for_biquad(s['w0'], s['Q'], P.STATIC_M_RATIOS, P.STATIC_N_RATIOS,
                                          r_lib, c_lib, stage_idx=i, stage_count=len(sections),
                                          show_progress=show_progress)
        else:
            r = pick_sk_parts_for_first_order(s['w0'], r_lib, c_lib,
                                               stage_idx=i, stage_count=len(sections),
                                               show_progress=show_progress)
        if r is None:
            failures.append(i)
        else:
            realised.append(r)
    if failures:
        raise RuntimeError(f"Stage(s) {failures} could not be realised from the part "
                            f"libraries at all -- try a larger E-series or a looser spec.")
    return realised


def verify_response(b, a, wp_hz, ws_hz, gpass_db, gstop_db):
    """Check a realised cascade's actual passband ripple and stopband
    attenuation against the spec, and report exactly where it fails.

    Ripple is measured as the true peak-to-peak deviation across the whole
    passband (its highest point minus its lowest), not deviation from an
    arbitrary single reference frequency -- a filter can rise above its own
    low-frequency level just as easily as it can dip below it, and both
    count against the ripple budget. Attenuation is then measured relative
    to that same passband peak, since that peak is the worst-case level a
    signal in the passband can actually reach.
    """
    f_pass = np.linspace(1.0, wp_hz, 2000)
    f_stop = np.geomspace(ws_hz, 20 * ws_hz, 400)
    _, Hp = freqs(b, a, 2 * np.pi * f_pass)
    _, Hs = freqs(b, a, 2 * np.pi * f_stop)
    mag_p = 20 * np.log10(np.maximum(np.abs(Hp), 1e-30))
    mag_s = 20 * np.log10(np.maximum(np.abs(Hs), 1e-30))

    i_peak, i_dip = int(np.argmax(mag_p)), int(np.argmin(mag_p))
    i_leak = int(np.argmax(mag_s))
    ripple_achieved = mag_p[i_peak] - mag_p[i_dip]
    atten_achieved = mag_p[i_peak] - mag_s[i_leak]

    return {
        'ripple_db_achieved': float(ripple_achieved),
        'ripple_ok': bool(ripple_achieved <= gpass_db + 1e-6),
        'ripple_peak_db': float(mag_p[i_peak]), 'ripple_peak_hz': float(f_pass[i_peak]),
        'ripple_dip_db': float(mag_p[i_dip]), 'ripple_dip_hz': float(f_pass[i_dip]),
        'atten_db_achieved': float(atten_achieved),
        'atten_ok': bool(atten_achieved >= gstop_db - 1e-6),
        'atten_worst_db': float(mag_s[i_leak]), 'atten_worst_hz': float(f_stop[i_leak]),
    }


def design_filter(filter_type, wp_hz, ws_hz, gpass_db, gstop_db, order_override=None,
                   e_series=24, retune_enabled=True, retune_margin_db=0.0,
                   target_gain_db=0.0, show_progress=False):
    """Run the full pipeline and return a dict with everything a caller
    needs to report on, plot, or save the result. Raises RuntimeError if a
    stage can't be realised from the part libraries at all (see
    realise_sections)."""
    r_lib, c_lib = P.build_libraries(e_series)

    b, a, N, Wn = generate_prototype(filter_type, wp_hz, ws_hz, gpass_db, gstop_db, order=order_override)
    n_min = compute_min_order(filter_type, wp_hz, ws_hz, gpass_db, gstop_db)

    sections = build_lpf_sections_from_poles(b, a)
    biquads = [s for s in sections if s['kind'] == 'biquad']
    first_orders = [s for s in sections if s['kind'] == 'first']

    retune_info = None
    if retune_enabled and biquads:
        f0_list = [s['w0'] / (2 * np.pi) for s in biquads]
        Q_list = [s['Q'] for s in biquads]
        fixed_w0_hz = (first_orders[0]['w0'] / (2 * np.pi)) if first_orders else None
        f0_new, Q_new, retune_info = retune_poles(
            f0_list, Q_list, wp_hz, ws_hz, gpass_db, gstop_db,
            margin_db=retune_margin_db, fixed_w0_hz=fixed_w0_hz)
        if retune_info['feasible']:
            for s, f0v, Qv in zip(biquads, f0_new, Q_new):
                s['w0'], s['Q'] = 2 * np.pi * f0v, Qv

    retuned_b, retuned_a = compute_ideal_cascade_tf(sections)

    realised = realise_sections(sections, r_lib, c_lib, show_progress=show_progress)

    # Each SK stage's gain K sets its Q *and* its DC gain, so the cascade
    # always ends up with excess passband gain. Size an output attenuator
    # to bring the overall gain to the target.
    total_gain = 1.0
    for s in realised:
        if s['kind'] == 'biquad':
            total_gain *= s['K_act']
    total_gain_db = 20 * np.log10(total_gain)
    needed_atten_db = total_gain_db - target_gain_db
    attenuator = None
    if needed_atten_db > 1e-6:
        attenuator = pick_attenuator(needed_atten_db, r_lib)
        realised.append(attenuator)

    real_b, real_a = compute_cascade_tf(realised)
    check = verify_response(real_b, real_a, wp_hz, ws_hz, gpass_db, gstop_db)

    return {
        'N': N, 'Wn': Wn, 'n_min': n_min,
        'stages': len(biquads) + len(first_orders),
        'ideal_b': b, 'ideal_a': a,
        'sections': sections, 'retune_info': retune_info,
        'retuned_b': retuned_b, 'retuned_a': retuned_a,
        'realised': realised, 'total_gain_db': total_gain_db,
        'needed_atten_db': needed_atten_db, 'attenuator': attenuator,
        'real_b': real_b, 'real_a': real_a, 'check': check,
    }
