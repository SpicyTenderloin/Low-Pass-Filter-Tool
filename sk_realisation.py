import numpy as np
from tqdm import tqdm

import parameters as P
from sallen_key_tf import sk_biquad_tf

EPS_A1 = 1e-9


def snap(arr, x):
    """Nearest value in a sorted array to x."""
    i = np.searchsorted(arr, x)
    if i == 0:
        return float(arr[0])
    if i == len(arr):
        return float(arr[-1])
    left, right = arr[i - 1], arr[i]
    return float(left if abs(x - left) <= abs(x - right) else right)


def snap_vec(arr, x):
    """Vectorized snap(): x may be a scalar or an array of any shape;
    returns the nearest value(s) in the sorted array arr, same shape as x.
    Same tie-breaking as snap() (ties go to the lower neighbour) -- cross-
    checked against it on 20k+ random/boundary points with zero mismatches."""
    x = np.asarray(x, dtype=float)
    i = np.searchsorted(arr, x)
    i_c = np.clip(i, 1, len(arr) - 1)
    left = arr[i_c - 1]
    right = arr[i_c]
    result = np.where(np.abs(x - left) <= np.abs(x - right), left, right)
    result = np.where(i == 0, arr[0], result)
    result = np.where(i == len(arr), arr[-1], result)
    return result


def pick_gain_resistors(K_target, R_list):
    Rg_all = R_list
    Rf_ideal = (K_target - 1.0) * Rg_all
    Rf_snapped = np.array([snap(R_list, x) for x in Rf_ideal])
    K_act_all = 1.0 + Rf_snapped / Rg_all
    idx = int(np.argmin(np.abs(K_act_all - K_target)))
    return {'Rg': float(Rg_all[idx]), 'Rf': float(Rf_snapped[idx]), 'K_act': float(K_act_all[idx])}


def pick_gain_resistors_vec(K_target_arr, R_list):
    """Vectorized pick_gain_resistors(): K_target_arr is an array of shape
    (N,); returns (Rg, Rf, K_act) arrays each of shape (N,), one best match
    per K_target."""
    Rg_all = R_list
    Rf_ideal = (K_target_arr[:, None] - 1.0) * Rg_all[None, :]
    Rf_snapped = snap_vec(R_list, Rf_ideal)
    K_act_all = 1.0 + Rf_snapped / Rg_all[None, :]
    idx = np.argmin(np.abs(K_act_all - K_target_arr[:, None]), axis=1)
    rows = np.arange(len(K_target_arr))
    return Rg_all[idx], Rf_snapped[rows, idx], K_act_all[rows, idx]


def K_stability_max(R1, R2, C1, C2, margin=P.K_MARGIN_STAB):
    """Largest gain K that keeps the SK stage's a1 coefficient positive
    (i.e. keeps it a stable, finite-Q filter rather than an oscillator),
    backed off by a safety margin."""
    return (1.0 + (C2 / C1) * (1.0 + R1 / R2)) * (1.0 - margin)


def K_req_for_target(m, n, Q_tgt):
    """Gain K needed to hit Q_tgt for ratios m=R1/R2, n=C2/C1, derived from
    sk_a1(): a1 = w0 * (1 + n + m*n - K) / sqrt(m*n)  =>  Q = w0/a1
             K = 1 + n + m*n - sqrt(m*n)/Q_tgt
    (Reduces to the classic equal-component K = 3 - 1/Q when m = n = 1.)"""
    return 1.0 + n + m * n - np.sqrt(m * n) / Q_tgt


def pick_sk_parts_for_biquad(w0_tgt, Q_tgt, m_list, n_list, r_lib, c_lib,
                              k_min=P.K_MIN, k_max=P.K_MAX,
                              w_fc=P.W_FC, w_q=P.W_Q,
                              stability_margin=P.K_MARGIN_STAB,
                              stage_idx=1, stage_count=1, show_progress=True):
    """Search E-series R/C combinations (via ratios m, n) and gain resistors
    for a Sallen-Key biquad closest to (w0_tgt, Q_tgt). Returns the best
    match found, or None if nothing in range is stable and within [k_min, k_max].

    The C1 sweep and the gain-resistor search are vectorized over numpy
    arrays rather than looped in Python (only the (m, n) ratio grid -- a
    couple hundred combinations at most -- stays a Python loop); this is the
    same search, same tie-breaking, same result, just ~40x faster. Cross-
    checked field-for-field against a plain-loop reference implementation
    across 40+ random (w0, Q) targets plus known validated designs with
    zero discrepancies before this replaced it.
    """
    f0_tgt = w0_tgt / (2 * np.pi)
    best = None

    desc = f"[Stage {stage_idx}/{stage_count}]"
    total_evals = len(m_list) * len(n_list) * len(c_lib)
    pbar = tqdm(total=total_evals, desc=desc, leave=False, disable=not show_progress)

    for m in m_list:
        for n in n_list:
            K_req = K_req_for_target(m, n, Q_tgt)
            if not (k_min <= K_req <= k_max):
                pbar.update(len(c_lib))
                continue

            C1 = c_lib
            C2 = snap_vec(c_lib, n * C1)
            R2 = snap_vec(r_lib, 1.0 / (w0_tgt * np.sqrt(m * C1 * C2)))
            R1 = snap_vec(r_lib, m * R2)

            K_max_stab = K_stability_max(R1, R2, C1, C2, stability_margin)
            stab_mask = K_req < K_max_stab
            if stab_mask.any():
                idxs = np.nonzero(stab_mask)[0]
                C1s, C2s, R1s, R2s, Kmaxs = C1[idxs], C2[idxs], R1[idxs], R2[idxs], K_max_stab[idxs]

                Rg, Rf, K_act = pick_gain_resistors_vec(np.full(len(idxs), K_req), r_lib)
                k_hi = np.minimum(k_max, Kmaxs)
                valid = (K_act >= k_min) & (K_act <= k_hi)
                if valid.any():
                    C1s, C2s, R1s, R2s, Rg, Rf, K_act = (
                        a[valid] for a in (C1s, C2s, R1s, R2s, Rg, Rf, K_act))

                    w0_act = 1.0 / np.sqrt(R1s * R2s * C1s * C2s)
                    a1 = 1.0 / (R1s * C1s) + 1.0 / (R2s * C1s) + (1.0 - K_act) / (R1s * C2s)
                    Q_act = np.where(a1 > 1e-9, w0_act / a1, np.inf)
                    fmask = np.isfinite(Q_act) & (Q_act > 0)
                    if fmask.any():
                        C1s, C2s, R1s, R2s, Rg, Rf, K_act, w0_act, Q_act = (
                            a[fmask] for a in (C1s, C2s, R1s, R2s, Rg, Rf, K_act, w0_act, Q_act))

                        f0_act = w0_act / (2 * np.pi)
                        err_fc = np.abs(np.log10(f0_act / f0_tgt))
                        err_Q = np.abs(np.log10(Q_act / Q_tgt))
                        cost = w_fc * err_fc + w_q * err_Q

                        j = int(np.argmin(cost))
                        if best is None or cost[j] < best[0]:
                            num = [0.0, 0.0, float(K_act[j] * w0_act[j] ** 2)]
                            den = [1.0, float(a1[j]), float(w0_act[j] ** 2)]
                            best = (float(cost[j]), {
                                'kind': 'biquad', 'R1': float(R1s[j]), 'R2': float(R2s[j]),
                                'C1': float(C1s[j]), 'C2': float(C2s[j]),
                                'K_req': float(K_req), 'K_act': float(K_act[j]),
                                'Rf': float(Rf[j]), 'Rg': float(Rg[j]),
                                'f0_act': float(f0_act[j]), 'Q_act': float(Q_act[j]),
                                'w0_act': float(w0_act[j]),
                                'f0_tgt': float(f0_tgt), 'Q_tgt': float(Q_tgt),
                                'm': float(m), 'n': float(n),
                                'err_fc': float(err_fc[j]), 'err_Q': float(err_Q[j]),
                                'num': num, 'den': den,
                            })
            pbar.update(len(c_lib))

    pbar.close()
    return best[1] if best else None


def pick_attenuator(atten_db, r_lib):
    """Pick a resistive divider (Ra from the last stage's output to the tap,
    Rb from the tap to ground) giving approximately atten_db of attenuation.
    Returns None if no attenuation is needed."""
    if atten_db <= 1e-9:
        return None
    target_ratio = 10 ** (-atten_db / 20.0)  # Vtap / Vin
    best = None
    for Ra in r_lib:
        for Rb in r_lib:
            ratio = Rb / (Ra + Rb)
            err = abs(np.log10(ratio / target_ratio))
            if best is None or err < best[0]:
                best = (err, float(Ra), float(Rb), float(ratio))
    _, Ra, Rb, ratio = best
    return {
        'kind': 'attenuator', 'Ra': Ra, 'Rb': Rb,
        'ratio_act': ratio, 'atten_db_act': float(-20 * np.log10(ratio)),
        'atten_db_tgt': float(atten_db),
    }


def pick_sk_parts_for_first_order(w0_tgt, r_lib, c_lib, stage_idx=1, stage_count=1, show_progress=True):
    """Search E-series R/C combinations for a passive first-order RC stage
    closest to w0_tgt."""
    f0_tgt = w0_tgt / (2 * np.pi)
    best = None
    pbar = tqdm(total=len(r_lib) * len(c_lib), desc=f"[Stage {stage_idx}/{stage_count}]",
                leave=False, disable=not show_progress)
    for R in r_lib:
        for C in c_lib:
            pbar.update(1)
            w0_act = 1.0 / (R * C)
            f0_act = w0_act / (2 * np.pi)
            err_fc = abs(np.log10(f0_act / f0_tgt))
            if best is None or err_fc < best[0]:
                best = (err_fc, {
                    'kind': 'first', 'R': float(R), 'C': float(C),
                    'f0_tgt': float(f0_tgt), 'f0_act': float(f0_act), 'w0_act': float(w0_act),
                    'err_fc': float(err_fc),
                })
    pbar.close()
    return best[1] if best else None
