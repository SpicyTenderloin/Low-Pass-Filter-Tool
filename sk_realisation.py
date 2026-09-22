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


def pick_gain_resistors(K_target, R_list):
    Rg_all = R_list
    Rf_ideal = (K_target - 1.0) * Rg_all
    Rf_snapped = np.array([snap(R_list, x) for x in Rf_ideal])
    K_act_all = 1.0 + Rf_snapped / Rg_all
    idx = int(np.argmin(np.abs(K_act_all - K_target)))
    return {'Rg': float(Rg_all[idx]), 'Rf': float(Rf_snapped[idx]), 'K_act': float(K_act_all[idx])}


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
    match found, or None if nothing in range is stable and within [k_min, k_max]."""
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
            for C1 in c_lib:
                pbar.update(1)
                C2 = snap(c_lib, n * C1)
                R2 = snap(r_lib, 1.0 / (w0_tgt * np.sqrt(m * C1 * C2)))
                R1 = snap(r_lib, m * R2)

                K_max_stab = K_stability_max(R1, R2, C1, C2, stability_margin)
                if K_req >= K_max_stab:
                    continue

                gain = pick_gain_resistors(K_req, r_lib)
                K_act = gain['K_act']
                if not (k_min <= K_act <= min(k_max, K_max_stab)):
                    continue

                num, den, w0_act, Q_act = sk_biquad_tf(R1, R2, C1, C2, K_act)
                if den is None or not np.isfinite(Q_act) or Q_act <= 0:
                    continue

                f0_act = w0_act / (2 * np.pi)
                err_fc = abs(np.log10(f0_act / f0_tgt))
                err_Q = abs(np.log10(Q_act / Q_tgt))
                cost = w_fc * err_fc + w_q * err_Q

                if best is None or cost < best[0]:
                    best = (cost, {
                        'kind': 'biquad', 'R1': R1, 'R2': R2, 'C1': C1, 'C2': C2,
                        'K_req': float(K_req), 'K_act': K_act, 'Rf': gain['Rf'], 'Rg': gain['Rg'],
                        'f0_act': float(f0_act), 'Q_act': float(Q_act), 'w0_act': float(w0_act),
                        'f0_tgt': float(f0_tgt), 'Q_tgt': float(Q_tgt), 'm': float(m), 'n': float(n),
                        'err_fc': float(err_fc), 'err_Q': float(err_Q),
                        'num': list(map(float, num)), 'den': list(map(float, den)),
                    })

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
