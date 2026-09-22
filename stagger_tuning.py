"""Pole retuning ("stagger tuning") for cascaded Sallen-Key biquad stages.

--- What "stagger tuning" means here ---

The term comes from stagger-tuned IF amplifiers: several resonant stages,
each tuned to a slightly different centre frequency, cascaded so their
combined response is broader and flatter than any single stage's. That is
not quite what this tool needs, but the same idea -- deliberately moving
each stage away from its textbook-exact position -- is repurposed here for
a different problem: some poles of a Chebyshev/Butterworth prototype need a
Q that is impossible to build accurately from standard-value resistors and
capacitors (see the note in sk_realisation.K_stability_max / K_MARGIN_STAB
-- a Sallen-Key stage's gain K sets Q *and* fights the stage's own
stability margin, which puts a hard ceiling on achievable Q well before the
E-series discreteness even becomes the limiting factor).

retune_poles() nudges every stage's (f0, Q) pair together -- not just the
worst one -- so the cascaded response still fits within the passband
ripple / stopband attenuation mask (optionally relaxed by `margin_db`),
while minimising the *highest* Q required by any single stage. Lower peak
Q is easier to hit exactly with discrete parts, because Q's sensitivity to
the gain resistor ratio grows with Q itself.

--- Why the previous implementation didn't do anything ---

`stagger_q_spread()` mirrored each Q around the peak and averaged
left/right pairs, but only if the counts on both sides of the peak
matched. For a Chebyshev low-pass, the highest-Q pole is always the one
closest to the band edge (index 0), so there is nothing to its left -- the
function always fell into a mismatched-length branch and returned the
input unchanged. `stagger_peaking_shift()` did move the centre
frequencies, spacing them evenly in log-frequency, but with no reference
to the pass/stopband mask at all, so it had no way to know whether the
result still met the spec (in general it doesn't).
"""

import numpy as np
from scipy.optimize import minimize


def _cascade_response_db(f_hz, f0_hz, Q):
    """dB magnitude of a cascade of unity-gain 2nd-order low-pass sections
    at frequencies f_hz, given each section's (f0, Q)."""
    f0_hz = np.asarray(f0_hz)
    Q = np.asarray(Q)
    u = f_hz[:, None] / f0_hz[None, :]
    stage_db = -10 * np.log10((1 - u ** 2) ** 2 + (u / Q[None, :]) ** 2)
    return stage_db.sum(axis=1)


def _unpack(x, M):
    return x[:M], np.exp(x[M:2 * M]), x[2 * M], x[2 * M + 1]


def retune_poles(f0_hz, Q, wp_hz, ws_hz, gpass_db, gstop_db, margin_db=0.5,
                  f_max_mult=20.0, n_pass=120, n_stop=80, restarts=6, seed=0,
                  fixed_w0_hz=None):
    """Retune a set of biquad pole targets to reduce the worst-case Q.

    Parameters
    ----------
    f0_hz, Q : the ideal prototype pole targets, one per biquad stage
        (retuned).
    wp_hz, ws_hz, gpass_db, gstop_db : the design spec the cascade must
        still meet (its passband/stopband mask).
    margin_db : how much extra ripple / how much less stopband attenuation
        is acceptable while retuning, i.e. how far outside the exact spec
        the retuned response is allowed to sit.
    fixed_w0_hz : f0 of a single real pole (first-order section) present
        in an odd-order filter, left untouched but whose fixed contribution
        to the mask is still accounted for. None if the order is even.

    Returns
    -------
    (f0_new, Q_new, info) -- info['feasible'] is False if no retuning
    within the given margin could be found (the ideal poles are returned
    unchanged in that case); info['max_Q_before'/'max_Q_after'] let the
    caller report the improvement.
    """
    f0_hz = np.asarray(f0_hz, dtype=float)
    Q = np.asarray(Q, dtype=float)
    M = len(f0_hz)
    max_q_before = float(Q.max()) if M else 0.0

    if M <= 1:
        return f0_hz.tolist(), Q.tolist(), {
            'feasible': True, 'changed': False,
            'max_Q_before': max_q_before, 'max_Q_after': max_q_before,
            'note': 'Only one stage -- nothing to stagger.',
        }

    f_pass = np.linspace(1.0, wp_hz, n_pass)
    f_stop = np.geomspace(ws_hz, f_max_mult * ws_hz, n_stop)

    if fixed_w0_hz:
        extra_pass = -10 * np.log10(1 + (f_pass / fixed_w0_hz) ** 2)
        extra_stop = -10 * np.log10(1 + (f_stop / fixed_w0_hz) ** 2)
    else:
        extra_pass = extra_stop = 0.0

    def constraints(x):
        lf0, Qv, P, t = _unpack(x, M)
        f0v = np.exp(lf0)
        cp = _cascade_response_db(f_pass, f0v, Qv) + extra_pass
        cs = _cascade_response_db(f_stop, f0v, Qv) + extra_stop
        return np.r_[
            P - cp,                              # response <= reference level P
            cp - (P - gpass_db - margin_db),      # response >= P - ripple - margin
            (P - gstop_db + margin_db) - cs,      # stopband response low enough
            t - Qv,                               # t bounds every stage's Q from above
        ]

    rng = np.random.default_rng(seed)
    bounds = (
        [(np.log(f0_hz.min() / 4), np.log(f0_hz.max() * 4))] * M
        + [(np.log(0.3), np.log(60.0))] * M
        + [(-60.0, 80.0), (0.3, 100.0)]
    )

    best = None
    for s in range(restarts):
        jitter = 0.0 if s == 0 else 0.05
        x0 = np.r_[np.log(f0_hz) + rng.normal(0, jitter, M),
                   np.log(Q) + rng.normal(0, jitter, M),
                   0.0, Q.max()]
        res = minimize(lambda x: x[-1], x0,
                        constraints=[{'type': 'ineq', 'fun': constraints}],
                        bounds=bounds, method='SLSQP',
                        options={'maxiter': 400, 'ftol': 1e-9})
        if res.success and constraints(res.x).min() > -1e-4:
            if best is None or res.fun < best.fun:
                best = res

    if best is None:
        return f0_hz.tolist(), Q.tolist(), {
            'feasible': False, 'changed': False,
            'max_Q_before': max_q_before, 'max_Q_after': max_q_before,
            'note': f'No feasible retuning found within a {margin_db:.2f} dB margin; '
                    f'using the ideal pole positions. Try a larger margin.',
        }

    lf0, Qv, _, t = _unpack(best.x, M)
    f0_new = np.exp(lf0)
    return f0_new.tolist(), Qv.tolist(), {
        'feasible': True, 'changed': True,
        'max_Q_before': max_q_before, 'max_Q_after': float(t),
        'note': f'Retuned within a {margin_db:.2f} dB spec margin: '
                f'worst-case Q {max_q_before:.2f} -> {float(t):.2f}.',
    }
