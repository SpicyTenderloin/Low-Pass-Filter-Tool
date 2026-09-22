"""Monte Carlo component-tolerance analysis for a realised Sallen-Key cascade.

Every resistor and capacitor in the BOM -- including the op-amp gain
resistors Rf/Rg and the output attenuator -- is randomly perturbed within
its tolerance, the cascaded response is recomputed, and the spread across
many trials shows how much a real build (with real part tolerances) can be
expected to vary from the nominal design.

This matters more than it might look for this topology: a Sallen-Key
stage's gain K sets its Q, and Q is most sensitive to K exactly where Q is
already high (see the note on K_stability_max in sk_realisation.py). So
Rf/Rg tolerance on a high-Q stage doesn't just blur its response a little,
it can push the stage's denominator coefficient a1 through zero -- genuine
instability, not just an off-spec response -- which is why trials are
checked for that explicitly rather than just re-measured against the mask.

Can be used either as a library (run_monte_carlo / summarize_monte_carlo,
called from main.py right after a design is realised) or run directly as a
script against any previously saved filter design JSON.
"""
import json

import numpy as np
from tqdm import tqdm
from scipy.signal import freqs

from sallen_key_tf import sk_biquad_tf, sk_first_order_tf, sk_a1
from report import log


def _perturb(value, tol_pct, rng, distribution):
    tol = tol_pct / 100.0
    if distribution == "uniform":
        return value * (1.0 + rng.uniform(-tol, tol))
    else:  # gaussian: tolerance treated as a 3-sigma bound
        return value * (1.0 + rng.normal(0.0, tol / 3.0))


def _sample_stage_tf(s, r_tol_pct, c_tol_pct, rng, distribution):
    """Return (num, den, stable) for one stage with perturbed part values."""
    if s['kind'] == 'biquad':
        R1 = _perturb(s['R1'], r_tol_pct, rng, distribution)
        R2 = _perturb(s['R2'], r_tol_pct, rng, distribution)
        C1 = _perturb(s['C1'], c_tol_pct, rng, distribution)
        C2 = _perturb(s['C2'], c_tol_pct, rng, distribution)
        Rf = _perturb(s['Rf'], r_tol_pct, rng, distribution)
        Rg = _perturb(s['Rg'], r_tol_pct, rng, distribution)
        K = 1.0 + Rf / Rg
        num, den, _, _ = sk_biquad_tf(R1, R2, C1, C2, K)
        stable = sk_a1(R1, R2, C1, C2, K) > 0
        return num, den, stable
    elif s['kind'] == 'first':
        R = _perturb(s['R'], r_tol_pct, rng, distribution)
        C = _perturb(s['C'], c_tol_pct, rng, distribution)
        num, den, _ = sk_first_order_tf(R, C)
        return num, den, True
    elif s['kind'] == 'attenuator':
        Ra = _perturb(s['Ra'], r_tol_pct, rng, distribution)
        Rb = _perturb(s['Rb'], r_tol_pct, rng, distribution)
        ratio = Rb / (Ra + Rb)
        return np.array([ratio]), np.array([1.0]), True
    else:
        raise ValueError(f"Unknown stage kind: {s['kind']}")


def _sample_cascade_tf(stages, r_tol_pct, c_tol_pct, rng, distribution):
    num_all, den_all = np.array([1.0]), np.array([1.0])
    stable = True
    for s in stages:
        num, den, ok = _sample_stage_tf(s, r_tol_pct, c_tol_pct, rng, distribution)
        stable = stable and ok
        num_all = np.polymul(num_all, num)
        den_all = np.polymul(den_all, den)
    return num_all, den_all, stable


def run_monte_carlo(stages, wp_hz, ws_hz, gpass_db, gstop_db, r_tol_pct, c_tol_pct,
                     n_trials, f_min=1.0, f_max=None, n_freq=500,
                     distribution="uniform", seed=0, show_progress=True):
    """Run a Monte Carlo tolerance analysis on a realised stage cascade.

    Returns a dict: f (freq axis), mag (n_trials x n_freq, NaN rows for
    unstable trials), nominal_mag, n_unstable, yield_frac (fraction of ALL
    trials, unstable ones included, that meet both the ripple and stopband
    spec), and per-trial ripple_db / atten_db arrays (stable trials only).
    """
    f_max = f_max or (ws_hz * 3)
    f = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
    w = 2 * np.pi * f
    pass_mask = f <= wp_hz
    stop_mask = f >= ws_hz

    rng = np.random.default_rng(seed)
    mag = np.full((n_trials, n_freq), np.nan)
    ripple_list, atten_list = [], []
    n_unstable = 0

    for i in tqdm(range(n_trials), desc="Monte Carlo", leave=False, disable=not show_progress):
        num, den, stable = _sample_cascade_tf(stages, r_tol_pct, c_tol_pct, rng, distribution)
        if not stable:
            n_unstable += 1
            continue
        _, H = freqs(num, den, w)
        m = 20 * np.log10(np.maximum(np.abs(H), 1e-30))
        mag[i] = m
        ref = m[pass_mask][0] if pass_mask.any() else m[0]
        ripple = (ref - m[pass_mask].min()) if pass_mask.any() else np.nan
        atten = (ref - m[stop_mask].max()) if stop_mask.any() else np.nan
        ripple_list.append(ripple)
        atten_list.append(atten)

    num0, den0, _ = _sample_cascade_tf(stages, 0.0, 0.0, rng, distribution)  # nominal (no perturbation)
    _, Hn = freqs(num0, den0, w)
    nominal_mag = 20 * np.log10(np.maximum(np.abs(Hn), 1e-30))

    ripple_arr = np.array(ripple_list)
    atten_arr = np.array(atten_list)
    meets_spec = (ripple_arr <= gpass_db) & (atten_arr >= gstop_db)
    yield_frac = float(meets_spec.sum()) / n_trials if n_trials else 0.0

    return {
        'f': f, 'mag': mag, 'nominal_mag': nominal_mag,
        'n_trials': n_trials, 'n_unstable': n_unstable,
        'ripple_db': ripple_arr, 'atten_db': atten_arr,
        'yield_frac': yield_frac,
        'r_tol_pct': r_tol_pct, 'c_tol_pct': c_tol_pct, 'distribution': distribution,
    }


def summarize_monte_carlo(result, gpass_db, gstop_db):
    log("\n===== Monte Carlo tolerance analysis =====")
    log(f"  R tolerance: +/-{result['r_tol_pct']:.2f}%   C tolerance: +/-{result['c_tol_pct']:.2f}%   "
        f"({result['distribution']}, {result['n_trials']} trials)")
    log(f"  Unstable trials: {result['n_unstable']}")
    if len(result['ripple_db']):
        rp = np.percentile(result['ripple_db'], [5, 50, 95])
        ap = np.percentile(result['atten_db'], [5, 50, 95])
        log(f"  Passband ripple  (5th/50th/95th pct): {rp[0]:.2f} / {rp[1]:.2f} / {rp[2]:.2f} dB "
            f"(spec <= {gpass_db:.2f} dB)")
        log(f"  Stopband atten.  (5th/50th/95th pct): {ap[0]:.2f} / {ap[1]:.2f} / {ap[2]:.2f} dB "
            f"(spec >= {gstop_db:.2f} dB)")
    log(f"  Yield (meets both criteria, out of all {result['n_trials']} trials): "
        f"{result['yield_frac'] * 100:.1f}%")
    if result['n_unstable'] > 0:
        log(f"  [!] {result['n_unstable']} trial(s) went unstable: gain-resistor tolerance on a "
            f"high-Q stage pushed it past its stability boundary (not just off-spec -- a real "
            f"oscillator). Tighten that stage's Rf/Rg tolerance, lower its target Q, or increase "
            f"K_MARGIN_STAB's safety margin in parameters.py.")


def _standalone():
    """Run as `python monte_carlo.py`: pick a saved design, ask for tolerance
    settings, run the analysis, plot it, and save the results alongside the
    design."""
    import load_filter as lf
    import cli
    from plotting import plot_monte_carlo, show_and_save

    filters = lf.list_saved_filters()
    if not filters:
        return
    try:
        choice = int(input("Select a filter to run Monte Carlo on (index): ")) - 1
        assert 0 <= choice < len(filters)
    except (ValueError, AssertionError):
        print("Invalid selection.")
        return

    data = lf.load_json(filters[choice])
    print(f"\nLoaded: {filters[choice].name}  ({data['type']}, order {data['order']})")

    mc = cli.get_monte_carlo_params_interactive()

    result = run_monte_carlo(data['sections'], data['wp_hz'], data['ws_hz'],
                              data['ripple_db'], data['atten_db'],
                              mc.r_tol_pct, mc.c_tol_pct, mc.n_trials,
                              distribution=mc.distribution, seed=mc.seed)
    summarize_monte_carlo(result, data['ripple_db'], data['atten_db'])

    fig, ax = plot_monte_carlo(result, data['wp_hz'], data['ws_hz'], data['ripple_db'], data['atten_db'])
    show = input("Show plot on screen? [y/N]: ").strip().lower() in ("y", "yes")
    plot_path = filters[choice].with_name(filters[choice].stem + "_montecarlo.png")
    show_and_save(fig, plot_path, show=show)
    print(f"[OK] Monte Carlo plot saved to {plot_path}")

    results_path = filters[choice].with_name(filters[choice].stem + "_montecarlo.json")
    with open(results_path, "w") as f:
        json.dump({
            'source_design': filters[choice].name,
            'r_tol_pct': mc.r_tol_pct, 'c_tol_pct': mc.c_tol_pct,
            'n_trials': mc.n_trials, 'distribution': mc.distribution,
            'n_unstable': result['n_unstable'], 'yield_frac': result['yield_frac'],
            'ripple_db_p5_p50_p95': [float(x) for x in np.percentile(result['ripple_db'], [5, 50, 95])]
                if len(result['ripple_db']) else None,
            'atten_db_p5_p50_p95': [float(x) for x in np.percentile(result['atten_db'], [5, 50, 95])]
                if len(result['atten_db']) else None,
        }, f, indent=4)
    print(f"[OK] Monte Carlo stats saved to {results_path}")


if __name__ == "__main__":
    _standalone()
