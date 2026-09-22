import time
import json
from pathlib import Path

import numpy as np
from scipy.signal import freqs

import cli
import parameters as P
from prototype_filter import generate_prototype
from sallen_key_tf import build_lpf_sections_from_poles, compute_cascade_tf, compute_ideal_cascade_tf
from sk_realisation import pick_sk_parts_for_biquad, pick_sk_parts_for_first_order, pick_attenuator
from stagger_tuning import retune_poles
from monte_carlo import run_monte_carlo, summarize_monte_carlo
from report import log, write_report
from bom import print_bom
from plotting import plot_bode, mark_spec, plot_monte_carlo, show_and_save


def save_filter_json(design_data, out_dir="filter designs", filename=None):
    Path(out_dir).mkdir(exist_ok=True)
    if not filename:
        filename = time.strftime("filter_%Y%m%d-%H%M%S.json")
    if not filename.endswith(".json"):
        filename += ".json"
    filepath = Path(out_dir) / filename
    with open(filepath, "w") as f:
        json.dump(design_data, f, indent=4)
    log(f"\n[OK] Filter design saved as JSON -> {filepath}")
    return filepath


def realise_sections(sections, r_lib, c_lib):
    """Realise every ideal (biquad/first-order) section into real parts."""
    realised, failures = [], []
    for i, s in enumerate(sections, 1):
        if s['kind'] == 'biquad':
            r = pick_sk_parts_for_biquad(s['w0'], s['Q'], P.STATIC_M_RATIOS, P.STATIC_N_RATIOS,
                                          r_lib, c_lib, stage_idx=i, stage_count=len(sections))
        else:
            r = pick_sk_parts_for_first_order(s['w0'], r_lib, c_lib,
                                               stage_idx=i, stage_count=len(sections))
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
    attenuation against the spec."""
    f_pass = np.linspace(1.0, wp_hz, 400)
    f_stop = np.geomspace(ws_hz, 20 * ws_hz, 200)
    _, Hp = freqs(b, a, 2 * np.pi * f_pass)
    _, Hs = freqs(b, a, 2 * np.pi * f_stop)
    mag_p = 20 * np.log10(np.maximum(np.abs(Hp), 1e-30))
    mag_s = 20 * np.log10(np.maximum(np.abs(Hs), 1e-30))
    ref = mag_p[0]
    ripple_achieved = ref - mag_p.min()
    atten_achieved = ref - mag_s.max()
    return {
        'ripple_db_achieved': float(ripple_achieved),
        'ripple_ok': bool(ripple_achieved <= gpass_db + 1e-6),
        'atten_db_achieved': float(atten_achieved),
        'atten_ok': bool(atten_achieved >= gstop_db - 1e-6),
    }


def run(spec=None):
    if spec is None:
        spec = cli.get_spec_interactive()

    log(f"\n{spec.name} -- Type: {spec.filter_type}")
    r_lib, c_lib = P.build_libraries(spec.e_series)

    b, a, N, Wn = generate_prototype(spec.filter_type, spec.wp_hz, spec.ws_hz,
                                      spec.gpass_db, spec.gstop_db)
    log(f"Order: {N}  Prototype edge: {Wn / (2 * np.pi):.2f} Hz")
    log("Ideal transfer function:")
    log(f"b = {b}")
    log(f"a = {a}")

    sections = build_lpf_sections_from_poles(b, a)
    biquads = [s for s in sections if s['kind'] == 'biquad']
    first_orders = [s for s in sections if s['kind'] == 'first']
    log(f"\n{len(biquads)} biquad stage(s), {len(first_orders)} first-order stage(s).")

    retune_info = None
    if spec.retune_enabled and biquads:
        f0_list = [s['w0'] / (2 * np.pi) for s in biquads]
        Q_list = [s['Q'] for s in biquads]
        fixed_w0_hz = (first_orders[0]['w0'] / (2 * np.pi)) if first_orders else None
        log("\nRetuning poles (stagger tuning)...")
        f0_new, Q_new, retune_info = retune_poles(
            f0_list, Q_list, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db,
            margin_db=spec.retune_margin_db, fixed_w0_hz=fixed_w0_hz)
        log(f"  {retune_info['note']}")
        if retune_info['feasible']:
            for s, f0, Qv in zip(biquads, f0_new, Q_new):
                s['w0'], s['Q'] = 2 * np.pi * f0, Qv

    retuned_b, retuned_a = compute_ideal_cascade_tf(sections)

    realised = realise_sections(sections, r_lib, c_lib)

    # --- gain compensation: each SK stage's K sets its Q *and* its DC
    # gain, so the cascade always ends up with excess passband gain. Size
    # an output attenuator to bring the overall gain to the target. ---
    total_gain = 1.0
    for s in realised:
        if s['kind'] == 'biquad':
            total_gain *= s['K_act']
    total_gain_db = 20 * np.log10(total_gain)
    log(f"\nRealised cascade passband gain before compensation: {total_gain_db:.2f} dB")
    needed_atten_db = total_gain_db - spec.target_gain_db
    if needed_atten_db > 1e-6:
        att = pick_attenuator(needed_atten_db, r_lib)
        realised.append(att)
        log(f"Added output attenuator: target {att['atten_db_tgt']:.2f} dB, "
            f"actual {att['atten_db_act']:.2f} dB")
    elif needed_atten_db < -1e-6:
        log(f"[!] Target gain is {-needed_atten_db:.2f} dB higher than the cascade "
            f"reaches passively; you would need an extra amplifier stage for that.")

    print_bom(realised)

    real_b, real_a = compute_cascade_tf(realised)
    check = verify_response(real_b, real_a, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db)
    log("\n===== Verification against spec =====")
    log(f"  Passband ripple : {check['ripple_db_achieved']:.2f} dB "
        f"(spec <= {spec.gpass_db:.2f} dB)  {'OK' if check['ripple_ok'] else 'FAIL'}")
    log(f"  Stopband atten. : {check['atten_db_achieved']:.2f} dB "
        f"(spec >= {spec.gstop_db:.2f} dB)  {'OK' if check['atten_ok'] else 'FAIL'}")
    if not (check['ripple_ok'] and check['atten_ok']):
        log("  [!] The realised design does not meet the spec with the parts available. "
            "Try a larger E-series, a bigger retune margin, or a looser spec.")

    f_max_plot = max(spec.ws_hz * 3, P.F_MAX)
    fig, axes = plot_bode(b, a, label="Ideal", color=P.COLORS['ideal'], f_max=f_max_plot)
    if retune_info and retune_info.get('changed'):
        plot_bode(retuned_b, retuned_a, label="Retuned target", color=P.COLORS['retuned'],
                   axes=axes, f_max=f_max_plot)
    plot_bode(real_b, real_a, label="Realised", color=P.COLORS['realised'], axes=axes, f_max=f_max_plot)
    mark_spec(axes, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db)

    out_dir = Path("filter designs")
    out_dir.mkdir(exist_ok=True)
    base = spec.out_name or time.strftime("filter_%Y%m%d-%H%M%S")
    plot_path = out_dir / f"{base}_bode.png"
    show_and_save(fig, plot_path, show=spec.show_plots)
    log(f"[OK] Plot saved to {plot_path}")

    filter_data = {
        "name": spec.name,
        "type": spec.filter_type,
        "order": int(N),
        "wp_hz": spec.wp_hz,
        "ws_hz": spec.ws_hz,
        "ripple_db": spec.gpass_db,
        "atten_db": spec.gstop_db,
        "target_gain_db": spec.target_gain_db,
        "e_series": spec.e_series,
        "retune": retune_info,
        "verification": check,
        "sections": realised,
        "monte_carlo": None,
    }

    if cli.ask_run_monte_carlo():
        mc = cli.get_monte_carlo_params_interactive()
        result = run_monte_carlo(realised, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db,
                                  mc.r_tol_pct, mc.c_tol_pct, mc.n_trials, distribution=mc.distribution)
        summarize_monte_carlo(result, spec.gpass_db, spec.gstop_db)

        mc_fig, mc_ax = plot_monte_carlo(result, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db)
        mc_path = out_dir / f"{base}_montecarlo.png"
        show_and_save(mc_fig, mc_path, show=spec.show_plots)
        log(f"[OK] Monte Carlo plot saved to {mc_path}")

        rp = np.percentile(result['ripple_db'], [5, 50, 95]).tolist() if len(result['ripple_db']) else None
        ap = np.percentile(result['atten_db'], [5, 50, 95]).tolist() if len(result['atten_db']) else None
        filter_data["monte_carlo"] = {
            "r_tol_pct": mc.r_tol_pct, "c_tol_pct": mc.c_tol_pct,
            "n_trials": mc.n_trials, "distribution": mc.distribution,
            "n_unstable": result['n_unstable'], "yield_frac": result['yield_frac'],
            "ripple_db_p5_p50_p95": rp, "atten_db_p5_p50_p95": ap,
        }

    save_filter_json(filter_data, filename=f"{base}.json")
    write_report()
    return filter_data


if __name__ == "__main__":
    run()
