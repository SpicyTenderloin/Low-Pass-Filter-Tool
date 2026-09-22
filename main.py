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
from plotting import (plot_bode, mark_spec, plot_passband_detail, mark_passband_spec,
                       plot_monte_carlo, show_fig, save_fig, block_until_closed)


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
    r_tag = "OK" if check['ripple_ok'] else "FAIL"
    log(f"  [{r_tag}] Passband ripple : {check['ripple_db_achieved']:.2f} dB peak-to-peak "
        f"(spec <= {spec.gpass_db:.2f} dB)")
    log(f"          peak {check['ripple_peak_db']:+.2f} dB @ {check['ripple_peak_hz']:.1f} Hz   "
        f"dip {check['ripple_dip_db']:+.2f} dB @ {check['ripple_dip_hz']:.1f} Hz")
    a_tag = "OK" if check['atten_ok'] else "FAIL"
    log(f"  [{a_tag}] Stopband atten. : {check['atten_db_achieved']:.2f} dB "
        f"(spec >= {spec.gstop_db:.2f} dB)")
    log(f"          weakest point {check['atten_worst_db']:+.2f} dB @ {check['atten_worst_hz']:.1f} Hz")
    if not (check['ripple_ok'] and check['atten_ok']):
        failed = [name for name, ok in (("passband ripple", check['ripple_ok']),
                                         ("stopband attenuation", check['atten_ok'])) if not ok]
        log(f"  [!] Failed: {', '.join(failed)}. The realised design does not meet the spec "
            "with the parts available. Try a larger E-series, a bigger retune margin, or a "
            "looser spec.")

    base = spec.out_name or time.strftime("filter_%Y%m%d-%H%M%S")
    out_dir = Path("filter designs")
    pending_plots = []  # (fig, filename) -- only written to disk if the user opts to save

    f_max_plot = max(spec.ws_hz * 3, P.F_MAX)
    fig, axes = plot_bode(b, a, label="Ideal", color=P.COLORS['ideal'], f_max=f_max_plot)
    if retune_info and retune_info.get('changed'):
        plot_bode(retuned_b, retuned_a, label="Retuned target", color=P.COLORS['retuned'],
                   axes=axes, f_max=f_max_plot)
    plot_bode(real_b, real_a, label="Realised", color=P.COLORS['realised'], axes=axes, f_max=f_max_plot)
    mark_spec(axes, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db)
    show_fig(fig, show=spec.show_plots)
    pending_plots.append((fig, f"{base}_bode.png"))

    pb_fig, pb_ax = plot_passband_detail(b, a, spec.wp_hz, label="Ideal", color=P.COLORS['ideal'])
    if retune_info and retune_info.get('changed'):
        plot_passband_detail(retuned_b, retuned_a, spec.wp_hz, label="Retuned target",
                              color=P.COLORS['retuned'], ax=pb_ax)
    plot_passband_detail(real_b, real_a, spec.wp_hz, label="Realised", color=P.COLORS['realised'], ax=pb_ax)
    mark_passband_spec(pb_ax, spec.gpass_db, ref_db=check['ripple_peak_db'])
    pb_ax.set_title("Passband detail (zoomed)")
    show_fig(pb_fig, show=spec.show_plots)
    pending_plots.append((pb_fig, f"{base}_passband.png"))

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
        show_fig(mc_fig, show=spec.show_plots)
        pending_plots.append((mc_fig, f"{base}_montecarlo.png"))

        rp = np.percentile(result['ripple_db'], [5, 50, 95]).tolist() if len(result['ripple_db']) else None
        ap = np.percentile(result['atten_db'], [5, 50, 95]).tolist() if len(result['atten_db']) else None
        filter_data["monte_carlo"] = {
            "r_tol_pct": mc.r_tol_pct, "c_tol_pct": mc.c_tol_pct,
            "n_trials": mc.n_trials, "distribution": mc.distribution,
            "n_unstable": result['n_unstable'], "yield_frac": result['yield_frac'],
            "ripple_db_p5_p50_p95": rp, "atten_db_p5_p50_p95": ap,
        }

    if cli.ask_save_design():
        out_dir.mkdir(exist_ok=True)
        for fig_, filename in pending_plots:
            save_fig(fig_, out_dir / filename)
            log(f"[OK] Plot saved to {out_dir / filename}")
        save_filter_json(filter_data, filename=f"{base}.json")
        write_report()
    else:
        log("\n[i] Design not saved (nothing written to \"filter designs/\").")

    block_until_closed()
    return filter_data


if __name__ == "__main__":
    run()
