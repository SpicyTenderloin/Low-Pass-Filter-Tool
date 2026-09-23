import time
import json
from pathlib import Path

import numpy as np

import cli
import parameters as P
from engine import design_filter
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


def run(spec=None):
    if spec is None:
        spec = cli.get_spec_interactive()

    log(f"\n{spec.name} -- Type: {spec.filter_type}")
    log("Designing (this searches the part library per stage, watch for progress bars)...")

    d = design_filter(spec.filter_type, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db,
                       order_override=spec.order_override, e_series=spec.e_series,
                       retune_enabled=spec.retune_enabled, retune_margin_db=spec.retune_margin_db,
                       target_gain_db=spec.target_gain_db, show_progress=True)

    if spec.order_override:
        log(f"Order: {d['N']} (manual override -- automatic minimum was {d['n_min']})  "
            f"Prototype edge: {d['Wn'] / (2 * np.pi):.2f} Hz")
    else:
        log(f"Order: {d['N']} (automatic minimum)  Prototype edge: {d['Wn'] / (2 * np.pi):.2f} Hz")
    log("Ideal transfer function:")
    log(f"b = {d['ideal_b']}")
    log(f"a = {d['ideal_a']}")

    biquads = [s for s in d['sections'] if s['kind'] == 'biquad']
    first_orders = [s for s in d['sections'] if s['kind'] == 'first']
    log(f"\n{len(biquads)} biquad stage(s), {len(first_orders)} first-order stage(s).")

    if spec.retune_enabled and d['retune_info']:
        log("\nRetuning poles (stagger tuning)...")
        log(f"  {d['retune_info']['note']}")

    log(f"\nRealised cascade passband gain before compensation: {d['total_gain_db']:.2f} dB")
    if d['attenuator']:
        att = d['attenuator']
        log(f"Added output attenuator: target {att['atten_db_tgt']:.2f} dB, "
            f"actual {att['atten_db_act']:.2f} dB")
    elif d['needed_atten_db'] < -1e-6:
        log(f"[!] Target gain is {-d['needed_atten_db']:.2f} dB higher than the cascade "
            f"reaches passively; you would need an extra amplifier stage for that.")

    print_bom(d['realised'])

    check = d['check']
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
    fig, axes = plot_bode(d['ideal_b'], d['ideal_a'], label="Ideal", color=P.COLORS['ideal'], f_max=f_max_plot)
    if d['retune_info'] and d['retune_info'].get('changed'):
        plot_bode(d['retuned_b'], d['retuned_a'], label="Retuned target", color=P.COLORS['retuned'],
                   axes=axes, f_max=f_max_plot)
    plot_bode(d['real_b'], d['real_a'], label="Realised", color=P.COLORS['realised'], axes=axes, f_max=f_max_plot)
    mark_spec(axes, spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db)
    show_fig(fig, show=spec.show_plots)
    pending_plots.append((fig, f"{base}_bode.png"))

    pb_fig, pb_ax = plot_passband_detail(d['ideal_b'], d['ideal_a'], spec.wp_hz, label="Ideal", color=P.COLORS['ideal'])
    if d['retune_info'] and d['retune_info'].get('changed'):
        plot_passband_detail(d['retuned_b'], d['retuned_a'], spec.wp_hz, label="Retuned target",
                              color=P.COLORS['retuned'], ax=pb_ax)
    plot_passband_detail(d['real_b'], d['real_a'], spec.wp_hz, label="Realised", color=P.COLORS['realised'], ax=pb_ax)
    mark_passband_spec(pb_ax, spec.gpass_db, ref_db=check['ripple_peak_db'])
    pb_ax.set_title("Passband detail (zoomed)")
    show_fig(pb_fig, show=spec.show_plots)
    pending_plots.append((pb_fig, f"{base}_passband.png"))

    filter_data = {
        "name": spec.name,
        "type": spec.filter_type,
        "order": int(d['N']),
        "order_override": spec.order_override,
        "wp_hz": spec.wp_hz,
        "ws_hz": spec.ws_hz,
        "ripple_db": spec.gpass_db,
        "atten_db": spec.gstop_db,
        "target_gain_db": spec.target_gain_db,
        "e_series": spec.e_series,
        "retune": d['retune_info'],
        "verification": check,
        "sections": d['realised'],
        "monte_carlo": None,
    }

    if cli.ask_run_monte_carlo():
        mc = cli.get_monte_carlo_params_interactive()
        result = run_monte_carlo(d['realised'], spec.wp_hz, spec.ws_hz, spec.gpass_db, spec.gstop_db,
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
