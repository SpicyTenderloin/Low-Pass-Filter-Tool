"""Interactive Sallen-Key filter designer.

Run with:  streamlit run app.py

A GUI counterpart to main.py -- same engine (engine.py), same part search,
same retuning, same Monte Carlo, but change a field/slider and the plots
and BOM update automatically instead of having to re-run and re-answer a
string of prompts. Designs saved here use the same JSON schema as the CLI,
so load_filter.py and monte_carlo.py's standalone script can open them too.
"""
import time
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import parameters as P
import cli as climod  # reuse its persisted "last run" defaults, not its prompts
from prototype_filter import compute_min_order
from engine import design_filter
from monte_carlo import run_monte_carlo
from bom import eng_unit
from plotting import plot_bode, mark_spec, plot_passband_detail, mark_passband_spec, plot_monte_carlo

st.set_page_config(page_title="Filter Designer", layout="wide")


# ------------------------------------------------------------------ cache
@st.cache_data(show_spinner="Designing filter (searching the part library)...")
def cached_design(filter_type, wp_hz, ws_hz, gpass_db, gstop_db, order_override,
                   e_series, retune_enabled, retune_margin_db, target_gain_db):
    try:
        return {'ok': True, 'error': None,
                'data': design_filter(filter_type, wp_hz, ws_hz, gpass_db, gstop_db,
                                       order_override=order_override, e_series=e_series,
                                       retune_enabled=retune_enabled, retune_margin_db=retune_margin_db,
                                       target_gain_db=target_gain_db, show_progress=False)}
    except RuntimeError as e:
        return {'ok': False, 'error': str(e), 'data': None}


@st.cache_data(show_spinner="Running Monte Carlo...")
def cached_monte_carlo(realised, wp_hz, ws_hz, gpass_db, gstop_db, r_tol_pct, c_tol_pct,
                        n_trials, distribution):
    return run_monte_carlo(realised, wp_hz, ws_hz, gpass_db, gstop_db, r_tol_pct, c_tol_pct,
                            n_trials, distribution=distribution, show_progress=False)


# ------------------------------------------------------------------ sidebar
last = climod._load_state().get('spec', {})
last_mc = climod._load_state().get('monte_carlo', {})

with st.sidebar:
    st.header("Spec")
    filter_type = st.selectbox("Filter type", ["chebyshev", "butterworth"],
                                index=["chebyshev", "butterworth"].index(last.get('filter_type', 'chebyshev')))
    wp_hz = st.number_input("Passband edge (Hz)", min_value=0.1,
                             value=float(last.get('wp_hz', 20000.0)))
    ws_hz = st.number_input("Stopband edge (Hz)", min_value=0.1,
                             value=float(last.get('ws_hz', wp_hz * 1.4)))
    gpass_db = st.number_input("Passband ripple / max loss (dB)", min_value=0.001,
                                value=float(last.get('gpass_db', 1.0)), format="%.3f")
    gstop_db = st.number_input("Stopband attenuation (dB)", min_value=0.001,
                                value=float(last.get('gstop_db', 40.0)))
    if ws_hz <= wp_hz:
        st.error("Stopband edge must be above the passband edge.")
        st.stop()
    if gstop_db <= gpass_db:
        st.error("Stopband attenuation must exceed the passband ripple.")
        st.stop()

    st.header("Build options")
    n_min = compute_min_order(filter_type, wp_hz, ws_hz, gpass_db, gstop_db)
    st.caption(f"Automatic minimum order: N={n_min} (no headroom once realised in real parts)")
    order_in = st.number_input("Filter order (>= minimum uses that many poles; "
                                "at/below the minimum, the automatic order is used)",
                                min_value=1, value=max(n_min, 1), step=1)
    order_override = int(order_in) if order_in > n_min else None
    stage_budget = st.number_input("Stage budget (warn if exceeded)", min_value=1, value=10, step=1)

    target_gain_db = st.number_input("Target passband gain (dB)",
                                      value=float(last.get('target_gain_db', 0.0)))
    e_series = int(st.selectbox("Component E-series", [24, 12],
                                 index=[24, 12].index(int(last.get('e_series', 24)))))
    retune_enabled = st.checkbox("Enable pole retuning (stagger tuning)",
                                  value=bool(last.get('retune_enabled', True)))
    retune_margin_db = 0.0
    if retune_enabled:
        retune_margin_db = st.number_input(
            "Retune margin (dB) -- positive relaxes the target to buy lower Q; "
            "negative tunes tighter than spec to reserve headroom for realisation error",
            value=float(last.get('retune_margin_db', 0.5)), step=0.05, format="%.3f")

    st.header("Monte Carlo tolerance")
    r_tol_pct = st.slider("Resistor tolerance (%)", 0.1, 20.0, float(last_mc.get('r_tol_pct', 1.0)), 0.1)
    c_tol_pct = st.slider("Capacitor tolerance (%)", 0.1, 20.0, float(last_mc.get('c_tol_pct', 5.0)), 0.1)
    mc_trials = st.number_input("Monte Carlo trials", min_value=20,
                                 value=int(last_mc.get('n_trials', 200)), step=20)
    mc_distribution = st.selectbox("Distribution", ["uniform", "gaussian"],
                                    index=["uniform", "gaussian"].index(last_mc.get('distribution', 'uniform')))


# ------------------------------------------------------------------ main
st.title("Sallen-Key Low-Pass Filter Designer")

result = cached_design(filter_type, wp_hz, ws_hz, gpass_db, gstop_db, order_override,
                        e_series, retune_enabled, retune_margin_db, target_gain_db)

if not result['ok']:
    st.error(f"Design failed: {result['error']}")
    st.stop()

d = result['data']
check = d['check']

if d['stages'] > stage_budget:
    st.warning(f"This design uses {d['stages']} stages, over your {stage_budget}-stage budget.")
if d['retune_info'] and not d['retune_info']['feasible']:
    st.warning(f"Retuning could not find a feasible solution within the given margin: "
               f"{d['retune_info']['note']} Falling back to the ideal (un-retuned) poles.")

order_note = (f"manual override, automatic minimum is {d['n_min']}" if order_override
              else "automatic minimum")
st.caption(f"Order {d['N']} ({order_note}) -- {d['stages']} stage(s)"
           + (" + output attenuator" if d['attenuator'] else ""))

col1, col2 = st.columns(2)
with col1:
    if check['ripple_ok']:
        st.success(f"Passband ripple: {check['ripple_db_achieved']:.3f} dB "
                    f"peak-to-peak (spec ≤ {gpass_db:.3f} dB)")
    else:
        st.error(f"Passband ripple: {check['ripple_db_achieved']:.3f} dB "
                  f"peak-to-peak (spec ≤ {gpass_db:.3f} dB)")
    st.caption(f"peak {check['ripple_peak_db']:+.2f} dB @ {check['ripple_peak_hz']:.0f} Hz   "
               f"dip {check['ripple_dip_db']:+.2f} dB @ {check['ripple_dip_hz']:.0f} Hz")
with col2:
    if check['atten_ok']:
        st.success(f"Stopband attenuation: {check['atten_db_achieved']:.2f} dB "
                    f"(spec ≥ {gstop_db:.2f} dB)")
    else:
        st.error(f"Stopband attenuation: {check['atten_db_achieved']:.2f} dB "
                  f"(spec ≥ {gstop_db:.2f} dB)")
    st.caption(f"weakest point {check['atten_worst_db']:+.2f} dB @ {check['atten_worst_hz']:.0f} Hz")

# --- Plot 1: response ---
st.subheader("Response")
f_max_plot = max(ws_hz * 3, P.F_MAX)
fig1, axes1 = plot_bode(d['ideal_b'], d['ideal_a'], label="Ideal", color=P.COLORS['ideal'], f_max=f_max_plot)
if d['retune_info'] and d['retune_info'].get('changed'):
    plot_bode(d['retuned_b'], d['retuned_a'], label="Retuned target", color=P.COLORS['retuned'],
              axes=axes1, f_max=f_max_plot)
plot_bode(d['real_b'], d['real_a'], label="Realised", color=P.COLORS['realised'], axes=axes1, f_max=f_max_plot)
mark_spec(axes1, wp_hz, ws_hz, gpass_db, gstop_db)
st.pyplot(fig1)
plt.close(fig1)

with st.expander("Passband detail (zoomed) -- ripple failures are often invisible on the plot above"):
    fig1b, ax1b = plot_passband_detail(d['ideal_b'], d['ideal_a'], wp_hz, label="Ideal", color=P.COLORS['ideal'])
    if d['retune_info'] and d['retune_info'].get('changed'):
        plot_passband_detail(d['retuned_b'], d['retuned_a'], wp_hz, label="Retuned target",
                              color=P.COLORS['retuned'], ax=ax1b)
    plot_passband_detail(d['real_b'], d['real_a'], wp_hz, label="Realised", color=P.COLORS['realised'], ax=ax1b)
    mark_passband_spec(ax1b, gpass_db)
    st.pyplot(fig1b)
    plt.close(fig1b)

# --- BOM table ---
st.subheader("Bill of Materials")
rows = []
for i, s in enumerate(d['realised'], 1):
    if s['kind'] == 'biquad':
        rows.append({
            'Stage': i, 'Kind': 'Biquad',
            'f0 target (Hz)': f"{s['f0_tgt']:.1f}", 'f0 actual (Hz)': f"{s['f0_act']:.1f}",
            'Q target': f"{s['Q_tgt']:.3f}", 'Q actual': f"{s['Q_act']:.3f}",
            'R1': eng_unit(s['R1'], 'ohm'), 'R2': eng_unit(s['R2'], 'ohm'),
            'C1': eng_unit(s['C1'], 'F'), 'C2': eng_unit(s['C2'], 'F'),
            'Rf': eng_unit(s['Rf'], 'ohm'), 'Rg': eng_unit(s['Rg'], 'ohm'),
            'K': f"{s['K_act']:.3f}",
        })
    elif s['kind'] == 'first':
        rows.append({
            'Stage': i, 'Kind': 'First-order',
            'f0 target (Hz)': f"{s['f0_tgt']:.1f}", 'f0 actual (Hz)': f"{s['f0_act']:.1f}",
            'Q target': '-', 'Q actual': '-',
            'R1': eng_unit(s['R'], 'ohm'), 'R2': '-', 'C1': eng_unit(s['C'], 'F'), 'C2': '-',
            'Rf': '-', 'Rg': '-', 'K': '-',
        })
    else:  # attenuator
        rows.append({
            'Stage': i, 'Kind': 'Output attenuator',
            'f0 target (Hz)': '-', 'f0 actual (Hz)': '-', 'Q target': '-', 'Q actual': '-',
            'R1': eng_unit(s['Ra'], 'ohm') + ' (series)', 'R2': eng_unit(s['Rb'], 'ohm') + ' (shunt)',
            'C1': '-', 'C2': '-', 'Rf': '-', 'Rg': '-',
            'K': f"{s['atten_db_act']:.2f} dB atten",
        })
st.dataframe(rows, width='stretch', hide_index=True)

# --- Plot 2: Monte Carlo ---
st.subheader("Monte Carlo (component tolerance)")
mc_result = cached_monte_carlo(d['realised'], wp_hz, ws_hz, gpass_db, gstop_db,
                                r_tol_pct, c_tol_pct, int(mc_trials), mc_distribution)
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Yield (meets spec)", f"{mc_result['yield_frac'] * 100:.1f}%")
mcol2.metric("Unstable trials", f"{mc_result['n_unstable']} / {mc_result['n_trials']}")
rp = np.percentile(mc_result['ripple_db'], [5, 50, 95]) if len(mc_result['ripple_db']) else [float('nan')] * 3
ap = np.percentile(mc_result['atten_db'], [5, 50, 95]) if len(mc_result['atten_db']) else [float('nan')] * 3
mcol3.metric("Median ripple / atten.", f"{rp[1]:.2f} dB / {ap[1]:.2f} dB")
st.caption(f"Ripple 5th/50th/95th pct: {rp[0]:.2f} / {rp[1]:.2f} / {rp[2]:.2f} dB   "
           f"Attenuation 5th/50th/95th pct: {ap[0]:.2f} / {ap[1]:.2f} / {ap[2]:.2f} dB")

fig2, ax2 = plot_monte_carlo(mc_result, wp_hz, ws_hz, gpass_db, gstop_db)
st.pyplot(fig2)
plt.close(fig2)

# --- Save ---
st.subheader("Save")
if "out_name" not in st.session_state:
    st.session_state.out_name = time.strftime("filter_%Y%m%d-%H%M%S")
out_name = st.text_input("Base filename", key="out_name")
if st.button("Save this design (JSON + plots) to \"filter designs/\""):
    out_dir = Path("filter designs")
    out_dir.mkdir(exist_ok=True)
    fig1.savefig(out_dir / f"{out_name}_bode.png", dpi=150)
    fig1b.savefig(out_dir / f"{out_name}_passband.png", dpi=150)
    fig2.savefig(out_dir / f"{out_name}_montecarlo.png", dpi=150)
    rp_list = rp.tolist() if hasattr(rp, 'tolist') else list(rp)
    ap_list = ap.tolist() if hasattr(ap, 'tolist') else list(ap)
    filter_data = {
        "name": P.FILTER_DESIGNER_NAME, "type": filter_type,
        "order": int(d['N']), "order_override": order_override,
        "wp_hz": wp_hz, "ws_hz": ws_hz, "ripple_db": gpass_db, "atten_db": gstop_db,
        "target_gain_db": target_gain_db, "e_series": e_series,
        "retune": d['retune_info'], "verification": check, "sections": d['realised'],
        "monte_carlo": {
            "r_tol_pct": r_tol_pct, "c_tol_pct": c_tol_pct, "n_trials": int(mc_trials),
            "distribution": mc_distribution, "n_unstable": mc_result['n_unstable'],
            "yield_frac": mc_result['yield_frac'],
            "ripple_db_p5_p50_p95": rp_list, "atten_db_p5_p50_p95": ap_list,
        },
    }
    import json
    with open(out_dir / f"{out_name}.json", "w") as f:
        json.dump(filter_data, f, indent=4)
    st.success(f"Saved to {out_dir / f'{out_name}.json'} "
                f"(plus _bode.png, _passband.png, and _montecarlo.png)")
