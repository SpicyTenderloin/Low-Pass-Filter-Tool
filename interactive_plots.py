"""Interactive Plotly versions of the response/passband/Monte Carlo plots,
built specifically for the Streamlit app (app.py).

The CLI (main.py, load_filter.py, monte_carlo.py) keeps using plotting.py's
matplotlib versions, since it saves static PNG files where interactivity
doesn't apply. app.py builds both: these for on-screen display (drag-zoom,
pan, hover, double-click to reset -- the standard Plotly toolbar, which
covers the same ground as MATLAB's figure zoom/pan/data-cursor tools), and
plotting.py's matplotlib figures purely for the "Save" button's PNG export.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import freqs

import parameters as P

PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}

# For the schematic specifically: Plotly.js defaults to `responsive: true`,
# which auto-resizes a chart to fill whatever container it's placed in --
# that silently overrides the explicit width/height render_plotly()
# computes from the data (so label spacing would still depend on the
# page's layout despite that fix). responsive: False makes it render at
# its own intrinsic size and stay there, so panning/scrolling is enough to
# see the rest of a wide cascade without ever needing to zoom in first.
SCHEMATIC_CONFIG = {**PLOTLY_CONFIG, "responsive": False}

# parameters.COLORS uses matplotlib's 'tab:' colour names (shared with the
# CLI's plots), which Plotly doesn't recognise -- translate the ones this
# project actually uses; anything already CSS-valid passes through as-is.
_MPL_TAB_COLORS = {
    'tab:blue': '#1f77b4', 'tab:orange': '#ff7f0e', 'tab:green': '#2ca02c',
    'tab:red': '#d62728', 'tab:purple': '#9467bd', 'tab:brown': '#8c564b',
    'tab:pink': '#e377c2', 'tab:gray': '#7f7f7f', 'tab:olive': '#bcbd22',
    'tab:cyan': '#17becf',
}


def _css_color(c):
    return _MPL_TAB_COLORS.get(c, c)


def _curve_xy(b, a, f_min, f_max, n=2000):
    f = np.logspace(np.log10(f_min), np.log10(f_max), n)
    w = 2 * np.pi * f
    _, H = freqs(b, a, w)
    mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))
    return f, mag, phase


def interactive_bode(curves, wp_hz, ws_hz, gpass_db, gstop_db, f_min=P.F_MIN, f_max=None):
    """curves: list of (b, a, label, color). Magnitude on top, phase on
    bottom, log-frequency x-axis shared between them. The y-axis defaults
    to just enough range to judge the spec (stopband depth plus a little
    headroom for the rolloff beyond it) instead of autoscaling to whatever
    the response happens to reach at f_max, which is usually far more
    negative than useful -- drag-zoom/scroll/double-click still reach the
    full range on demand."""
    f_max = f_max or P.F_MAX
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         row_heights=[0.65, 0.35])
    for b, a, label, color in curves:
        color = _css_color(color)
        f, mag, phase = _curve_xy(b, a, f_min, f_max)
        fig.add_trace(go.Scatter(x=f, y=mag, name=label, legendgroup=label,
                                  line=dict(color=color),
                                  hovertemplate=f"{label}: " + "%{y:.3f} dB<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=phase, name=label, legendgroup=label, showlegend=False,
                                  line=dict(color=color),
                                  hovertemplate=f"{label}: " + "%{y:.3f} rad<extra></extra>"),
                      row=2, col=1)

    for x in (wp_hz, ws_hz):
        fig.add_vline(x=x, line=dict(color="grey", dash="dash", width=1))
    fig.add_hline(y=-gpass_db, line=dict(color="grey", dash="dot", width=1), row=1, col=1)
    fig.add_hline(y=-gstop_db, line=dict(color="grey", dash="dot", width=1), row=1, col=1)

    y_bottom = -(gstop_db + 20)
    fig.update_xaxes(type="log", row=2, col=1, title_text="Frequency [Hz]")
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude [dB]", range=[y_bottom, 10], row=1, col=1)
    fig.update_yaxes(title_text="Phase [rad]", row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", dragmode="zoom",
                       margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.08))
    return fig


def interactive_passband_detail(curves, wp_hz, gpass_db, f_min=None, normalize=True):
    """curves: list of (b, a, label, color). Interactive counterpart of
    plotting.plot_passband_detail() + mark_passband_spec() -- see that
    docstring for why curves are normalized to their own peak by default."""
    f_min = f_min or max(0.1, wp_hz / 20000)
    f = np.linspace(f_min, wp_hz, 2000)
    w = 2 * np.pi * f

    fig = go.Figure()
    for b, a, label, color in curves:
        color = _css_color(color)
        _, H = freqs(b, a, w)
        mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        if normalize:
            mag = mag - mag.max()
        fig.add_trace(go.Scatter(x=f, y=mag, name=label, line=dict(color=color),
                                  hovertemplate=f"{label}: " + "%{y:.3f} dB<extra></extra>"))

    fig.add_hline(y=0, line=dict(color="grey", width=1))
    fig.add_hline(y=-gpass_db, line=dict(color="red", dash="dot", width=1.5),
                  annotation_text=f"-{gpass_db:.2f} dB spec", annotation_position="bottom right")
    fig.update_xaxes(type="log", title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="Magnitude rel. to own peak [dB]" if normalize else "Magnitude [dB]")
    fig.update_layout(height=420, hovermode="x unified", dragmode="zoom", margin=dict(t=20, b=10))
    return fig


def interactive_monte_carlo(result, wp_hz, ws_hz, gpass_db, gstop_db,
                             max_trials_shown=120, max_points_per_trial=150):
    """Every shown trial drawn as one combined WebGL trace (each trial's
    points separated by a None gap) rather than one Plotly trace per trial
    -- hundreds of separate SVG traces would be noticeably slower to render
    and interact with than a single WebGL (Scattergl) trace carrying the
    same data.

    More than ~100-150 translucent trial lines doesn't change the visual
    impression of spread but does slow down interaction (a 500-trial run
    at full frequency resolution serializes to several MB of JSON, which
    dominated per-rerun time in testing), so the overlay is capped and
    frequency-subsampled for display. Yield/percentile statistics, reported
    separately by the caller, always come from the full, uncapped result --
    only this drawing is trimmed.
    """
    f_full = result['f']
    step = max(1, len(f_full) // max_points_per_trial)
    f = f_full[::step]

    valid_rows = [row for row in result['mag'] if not np.all(np.isnan(row))]
    if len(valid_rows) > max_trials_shown:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(valid_rows), size=max_trials_shown, replace=False)
        valid_rows = [valid_rows[i] for i in idx]

    xs, ys = [], []
    for row in valid_rows:
        xs.extend(f.tolist()); xs.append(None)
        ys.extend(row[::step].tolist()); ys.append(None)

    fig = go.Figure()
    if xs:
        n_shown_label = f"Trials (showing {len(valid_rows)} of {result['n_trials'] - result['n_unstable']} stable)"
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines",
                                    line=dict(color="rgba(220,50,50,0.15)", width=1),
                                    name=n_shown_label, hoverinfo="skip"))
    # Nominal is a single trace (cheap either way) -- keep it at full
    # resolution rather than the trials' subsampled f, which would
    # otherwise mismatch nominal_mag's length.
    fig.add_trace(go.Scatter(x=f_full, y=result['nominal_mag'], mode="lines",
                              line=dict(color="black", width=2), name="Nominal",
                              hovertemplate="Nominal: %{y:.3f} dB<extra></extra>"))
    for x in (wp_hz, ws_hz):
        fig.add_vline(x=x, line=dict(color="grey", dash="dash", width=1))
    fig.add_hline(y=-gpass_db, line=dict(color="grey", dash="dot", width=1))
    fig.add_hline(y=-gstop_db, line=dict(color="grey", dash="dot", width=1))

    y_bottom = -(gstop_db + 20)
    fig.update_xaxes(type="log", title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="Magnitude [dB]", range=[y_bottom, 10])
    fig.update_layout(
        height=460, dragmode="zoom", margin=dict(t=40, b=10),
        title=f"{result['n_trials']} trials ({result['n_unstable']} unstable), "
              f"yield = {result['yield_frac'] * 100:.1f}%")
    return fig
