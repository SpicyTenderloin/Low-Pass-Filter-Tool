import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqs

import parameters as P


def new_bode_fig():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.set_ylabel("Magnitude [dB]")
    ax1.grid(True, which='both', linestyle=':')
    ax2.set_ylabel("Phase [rad]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.grid(True, which='both', linestyle=':')
    return fig, (ax1, ax2)


def plot_bode(b, a, label="TF", color="tab:blue", axes=None, f_min=P.F_MIN, f_max=P.F_MAX):
    """Plot magnitude/phase of a transfer function. Pass `axes` (from an
    earlier call's return value) to overlay several curves on one figure."""
    f = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
    w = 2 * np.pi * f
    _, H = freqs(b, a, w)
    mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))

    if axes is None:
        fig, (ax1, ax2) = new_bode_fig()
    else:
        ax1, ax2 = axes
        fig = ax1.figure

    ax1.semilogx(f, mag, label=label, color=color)
    ax2.semilogx(f, phase, label=label, color=color)
    ax1.legend()
    return fig, (ax1, ax2)


def mark_spec(axes, wp_hz, ws_hz, gpass_db, gstop_db):
    """Draw the passband ripple / stopband attenuation mask on a Bode plot
    for visual reference. `axes` is the (ax1, ax2) pair plot_bode returns."""
    ax1, ax2 = axes
    ax1.axvline(wp_hz, color='grey', linestyle='--', linewidth=0.8)
    ax1.axvline(ws_hz, color='grey', linestyle='--', linewidth=0.8)
    ax1.axhline(-gpass_db, color='grey', linestyle=':', linewidth=0.8)
    ax1.axhline(-gstop_db, color='grey', linestyle=':', linewidth=0.8)
    return ax1.figure, (ax1, ax2)


def plot_monte_carlo(result, wp_hz, ws_hz, gpass_db, gstop_db):
    """Overlay every Monte Carlo trial's magnitude response (thin, semi-
    transparent) with the nominal response highlighted, plus the spec mask."""
    fig, ax = plt.subplots(figsize=(8, 5))
    f = result['f']
    n_shown = 0
    for row in result['mag']:
        if np.all(np.isnan(row)):
            continue
        ax.semilogx(f, row, color='tab:red', alpha=max(0.02, min(0.3, 20.0 / result['n_trials'])),
                    linewidth=0.8)
        n_shown += 1
    ax.semilogx(f, result['nominal_mag'], color='black', linewidth=1.8, label='Nominal', zorder=5)
    ax.axvline(wp_hz, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(ws_hz, color='grey', linestyle='--', linewidth=0.8)
    ax.axhline(-gpass_db, color='grey', linestyle=':', linewidth=0.8)
    ax.axhline(-gstop_db, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(f"Monte Carlo: {result['n_trials']} trials "
                 f"({result['n_unstable']} unstable), yield = {result['yield_frac'] * 100:.1f}%")
    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    return fig, ax


def show_and_save(fig, path, show=False):
    """Always save the figure to disk; only open an interactive window if
    `show` is True (and only if a display is actually available)."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        try:
            plt.show()
        except Exception:
            pass  # no display available (e.g. headless/SSH session)
