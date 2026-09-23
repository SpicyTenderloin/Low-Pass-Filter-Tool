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


def plot_passband_detail(b, a, wp_hz, label="TF", color="tab:blue", ax=None, f_min=None, n=2000,
                          normalize=True):
    """Zoomed, auto-scaled magnitude-only view of just the passband.

    A full Bode plot's y-axis usually spans 100+ dB to fit the stopband
    rolloff, which hides a sub-dB or few-dB ripple failure completely --
    the trace just looks like a flat line. This plots magnitude only, over
    [f_min, wp_hz], and lets matplotlib autoscale the y-axis to the data so
    the actual ripple shape is visible.

    Each curve is normalized to its own peak (0 dB = that curve's highest
    point in the passband) by default. This plot exists purely to judge
    ripple, and curves shown together here -- ideal / retuned target /
    realised -- have no reason to share an absolute gain reference: pole
    retuning and the output gain-compensation stage both deliberately shift
    it (retuning doesn't preserve the ideal prototype's equiripple shape,
    it trades it away for lower Q). Normalizing means a single -gpass_db
    line is the correct ripple floor for every curve shown, instead of
    being correct for only whichever one curve it was drawn relative to --
    an absolute-dB version of this plot made the ideal curve look like it
    was constantly failing spec when it was behaving perfectly, just
    measured from a different peak. Absolute level is still shown in the
    full Bode plot, where it's meaningful (e.g. confirming gain compensation
    landed near the target).
    """
    f_min = f_min or max(0.1, wp_hz / 20000)
    f = np.linspace(f_min, wp_hz, n)
    w = 2 * np.pi * f
    _, H = freqs(b, a, w)
    mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    if normalize:
        mag = mag - mag.max()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude rel. to own peak [dB]" if normalize else "Magnitude [dB]")
        ax.grid(True, which='both', linestyle=':')
    else:
        fig = ax.figure

    ax.semilogx(f, mag, label=label, color=color)
    ax.legend()
    return fig, ax


def mark_passband_spec(ax, gpass_db):
    """Draw the ripple budget on a passband-detail plot. Assumes curves were
    plotted with plot_passband_detail's default normalize=True, so a single
    -gpass_db line is the correct floor for every curve on the axes."""
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.6)
    ax.axhline(-gpass_db, color='red', linestyle=':', linewidth=1.2,
               label=f'-{gpass_db:.2f} dB spec')
    ax.legend()
    return ax.figure, ax


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


_any_shown = False


def show_fig(fig, show=False):
    """Display a figure without blocking script execution, if `show` is
    True and a display is available (a no-op otherwise, e.g. headless/SSH).
    plt.show() with no arguments blocks until the window is closed, which
    is why several plots in a row would each stall the script -- this uses
    block=False plus a short pause to force the window to actually render
    without waiting. Call block_until_closed() once, at the very end of the
    script, to keep every window shown this way open until the user closes
    them -- otherwise they vanish the instant the process exits."""
    global _any_shown
    if not show:
        return
    try:
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        _any_shown = True
    except Exception:
        pass


def save_fig(fig, path):
    """Save a figure to disk."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def block_until_closed():
    """Block until every window opened via show_fig() (in this process) is
    closed. No-op if nothing was shown. Call once, right at the end of a
    script, after every plot for the run has already been created."""
    global _any_shown
    if _any_shown:
        try:
            plt.show()
        except Exception:
            pass
        _any_shown = False


def show_and_save(fig, path, show=False):
    """Save a figure to disk, and optionally display it (non-blocking --
    call block_until_closed() at the end of the script to keep it open)."""
    save_fig(fig, path)
    show_fig(fig, show=show)
