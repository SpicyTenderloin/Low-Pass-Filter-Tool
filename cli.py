"""Interactive collection of a filter design specification.

Most prompts show a default in [brackets]; pressing Enter accepts it. The
core spec values (passband/stopband edges, ripple, attenuation) have no
hardcoded default -- on the very first run ever they must be typed, but
every run after that defaults to whatever was typed last time (persisted
in STATE_PATH), so tweaking one field at a time just means overriding
that one prompt and pressing Enter through the rest.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import parameters as P

STATE_PATH = Path(".last_run.json")


def _load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state_section(section, data):
    state = _load_state()
    state[section] = data
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except OSError:
        pass  # non-fatal -- next run just won't have these as defaults


@dataclass
class MonteCarloParams:
    r_tol_pct: float
    c_tol_pct: float
    n_trials: int
    distribution: str = "uniform"   # "uniform" or "gaussian" (tolerance treated as 3-sigma)
    seed: int = 0


@dataclass
class DesignSpec:
    name: str
    filter_type: str            # 'chebyshev' | 'butterworth'
    wp_hz: float                 # passband edge frequency
    ws_hz: float                  # stopband edge frequency
    gpass_db: float                # passband ripple / max passband loss
    gstop_db: float                 # minimum stopband attenuation
    target_gain_db: float = 0.0     # desired overall passband gain
    e_series: int = 24
    retune_enabled: bool = True
    retune_margin_db: float = 0.5
    show_plots: bool = False
    out_name: Optional[str] = None


def _ask_float(prompt, default, validate=None, hint=None):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            value = float(default)
        else:
            try:
                value = float(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
        if validate and not validate(value):
            print(f"  Out of range{': ' + hint if hint else ''}, try again.")
            continue
        return value


def _ask_float_required(prompt, validate=None, hint=None):
    while True:
        raw = input(f"{prompt}: ").strip()
        if raw == "":
            print("  This value is required, please enter a number.")
            continue
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if validate and not validate(value):
            print(f"  Out of range{': ' + hint if hint else ''}, try again.")
            continue
        return value


def _ask_float_maybe_default(prompt, default, validate=None, hint=None):
    """Like _ask_float, but if `default` is None there is nothing to default
    to yet (no prior run) -- blank input re-prompts instead of silently
    accepting a value."""
    if default is None:
        return _ask_float_required(prompt, validate=validate, hint=hint)
    return _ask_float(prompt, default, validate=validate, hint=hint)


def _ask_choice(prompt, choices, default):
    choices_l = [c.lower() for c in choices]
    while True:
        raw = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip().lower()
        if raw == "":
            return default
        if raw in choices_l:
            return choices[choices_l.index(raw)]
        print(f"  Choose one of: {', '.join(choices)}")


def _ask_yesno(prompt, default):
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{d}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def get_spec_interactive() -> DesignSpec:
    last = _load_state().get('spec', {})

    print(f"=== {P.FILTER_DESIGNER_NAME}: interactive design spec ===")
    if last:
        print("(Defaults in [brackets] are what you entered last run --\n"
              " press Enter to reuse a value, or type a new one.)\n")
    else:
        print("(First run: the core spec has no default yet and must be\n"
              " typed in full. From the next run on, each field will\n"
              " default to what you entered this time.)\n")

    filter_type = _ask_choice("Filter type", ["chebyshev", "butterworth"],
                               last.get('filter_type', P.DEFAULT_FILTER_TYPE))

    wp = _ask_float_maybe_default("Passband edge frequency (Hz)", last.get('wp_hz'), lambda v: v > 0)
    ws = _ask_float_maybe_default("Stopband edge frequency (Hz)", last.get('ws_hz'), lambda v: v > wp,
                                   hint="must be above the passband edge")

    if filter_type == "chebyshev":
        gpass = _ask_float_maybe_default("Passband ripple (dB)", last.get('gpass_db'), lambda v: v > 0)
    else:
        gpass = _ask_float_maybe_default("Max passband loss at the edge (dB)", last.get('gpass_db'),
                                          lambda v: v > 0)
    gstop = _ask_float_maybe_default("Minimum stopband attenuation (dB)", last.get('gstop_db'),
                                      lambda v: v > gpass, hint="must exceed the passband figure")

    print("\n--- A few other parameters that affect the build ---")
    target_gain = _ask_float(
        "Target overall passband gain (dB) -- each Sallen-Key stage needs\n"
        "  gain K>1 to set its Q, so the cascade naturally has excess gain;\n"
        "  a compensation attenuator stage will be sized to hit this",
        last.get('target_gain_db', P.DEFAULT_TARGET_GAIN_DB))

    e_series = int(_ask_choice("Component E-series (24 gives finer value steps, "
                                "easier to hit high Q)", ["12", "24"],
                                str(last.get('e_series', P.DEFAULT_E_SERIES))))

    retune = _ask_yesno(
        "Allow automatic pole retuning ('stagger tuning') to ease stages\n"
        "  that need an unrealistically precise Q? This nudges pole\n"
        "  positions within a small spec margin -- see the explanation\n"
        "  printed after the run",
        last.get('retune_enabled', P.DEFAULT_RETUNE_ENABLED))
    retune_margin = last.get('retune_margin_db', P.DEFAULT_RETUNE_MARGIN_DB) if not retune else 0.0
    if retune:
        retune_margin = _ask_float(
            "  Spec margin to allow while retuning, added to ripple and\n"
            "  subtracted from stopband attenuation (dB)",
            last.get('retune_margin_db', P.DEFAULT_RETUNE_MARGIN_DB), lambda v: v >= 0)

    show_plots = _ask_yesno("Show plots on screen (they are always saved to disk too)?",
                             last.get('show_plots', P.DEFAULT_SHOW_PLOTS))

    out_name = input("Base filename for saved outputs [auto timestamp]: ").strip() or None

    spec = DesignSpec(
        name=P.FILTER_DESIGNER_NAME, filter_type=filter_type,
        wp_hz=wp, ws_hz=ws, gpass_db=gpass, gstop_db=gstop,
        target_gain_db=target_gain, e_series=e_series,
        retune_enabled=retune, retune_margin_db=retune_margin,
        show_plots=show_plots, out_name=out_name,
    )

    _save_state_section('spec', {
        'filter_type': spec.filter_type, 'wp_hz': spec.wp_hz, 'ws_hz': spec.ws_hz,
        'gpass_db': spec.gpass_db, 'gstop_db': spec.gstop_db,
        'target_gain_db': spec.target_gain_db, 'e_series': spec.e_series,
        'retune_enabled': spec.retune_enabled,
        # Only overwrite the remembered margin when retuning was actually
        # exercised this run -- otherwise a one-off "no" would blank out a
        # previously meaningful margin back to the hardcoded default.
        'retune_margin_db': spec.retune_margin_db if spec.retune_enabled
                             else last.get('retune_margin_db', P.DEFAULT_RETUNE_MARGIN_DB),
        'show_plots': spec.show_plots,
    })
    return spec


def ask_run_monte_carlo(default=False) -> bool:
    return _ask_yesno("\nRun a Monte Carlo component-tolerance analysis on this design now?", default)


def ask_save_design(default=True) -> bool:
    return _ask_yesno('\nSave this design (JSON, plots, and report) to "filter designs/"?', default)


def get_monte_carlo_params_interactive() -> MonteCarloParams:
    last = _load_state().get('monte_carlo', {})

    print("\n=== Monte Carlo component tolerance analysis ===")
    r_tol = _ask_float("Resistor tolerance (+/-%, e.g. 1 for metal-film, 5 for carbon-film)",
                        last.get('r_tol_pct', P.DEFAULT_R_TOL_PCT), lambda v: v > 0)
    c_tol = _ask_float("Capacitor tolerance (+/-%, e.g. 5 for film caps, 10-20 for others)",
                        last.get('c_tol_pct', P.DEFAULT_C_TOL_PCT), lambda v: v > 0)
    n_trials = int(_ask_float("Number of Monte Carlo trials",
                               last.get('n_trials', P.DEFAULT_MC_TRIALS),
                               lambda v: v >= 10, hint="need at least 10"))
    distribution = _ask_choice(
        "Tolerance distribution (uniform = manufacturer's guarantee that\n"
        "  every part falls somewhere in the tolerance band; gaussian = tolerance\n"
        "  treated as a 3-sigma bound, most parts land near nominal)",
        ["uniform", "gaussian"], last.get('distribution', P.DEFAULT_MC_DISTRIBUTION))

    mc = MonteCarloParams(r_tol_pct=r_tol, c_tol_pct=c_tol, n_trials=n_trials,
                           distribution=distribution)
    _save_state_section('monte_carlo', {
        'r_tol_pct': mc.r_tol_pct, 'c_tol_pct': mc.c_tol_pct,
        'n_trials': mc.n_trials, 'distribution': mc.distribution,
    })
    return mc
