"""Interactive collection of a filter design specification.

Most prompts show a default in [brackets]; pressing Enter accepts it. The
core spec values (passband/stopband edges, ripple, attenuation) have no
default -- they must be typed in every run.
"""
from dataclasses import dataclass
from typing import Optional

import parameters as P


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
    print(f"=== {P.FILTER_DESIGNER_NAME}: interactive design spec ===")
    print("(The spec below must be entered each run. Later prompts show a\n"
          " default in [brackets] -- press Enter to accept it.)\n")

    filter_type = _ask_choice("Filter type", ["chebyshev", "butterworth"], P.DEFAULT_FILTER_TYPE)

    wp = _ask_float_required("Passband edge frequency (Hz)", lambda v: v > 0)
    ws = _ask_float_required("Stopband edge frequency (Hz)", lambda v: v > wp,
                              hint="must be above the passband edge")

    if filter_type == "chebyshev":
        gpass = _ask_float_required("Passband ripple (dB)", lambda v: v > 0)
    else:
        gpass = _ask_float_required("Max passband loss at the edge (dB)", lambda v: v > 0)
    gstop = _ask_float_required("Minimum stopband attenuation (dB)",
                                 lambda v: v > gpass, hint="must exceed the passband figure")

    print("\n--- A few other parameters that affect the build ---")
    target_gain = _ask_float(
        "Target overall passband gain (dB) -- each Sallen-Key stage needs\n"
        "  gain K>1 to set its Q, so the cascade naturally has excess gain;\n"
        "  a compensation attenuator stage will be sized to hit this",
        P.DEFAULT_TARGET_GAIN_DB)

    e_series = int(_ask_choice("Component E-series (24 gives finer value steps, "
                                "easier to hit high Q)", ["12", "24"], str(P.DEFAULT_E_SERIES)))

    retune = _ask_yesno(
        "Allow automatic pole retuning ('stagger tuning') to ease stages\n"
        "  that need an unrealistically precise Q? This nudges pole\n"
        "  positions within a small spec margin -- see the explanation\n"
        "  printed after the run",
        P.DEFAULT_RETUNE_ENABLED)
    retune_margin = 0.0
    if retune:
        retune_margin = _ask_float(
            "  Spec margin to allow while retuning, added to ripple and\n"
            "  subtracted from stopband attenuation (dB)",
            P.DEFAULT_RETUNE_MARGIN_DB, lambda v: v >= 0)

    show_plots = _ask_yesno("Show plots on screen (they are always saved to disk too)?",
                             P.DEFAULT_SHOW_PLOTS)

    out_name = input("Base filename for saved outputs [auto timestamp]: ").strip() or None

    return DesignSpec(
        name=P.FILTER_DESIGNER_NAME, filter_type=filter_type,
        wp_hz=wp, ws_hz=ws, gpass_db=gpass, gstop_db=gstop,
        target_gain_db=target_gain, e_series=e_series,
        retune_enabled=retune, retune_margin_db=retune_margin,
        show_plots=show_plots, out_name=out_name,
    )


def ask_run_monte_carlo(default=False) -> bool:
    return _ask_yesno("\nRun a Monte Carlo component-tolerance analysis on this design now?", default)


def ask_save_design(default=True) -> bool:
    return _ask_yesno('\nSave this design (JSON, plots, and report) to "filter designs/"?', default)


def get_monte_carlo_params_interactive() -> MonteCarloParams:
    print("\n=== Monte Carlo component tolerance analysis ===")
    r_tol = _ask_float("Resistor tolerance (+/-%, e.g. 1 for metal-film, 5 for carbon-film)",
                        P.DEFAULT_R_TOL_PCT, lambda v: v > 0)
    c_tol = _ask_float("Capacitor tolerance (+/-%, e.g. 5 for film caps, 10-20 for others)",
                        P.DEFAULT_C_TOL_PCT, lambda v: v > 0)
    n_trials = int(_ask_float("Number of Monte Carlo trials", P.DEFAULT_MC_TRIALS,
                               lambda v: v >= 10, hint="need at least 10"))
    distribution = _ask_choice(
        "Tolerance distribution (uniform = manufacturer's guarantee that\n"
        "  every part falls somewhere in the tolerance band; gaussian = tolerance\n"
        "  treated as a 3-sigma bound, most parts land near nominal)",
        ["uniform", "gaussian"], P.DEFAULT_MC_DISTRIBUTION)
    return MonteCarloParams(r_tol_pct=r_tol, c_tol_pct=c_tol, n_trials=n_trials,
                             distribution=distribution)
