import numpy as np

# Global identity
FILTER_DESIGNER_NAME = "Filter Designer"

# --- Suggested defaults shown in the interactive prompts (cli.py) ---
# These are NOT the spec any more -- the spec is collected interactively
# every run. They only pre-fill the prompts so testing/re-running is fast.
DEFAULT_FILTER_TYPE   = "chebyshev"   # "chebyshev" or "butterworth"
DEFAULT_WP_HZ          = 4300.0
DEFAULT_WS_HZ          = 6000.0
DEFAULT_GPASS_DB       = 2.0
DEFAULT_GSTOP_DB       = 50.0
DEFAULT_TARGET_GAIN_DB = 0.0
DEFAULT_E_SERIES       = 24
DEFAULT_RETUNE_ENABLED = True
DEFAULT_RETUNE_MARGIN_DB = 0.5
DEFAULT_SHOW_PLOTS     = False

DEFAULT_R_TOL_PCT        = 1.0
DEFAULT_C_TOL_PCT        = 5.0
DEFAULT_MC_TRIALS        = 500
DEFAULT_MC_DISTRIBUTION  = "uniform"   # "uniform" or "gaussian"

# E-series and libraries. Sized for audio work: R stays in the 1k-910k
# range op-amps drive comfortably; C is extended down to 1nF so the
# smallest achievable R*C reaches ~159 kHz (1k ohm x 1nF), well above the
# 20kHz audio edge -- the old C floor of 10nF only reached ~15.9kHz even
# with the smallest resistor, which is what broke a 20kHz+ design.
R_DECADES = [3, 4, 5]         # 1k..~910k ohms
C_DECADES = [-9, -8, -7, -6]  # 1nF..~9.1uF

def e_series(E):
    if E == 12:
        return np.array([1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2])
    elif E == 24:
        return np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                         3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])
    else:
        raise ValueError("Invalid E-series (use 12 or 24)")

def make_values(bases, decades):
    return np.sort([v * 10**d for d in decades for v in bases])

def build_libraries(e_series_n):
    """Build the R and C part-value libraries for a chosen E-series."""
    bases = e_series(e_series_n)
    r_lib = make_values(bases, R_DECADES)
    c_lib = make_values(bases, C_DECADES)
    return r_lib, c_lib

# Default libraries (E24) for scripts that don't collect a spec interactively
# (e.g. load_filter.py).
R_LIB, C_LIB = build_libraries(DEFAULT_E_SERIES)

# Ratios searched when realising each Sallen-Key biquad stage
STATIC_M_RATIOS = [0.33, 0.39, 0.47, 0.56, 0.68, 0.82, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3]
STATIC_N_RATIOS = STATIC_M_RATIOS.copy()

# Limits
K_MIN, K_MAX = 1.0, 3.0
K_MARGIN_STAB = 0.05   # keep K this fraction below the SK stability limit

# Cost weights for picking the best part combination (log-relative errors)
W_FC = 2.0
W_Q = 1.0

# Plotting
F_MIN = 1.0
F_MAX = 6000.0

COLORS = {
    'ideal': 'tab:blue',
    'retuned': 'tab:orange',
    'realised': 'tab:green',
}
