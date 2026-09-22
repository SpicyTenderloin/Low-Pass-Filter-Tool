import numpy as np
from scipy.signal import tf2zpk


def sk_w0(R1, R2, C1, C2):
    return 1.0 / np.sqrt(R1 * R2 * C1 * C2)

def sk_a1(R1, R2, C1, C2, K):
    return 1.0 / (R1 * C1) + 1.0 / (R2 * C1) + (1.0 - K) / (R1 * C2)

def sk_Q(R1, R2, C1, C2, K):
    w0 = sk_w0(R1, R2, C1, C2)
    a1 = sk_a1(R1, R2, C1, C2, K)
    return w0 / a1 if a1 > 1e-9 else np.inf

def sk_biquad_tf(R1, R2, C1, C2, K):
    """Transfer function, w0 and Q of a Sallen-Key low-pass biquad stage."""
    w0 = sk_w0(R1, R2, C1, C2)
    a1 = sk_a1(R1, R2, C1, C2, K)
    den = np.array([1.0, a1, w0 ** 2])
    num = np.array([0.0, 0.0, K * w0 ** 2])
    return num, den, w0, sk_Q(R1, R2, C1, C2, K)

def sk_first_order_tf(R, C):
    """Transfer function of a passive first-order RC low-pass stage."""
    w0 = 1.0 / (R * C)
    num = np.array([0.0, w0])
    den = np.array([1.0, w0])
    return num, den, w0

def dc_gain(b, a):
    return b[-1] / a[-1] if a[-1] != 0 else np.inf

def build_lpf_sections_from_poles(b, a):
    """Split a low-pass prototype's poles into cascadable biquad/first-order
    sections, each carrying the ideal (unrealised) target w0 and Q."""
    z, p, k = tf2zpk(b, a)
    used = np.zeros(len(p), dtype=bool)
    sections = []
    for i, pi in enumerate(p):
        if used[i]: continue
        if np.iscomplex(pi):
            pj = np.conj(pi)
            j = np.where((~used) & np.isclose(p, pj, atol=1e-9))[0]
            if len(j):
                used[i] = used[j[0]] = True
                sigma = -np.real(pi)
                omega_d = np.imag(pi)
                w0 = np.sqrt(sigma**2 + omega_d**2)
                Q = w0 / (2 * sigma)
                den = [1.0, w0 / Q, w0 ** 2]
                num = [0.0, 0.0, w0 ** 2]
                sections.append({'kind': 'biquad', 'w0': w0, 'Q': Q, 'num': num, 'den': den})
        else:
            used[i] = True
            w0 = -np.real(pi)
            sections.append({'kind': 'first', 'w0': w0, 'Q': 0.5, 'num': [w0], 'den': [1.0, w0]})
    return sections

def compute_ideal_cascade_tf(sections):
    """Combine a list of ideal target sections (each carrying w0 and, for
    biquads, Q -- as produced/retuned by build_lpf_sections_from_poles /
    stagger_tuning.retune_poles) into a single transfer function. Used to
    preview a retuned target set before it's realised in real parts."""
    num_all, den_all = np.array([1.0]), np.array([1.0])
    for s in sections:
        if s['kind'] == 'biquad':
            w0, Q = s['w0'], s['Q']
            num, den = np.array([w0 ** 2]), np.array([1.0, w0 / Q, w0 ** 2])
        else:
            w0 = s['w0']
            num, den = np.array([w0]), np.array([1.0, w0])
        num_all = np.polymul(num_all, num)
        den_all = np.polymul(den_all, den)
    return num_all, den_all

def compute_cascade_tf(stages):
    """Combine a list of realised stage dicts (biquad / first / attenuator)
    into a single overall transfer function (num, den)."""
    num_all, den_all = np.array([1.0]), np.array([1.0])
    for s in stages:
        if s['kind'] == 'biquad':
            num, den, _, _ = sk_biquad_tf(s['R1'], s['R2'], s['C1'], s['C2'], s['K_act'])
        elif s['kind'] == 'first':
            num, den, _ = sk_first_order_tf(s['R'], s['C'])
        elif s['kind'] == 'attenuator':
            num, den = np.array([s['ratio_act']]), np.array([1.0])
        else:
            raise ValueError(f"Unknown stage kind: {s['kind']}")
        num_all = np.polymul(num_all, num)
        den_all = np.polymul(den_all, den)
    return num_all, den_all
