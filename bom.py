from report import log
import numpy as np

def eng_unit(value, unit):
    prefixes = {
        -9: "n", -6: "u", -3: "m",
         0: "", 3: "k", 6: "M", 9: "G"
    }
    if value == 0: return f"0{unit}"
    exp3 = int(np.floor(np.log10(abs(value)) / 3) * 3)
    prefix = prefixes.get(exp3, f"e{exp3}")
    scaled = value / (10 ** exp3)
    return f"{scaled:.3g}{prefix}{unit}"

def print_bom(stages):
    log("\n===== Bill of Materials (BOM) =====")
    for i, d in enumerate(stages, 1):
        if d["kind"] == "biquad":
            log(f"\nStage {i:02d} - Biquad  (target f0={d['f0_tgt']:.1f} Hz Q={d['Q_tgt']:.3f} "
                f"-> actual f0={d['f0_act']:.1f} Hz Q={d['Q_act']:.3f})")
            log(f"  R1 = {eng_unit(d['R1'], 'ohm')}   R2 = {eng_unit(d['R2'], 'ohm')}")
            log(f"  C1 = {eng_unit(d['C1'], 'F')}   C2 = {eng_unit(d['C2'], 'F')}")
            log(f"  Rf = {eng_unit(d['Rf'], 'ohm')}   Rg = {eng_unit(d['Rg'], 'ohm')}  (K = {d['K_act']:.3f})")
            if d['err_fc'] > 0.02 or d['err_Q'] > 0.15:
                log(f"  [!] Best available parts miss the target "
                    f"(f0 err {d['err_fc']*100:.1f}% in log-decades, Q ratio err {d['err_Q']:.2f} in log-decades)")
        elif d["kind"] == "first":
            log(f"\nStage {i:02d} - First-order  (target f0={d['f0_tgt']:.1f} Hz "
                f"-> actual f0={d['f0_act']:.1f} Hz)")
            log(f"  R = {eng_unit(d['R'], 'ohm')}   C = {eng_unit(d['C'], 'F')}")
        elif d["kind"] == "attenuator":
            log(f"\nStage {i:02d} - Output attenuator (gain compensation, "
                f"target {d['atten_db_tgt']:.2f} dB -> actual {d['atten_db_act']:.2f} dB)")
            log(f"  Ra = {eng_unit(d['Ra'], 'ohm')}   Rb = {eng_unit(d['Rb'], 'ohm')}  "
                f"(Ra in series from the last stage's output to the output node; "
                f"Rb from the output node to ground)")
        else:
            log(f"\nStage {i:02d} - unknown kind '{d['kind']}'")
