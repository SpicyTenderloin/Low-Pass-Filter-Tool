# Low-Pass Filter Tool

An interactive design tool for analog Sallen-Key low-pass filters, built entirely from
standard E12/E24 resistor and capacitor values.

Given a passband/stopband spec, it:

- builds a Chebyshev Type I or Butterworth analog prototype
- splits it into cascadable 2nd-order (biquad) Sallen-Key stages, plus one 1st-order
  RC stage for odd filter orders
- optionally retunes the pole positions ("stagger tuning") to ease stages that would
  otherwise need an unrealistically precise Q
- searches E12/E24 parts for the closest realisable R/C/gain-resistor combination
  for each stage
- sizes an output attenuator so the overall cascade hits a target passband gain
- verifies the realised design against the original spec and reports a clear pass/fail
- runs a Monte Carlo component-tolerance analysis and reports build yield

## Requirements

Python 3.10+, then:

```bash
pip install -r requirements.txt
```

## Usage

### Design a filter

```bash
python main.py
```

Interactive prompts walk through filter type, passband/stopband edges, ripple/attenuation,
and a few build options (target gain, E-series, pole retuning, plot display). Press Enter
on any prompt to accept the default shown in `[brackets]`.

Every run writes to `filter designs/`: a `.json` (full design + verification result), a
`_bode.png` plot, and a shared `filter_design_report.txt` log.

### Revisit a saved design

```bash
python load_filter.py
```

Lists saved designs, reprints the bill of materials, and re-plots the response.

### Monte Carlo tolerance analysis

```bash
python monte_carlo.py
```

Pick a saved design, enter resistor/capacitor tolerances and a trial count, and get an
overlaid response plot plus a yield estimate (the fraction of trials meeting spec) and
an instability count. It can also be run immediately after a design in `main.py`, without
saving and reloading first.

## How it works

A Sallen-Key stage's op-amp gain `K` sets both its Q *and* its passband gain -- that's
inherent to the topology, not a limitation of this tool. Two consequences follow directly:

- the cascade always ends up with excess gain (each stage multiplies by its own `K`),
  which is why an output attenuator is sized automatically to hit your target gain;
- Q becomes very sensitive to `K` as Q gets high, which puts a practical ceiling on how
  high a single stage's Q can go before standard-value parts (and a safety margin against
  outright instability) can't hit it accurately. This is usually the real bottleneck in a
  Sallen-Key build, not part-value resolution.

**Pole retuning ("stagger tuning")** exists to work around that ceiling. The name is
borrowed from stagger-tuned IF amplifiers, where several resonant stages are deliberately
tuned to slightly different frequencies so their cascade is broader and flatter than any
one stage alone. Here it's repurposed as a constrained optimisation: every stage's target
(f0, Q) is nudged together to minimise the worst-case Q, while keeping the cascaded
response inside the passband ripple / stopband attenuation mask (optionally relaxed by a
margin you choose). Lower peak Q is easier to hit exactly with discrete parts.

The **verification** step after realisation is the tool being honest: it measures the
actual realised response against your spec and reports the real numbers, rather than
assuming success. A design that doesn't meet spec with the parts available usually means
the spec needs loosening, the retune margin needs widening, or the deviation is one you
can live with.

## Project layout

| File | Role |
|---|---|
| `main.py` | orchestrates a full design run |
| `cli.py` | interactive prompts / spec collection |
| `parameters.py` | component libraries, ratios, limits, defaults |
| `prototype_filter.py` | analog Chebyshev/Butterworth prototype |
| `sallen_key_tf.py` | Sallen-Key stage transfer functions, pole splitting |
| `sk_realisation.py` | E-series part search for each stage |
| `stagger_tuning.py` | pole retuning optimiser |
| `monte_carlo.py` | component-tolerance Monte Carlo analysis |
| `plotting.py` / `bom.py` / `report.py` | plotting, bill of materials, logging |
| `load_filter.py` | reload / re-plot a saved design |

## Notes / limitations

- The default R/C libraries in `parameters.py` are sized for audio work (roughly sub-Hz
  up to ~150 kHz ceiling). A very different frequency range may need wider `R_DECADES`/
  `C_DECADES`.
- Op-amps are treated as ideal -- no gain-bandwidth, slew rate, or noise modelling.
- Monte Carlo perturbs every resistor and capacitor, including the gain-setting `Rf`/`Rg`
  pair, and checks each trial for genuine instability (not just an off-spec response).
