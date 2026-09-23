# Low-Pass Filter Tool

An interactive design tool for analog Sallen-Key low-pass filters, built entirely from
standard E12/E24 resistor and capacitor values.

Given a passband/stopband spec, it:

- builds a Chebyshev Type I or Butterworth analog prototype, at the automatic minimum
  order or a manually forced higher one
- splits it into cascadable 2nd-order (biquad) Sallen-Key stages, plus one 1st-order
  RC stage for odd filter orders
- optionally retunes the pole positions ("stagger tuning") to ease stages that would
  otherwise need an unrealistically precise Q
- searches E12/E24 parts for the closest realisable R/C/gain-resistor combination
  for each stage
- sizes an output attenuator so the overall cascade hits a target passband gain
- verifies the realised design against the original spec and reports exactly which
  criteria fail, and where in frequency
- runs a Monte Carlo component-tolerance analysis and reports build yield
- draws the actual schematic (every resistor, capacitor, op-amp, and ground connection)
  for the realised cascade

Two front ends share the same engine (`engine.py`) and produce the same JSON schema:
an interactive Streamlit app for exploring a spec live, and a CLI for scripted/repeatable
runs.

## Requirements

Python 3.10+, then:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive app (recommended for exploring a spec)

```bash
streamlit run app.py
```

Opens in your browser. Passband/stopband/ripple/attenuation are fields, build options
(order override, E-series, retuning, target gain) are checkboxes/number inputs, resistor
and capacitor tolerance are sliders, and the response plot, passband-ripple zoom, bill of
materials, circuit diagram, and Monte Carlo plot all recompute automatically as you change anything -- no
need to re-run and re-answer a string of prompts to try a different number. Plots are
interactive (Plotly): drag to zoom into a region, scroll to zoom, double-click to reset,
click a legend entry to hide/show a curve, hover for exact values -- the magnitude axis
defaults to a sensible range around the spec rather than however low the response happens
to reach at the plotted frequency ceiling, but the full range is always a zoom-out away.
A "Save this
design" button writes the same JSON/plot files `main.py` would, into `filter designs/`.

### CLI (scripted / repeatable runs)

```bash
python main.py
```

Interactive prompts walk through filter type, passband/stopband edges, ripple/attenuation
(no default -- must be typed the first time you ever run it, then each defaults to what you
entered last run), and a few build options (order override, target gain, E-series, pole
retuning, plot display). Press Enter on any prompt to accept the default shown in
`[brackets]`.

At the end it asks once whether to save; if you do, it writes to `filter designs/`: a
`.json` (full design + verification result), `_bode.png`, `_passband.png` (zoomed
ripple detail), and `_schematic.png` (the actual circuit) plots, and a shared
`filter_design_report.txt` log.

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

Raising the order *without* retuning does not help -- for a fixed ripple and passband
edge, more poles demands a *steeper* transition, which pushes every pole *closer* to the
imaginary axis (higher Q), not further. It only helps combined with retuning, which uses
the extra poles to spread the required selectivity across more, gentler stages instead of
a few sharp ones -- confirmed against a real spec: order 8 retuned to a peak Q of ~9 (right
at the realisable ceiling); order 12 on the *same* spec retuned to ~2. The order field in
both front ends exists for exactly this: press Enter/leave at the automatic minimum unless
you're also enabling retuning to spend the extra poles productively.

The retune margin can also go *negative*: at margin 0 the ideal retuned target already
sits exactly on the ripple boundary, leaving no room for the E-series realisation error
that's added on top, so a design can retune to a very comfortable Q and still miss spec by
a hundredth of a dB. A small negative margin (e.g. -0.1 dB) tunes tighter than spec on
purpose, reserving exactly that headroom.

The **verification** step after realisation is the tool being honest: it measures the
actual realised response against your spec -- true peak-to-peak passband ripple, and
attenuation relative to the passband's own peak -- and reports the real numbers and where
each failure occurs in frequency, rather than assuming success. A design that doesn't meet
spec with the parts available usually means the spec needs loosening, the retune margin
needs adjusting, or the deviation is one you can live with.

**Monte Carlo is the other half of "does it actually work."** A design can verify cleanly
against its nominal (as-designed) component values and still have poor real-world yield
once ordinary part tolerances are applied -- ripple in particular is often far more
sensitive to tolerance than the nominal check suggests, even when Q and attenuation stay
comfortable. Check it before trusting a tight ripple spec.

## Project layout

| File | Role |
|---|---|
| `app.py` | interactive Streamlit app |
| `main.py` | CLI: orchestrates a full design run |
| `engine.py` | shared pipeline (prototype -> retune -> realise -> verify) used by both front ends |
| `cli.py` | interactive prompts / spec collection for the CLI |
| `parameters.py` | component libraries, ratios, limits, defaults |
| `prototype_filter.py` | analog Chebyshev/Butterworth prototype, incl. order override |
| `sallen_key_tf.py` | Sallen-Key stage transfer functions, pole splitting |
| `sk_realisation.py` | E-series part search for each stage |
| `stagger_tuning.py` | pole retuning optimiser |
| `monte_carlo.py` | component-tolerance Monte Carlo analysis |
| `schematic.py` | circuit diagram: geometry computed once, rendered as both a static (matplotlib) and interactive (Plotly) figure |
| `plotting.py` / `bom.py` / `report.py` | plotting, bill of materials, logging |
| `interactive_plots.py` | Plotly versions of the response/Monte Carlo plots, for the app |
| `load_filter.py` | reload / re-plot a saved design |

## Notes / limitations

- The default R/C libraries in `parameters.py` are sized for audio work (roughly sub-Hz
  up to ~150 kHz ceiling). A very different frequency range may need wider `R_DECADES`/
  `C_DECADES`.
- Op-amps are treated as ideal -- no gain-bandwidth, slew rate, or noise modelling.
- Monte Carlo perturbs every resistor and capacitor, including the gain-setting `Rf`/`Rg`
  pair, and checks each trial for genuine instability (not just an off-spec response).
