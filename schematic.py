"""Circuit schematic generation for a realised Sallen-Key cascade.

The topology (where every wire, component, and op-amp pin goes) is
computed once as a small list of library-agnostic drawing primitives --
Wire, Resistor, Capacitor, OpAmp, Ground, Node, Text -- with plain (x, y)
coordinates. Two renderers turn that same primitive list into a picture:
render_matplotlib() for the CLI's saved PNGs, render_plotly() for the
app's interactive on-screen view (zoom/pan on a wide multi-stage cascade).
Keeping the geometry in one place means both pictures are always the same
circuit, not two hand-maintained drawings that can drift apart.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from bom import eng_unit


# ------------------------------------------------------------- primitives
@dataclass
class Wire:
    x0: float; y0: float; x1: float; y1: float

@dataclass
class Resistor:
    x0: float; y0: float; x1: float; y1: float
    label: str

@dataclass
class Capacitor:
    x0: float; y0: float; x1: float; y1: float
    label: str

@dataclass
class OpAmp:
    x: float; y: float  # base-centre; triangle points in +x direction
    size: float = 0.8

@dataclass
class Ground:
    x: float; y: float

@dataclass
class Node:
    x: float; y: float

@dataclass
class Text:
    x: float; y: float
    text: str
    ha: str = "center"
    va: str = "center"
    size: float = 9
    weight: str = "normal"

Primitive = object  # Wire | Resistor | Capacitor | OpAmp | Ground | Node | Text


# ------------------------------------------------------------ stage layout
# Each stage is built in its own local coordinate frame (origin at its
# input node, y=0 as the main signal rail) and then shifted by x_off when
# assembled into the full cascade.

_STAGE_H = 3.4   # height reserved per stage (signal rail sits at local y=0,
                  # ground rail near the bottom, feedback loop near the top)
_Y_SIGNAL = 0.0
_Y_GND = -1.6
_Y_TOP = 1.3


def _biquad_primitives(s, x_off, stage_no) -> Tuple[List[Primitive], float]:
    """Standard non-inverting Sallen-Key low-pass: R1-R2 from input to the
    op-amp's + input, C2 from that node to ground, C1 feeding back from
    the R1/R2 junction to the output, and an Rf/Rg divider from the output
    setting the gain that also drives the op-amp's - input."""
    y = _Y_SIGNAL
    xA = x_off + 1.1   # R1/R2/C1 junction
    xB = x_off + 2.2   # op-amp + input
    x_op = x_off + 2.6
    op_size = 0.8
    x_out = x_op + op_size  # op-amp output / stage output node

    pin_off = op_size * 0.275  # +/- input pin height above/below the op-amp's centreline

    p: List[Primitive] = []
    if stage_no == 1:
        p.append(Text(x_off, y + 0.35, "Vin", ha="left", size=8))
    p.append(Resistor(x_off, y, xA, y, f"R1={eng_unit(s['R1'], 'ohm')}"))
    p.append(Resistor(xA, y, xB, y, f"R2={eng_unit(s['R2'], 'ohm')}"))
    p.append(Wire(xB, y, x_op, y + pin_off))  # into the + pin, at its actual height
    p.append(Node(xA, y))
    p.append(Node(xB, y))

    p.append(OpAmp(x_op, y, size=op_size))
    p.append(Text(x_op + 0.14, y + pin_off + 0.05, "+", size=9, ha="left", va="bottom"))
    p.append(Text(x_op + 0.14, y - pin_off - 0.05, "-", size=9, ha="left", va="top"))
    p.append(Node(x_out, y))

    # C2: B down to ground
    p.append(Capacitor(xB, y, xB, _Y_GND + 0.4, f"C2={eng_unit(s['C2'], 'F')}"))
    p.append(Wire(xB, _Y_GND + 0.4, xB, _Y_GND))
    p.append(Ground(xB, _Y_GND))

    # C1: feedback from A, over the top, down into the output node
    p.append(Wire(xA, y, xA, _Y_TOP))
    p.append(Capacitor(xA, _Y_TOP, (xA + x_out) / 2 - 0.3, _Y_TOP, f"C1={eng_unit(s['C1'], 'F')}"))
    p.append(Wire((xA + x_out) / 2 - 0.3, _Y_TOP, x_out, _Y_TOP))
    p.append(Wire(x_out, _Y_TOP, x_out, y))

    # Rf/Rg gain divider below the op-amp, tapped into the - input
    x_div = x_out
    y_tap = y - 0.9
    p.append(Wire(x_out, y, x_div, y - 0.25))
    p.append(Resistor(x_div, y - 0.25, x_div, y_tap, f"Rf={eng_unit(s['Rf'], 'ohm')}"))
    p.append(Node(x_div, y_tap))
    p.append(Resistor(x_div, y_tap, x_div, _Y_GND, f"Rg={eng_unit(s['Rg'], 'ohm')}"))
    p.append(Ground(x_div, _Y_GND))
    minus_x = x_op
    p.append(Wire(x_div, y_tap, minus_x, y_tap))
    p.append(Wire(minus_x, y_tap, minus_x, y - pin_off))

    p.append(Text((x_off + x_out) / 2, _Y_TOP + 0.55,
                   f"Stage {stage_no}: biquad  f0={s['f0_act']:.0f} Hz  Q={s['Q_act']:.2f}  K={s['K_act']:.2f}",
                   size=8, weight="bold"))
    return p, x_out


def _first_order_primitives(s, x_off, stage_no) -> Tuple[List[Primitive], float]:
    y = _Y_SIGNAL
    x_node = x_off + 1.4
    x_out = x_node + 0.5

    p: List[Primitive] = [
        Resistor(x_off, y, x_node, y, f"R={eng_unit(s['R'], 'ohm')}"),
        Wire(x_node, y, x_out, y),
        Node(x_node, y),
        Capacitor(x_node, y, x_node, _Y_GND + 0.4, f"C={eng_unit(s['C'], 'F')}"),
        Wire(x_node, _Y_GND + 0.4, x_node, _Y_GND),
        Ground(x_node, _Y_GND),
        Text((x_off + x_out) / 2, _Y_TOP,
             f"Stage {stage_no}: first-order  f0={s['f0_act']:.0f} Hz", size=8, weight="bold"),
    ]
    return p, x_out


def _attenuator_primitives(s, x_off, stage_no) -> Tuple[List[Primitive], float]:
    y = _Y_SIGNAL
    x_node = x_off + 1.4
    x_out = x_node + 0.5

    p: List[Primitive] = [
        Resistor(x_off, y, x_node, y, f"Ra={eng_unit(s['Ra'], 'ohm')}"),
        Wire(x_node, y, x_out, y),
        Node(x_node, y),
        Resistor(x_node, y, x_node, _Y_GND + 0.4, f"Rb={eng_unit(s['Rb'], 'ohm')}"),
        Wire(x_node, _Y_GND + 0.4, x_node, _Y_GND),
        Ground(x_node, _Y_GND),
        Text((x_off + x_out) / 2, _Y_TOP,
             f"Stage {stage_no}: output pad  {s['atten_db_act']:.1f} dB", size=8, weight="bold"),
    ]
    return p, x_out


_STAGE_WIDTH = {'biquad': 4.4, 'first': 2.1, 'attenuator': 2.1}
_BUILDERS = {'biquad': _biquad_primitives, 'first': _first_order_primitives,
             'attenuator': _attenuator_primitives}


def build_cascade_schematic(realised_stages) -> Tuple[List[Primitive], Tuple[float, float, float, float]]:
    """Lay out every realised stage left to right, wired output-to-input.
    Returns (primitives, (xmin, xmax, ymin, ymax)) for sizing a canvas."""
    primitives: List[Primitive] = []
    x = 0.0
    for i, s in enumerate(realised_stages, 1):
        kind = s['kind']
        build = _BUILDERS.get(kind)
        if build is None:
            continue
        stage_prims, x_out = build(s, x, i)
        primitives.extend(stage_prims)
        x_next = x + _STAGE_WIDTH[kind]
        primitives.append(Wire(x_out, _Y_SIGNAL, x_next, _Y_SIGNAL))
        x = x_next

    primitives.append(Text(x + 0.3, _Y_SIGNAL, "Vout", ha="left", size=9, weight="bold"))
    xmax = x + 0.9
    return primitives, (-0.3, xmax, _Y_GND - 0.6, _Y_TOP + 0.9)


# ------------------------------------------------------------- renderers
def render_matplotlib(primitives, bounds, figsize=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    xmin, xmax, ymin, ymax = bounds
    if figsize is None:
        figsize = (max(6.0, (xmax - xmin) * 1.15), (ymax - ymin) * 1.15 + 0.5)
    fig, ax = plt.subplots(figsize=figsize)

    r_hw = 0.09   # resistor box half-width (perpendicular to its axis)
    r_pad = 0.18  # gap left/right of the resistor box along its axis
    c_gap = 0.06  # capacitor plate half-gap
    c_len = 0.22  # capacitor plate half-length

    for pr in primitives:
        if isinstance(pr, Wire):
            ax.plot([pr.x0, pr.x1], [pr.y0, pr.y1], color="black", linewidth=1.2, zorder=1)

        elif isinstance(pr, Resistor):
            horiz = abs(pr.x1 - pr.x0) >= abs(pr.y1 - pr.y0)
            if horiz:
                x0, x1, y = pr.x0, pr.x1, pr.y0
                ax.plot([x0, x0 + r_pad], [y, y], color="black", linewidth=1.2, zorder=1)
                ax.plot([x1 - r_pad, x1], [y, y], color="black", linewidth=1.2, zorder=1)
                ax.add_patch(Rectangle((x0 + r_pad, y - r_hw), (x1 - r_pad) - (x0 + r_pad), 2 * r_hw,
                                        fill=False, edgecolor="black", linewidth=1.2, zorder=2))
                ax.text((x0 + x1) / 2, y + r_hw + 0.15, pr.label, ha="center", va="bottom", fontsize=7.5)
            else:
                y0, y1, x = pr.y0, pr.y1, pr.x0
                ylo, yhi = min(y0, y1), max(y0, y1)
                ax.plot([x, x], [yhi - r_pad, yhi], color="black", linewidth=1.2, zorder=1)
                ax.plot([x, x], [ylo, ylo + r_pad], color="black", linewidth=1.2, zorder=1)
                ax.add_patch(Rectangle((x - r_hw, ylo + r_pad), 2 * r_hw, (yhi - r_pad) - (ylo + r_pad),
                                        fill=False, edgecolor="black", linewidth=1.2, zorder=2))
                ax.text(x + r_hw + 0.08, (y0 + y1) / 2, pr.label, ha="left", va="center", fontsize=7.5)

        elif isinstance(pr, Capacitor):
            horiz = abs(pr.x1 - pr.x0) >= abs(pr.y1 - pr.y0)
            if horiz:
                xm = (pr.x0 + pr.x1) / 2
                y = pr.y0
                ax.plot([pr.x0, xm - c_gap], [y, y], color="black", linewidth=1.2, zorder=1)
                ax.plot([xm + c_gap, pr.x1], [y, y], color="black", linewidth=1.2, zorder=1)
                ax.plot([xm - c_gap, xm - c_gap], [y - c_len, y + c_len], color="black", linewidth=1.6, zorder=2)
                ax.plot([xm + c_gap, xm + c_gap], [y - c_len, y + c_len], color="black", linewidth=1.6, zorder=2)
                ax.text(xm, y + c_len + 0.15, pr.label, ha="center", va="bottom", fontsize=7.5)
            else:
                ym = (pr.y0 + pr.y1) / 2
                x = pr.x0
                ax.plot([x, x], [pr.y0, ym - c_gap], color="black", linewidth=1.2, zorder=1)
                ax.plot([x, x], [ym + c_gap, pr.y1], color="black", linewidth=1.2, zorder=1)
                ax.plot([x - c_len, x + c_len], [ym - c_gap, ym - c_gap], color="black", linewidth=1.6, zorder=2)
                ax.plot([x - c_len, x + c_len], [ym + c_gap, ym + c_gap], color="black", linewidth=1.6, zorder=2)
                ax.text(x + c_len + 0.15, ym, pr.label, ha="left", va="center", fontsize=7.5)

        elif isinstance(pr, OpAmp):
            tri = Polygon([(pr.x, pr.y + pr.size / 2), (pr.x, pr.y - pr.size / 2),
                           (pr.x + pr.size, pr.y)],
                          closed=True, fill=True, facecolor="white", edgecolor="black",
                          linewidth=1.2, zorder=2)
            ax.add_patch(tri)

        elif isinstance(pr, Ground):
            x, y = pr.x, pr.y
            ax.plot([x, x], [y + 0.25, y], color="black", linewidth=1.2, zorder=1)
            for i, w in enumerate([0.22, 0.14, 0.06]):
                yy = y - i * 0.09
                ax.plot([x - w, x + w], [yy, yy], color="black", linewidth=1.2, zorder=2)

        elif isinstance(pr, Node):
            ax.plot([pr.x], [pr.y], marker="o", color="black", markersize=3, zorder=3)

        elif isinstance(pr, Text):
            ax.text(pr.x, pr.y, pr.text, ha=pr.ha, va=pr.va, fontsize=pr.size,
                    fontweight=pr.weight, zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_plotly(primitives, bounds):
    """Interactive counterpart of render_matplotlib(): same primitives,
    same geometry, drawn with Plotly shapes/annotations so a wide
    multi-stage schematic can be dragged/zoomed/panned instead of only
    viewed at whatever size it was saved at."""
    import plotly.graph_objects as go

    xmin, xmax, ymin, ymax = bounds
    fig = go.Figure()

    r_hw = 0.09
    r_pad = 0.18
    c_gap = 0.06
    c_len = 0.22
    shapes = []
    annotations = []

    def line(x0, y0, x1, y1, width=1.4):
        shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                            line=dict(color="black", width=width)))

    for pr in primitives:
        if isinstance(pr, Wire):
            line(pr.x0, pr.y0, pr.x1, pr.y1)

        elif isinstance(pr, Resistor):
            horiz = abs(pr.x1 - pr.x0) >= abs(pr.y1 - pr.y0)
            if horiz:
                x0, x1, y = pr.x0, pr.x1, pr.y0
                line(x0, y, x0 + r_pad, y)
                line(x1 - r_pad, y, x1, y)
                shapes.append(dict(type="rect", x0=x0 + r_pad, x1=x1 - r_pad, y0=y - r_hw, y1=y + r_hw,
                                    line=dict(color="black", width=1.4), fillcolor="white"))
                annotations.append(dict(x=(x0 + x1) / 2, y=y + r_hw + 0.15, text=pr.label,
                                         showarrow=False, font=dict(size=11, color="black"), xanchor="center", yanchor="bottom"))
            else:
                y0, y1, x = pr.y0, pr.y1, pr.x0
                ylo, yhi = min(y0, y1), max(y0, y1)
                line(x, yhi - r_pad, x, yhi)
                line(x, ylo, x, ylo + r_pad)
                shapes.append(dict(type="rect", x0=x - r_hw, x1=x + r_hw, y0=ylo + r_pad, y1=yhi - r_pad,
                                    line=dict(color="black", width=1.4), fillcolor="white"))
                annotations.append(dict(x=x + r_hw + 0.08, y=(y0 + y1) / 2, text=pr.label,
                                         showarrow=False, font=dict(size=11, color="black"), xanchor="left", yanchor="middle"))

        elif isinstance(pr, Capacitor):
            horiz = abs(pr.x1 - pr.x0) >= abs(pr.y1 - pr.y0)
            if horiz:
                xm, y = (pr.x0 + pr.x1) / 2, pr.y0
                line(pr.x0, y, xm - c_gap, y)
                line(xm + c_gap, y, pr.x1, y)
                line(xm - c_gap, y - c_len, xm - c_gap, y + c_len, width=2.0)
                line(xm + c_gap, y - c_len, xm + c_gap, y + c_len, width=2.0)
                annotations.append(dict(x=xm, y=y + c_len + 0.15, text=pr.label, showarrow=False,
                                         font=dict(size=11, color="black"), xanchor="center", yanchor="bottom"))
            else:
                ym, x = (pr.y0 + pr.y1) / 2, pr.x0
                line(x, pr.y0, x, ym - c_gap)
                line(x, ym + c_gap, x, pr.y1)
                line(x - c_len, ym - c_gap, x + c_len, ym - c_gap, width=2.0)
                line(x - c_len, ym + c_gap, x + c_len, ym + c_gap, width=2.0)
                annotations.append(dict(x=x + c_len + 0.15, y=ym, text=pr.label, showarrow=False,
                                         font=dict(size=11, color="black"), xanchor="left", yanchor="middle"))

        elif isinstance(pr, OpAmp):
            x, y, sz = pr.x, pr.y, pr.size
            path = f"M{x},{y + sz / 2} L{x},{y - sz / 2} L{x + sz},{y} Z"
            shapes.append(dict(type="path", path=path, line=dict(color="black", width=1.4),
                                fillcolor="white"))

        elif isinstance(pr, Ground):
            x, y = pr.x, pr.y
            line(x, y + 0.25, x, y)
            for i, w in enumerate([0.22, 0.14, 0.06]):
                yy = y - i * 0.09
                line(x - w, yy, x + w, yy)

        elif isinstance(pr, Node):
            shapes.append(dict(type="circle", x0=pr.x - 0.035, x1=pr.x + 0.035,
                                y0=pr.y - 0.035, y1=pr.y + 0.035,
                                line=dict(width=0), fillcolor="black"))

        elif isinstance(pr, Text):
            xanchor = {"left": "left", "right": "right", "center": "center"}.get(pr.ha, "center")
            yanchor = {"top": "top", "bottom": "bottom", "center": "middle"}.get(pr.va, "middle")
            annotations.append(dict(x=pr.x, y=pr.y, text=pr.text, showarrow=False,
                                     font=dict(size=pr.size * 1.35,
                                               color="black",
                                               family="Arial Black" if pr.weight == "bold" else "Arial"),
                                     xanchor=xanchor, yanchor=yanchor))

    fig.update_layout(shapes=shapes, annotations=annotations,
                       xaxis=dict(range=[xmin, xmax], visible=False),
                       yaxis=dict(range=[ymin, ymax], visible=False, scaleanchor="x", scaleratio=1),
                       height=420, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="white", dragmode="pan")
    return fig
