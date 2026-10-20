"""Two-class examples of an empty Omega and of an Omega cut by w_1 = 0
(illustrates the "Empty regions" paragraph of Appendix K.5; not a paper figure).

    Omega = { w >= 0 : C_up w >= q_low,  C_low w <= q_up }

* Example 1: all entries of C and q lie in (0, 1), yet the region defined by
  the four constraints lies entirely in w_1 < 0, so Omega is empty.
* Example 2: the region crosses w_1 = 0, so one edge of Omega comes from the
  constraint w_1 >= 0 rather than from C and q.

For each example the script prints Omega_LP, the phase-1 slack
(``regions.Omega_LPML_VariableElastic``; zero iff Omega is nonempty) and the
depth of the region into the negative orthant (``regions.Omega_LowBD``), and
draws both regions into ``figures/omega_examples.pdf``.

Example:
    python synthetic/empty_region_examples.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import FIGURES_ROOT
from exputils.plotting import set_paper_style
from exputils.regions import Omega_LowBD, Omega_LPML, Omega_LPML_VariableElastic

EXAMPLES = {
    "empty": {"C_low": np.array([[0.15, 0.70], [0.85, 0.80]]), "C_up": np.array([[0.25, 0.85], [0.95, 0.90]]),
              "q_low": np.array([0.50, 0.10]), "q_up": np.array([0.75, 0.15]), "box": ((-2.2, 1.4), (-0.5, 2.6))},
    "crossing": {"C_low": np.array([[0.35, 0.60], [0.45, 0.15]]), "C_up": np.array([[0.45, 0.70], [0.55, 0.20]]),
                 "q_low": np.array([0.55, 0.20]), "q_up": np.array([0.75, 0.35]), "box": ((-0.9, 1.1), (-0.3, 1.7))},
}


def clip(poly, a, b, c):
    """Part of the convex polygon ``poly`` with a*x + b*y <= c (Sutherland-Hodgman)."""
    out = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        d1, d2 = a * x1 + b * y1 - c, a * x2 + b * y2 - c
        in1, in2 = d1 <= 1e-12, d2 <= 1e-12
        if in1:
            out.append((x1, y1))
        if in1 != in2:
            t = d1 / (d1 - d2)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return out


def polygon(constraints, box):
    """Intersection of the bounding box with the half-planes a*x + b*y <= c."""
    (xmin, xmax), (ymin, ymax) = box
    poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    for a, b, c in constraints:
        if not poly:
            break
        poly = clip(poly, a, b, c)
    return poly


def half_planes(ex):
    """The four constraints of Omega as (a, b, c) with a*w1 + b*w2 <= c."""
    C_low, C_up, q_low, q_up = ex["C_low"], ex["C_up"], ex["q_low"], ex["q_up"]
    return [(-C_up[0, 0], -C_up[0, 1], -q_low[0]), (C_low[0, 0], C_low[0, 1], q_up[0]),
            (-C_up[1, 0], -C_up[1, 1], -q_low[1]), (C_low[1, 0], C_low[1, 1], q_up[1])]


def draw_line(ax, a, b, c, xr, yr, **kwargs):
    xs = np.linspace(xr[0], xr[1], 200)
    if abs(b) > 1e-9:
        ax.plot(xs, (c - a * xs) / b, **kwargs)
    else:
        ax.plot([c / a, c / a], yr, **kwargs)


def report(name, ex):
    args = (ex["C_low"], ex["C_up"], ex["q_low"], ex["q_up"])
    print(f"\n=== {name} ===")
    print("Omega_LP:\n", np.round(Omega_LPML(*args), 4))
    Omega_LPML_VariableElastic(*args)
    Omega_LowBD(*args)


def bounds_text(ex):
    def fmt(m):
        return "[" + ",".join("[" + ",".join(f"{v:.2f}" for v in row) + "]" for row in m) + "]"

    return (f"C_low={fmt(ex['C_low'])}, C_up={fmt(ex['C_up'])}\n"
            f"q_low=({ex['q_low'][0]:.2f},{ex['q_low'][1]:.2f}), q_up=({ex['q_up'][0]:.2f},{ex['q_up'][1]:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    args = parser.parse_args()

    for name, ex in EXAMPLES.items():
        report(name, ex)

    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))

    # left: empty Omega
    ex = EXAMPLES["empty"]
    (xr, yr), cons = ex["box"], half_planes(ex)
    raw = polygon(cons, ex["box"])
    ax1.set_xlim(*xr)
    ax1.set_ylim(*yr)
    ax1.fill_between([0, xr[1]], 0, yr[1], color="#bfe3bf", alpha=0.4, zorder=0)
    if len(raw) >= 3:
        ax1.add_patch(plt.Polygon(raw, closed=True, facecolor="#e07b7b", edgecolor="#8a3b3b", linewidth=1.2,
                                  alpha=0.65, zorder=2))
    for (a, b, c), ls, col in zip(cons, ["--", "--", ":", ":"], ["#8a3b3b", "#8a3b3b", "#3b5b8a", "#3b5b8a"]):
        draw_line(ax1, a, b, c, xr, yr, linestyle=ls, color=col, linewidth=1.6, zorder=3)
    ax1.text(0.65, 2.2, r"$\Omega\subset$ this region" + "\n" + r"(needs $\omega_1,\omega_2\geq 0$)",
             fontsize=10, color="#2e6b2e", ha="center", va="center")
    ax1.annotate(r"region lies entirely at $\omega_1<0$" + "\n" + r"$\Rightarrow\ \Omega=\varnothing$",
                 xy=(-0.8, 1.0), xytext=(-2.1, -0.15), fontsize=10.5, color="#8a3b3b", ha="left", va="center",
                 bbox=dict(boxstyle="round", fc="white", ec="#8a3b3b", alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color="#8a3b3b"))
    ax1.set_title(r"Example 1: $\Omega=\varnothing$ (all $C, q$ entries in $(0,1)$)" + "\n" + bounds_text(ex))

    # right: Omega cut by w_1 = 0
    ex = EXAMPLES["crossing"]
    (xr, yr), cons = ex["box"], half_planes(ex)
    raw = polygon(cons, ex["box"])
    omega = polygon(cons + [(-1, 0, 0), (0, -1, 0)], ex["box"])
    ax2.set_xlim(*xr)
    ax2.set_ylim(*yr)
    if len(raw) >= 3:
        ax2.add_patch(plt.Polygon(raw, closed=True, facecolor="#f0c98a", edgecolor="none", alpha=0.6, zorder=2,
                                  label=r"region without $\omega\geq 0$"))
    if len(omega) >= 3:
        ax2.add_patch(plt.Polygon(omega, closed=True, facecolor="#3b7dd8", edgecolor="none", alpha=0.55, zorder=3,
                                  label=r"$\Omega$ (with $\omega_1,\omega_2\geq 0$)"))
    for (a, b, c), ls, col in zip(cons, ["--", "--", ":", ":"], ["#a06a1a", "#a06a1a", "#5a3f9c", "#5a3f9c"]):
        draw_line(ax2, a, b, c, xr, yr, linestyle=ls, color=col, linewidth=1.6, zorder=4)
    ax2.annotate("this edge of $\\Omega$ comes from\n" + r"$\omega_1\geq 0$, not from $C, q$",
                 xy=(0.0, 1.1), xytext=(-0.85, 0.15), fontsize=9.5, color="#1f4e99",
                 arrowprops=dict(arrowstyle="->", color="#1f4e99"))
    ax2.set_title(r"Example 2: the region crosses $\omega_1=0$ (all $C, q$ entries in $(0,1)$)" + "\n"
                  + bounds_text(ex))
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)

    for ax in (ax1, ax2):
        ax.axhline(0, color="black", linewidth=1.8, zorder=5)
        ax.axvline(0, color="black", linewidth=1.8, zorder=5)
        ax.set_xlabel(r"$\omega_1$")
        ax.set_ylabel(r"$\omega_2$")
        ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "omega_examples.pdf")
    print(f"\nFigure saved to {fig_dir / 'omega_examples.pdf'}")


if __name__ == "__main__":
    main()
