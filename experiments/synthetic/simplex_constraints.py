"""Figure S1: the two-class example with its constraints, Omega_LP and Omega_GE.

Interval bounds of the example (Appendix C):

    C_low = [[0.3, 0.1], [0.0, 0.4]],  C_up = [[0.4, 0.2], [0.1, 0.6]],
    q_low = (0.6, 0.3),                q_up = (0.7, 0.4).

The four dashed lines are the constraints of Omega (C_up w >= q_low and
C_low w <= q_up).  The boxes are computed with the LP and GE solvers.  The
shaded points are the weights w = C^-1 q of admissible pairs (C, q) drawn
uniformly from the intervals that also satisfy sum_ij C_ij = 1 and
sum_k q_k = 1, with their density contours.

Writes ``figures/Simplex_constraints.pdf``.

Example:
    python synthetic/simplex_constraints.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils import FIGURES_ROOT
from exputils.plotting import set_paper_style
from exputils.regions import Omega_GEKH, Omega_LPML

C_LOW = np.array([[0.3, 0.1], [0.0, 0.4]])
C_UP = np.array([[0.4, 0.2], [0.1, 0.6]])
Q_LOW = np.array([0.6, 0.3])
Q_UP = np.array([0.7, 0.4])
OMEGA_STAR = (5 / 3, 3 / 5)       # true weights of the workflow example (Figure S2)


def sample_weights(n_raw, rng):
    """w = C^-1 q for (C, q) uniform on the intervals, restricted to the simplex."""
    C11 = rng.uniform(C_LOW[0, 0], C_UP[0, 0], n_raw)
    C12 = rng.uniform(C_LOW[0, 1], C_UP[0, 1], n_raw)
    C21 = rng.uniform(C_LOW[1, 0], C_UP[1, 0], n_raw)
    C22 = 1.0 - (C11 + C12 + C21)
    keep = (C22 >= C_LOW[1, 1]) & (C22 <= C_UP[1, 1])
    C11, C12, C21, C22 = C11[keep], C12[keep], C21[keep], C22[keep]

    q1 = rng.uniform(Q_LOW[0], Q_UP[0], keep.sum())
    q2 = 1.0 - q1
    det = C11 * C22 - C12 * C21
    return (C22 * q1 - C12 * q2) / det, (-C21 * q1 + C11 * q2) / det


def constraint_lines(ax, w1_grid):
    """Boundary lines of the four constraints of Omega."""
    lines = [(C_LOW[0], Q_UP[0], "C4"), (C_UP[0], Q_LOW[0], "C1"), (C_LOW[1], Q_UP[1], "C9"), (C_UP[1], Q_LOW[1], "C6")]
    for (a1, a2), b, color in lines:
        ax.plot(w1_grid, (b - a1 * w1_grid) / a2, "--", lw=1.2, color=color, zorder=5,
                label=rf"${a1:.1f}\omega_1 + {a2:.1f}\omega_2 = {b:.1f}$")


def annotate_box(ax, box, name, color, offsets):
    """Write the four bounds next to the box; ``offsets`` maps a bound to (x, y, ha, va)."""
    for key, (x, y, ha, va) in offsets.items():
        k, side = int(key[0]), key[1:]
        value = box[k, 0] if side == "min" else box[k, 1]
        digits = 1 if abs(value - round(value, 1)) < 1e-9 else 3
        ax.text(x, y, rf"{name} $\omega_{{{k + 1},{side}}}={value:.{digits}f}$", color=color, fontsize=9,
                fontweight="bold", ha=ha, va=va, zorder=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=10_000_000, help="draws before the simplex restriction")
    parser.add_argument("--seed", type=int, default=1652)
    parser.add_argument("--fig-dir", default=str(FIGURES_ROOT))
    args = parser.parse_args()

    lp_box = Omega_LPML(C_LOW, C_UP, Q_LOW, Q_UP)
    ge_box = Omega_GEKH(C_LOW, C_UP, Q_LOW, Q_UP)
    print(f"Omega_LP: w1 in [{lp_box[0, 0]:.4f}, {lp_box[0, 1]:.4f}], w2 in [{lp_box[1, 0]:.4f}, {lp_box[1, 1]:.4f}]")
    print(f"Omega_GE: w1 in [{ge_box[0, 0]:.4f}, {ge_box[0, 1]:.4f}], w2 in [{ge_box[1, 0]:.4f}, {ge_box[1, 1]:.4f}]")

    omega1, omega2 = sample_weights(args.n_samples, np.random.default_rng(args.seed))
    n = omega1.size
    print(f"{n} admissible draws")

    # density of the sampled weights
    heatmap, xedges, yedges = np.histogram2d(omega1, omega2, bins=200)
    density = heatmap / (n * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))
    density = gaussian_filter(density.T, sigma=1.2)
    density = np.ma.masked_where(density < 0.001, density)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)

    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(omega1[::100], omega2[::100], ".", markersize=2, alpha=0.9, markerfacecolor="black",
            markeredgecolor="none", zorder=1,
            label=r"Sampled $\boldsymbol{\omega}$" + "\n" + r"($\sum_{ij} C_{ij}=1, \sum_{k} q_k=1$)")
    cf = ax.contourf(X, Y, density, levels=14, cmap="cividis_r", alpha=0.9, zorder=2)
    cax = fig.add_axes([0.38, 0.93, 0.46, 0.028])
    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cbar.ax.invert_xaxis()
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_label(r"Contour of $\boldsymbol{\omega}$", fontsize=11, labelpad=6)
    cbar.locator = ticker.MaxNLocator(nbins=8)
    cbar.update_ticks()

    constraint_lines(ax, np.linspace(0.8, 2.5, 200))
    for box, color, style, label in [(lp_box, "blue", "-", "LP Box"), (ge_box, "darkgreen", "--", "GE Box")]:
        ax.add_patch(patches.Rectangle((box[0, 0], box[1, 0]), box[0, 1] - box[0, 0], box[1, 1] - box[1, 0],
                                       linewidth=2, edgecolor=color, facecolor="none", linestyle=style,
                                       label=label, zorder=6))
    annotate_box(ax, lp_box, "LP", "blue", {
        "0min": (lp_box[0, 0], lp_box[1, 0] - 0.09, "right", "baseline"),
        "1min": (lp_box[0, 1] + 0.01, lp_box[1, 0] + 0.05, "left", "center"),
        "0max": (lp_box[0, 1] + 0.02, lp_box[1, 1] - 0.06, "left", "baseline"),
        "1max": (lp_box[0, 0] + 0.015, lp_box[1, 1] + 0.04, "left", "center")})
    annotate_box(ax, ge_box, "GE", "darkgreen", {
        "0min": (ge_box[0, 0] - 0.01, ge_box[1, 0] + 0.04, "right", "baseline"),
        "0max": (ge_box[0, 1] + 0.01, ge_box[1, 1] - 0.07, "left", "baseline"),
        "1min": (ge_box[0, 1] + 0.02, ge_box[1, 0] - 0.07, "left", "center"),
        "1max": (ge_box[0, 0] + 0.1, ge_box[1, 1] + 0.05, "left", "center")})

    ax.plot(*OMEGA_STAR, linestyle="None", marker=(6, 2, 0), color="orange", mec="orange", mew=1.2, markersize=6,
            zorder=7, label=rf"$\boldsymbol{{\omega}}^* = ({OMEGA_STAR[0]:.2f}, {OMEGA_STAR[1]:.2f})$")

    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlim(-0.08, 2.7)
    ax.set_ylim(-0.05, 1.7)
    ax.set_xlabel(r"$\omega_1$", fontsize=12)
    ax.set_ylabel(r"$\omega_2$", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(bbox_to_anchor=(0.005, 0.65), loc="center left", fontsize=8.5, framealpha=0.95)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.93)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "Simplex_constraints.pdf")
    print(f"Figure saved to {fig_dir / 'Simplex_constraints.pdf'}")


if __name__ == "__main__":
    main()
