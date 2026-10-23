"""
Figures for the FLFM pages of the documentation site.

Two kinds of figure, both produced here from first principles or from the
numbers reported in the text, so that no manuscript figure is reused:

  * physics / toy-model figures (focal length vs in-focus depth, Fourier-plane
    phase masks, per-view blur against depth). These use the system parameters
    of the FLFM study but simple geometric + diffraction approximations, so they
    illustrate the mechanism; they are not the paper's simulations.
  * charts of reported results (overall comparison, lateral resolution at the
    reported depths, learned vs random focal maps). The values are typed in
    below from the manuscript's tables and text.

Run:
    python examples/flfm_figures.py          # writes PNGs to docs/assets/figures/flfm/
Requirements: numpy, matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "flfm"

# ---------------------------------------------------------------- style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARM_COLORS = {                      # fixed categorical order, one colour per arm
    "Fixed optics + RL-TV": "#2a78d6",
    "Fixed optics + CNN": "#eb6834",
    "Learned f + CNN": "#1baf7a",
    "Learned f, D + CNN": "#eda100",
}
ENC_COLORS = {"Fixed MLA": "#eb6834", "Random f (mean of 40)": "#6250d6", "Learned f": "#1baf7a"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 2,
})

# ---------------------------------------------------------------- system parameters
LAM = 633e-6           # mm
N_IMM = 1.33
F_OBJ = 3.0            # mm
M_PUPIL = 300 / 180    # pupil -> Fourier plane magnification
F_MLA = 43.1           # mm (sensor distance behind the encoder)
PITCH = 2.12           # mm
PUPIL_D = 11.0         # mm, illuminated pupil at the Fourier plane
Z_HALF = 0.030         # mm, half depth of the reconstructed volume
M_T = (F_MLA / F_OBJ) * (180 / 300)   # lateral magnification object -> sensor
C = N_IMM * M_PUPIL**2 * F_OBJ**2     # mm^2, z = C (1/f - 1/f_MLA)


def depth_in_focus(f):
    return C * (1.0 / f - 1.0 / F_MLA)


F_MIN = 1.0 / (1.0 / F_MLA + Z_HALF / C)
F_MAX = 1.0 / (1.0 / F_MLA - Z_HALF / C)


def illustrative_focal_lengths(n=25, seed=3):
    """Focal lengths evenly spread in 1/f across the allowed interval, shuffled.
    Illustrative only: these are NOT the learned values."""
    inv = np.linspace(1 / F_MAX, 1 / F_MIN, n)
    rng = np.random.default_rng(seed)
    return 1.0 / rng.permutation(inv)


def hex_centres(n=5, pitch=PITCH):
    pts = []
    for r in range(n):
        for c in range(n):
            # same layout as hex_centers_px in code/flfm/F_Optics_fixed.py:
            # even rows centred on the axis, odd rows shifted by half a pitch
            x = (c - (n - 1) / 2) * pitch + (0.5 * pitch if r % 2 else 0.0)
            y = (r - (n - 1) / 2) * pitch * np.sqrt(3) / 2
            pts.append((x, y))
    return np.array(pts)


# ---------------------------------------------------------------- figures
def fig_focal_vs_depth():
    f = np.linspace(41.0, 45.4, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axhspan(-30, 30, color="#1baf7a", alpha=0.08, lw=0)
    ax.axvspan(F_MIN, F_MAX, color="#1baf7a", alpha=0.08, lw=0)
    ax.plot(f, depth_in_focus(f) * 1e3, color=INK)
    ax.plot([F_MLA], [0], "o", ms=8, color="#eb6834", mec="white", mew=2, zorder=3)
    ax.annotate("conventional MLA\nf = 43.1 mm → z = 0", (F_MLA, 0), (43.6, 14), color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    for fv, lab in [(F_MIN, f"f_min = {F_MIN:.2f} mm"), (F_MAX, f"f_max = {F_MAX:.2f} mm")]:
        ax.axvline(fv, color="#1baf7a", lw=1, ls="--")
        ax.text(fv, 41, lab, ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_xlabel("lenslet focal length f (mm)")
    ax.set_ylabel("in-focus depth z (µm)")
    ax.set_ylim(-48, 52)
    ax.text(41.1, -44, "allowed interval: lenslets focus within the ±30 µm volume", color=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def fig_phase_masks(npx=2400):
    x = np.linspace(-PUPIL_D / 2, PUPIL_D / 2, npx)
    X, Y = np.meshgrid(x, x)
    pupil = X**2 + Y**2 <= (PUPIL_D / 2) ** 2
    centres = hex_centres()
    fl = illustrative_focal_lengths()
    wrapped = np.full(X.shape, np.nan)
    extra = np.full(X.shape, np.nan)
    for (cx, cy), fj in zip(centres, fl):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        m = (r2 <= (PITCH / 2) ** 2) & pupil
        wrapped[m] = np.mod(-np.pi * r2[m] / (LAM * F_MLA), 2 * np.pi)
        extra[m] = -np.pi * r2[m] / LAM * (1.0 / fj - 1.0 / F_MLA)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1))
    axes[0].imshow(wrapped, cmap="twilight", extent=[x[0], x[-1], x[0], x[-1]], origin="lower",
                   interpolation="nearest")
    axes[0].set_title("Conventional MLA: wrapped phase, f = 43.1 mm", fontsize=10)
    lim = np.nanmax(np.abs(extra))
    im = axes[1].imshow(extra, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[x[0], x[-1], x[0], x[-1]],
                        origin="lower", interpolation="nearest")
    axes[1].set_title("Multifocal array: extra phase vs conventional", fontsize=10)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    cb.set_label("extra phase (rad)"); cb.outline.set_visible(False)
    for ax in axes:
        ax.add_patch(plt.Circle((0, 0), PUPIL_D / 2, fill=False, ec=INK2, lw=1, ls="--"))
        ax.set_xlabel("x₃ (mm)"); ax.grid(False)
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    axes[0].set_ylabel("y₃ (mm)")
    fig.tight_layout()
    return fig


def view_width_um(f, z_mm, D=PITCH):
    """Toy object-space width of one lenslet's view of a point at depth z:
    geometric defocus blur combined in quadrature with the diffraction spot."""
    delta = (1.0 / f - 1.0 / F_MLA) - z_mm / C          # 1/mm, focus mismatch
    geometric = D * F_MLA * np.abs(delta)               # mm at the sensor
    diffraction = LAM * F_MLA / D                       # mm at the sensor (~FWHM)
    return np.sqrt(geometric**2 + diffraction**2) / M_T * 1e3


def fig_view_blur():
    z = np.linspace(-0.030, 0.030, 241)
    fl = illustrative_focal_lengths()
    w_fixed = view_width_um(F_MLA, z)
    ws = np.array([view_width_um(fj, z) for fj in fl])
    # Sharp views carry the high spatial frequencies, so the views are combined
    # by their sharpness (1/width): effective width = harmonic mean over views.
    w_eff = 1.0 / np.mean(1.0 / ws, axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for i, w in enumerate(ws):
        ax.plot(z * 1e3, w, color="#1baf7a", lw=0.7, alpha=0.3, label="single views (multifocal)" if i == 0 else None)
    ax.plot(z * 1e3, w_fixed, color="#eb6834", label="conventional MLA (all 25 views)")
    ax.plot(z * 1e3, w_eff, color="#1baf7a", lw=2.4, label="multifocal, combined over views")
    ax.set_xlabel("depth z (µm)"); ax.set_ylabel("view blur in object space (µm)")
    ax.set_ylim(0, 12.5)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- reported results
TABLE1 = {   # arm: (depth spread, PSNR dB, SSIM, rho, time s) -- Publication III, Table 1
    "Fixed optics + RL-TV": (3.97, 20.43, 0.39, 0.79, 15.73),
    "Fixed optics + CNN": (3.83, 20.70, 0.44, 0.79, 2.19),
    "Learned f + CNN": (2.14, 21.27, 0.45, 0.80, 2.16),
    "Learned f, D + CNN": (2.17, 21.32, 0.45, 0.80, 2.17),
}
LATERAL = {  # effective lateral resolution (µm) at the depths reported in the text
    "z": [-30, 0, 20, 30],
    "Fixed optics + CNN": [6.86, 1.80, 3.91, 6.90],
    "Learned f + CNN": [4.55, 2.58, 2.78, 5.30],
}
RANDOM = {   # spread (worst / best over depth) from PSF stacks -- Supplement S9
    "Fixed MLA": (4.61, 5.20),
    "Random f (mean of 40)": (2.59, 3.17),
    "Learned f": (2.02, 1.96),
}


def _hbar(ax, labels, values, colors, fmt, title, better):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", color=INK, fontsize=9.5)
    ax.set_yticks(y, labels)
    ax.set_title(f"{title}  ({better})", fontsize=10, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.22)


def fig_overall():
    arms = list(TABLE1)
    cols = [ARM_COLORS[a] for a in arms]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.9))
    _hbar(axes[0], arms, [TABLE1[a][0] for a in arms], cols, "{:.2f}", "Depth spread", "lower is better")
    y = np.arange(len(arms))[::-1]
    ps = [TABLE1[a][1] for a in arms]
    axes[1].scatter(ps, y, s=90, c=cols, edgecolors="white", linewidths=2, zorder=3)
    for yi, v in zip(y, ps):
        axes[1].text(v, yi + 0.28, f"{v:.2f} dB", ha="center", va="bottom", color=INK, fontsize=9.5)
    axes[1].set_yticks(y, [])
    axes[1].set_ylim(-0.6, len(arms) - 0.2)
    axes[1].set_xlim(20.2, 21.6)
    axes[1].grid(axis="y", visible=False)
    axes[1].set_title("PSNR, volume average  (higher is better)", fontsize=10, loc="left")
    axes[1].set_xlabel("dB")
    fig.tight_layout()
    return fig


def fig_lateral_depth():
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    z = LATERAL["z"]
    for arm in ["Fixed optics + CNN", "Learned f + CNN"]:
        ax.plot(z, LATERAL[arm], "-o", color=ARM_COLORS[arm], ms=8, mec="white", mew=2)
        ax.text(z[-1] + 1.2, LATERAL[arm][-1], arm, va="center", color=INK, fontsize=9)
    ax.set_xlabel("depth z (µm)"); ax.set_ylabel("effective lateral resolution (µm)")
    ax.set_xlim(-34, 47); ax.set_ylim(0, 7.8); ax.set_xticks(z)
    ax.text(-33, 0.35, "values at the depths reported in the text; lower is sharper", color=INK2, fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_random_vs_learned():
    encs = list(RANDOM)
    cols = [ENC_COLORS[e] for e in encs]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.5))
    _hbar(axes[0], encs, [RANDOM[e][0] for e in encs], cols, "{:.2f}", "Lateral PSF-width spread", "lower is better")
    _hbar(axes[1], encs, [RANDOM[e][1] for e in encs], cols, "{:.2f}", "Axial decorrelation spread", "lower is better")
    axes[1].set_yticklabels([])
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {
        "focal_vs_depth.png": fig_focal_vs_depth(),
        "phase_masks_illustrative.png": fig_phase_masks(),
        "view_blur_toy.png": fig_view_blur(),
        "overall_comparison.png": fig_overall(),
        "lateral_vs_depth.png": fig_lateral_depth(),
        "random_vs_learned.png": fig_random_vs_learned(),
    }
    for name, fig in figs.items():
        fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
        print("wrote", OUT / name)
    print(f"f_min = {F_MIN:.3f} mm, f_max = {F_MAX:.3f} mm, M_T = {M_T:.2f}")


if __name__ == "__main__":
    main()
