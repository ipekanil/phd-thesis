"""
Toy wave-optics forward model for wavefront-coded EDoF imaging.

This is a small, self-contained NumPy script that illustrates the physics used
throughout the EDoF chapter of the thesis:

  pupil  P(rho) = A(rho) * exp(i [ psi * rho^2  +  phi_mask ])
  PSF    h      = |F{P}|^2                     (incoherent, fluorescence)
  MTF           = |F{h}| / |F{h}|(0)
  sensor y      = x * h + n                    (shift-invariant convolution)

It is NOT the differentiable model used in the publications (that one is
available from the authors on request). It only reproduces the qualitative
behaviour so readers can build intuition.

Run:
    python examples/toy_forward_model.py            # writes PNGs to docs/assets/figures/generated
    python examples/toy_forward_model.py --show     # also opens the figures

Requirements: numpy, matplotlib (scipy optional, only for the test image).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "generated"


# ----------------------------------------------------------------------------
# Core model
# ----------------------------------------------------------------------------
def pupil_grid(n: int, fill: float = 0.5):
    """Square grid of n x n samples. The pupil radius is fill * n / 2 samples.

    Returns normalised pupil coordinates (u, v) with the pupil edge at rho = 1,
    and the binary circular aperture A.
    """
    r_samples = fill * n / 2
    c = (np.arange(n) - n / 2) / r_samples
    u, v = np.meshgrid(c, c)
    rho2 = u**2 + v**2
    aperture = (rho2 <= 1.0).astype(float)
    return u, v, rho2, aperture


def defocus_phase(rho2: np.ndarray, psi: float) -> np.ndarray:
    """Quadratic defocus phase psi * rho^2 (psi = defocus coefficient, in rad)."""
    return psi * rho2


def cubic_phase(u: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """Classical Dowski-Cathey cubic phase mask alpha * (u^3 + v^3)."""
    return alpha * (u**3 + v**3)


def psf_from_pupil(aperture: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Incoherent PSF = |Fourier transform of the pupil|^2, normalised to sum 1."""
    field = aperture * np.exp(1j * phase)
    E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    h = np.abs(E) ** 2
    return h / h.sum()


def mtf_from_psf(h: np.ndarray) -> np.ndarray:
    """MTF = |OTF| normalised at DC. OTF is the Fourier transform of the PSF."""
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h)))
    m = np.abs(otf)
    return m / m.max()


def convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (fine for a toy example)."""
    H = np.fft.fft2(np.fft.ifftshift(h))
    return np.real(np.fft.ifft2(np.fft.fft2(x) * H))


def test_image(n: int) -> np.ndarray:
    """Synthetic 'cells': random blobs plus a few thin filaments."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n))
    for _ in range(40):
        cx, cy = rng.uniform(0, n, 2)
        r = rng.uniform(n / 60, n / 25)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2))
    for _ in range(12):
        x0, y0 = rng.uniform(0, n, 2)
        th = rng.uniform(0, np.pi)
        d = np.abs((xx - x0) * np.sin(th) - (yy - y0) * np.cos(th))
        img += 0.6 * np.exp(-(d**2) / 2.0)
    return img / img.max()


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def crop(a: np.ndarray, size: int) -> np.ndarray:
    c = a.shape[0] // 2
    return a[c - size // 2 : c + size // 2, c - size // 2 : c + size // 2]


def fig_psf_vs_defocus(n=512, psis=(0, 10, 20, 30), alpha=60.0):
    u, v, rho2, A = pupil_grid(n, fill=0.25)
    fig, axes = plt.subplots(2, len(psis), figsize=(2.3 * len(psis), 4.9))
    for j, psi in enumerate(psis):
        for i, (label, extra) in enumerate(
            [("Clear pupil", 0.0), (f"Cubic mask (α = {alpha:g} rad)", cubic_phase(u, v, alpha))]
        ):
            h = psf_from_pupil(A, defocus_phase(rho2, psi) + extra)
            ax = axes[i, j]
            ax.imshow(crop(h, 256) ** 0.5, cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"ψ = {psi} rad", fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("PSF versus defocus (√ intensity shown)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_mtf_vs_defocus(n=512, psis=(0, 10, 20, 30), alpha=60.0):
    u, v, rho2, A = pupil_grid(n, fill=0.25)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(psis)))
    f = np.linspace(-1, 1, n)  # frequency axis, normalised below
    for ax, (title, extra) in zip(axes, [("Clear pupil", 0.0), (f"Cubic mask (α = {alpha:g} rad)", cubic_phase(u, v, alpha))]):
        for c, psi in zip(colors, psis):
            m = mtf_from_psf(psf_from_pupil(A, defocus_phase(rho2, psi) + extra))
            row = m[n // 2, :]
            # incoherent cutoff = pupil diameter = fill * n samples  ->  normalise so cutoff = 1
            fx = (np.arange(n) - n / 2) / (0.25 * n)
            keep = (fx >= 0) & (fx <= 1.05)
            ax.plot(fx[keep], row[keep], color=c, lw=1.6, label=f"ψ = {psi}")
        ax.axhline(0.1, ls="--", lw=0.8, color="grey")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("spatial frequency / incoherent cutoff")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1.05)
    axes[0].set_ylabel("MTF (log scale)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Defocus drives the clear-pupil MTF to zeros; the coded pupil keeps it non-zero and similar", fontsize=10)
    fig.tight_layout()
    return fig


def fig_imaging_example(n=512, psis=(0, 15, 30), alpha=60.0, sigma=0.005):
    u, v, rho2, A = pupil_grid(n, fill=0.25)
    x = test_image(n)
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, len(psis) + 1, figsize=(2.4 * (len(psis) + 1), 5))
    axes[0, 0].imshow(x, cmap="gray"); axes[0, 0].set_title("object x", fontsize=10)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, "y = x ∗ h + n\n\nSame object,\nthree depths.\n\nCoded images\nlook blurrier,\nbut they look\nalike across\ndepth, so one\nnetwork can\ninvert them.",
                    fontsize=8.5, va="center")
    for j, psi in enumerate(psis):
        for i, extra in enumerate([0.0, cubic_phase(u, v, alpha)]):
            h = psf_from_pupil(A, defocus_phase(rho2, psi) + extra)
            y = convolve_fft(x, h) + sigma * rng.standard_normal((n, n))
            ax = axes[i, j + 1]
            ax.imshow(np.clip(y, 0, None), cmap="gray")
            ax.set_title(("clear" if i == 0 else "coded") + f", ψ = {psi}", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_sampling(psi=20.0, n_coarse=64, n_fine=512, fill=0.5):
    """Same physical PSF, simulated with a too-coarse and a fine pupil grid.

    With a fixed fill ratio the PSF pixel pitch is identical for both grids,
    so the two centre crops can be compared directly.
    Criterion (Publication I, Eq. 8):  Δs <= π r / (8 ψ)  ->  r_samples >= 8 ψ / π.
    """
    need = 8 * psi / np.pi
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    res = {}
    for n in (n_coarse, n_fine):
        u, v, rho2, A = pupil_grid(n, fill=fill)
        res[n] = (np.angle(np.exp(1j * defocus_phase(rho2, psi))) * A, psf_from_pupil(A, defocus_phase(rho2, psi)))
    r_c, r_f = fill * n_coarse / 2, fill * n_fine / 2
    axes[0].imshow(res[n_coarse][0], cmap="twilight", interpolation="nearest")
    axes[0].set_title(f"wrapped pupil phase, r = {r_c:.0f} samples\n(needs ≥ {need:.0f}: undersampled)", fontsize=9)
    size = n_coarse
    axes[1].imshow(crop(res[n_coarse][1], size) ** 0.5, cmap="magma")
    axes[1].set_title(f"PSF from coarse grid\n(aliased, wrong)", fontsize=9)
    big = 3 * size
    axes[2].imshow(crop(res[n_fine][1], big) ** 0.5, cmap="magma")
    axes[2].add_patch(plt.Rectangle((big / 2 - size / 2, big / 2 - size / 2), size, size, fill=False, ec="cyan", ls="--", lw=1))
    axes[2].set_title(f"PSF from fine grid, r = {r_f:.0f} samples\n(meets the bound; dashed = coarse-grid window)", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Why the Fourier-plane sampling bound matters (ψ = {psi:g} rad)", fontsize=10)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    figs = {
        "psf_vs_defocus.png": fig_psf_vs_defocus(),
        "mtf_vs_defocus.png": fig_mtf_vs_defocus(),
        "imaging_example.png": fig_imaging_example(),
        "sampling_aliasing.png": fig_sampling(),
    }
    for name, fig in figs.items():
        fig.savefig(args.out / name, dpi=150, bbox_inches="tight")
        print("wrote", args.out / name)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
