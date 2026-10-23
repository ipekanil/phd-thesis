"""
make_figure2_optics.py -- the learned-optics figure: parameter maps + PSF profiles.

Rebuilds the manuscript's optics figure from the saved checkpoints:

  left   per-lenslet parameter map, one row per arm. Colour is focal length,
         circle size is effective aperture diameter, on the staggered hex
         lattice the array actually uses.
  right  xz and yz PSF profiles over the full axial range, log intensity.

The PSF panels are a single horizontal and a single vertical line through the
sensor-plane PSF stack at the centre of the array -- which is what makes the
five streaks in the xz panel the five lenslets of the middle row, and the three
in yz the middle column. It is the same extraction F_Optics_fixed.plot_xz_yz_psf
already does, at y = H/2 and x = W/2.

    python make_figure2_optics.py --root ..

Runs one optics forward pass per arm, so it needs the GPU free -- about a
minute per arm. The input volume only drives the shape; the PSF stack is a
property of the optics, not of the sample.

A NOTE ON THE X AXIS. The published figure's xz panels span 0-20 mm. This
script computes the extent from config (sensor samples x psf_sampling_sp) and
prints what that comes to, rather than hard-coding a range I cannot verify
against the original. If the number it prints disagrees with the manuscript,
--extent-mm overrides it, but check which sampling the original used before
assuming this one is wrong.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from F_Optics_fixed import F_Optics
from make_paper_table import DEVICE, SEED, infer_learn_D

ROWS = [("fixed", "Fixed"),
        ("f",     r"Learned $f$"),
        ("fD",    r"Learned $f$ and $D$")]
INK, MUTED = "#1a1a19", "#6b6a63"


def load_optics(ckpt, fixed):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    learn_D = infer_learn_D(sd)
    m = F_Optics(config.optics_params, seed=SEED, learn_D=learn_D,
                 init_f="nominal" if fixed else "random").to(DEVICE)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, learn_D


def lenslet_lines(psf_mid, ns):
    """Row and column indices of a real lenslet, nearest the array centre.

    psf_mid: one depth slice of the sensor-plane PSF stack, [H, W].

    The array centre is NOT in general a lenslet centre. On the staggered hex
    lattice the odd rows are offset by half a pitch, so a line at H/2 or W/2
    can fall in the gap between lenslets. When it does, the extracted profile
    contains only PSF tails, its own maximum is a tail, and normalising by that
    maximum lifts the noise floor to full scale -- which is what turned the xz
    panels into orange mush.

    So the lines are found from the data: collapse the slice onto each axis,
    which gives one bump per lenslet row (or column), and take the strong peak
    closest to the centre. A vertical line through a column crosses only the
    rows that have a lenslet at that offset -- 3 of 5 here, which is why the
    manuscript's yz panels show three lenslets and not five.
    """
    out = []
    for axis in (1, 0):                      # collapse x -> rows; y -> cols
        prof = np.asarray(psf_mid).sum(axis=axis).astype(np.float64)
        n = prof.size
        # smooth over ~1% of the span so single hot pixels are not peaks
        k = max(3, int(round(n * 0.01)) | 1)
        sm = np.convolve(prof, np.ones(k) / k, mode="same")
        thr = sm.max() * 0.2
        idx = np.where((sm[1:-1] >= sm[:-2]) & (sm[1:-1] > sm[2:])
                       & (sm[1:-1] > thr))[0] + 1
        if idx.size == 0:                    # nothing found: fall back, loudly
            print(f"  !! no lenslet peaks on axis {axis}; using the centre, "
                  f"which may land in a gap")
            out.append(n // 2)
            continue
        out.append(int(idx[np.argmin(np.abs(idx - n / 2))]))
    return out[0], out[1]                    # (row for xz, col for yz)


def param_map(ax, model, learn_D, ns, fmin_mm, fmax_mm, pitch_m,
              cmap="cividis"):
    """Circles on the staggered hex lattice: colour = f, diameter = D.

    The circle scale is referenced to the LENSLET PITCH, not to the largest
    diameter in the figure: at D = pitch neighbouring lenslets touch exactly,
    and above it they overlap, which is the realizability limit the (f, D) arm
    can cross. Scaling to the largest diameter instead -- which this script did
    first -- shrinks the fixed and learned-f arms below touching even though
    their D sits exactly at the pitch, and hides the overlap entirely.
    """
    f_mm = model.f_rand.detach().cpu().numpy().reshape(ns, ns) * 1e3
    if learn_D:
        d_m = model.D_map.detach().cpu().numpy().reshape(ns, ns)
    else:
        # a conventional MLA has one diameter for every lenslet
        d_m = np.full((ns, ns), float(model.D_pitch))

    # Same staggered lattice the optics uses (hex_centers_px): odd rows are
    # offset by half a pitch. Drawing them on a square grid would show an
    # array the system does not have.
    for r in range(ns):
        shift = 0.5 if (r % 2 == 1) else 0.0
        for c in range(ns):
            x, y = c + shift, r
            # circle radius in DATA units: 0.5 would touch at full pitch
            rad = 0.5 * (d_m[r, c] / pitch_m)   # 0.5 = touching at D = pitch
            col = plt.get_cmap(cmap)(
                (f_mm[r, c] - fmin_mm) / max(fmax_mm - fmin_mm, 1e-9))
            ax.add_patch(plt.Circle((x, y), rad, facecolor=col,
                                    edgecolor="#333", lw=0.4, zorder=2))
            ax.text(x, y, f"{f_mm[r, c]:.1f}", ha="center", va="center",
                    fontsize=5.6, color="#111", zorder=3)
    ax.set_xlim(-0.75, ns - 0.25 + 0.5)
    ax.set_ylim(ns - 0.5, -0.5)            # row 0 at the top
    ax.set_aspect("equal")
    ax.set_xticks(range(ns)); ax.set_yticks(range(ns))
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    for sp in ax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.6)
    return f_mm, d_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..",
                    help="directory holding the optics checkpoints")
    ap.add_argument("--extent-um", type=float, default=None,
                    help="override the full lateral extent of the PSF panels, "
                         "in µm. Default: W * model.pixel_size, as plot_psf.py "
                         "computes it.")
    ap.add_argument("--floor", type=float, default=10 ** -2.5,
                    help="log-scale floor as a fraction of the panel maximum. "
                         "The default matches plot_psf.py's LOG_FLOOR = -2.5; "
                         "1e-5 reaches so far into the background that the "
                         "panel fills with amplified noise.")
    ap.add_argument("--psf-cmap", default="magma",
                    help="colour map for the PSF panels. 'magma' matches the "
                         "orthoview figures; 'inferno' is the lighter variant "
                         "the submitted figure used.")
    ap.add_argument("--param-cmap", default="cividis",
                    help="colour map for the focal-length maps. 'cividis' is "
                         "perceptually uniform and colour-vision safe, and "
                         "its blue-to-yellow range reads as an ordered scale "
                         "rather than as categories.")
    ap.add_argument("--diagnose", action="store_true",
                    help="also write <out>_diag_<arm>.png: the mid-depth "
                         "sensor slice with the chosen xz row and yz column "
                         "drawn on it, plus the collapsed profiles. One run "
                         "shows where the lenslets actually are.")
    ap.add_argument("--xz-row", type=int, default=None,
                    help="sensor row for the xz profile. Default: found from "
                         "the data (the lenslet row nearest the centre).")
    ap.add_argument("--yz-col", type=int, default=None,
                    help="sensor column for the yz profile. Default: found "
                         "from the data.")
    ap.add_argument("--out", default="figure2_optics")
    args = ap.parse_args()

    root = Path(args.root)
    ns = int(config.optics_params["microlens"]["array_size"])

    zmin = float(config.optics_params["volume_depth_min"]) * 1e6
    zmax = float(config.optics_params["volume_depth_max"]) * 1e6

    # A single point at the centre of every depth, as plot_psf.py does. The
    # returned stack is the system PSF either way, but a probe needs no dataset
    # and makes the panels unambiguous.
    nz = int(round((-float(config.optics_params["volume_depth_min"])
                    + float(config.optics_params["volume_depth_max"]))
                   / float(config.optics_params["depth_resolution"]))) + 1
    probe = torch.zeros((nz, 512, 512), dtype=torch.float32, device=DEVICE)
    probe[:, 256, 256] = 1.0

    # ---- pass 1: load every arm and collect parameters, so the colour scale
    # and the circle scale are SHARED across rows. Scaling each row to its own
    # range would make a conventional MLA's single focal length look as varied
    # as a learned one's.
    arms = {}
    for tag, _ in ROWS:
        ck = root / f"optics_model_{tag}.pth"
        if not ck.exists():
            raise SystemExit(f"!! {ck} not found")
        arms[tag] = load_optics(str(ck), tag == "fixed")

    fs, ds_ = [], []
    for tag, _ in ROWS:
        m, lD = arms[tag]
        fs.append(m.f_rand.detach().cpu().numpy() * 1e3)
        ds_.append(m.D_map.detach().cpu().numpy().ravel() if lD
                   else np.full(ns * ns, float(m.D_pitch)))
    fmin_mm, fmax_mm = float(np.min(fs)), float(np.max(fs))
    pitch_m = float(config.optics_params["microlens"]["pitch"])
    dmax = float(np.max(ds_))
    print(f"focal length {fmin_mm:.2f}..{fmax_mm:.2f} mm; "
          f"pitch {pitch_m*1e3:.3f} mm, max diameter {dmax*1e3:.3f} mm "
          f"({dmax/pitch_m:.2f}x pitch)")
    if dmax > pitch_m:
        print(f"!! largest diameter exceeds the pitch by "
              f"{(dmax/pitch_m - 1)*100:.1f}% -- those lenslets overlap and "
              f"the array is not physically realizable as drawn")

    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(len(ROWS), 2, width_ratios=[1.0, 1.45],
                          hspace=0.22, wspace=0.18,
                          left=0.11, right=0.97, top=0.945, bottom=0.11)

    half_um = None
    for i, (tag, label) in enumerate(ROWS):
        model, learn_D = arms[tag]

        axp = fig.add_subplot(gs[i, 0])
        param_map(axp, model, learn_D, ns, fmin_mm, fmax_mm, pitch_m,
                  args.param_cmap)
        axp.set_ylabel(label, fontsize=9, color=INK, labelpad=18)
        if i == 0:
            axp.set_title("Parameter maps", fontsize=9.5, color=INK, pad=6)
        if i == len(ROWS) - 1:
            axp.set_xlabel("Lenslet column", fontsize=8, color=INK)

        # ---- PSF stack for this arm
        with torch.no_grad():
            _out, _pitch, psf = model(probe)
        # THE PSF STACK COMES BACK IN FFT LAYOUT -- DC at the corner, so the
        # lenslets are wrapped around the edges and the array centre is empty.
        # Slicing it unshifted is what turned the xz panels into noise: the
        # line sampled background, and normalising by its own maximum lifted
        # that background to full scale. plot_psf.py shifts first; so does this
        # now.
        psf = torch.fft.fftshift(psf.detach(), dim=(-2, -1))
        # normalise EACH DEPTH by its own maximum before slicing, or the
        # defocused depths vanish against the in-focus one
        psf = psf / (psf.amax(dim=(-2, -1), keepdim=True) + 1e-12)
        D, Hs, Ws = psf.shape
        # one line through the centre of the array: the middle lenslet ROW for
        # xz, the middle COLUMN for yz
        mid = psf[D // 2].float().cpu().numpy()
        ry = args.xz_row if args.xz_row is not None else None
        cx = args.yz_col if args.yz_col is not None else None
        if ry is None or cx is None:
            fr, fc = lenslet_lines(mid, ns)
            ry = fr if ry is None else ry
            cx = fc if cx is None else cx
        print(f"  [{tag}] xz row {ry} of {Hs} ({ry/Hs:.3f}), "
              f"yz col {cx} of {Ws} ({cx/Ws:.3f})")
        if args.diagnose:
            dfig, dax = plt.subplots(1, 3, figsize=(13, 3.6))
            lo = np.percentile(mid, 99.0) or mid.max() or 1.0
            dax[0].imshow(mid, cmap=args.psf_cmap,
                          norm=LogNorm(vmin=max(lo * 1e-4, 1e-20), vmax=lo))
            dax[0].axhline(ry, color="#2a78d6", lw=0.8)
            dax[0].axvline(cx, color="#eb6834", lw=0.8)
            dax[0].set_title(f"{tag}: mid-depth sensor slice", fontsize=9)
            dax[1].plot(mid.sum(1), lw=0.7, color="#2a78d6")
            dax[1].axvline(ry, color="k", lw=0.6, ls=":")
            dax[1].set_title("collapsed onto rows (xz line in blue)", fontsize=9)
            dax[2].plot(mid.sum(0), lw=0.7, color="#eb6834")
            dax[2].axvline(cx, color="k", lw=0.6, ls=":")
            dax[2].set_title("collapsed onto columns (yz line in orange)",
                             fontsize=9)
            for a in dax[1:]:
                a.set_yscale("log")
            dfig.tight_layout()
            dfig.savefig(f"{args.out}_diag_{tag}.png", dpi=130,
                         facecolor="white")
            plt.close(dfig)
            print(f"  wrote {args.out}_diag_{tag}.png")

        xz = psf[:, ry, :].float().cpu().numpy()
        yz = psf[:, :, cx].float().cpu().numpy()
        # a line that missed every lenslet has no dynamic range; say so rather
        # than drawing a panel of amplified noise
        for nm, im in (("xz", xz), ("yz", yz)):
            dyn = float(im.max()) / max(float(np.median(im)), 1e-30)
            if dyn < 50:
                print(f"  !! [{tag}] {nm} line has only {dyn:.0f}x dynamic "
                      f"range -- it is probably between lenslets, not on one")
        del psf, _out
        torch.cuda.empty_cache()

        if half_um is None:
            # plot_psf.py uses model.pixel_size and labels the axis in µm,
            # centred on zero -- not the sensor-plane sampling, and not mm
            px = float(model.pixel_size)
            half_um = (args.extent_um / 2.0) if args.extent_um is not None \
                else (Ws / 2.0) * px * 1e6
            print(f"PSF stack {D} x {Hs} x {Ws}; pixel_size {px*1e6:.4f} µm, "
                  f"lateral half-extent {half_um:.1f} µm "
                  f"(full {2*half_um:.1f} µm = {2*half_um/1e3:.2f} mm)")

        sub = gs[i, 1].subgridspec(2, 1, hspace=0.42)
        for k, (img, axis_lab) in enumerate(((xz, "x (µm)"), (yz, "y (µm)"))):
            ax = fig.add_subplot(sub[k])
            m = float(img.max()) or 1.0
            ax.imshow(img / m, aspect="auto", cmap=args.psf_cmap,
                      norm=LogNorm(vmin=args.floor, vmax=1.0),
                      extent=[-half_um, half_um, zmax, zmin])
            ax.set_xlabel(axis_lab, fontsize=7.5, color=INK, labelpad=1)
            ax.set_ylabel("z (µm)", fontsize=7.5, color=INK, labelpad=1)
            ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
            for sp in ax.spines.values():
                sp.set_color(MUTED); sp.set_linewidth(0.6)
            if i == 0 and k == 0:
                ax.set_title("XZ and YZ PSF", fontsize=9.5, color=INK, pad=6)

    # ---- the two colour bars, along the bottom
    cax1 = fig.add_axes([0.11, 0.045, 0.30, 0.013])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(fmin_mm, fmax_mm), cmap=args.param_cmap),
        cax=cax1, orientation="horizontal")
    cb1.set_label("Focal length (mm)", fontsize=7.5, color=MUTED, labelpad=2)
    cb1.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)

    cax2 = fig.add_axes([0.52, 0.045, 0.45, 0.013])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(
        norm=LogNorm(args.floor, 1.0), cmap=args.psf_cmap),
        cax=cax2, orientation="horizontal")
    cb2.set_label("Normalized intensity (log scale)", fontsize=7.5,
                  color=MUTED, labelpad=2)
    cb2.ax.tick_params(labelsize=6.5, colors=MUTED, length=2)

    for cb in (cb1, cb2):
        cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(MUTED)

    for e in ("png", "svg", "pdf"):
        fig.savefig(f"{args.out}.{e}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
