"""
make_paper_table.py -- the paper's Table 1, regenerated under the current pipeline,
plus the XZ / YZ / 3D figures.

WHY THIS EXISTS, AND WHY IT IS NOT test_img.py
----------------------------------------------
test_img.py is a single-volume diagnostic swept over photon budgets. The paper's
Table 1 is a different object: metrics at five reporting depths, mean +/- sd over
the held-out test volumes, for one configuration. This script produces that, for
whichever checkpoint pair you point it at, so the f-only and (f, D) runs can be
put side by side.

WHAT IT MEASURES, AND ONE THING TO KNOW ABOUT IT
------------------------------------------------
Lateral resolution, RSE and RSP all come out of ONE optimisation, because they
are the same measurement read three ways. Following the manuscript:

    blur the GROUND TRUTH with a Gaussian of width sigma, rescale it linearly
    (alpha * blurred + beta) to best match the reconstruction, and take the
    sigma that minimises MSE.

    lateral resolution = 2.355 * sigma * dx      (Gaussian FWHM)
    RSE               = RMSE at that optimum
    RSP               = Pearson correlation at that optimum

alpha and beta are solved in closed form at every sigma, so the only search is
over sigma. That matches "linearly intensity matched" in the manuscript and makes
RSE / RSP resolution-scaled in the SQUIRREL sense rather than plain RMSE.

The caveat, which this script makes explicit: this measures how Gaussian-like the reconstruction's error is, not
optical resolution. If the reconstruction is SHARPER than the blurred ground
truth can get, the optimum sits at the bottom of the sigma grid and the number
is meaningless. Every such case is counted and printed as `sigma_at_floor`. If
that count is not ~0, do not quote the resolution column.

Also reported, because the manuscript has no axial metric at all:

    rho    Pearson correlation over the whole volume
    adj-Z  mean correlation between adjacent depth slices, against the
           volume's OWN ground-truth value. Positive gap = too smooth in z.

And per population, since the test set splits into two groups that behave
completely differently (see GROUP_HI_THRESHOLD in train_dp.py).

USAGE
-----
    python make_paper_table.py --tag learned_f
    python make_paper_table.py --tag learned_fD \
        --optics optics_model_fD.pth --recon recon_model_fD.pth
    python make_paper_table.py --tag learned_f --photons 10 30 100 300 1000

learn_D is detected from the optics checkpoint, so an (f, D) checkpoint needs no
flag. Outputs land in paper_table_<tag>/.
"""
import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d)
import tifffile as tiff
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.metrics import structural_similarity as compare_ssim

from config import config
import F_Optics_fixed as optics_mod
from F_Optics_fixed import F_Optics
from F_VCD_pytorch_multiple_views import F_Recon
from dataset import MyDataset
from adjoint import FLFMAdjointBackprojector
from sensor_noise import add_sensor_noise
from rltv import rl_tv_solve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA_DIR = config.paths["test_stack_dir"]
GROUP_HI_THRESHOLD = 2500.0
# Table columns. Overridden by --depth-count; the metrics themselves are
# computed on EVERY slice regardless (see --depths), so changing the table
# columns never needs a re-run of the expensive part.
REPORT_DEPTHS_UM = [-30.0, -15.0, 0.0, 15.0, 30.0]

# sigma search for the blur matching, in object pixels.
SIGMA_MIN, SIGMA_MAX, SIGMA_COARSE = 0.25, 14.0, 56      # coarse grid
SIGMA_REFINE = 21                                        # local refinement points

# Axial blur matching. Same procedure, 1D along z, so sigma is in SLICES and the
# FWHM comes out in microns through the 1.4286 um slice spacing. The grid stops
# at 6 slices (FWHM ~20 um) because beyond that the window itself is the limit.
SIGMA_Z_MIN, SIGMA_Z_MAX, SIGMA_Z_COARSE = 0.25, 6.0, 48
AXIAL_HALF_WINDOW = 7    # slices either side of the reported depth (+-10 um)
# Lateral subsampling for the AXIAL fit only. sigma_z is one scalar per depth,
# so it does not need all 512x512 pixels to estimate: measured against known
# blurs, stride 4 returns sigma identical to four decimals (1.0 -> 0.996,
# 2.5 -> 2.501, 4.0 -> 4.006, the same as stride 1) while running 17x faster
# -- 22.7 s per depth down to 1.3 s. Without it the axial metric costs ~30 min
# per arm on its own. The LATERAL fit is unaffected and still uses every pixel.
AXIAL_XY_STRIDE = 4

_NOISE = config.train_params["sensor_noise"]
READ_NOISE_E = _NOISE["read_noise_e"]
TRAIN_PHOTONS = _NOISE["photons"]
SEED = config.train_params.get("seed", 1234)


# ---------------------------------------------------------------- helpers
def normalize_sensor(x, eps=1e-8):
    """Byte-identical to train_dp.py. If you change one, change both."""
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def psnr(a, b):
    m = float(np.mean((a - b) ** 2))
    return float("inf") if m == 0 else 20.0 * math.log10(1.0 / math.sqrt(m))


def nrmse(gt, rec, eps=1e-8):
    """RMSE / std(GT), per slice -- same definition as train_dp.calculate_nrmse."""
    return float(np.sqrt(np.mean((gt - rec) ** 2)) / (gt.std() + eps))


def psnr_affine(gt, rec):
    """PSNR after the single global (alpha, beta) that minimises MSE.

    The scale-free PSNR. The learned arms' output is trained to sit on the
    target's own normalisation; RL-TV converges to an arbitrary intensity scale
    and gets min-max normalised instead. Raw PSNR therefore penalises RL-TV for
    a bookkeeping difference rather than for its reconstruction. This column
    removes that: it is the PSNR of the best affine rescaling of the
    reconstruction, so it measures structure only, and it is comparable across
    all arms. Resolution, RSP and RSE were already affine-invariant.
    """
    _, _, mse, _ = _fit_affine_mse(rec, gt)
    return float("inf") if mse <= 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def pearson(a, b, eps=1e-12):
    a = a.astype(np.float64).ravel(); a = a - a.mean()
    b = b.astype(np.float64).ravel(); b = b - b.mean()
    d = math.sqrt((a * a).sum()) * math.sqrt((b * b).sum()) + eps
    return float((a * b).sum() / d)


def adj_z(vol):
    """Mean correlation between adjacent depth slices of a [D,H,W] volume."""
    return float(np.mean([pearson(vol[z], vol[z + 1])
                          for z in range(vol.shape[0] - 1)]))


def _fit_affine_mse(blurred, target):
    """Least-squares alpha, beta for alpha*blurred + beta ~ target. Closed form."""
    x = blurred.ravel().astype(np.float64)
    y = target.ravel().astype(np.float64)
    xm, ym = x.mean(), y.mean()
    xc, yc = x - xm, y - ym
    denom = float((xc * xc).sum())
    if denom < 1e-20:
        return 0.0, ym, float(np.mean((y - ym) ** 2)), np.full_like(x, ym)
    alpha = float((xc * yc).sum() / denom)
    beta = float(ym - alpha * xm)
    fit = alpha * x + beta
    return alpha, beta, float(np.mean((fit - y) ** 2)), fit


def blur_match(gt_slice, rec_slice, dx_um):
    """Gaussian blur matching, as the manuscript defines it.

    Blurs the GROUND TRUTH toward the reconstruction and returns
    (fwhm_um, sigma_px, rse, rsp, at_floor).

    at_floor=True means the optimum landed on the smallest sigma tried, i.e. the
    reconstruction is at least as sharp as the metric can represent and the
    number is not interpretable.
    """
    def score(sig):
        b = gaussian_filter(gt_slice.astype(np.float64), sigma=float(sig),
                            mode="nearest")
        _, _, mse, fit = _fit_affine_mse(b, rec_slice)
        return mse, fit

    grid = np.linspace(SIGMA_MIN, SIGMA_MAX, SIGMA_COARSE)
    mses = np.array([score(s)[0] for s in grid])
    k = int(np.argmin(mses))

    # local refinement around the coarse winner
    step = grid[1] - grid[0]
    lo = max(SIGMA_MIN, grid[k] - step)
    hi = min(SIGMA_MAX, grid[k] + step)
    fine = np.linspace(lo, hi, SIGMA_REFINE)
    fmses = np.array([score(s)[0] for s in fine])
    j = int(np.argmin(fmses))
    sigma = float(fine[j])

    _, fit = score(sigma)
    rse = float(np.sqrt(fmses[j]))
    rsp = pearson(fit.reshape(rec_slice.shape), rec_slice)
    at_floor = sigma <= SIGMA_MIN + 1e-6
    return 2.355 * sigma * dx_um, sigma, rse, rsp, at_floor


def blur_match_axial(gt_vol, rec_vol, z0, dz_um, half=AXIAL_HALF_WINDOW):
    """Axial blur matching -- the z analogue of blur_match.

    Identical procedure, one dimension over: the GROUND TRUTH is blurred ALONG z
    with a 1D Gaussian, linearly intensity matched to the reconstruction, and the
    sigma minimising MSE is converted to an axial FWHM through the slice spacing.
    Using the same estimator laterally and axially is the point -- the two numbers
    are then directly comparable, and the manuscript only has to describe one
    method.

    Depth-resolved: the fit is scored only inside a window of +-`half` slices
    around z0, so each reported depth gets its own number instead of one figure
    for the whole volume.

    The blur is applied to the WHOLE volume and only the SCORE is windowed. That
    distinction matters and was measured: blurring inside the window instead lets
    its edges see `mode="nearest"` padding rather than the neighbouring slices,
    which biases sigma LOW by 11% at sigma_z = 3.5 slices and 20% at 5 -- exactly
    the wide-blur regime these reconstructions sit in, so the error would have
    flattered every arm. Blurring the full volume recovers the true sigma to
    within 0.5% at every sigma and at every depth, edges included.

    Returns (fwhm_um, sigma_slices, at_floor).
    """
    D = gt_vol.shape[0]
    a, b = max(0, z0 - half), min(D, z0 + half + 1)
    s = AXIAL_XY_STRIDE
    g = gt_vol[:, ::s, ::s].astype(np.float64)
    r = rec_vol[a:b, ::s, ::s]

    def score(sig):
        bl = gaussian_filter1d(g, sigma=float(sig), axis=0, mode="nearest")
        _, _, mse, _ = _fit_affine_mse(bl[a:b], r)
        return mse

    grid = np.linspace(SIGMA_Z_MIN, SIGMA_Z_MAX, SIGMA_Z_COARSE)
    mses = np.array([score(s) for s in grid])
    k = int(np.argmin(mses))
    step = grid[1] - grid[0]
    fine = np.linspace(max(SIGMA_Z_MIN, grid[k] - step),
                       min(SIGMA_Z_MAX, grid[k] + step), SIGMA_REFINE)
    j = int(np.argmin([score(s) for s in fine]))
    sigma = float(fine[j])
    return 2.355 * sigma * dz_um, sigma, sigma <= SIGMA_Z_MIN + 1e-6


def infer_learn_D(sd):
    return any(k.endswith("D_map") or k == "D_map" for k in sd)


# ---------------------------------------------------------------- figures
def fig_xz_yz(gt, rec, dx_um, z_um, outdir, tag):
    """GT vs reconstruction, central XZ and YZ cross-sections."""
    y0, x0 = gt.shape[1] // 2, gt.shape[2] // 2
    for axis, fname in (("XZ", "xz_sections"), ("YZ", "yz_sections")):
        g = gt[:, y0, :] if axis == "XZ" else gt[:, :, x0]
        r = rec[:, y0, :] if axis == "XZ" else rec[:, :, x0]
        lat_um = g.shape[1] * dx_um
        extent = [0, lat_um, z_um[-1], z_um[0]]
        vmax = float(max(g.max(), r.max()))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for a, img, ttl in ((ax[0], g, "Ground truth"), (ax[1], r, f"Reconstruction {tag}")):
            im = a.imshow(img, cmap="magma", vmin=0, vmax=vmax,
                          extent=extent, aspect="auto")
            a.set_title(f"{ttl} — {axis}")
            a.set_xlabel("x (µm)" if axis == "XZ" else "y (µm)")
            a.set_ylabel("z (µm)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.savefig(outdir / f"{fname}.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / f"{fname}.svg", bbox_inches="tight")
        plt.close(fig)


def fig_orthoview(gt, rec, dx_um, z_um, outdir):
    """Max-intensity projections along all three axes, GT above reconstruction.

    This is the orthoview microscopy papers normally show, and it is the honest
    way to display a 60 um x 193 um x 193 um volume: the axial extent is 3x
    smaller than the lateral one, so a cube rendering misleads.
    """
    lat = gt.shape[1] * dx_um
    dz = abs(z_um[-1] - z_um[0])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for row, (vol, lbl) in enumerate(((gt, "Ground truth"), (rec, "Reconstruction"))):
        mips = [vol.max(0), vol.max(1), vol.max(2)]
        exts = [[0, lat, lat, 0], [0, lat, z_um[-1], z_um[0]], [0, lat, z_um[-1], z_um[0]]]
        names = ["XY (max over z)", "XZ (max over y)", "YZ (max over x)"]
        for c in range(3):
            a = axes[row, c]
            a.imshow(mips[c], cmap="magma", extent=exts[c], aspect="auto")
            a.set_title(f"{lbl} — {names[c]}", fontsize=10)
            a.set_xlabel("x (µm)" if c != 2 else "y (µm)")
            a.set_ylabel("y (µm)" if c == 0 else "z (µm)")
    fig.suptitle(f"lateral {lat:.0f} × {lat:.0f} µm, axial {dz:.0f} µm", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "orthoview_mip.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "orthoview_mip.svg", bbox_inches="tight")
    plt.close(fig)


def fig_volume_3d(gt, rec, dx_um, z_um, outdir, percentile=99.5, max_pts=60000):
    """3D scatter of the brightest voxels, GT beside reconstruction.

    Only voxels above `percentile` are drawn, subsampled to max_pts, coloured by
    depth. A threshold is unavoidable for a 11M-voxel volume; it is stated in the
    title so the figure cannot be read as a full rendering. For a publication
    figure, the TIFFs this script writes will render better in napari or
    ImageJ's 3D viewer than matplotlib can.
    """
    fig = plt.figure(figsize=(13, 5.6))
    for i, (vol, lbl) in enumerate(((gt, "Ground truth"), (rec, "Reconstruction"))):
        thr = np.percentile(vol, percentile)
        zi, yi, xi = np.nonzero(vol > thr)
        if zi.size > max_pts:
            sel = np.random.default_rng(0).choice(zi.size, max_pts, replace=False)
            zi, yi, xi = zi[sel], yi[sel], xi[sel]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if zi.size:
            ax.scatter(xi * dx_um, yi * dx_um, z_um[zi], c=z_um[zi],
                       cmap="magma", s=1.2, alpha=0.35, linewidths=0)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
        ax.set_zlim(z_um[0], z_um[-1])
        ax.set_title(f"{lbl}  (> {percentile}th pct)", fontsize=10)
        ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    fig.savefig(outdir / "volume_3d.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "volume_3d.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- latex
def write_latex(depth_df, path, tag, photons):
    """Table 1 in the manuscript's layout: metric rows x depth columns."""
    metrics = [("lat_res_um", "Lat. res. (µm)", "{:.2f}"),
               ("axial_res_um", "Ax. res. (µm)", "{:.2f}"),
               ("psnr_db",    "PSNR (dB)",      "{:.2f}"),
               ("psnr_db_affine", "PSNR, scale-free (dB)", "{:.2f}"),
               ("ssim",       "SSIM",           "{:.3f}"),
               ("nrmse",      "NRMSE",          "{:.3f}"),
               ("rsp",        "RSP",            "{:.3f}"),
               ("rse",        "RSE",            "{:.4f}")]
    depths = list(depth_df["depth_um"])
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Depth-dependent reconstruction metrics, {tag}, "
             rf"$N = {photons}$ photons. Mean $\pm$ s.d. over "
             rf"{int(depth_df['n_volumes'].iloc[0])} held-out test volumes.}}",
             r"\begin{tabular}{l" + "c" * len(depths) + "}", r"\toprule",
             "Metric & " + " & ".join(rf"${d:+.0f}~\mu$m" for d in depths) + r" \\",
             r"\midrule"]
    for key, label, fmt in metrics:
        cells = []
        for _, row in depth_df.iterrows():
            m, s = row[f"{key}_mean"], row[f"{key}_sd"]
            cells.append(f"${fmt.format(m)} \\pm {fmt.format(s)}$")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--optics", default="optics_model.pth")
    ap.add_argument("--recon", default="recon_model.pth",
                    help="ignored when --method rltv")
    ap.add_argument("--method", choices=["cnn", "rltv"], default="cnn",
                    help="'cnn' (default) = CG lift + learned post-processing, "
                         "the learned arms. 'rltv' = Richardson-Lucy + TV on "
                         "the same forward operator, no network and no "
                         "training: the model-based baseline. --recon is not "
                         "read in that case.")
    ap.add_argument("--rltv-iters", type=int, default=100,
                    help="RL-TV iterations (submitted paper: 100)")
    ap.add_argument("--rltv-tv", type=float, default=1e-5,
                    help="RL-TV total-variation weight (submitted paper: 1e-5)")
    ap.add_argument("--rltv-norm", choices=["percentile", "minmax"],
                    default="percentile",
                    help="'percentile' (default) puts RL-TV on the same scale "
                         "as the learned arms, which is what makes PSNR and "
                         "SSIM comparable. 'minmax' reproduces solver.py and "
                         "is destroyed by a single outlier voxel.")
    ap.add_argument("--rltv-margin", type=int, default=0,
                    help="extra pixels solved on each side and then discarded, "
                         "the analogue of solver.py solving at sensor size and "
                         "cropping. 0 keeps the operator identical to the "
                         "learned arms'.")
    ap.add_argument("--rltv-sens-floor", type=float, default=0.0,
                    help="floor on A^T(1) relative to its max. Off by default: "
                         "it was a guess at the cause of the corner spikes and "
                         "a toy reproduction says it is not the cause.")
    ap.add_argument("--rltv-tv-sign", type=float, default=1.0,
                    help="+1 reproduces the submitted solver.py, whose TV term "
                         "is signed so that it sharpens rather than smooths "
                         "(negligible at 1e-5). -1 is the intended sign.")
    ap.add_argument("--photons", type=float, nargs="*", default=None,
                    help="photon budgets; default = the training budget only")
    ap.add_argument("--max-volumes", type=int, default=None)
    ap.add_argument("--figure-photons", default="first",
                    help="'first' (default) saves figures/TIFFs only at the "
                         "first budget; 'all' saves them at every budget, "
                         "which is what the qualitative sweep grid needs. "
                         "Each budget costs another set of volumes on disk, "
                         "so 'all' with 5 budgets is 5x the disk.")
    ap.add_argument("--figure-volumes", type=int, nargs="*", default=[1, 3],
                    help="volume indices to render XZ/YZ/3D figures for. "
                         "Default 1 and 3: volume 1 is group B (sparse, bright) "
                         "and volume 3 is group A (dense, diffuse), so the pair "
                         "shows one of each population rather than two of the "
                         "same kind.")
    ap.add_argument("--fixed-optics", action="store_true",
                    help="cosmetic: this checkpoint is the conventional MLA arm")
    ap.add_argument("--depth-count", type=int, default=9,
                    help="how many evenly spaced depths the TABLE reports "
                         "(default 9; use 5 for the manuscript's layout, 7, 9, "
                         "or 43 for every slice). Odd numbers keep z=0 centred.")
    ap.add_argument("--out-dir", default=".",
                    help="parent directory for paper_table_<tag>/. Use "
                         "separate directories for runs made with different "
                         "settings, or the later one overwrites the earlier.")
    ap.add_argument("--axial-depths", choices=["table", "all"], default="table",
                    help="where to measure AXIAL resolution. 'table' (default) "
                         "does the reported depths only; 'all' does every "
                         "measured slice, which is what the depth-dependence "
                         "figure wants. Each axial fit runs its own sigma "
                         "search over a 15-slice window, so 'all' costs about "
                         "+45 s per volume per photon budget -- roughly 10 "
                         "extra minutes per arm for 12 volumes at one budget, "
                         "and five times that across a 5-point photon sweep.")
    ap.add_argument("--depths", choices=["all", "table"], default="all",
                    help="'all' (default) computes metrics on every one of the "
                         "43 slices, so per_slice.csv is the full "
                         "resolution-vs-depth curve and the table can be "
                         "re-cut without re-running. 'table' computes only the "
                         "reported depths, which is ~8x faster.")
    ap.add_argument("--tiff", choices=["all", "min", "none"], default="min",
                    help="volumes to save per figure volume. 'min' (default) = "
                         "gt + reconstruction; 'all' adds the coarse lift; "
                         "'none' = figures only. One volume is 45 MB.")
    args = ap.parse_args()

    # --out-dir keeps runs with different settings apart. The depth figure
    # wants axial at all 43 slices at one budget; the photon sweep wants 5
    # budgets and no axial at all. Sharing one directory would mean the second
    # run silently overwrites the first's per_slice.csv with coarser axial
    # data, and the depth figure would quietly lose 36 of its 43 points.
    out = Path(args.out_dir) / f"paper_table_{args.tag}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    budgets = [float(p) for p in (args.photons or [TRAIN_PHOTONS])]

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- models. learn_D read off the checkpoint, so no flag is needed.
    sd_o = torch.load(args.optics, map_location="cpu")
    sd_o = sd_o.get("state_dict", sd_o) if isinstance(sd_o, dict) else sd_o
    learn_D = infer_learn_D(sd_o)
    # init_f does not matter here: f_rand is overwritten by load_state_dict
    # either way. It is passed only so the constructor prints something
    # consistent with the arm.
    model_optics = F_Optics(config.optics_params, seed=SEED, learn_D=learn_D,
                            init_f="nominal" if args.fixed_optics else "random").to(DEVICE)
    model_optics.load_state_dict(sd_o, strict=True)
    model_optics.eval()

    # The network is not loaded at all for the RL-TV baseline -- that arm has
    # no learned stage, which is the whole point of reporting it.
    model_recon = None
    if args.method == "cnn":
        model_recon = F_Recon(**config.recon_params).to(DEVICE)
        sd_r = torch.load(args.recon, map_location="cpu")
        model_recon.load_state_dict(
            sd_r.get("state_dict", sd_r) if isinstance(sd_r, dict) else sd_r,
            strict=True)
        model_recon.eval()

    bp = FLFMAdjointBackprojector(
        out_hw=(config.recon_params["output_size"][0],
                config.recon_params["output_size"][1]),
        normalize="percentile", p_low=10.0, p_high=99.7,
        deconv="cg", cg_reg=3.0, cg_iters=10, cg_chunk=4, cg_cache_H=True,
    ).to(DEVICE)

    ds = MyDataset(stack_dir=TEST_DATA_DIR, config=config,
                   transform_3d=config.transform_3d)
    n_vol = len(ds) if args.max_volumes is None else min(len(ds), args.max_volumes)

    dx_um = float(config.optics_params["object_pixel_size"]) * 1e6
    zmin = float(config.optics_params["volume_depth_min"]) * 1e6
    zmax = float(config.optics_params["volume_depth_max"]) * 1e6

    print(f"[{args.tag}] learn_D={learn_D}  pupil="
          f"{getattr(optics_mod, 'PUPIL_MODEL', 'n/a')}  volumes={n_vol}")
    print(f"[{args.tag}] trained at {TRAIN_PHOTONS:.0f} photons, "
          f"testing at {budgets}, read noise {READ_NOISE_E} e-")
    print(f"[{args.tag}] object pixel {dx_um:.4f} µm, z {zmin:+.1f}..{zmax:+.1f} µm")

    # Disk pre-check. A full disk is what killed this script's first real run:
    # it died writing a 45 MB volume on the second test volume, after the arm
    # had already trained for 20 hours. Say so up front instead.
    _per_vol_mb = 43 * 512 * 512 * 4 / 1e6                     # one float32 volume
    _n_tiff = {"all": 3, "min": 2, "none": 0}[args.tiff]
    _n_budget_fig = len(budgets) if args.figure_photons == "all" else 1
    _need_mb = (len(args.figure_volumes) * _n_budget_fig
                * (_n_tiff * _per_vol_mb + 25))        # +figures
    _free_mb = os.statvfs(".").f_bavail * os.statvfs(".").f_frsize / 1e6
    print(f"[{args.tag}] disk: need ~{_need_mb:.0f} MB for figures/TIFFs "
          f"(--tiff {args.tiff}), {_free_mb:.0f} MB free")
    if _need_mb > 0.9 * _free_mb:
        print(f"[{args.tag}] !! not enough room. The METRICS will still be "
              f"written (they are tiny); figures will be skipped as they fail.")
        print(f"[{args.tag}] !! free space, or re-run with --tiff none.")

    slice_rows, vol_rows = [], []
    floor_hits = 0
    floor_total = 0
    axial_floor_hits = 0
    axial_floor_total = 0
    # slice spacing, the axial counterpart of dx_um. 60 um over 42 gaps = 1.4286.
    dz_um = (zmax - zmin) / (config.optics_params["n_slices"] - 1)

    for idx in range(n_vol):
        target3d, lo, hi = ds[idx]
        target3d = target3d.to(DEVICE)
        group = "A" if float(hi) < GROUP_HI_THRESHOLD else "B"

        with torch.no_grad():
            out_optics, _pitch, psf_stack = model_optics(target3d)
            sensor_clean = normalize_sensor(out_optics)

        gt = target3d.squeeze().detach().cpu().numpy().astype(np.float32)
        D = gt.shape[0]
        z_um = np.linspace(zmin, zmax, D)

        # Which slices to measure. The reporting depths are the nearest slices
        # to an evenly spaced set across the volume; with --depths all we
        # measure every slice instead and select the table columns afterwards,
        # which costs ~8x the metric time but makes the table re-cuttable and
        # gives the full resolution-vs-depth curve in per_slice.csv.
        table_targets = np.linspace(zmin, zmax, max(2, args.depth_count))
        table_idx = sorted({int(np.argmin(np.abs(z_um - t))) for t in table_targets})
        measure_idx = list(range(D)) if args.depths == "all" else table_idx
        if idx == 0:
            print(f"[{args.tag}] table depths (µm): "
                  f"{[round(float(z_um[i]), 1) for i in table_idx]}")
            print(f"[{args.tag}] measuring {len(measure_idx)} of {D} slices "
                  f"per volume (--depths {args.depths})")

        for photons in budgets:
            torch.manual_seed(SEED + int(photons) + 1000 * idx)
            with torch.no_grad():
                noisy = add_sensor_noise(sensor_clean, photons,
                                         read_noise_e=READ_NOISE_E, exact=True)
                if args.method == "rltv":
                    # No lift and no network: RL-TV iterates on the same A/A^T
                    # pair the CG lift uses. coarse_np stays None, so the
                    # lift-reference columns are NaN for this arm rather than
                    # silently borrowing another arm's numbers.
                    rec_t = rl_tv_solve(
                        noisy, psf_stack,
                        out_hw=(config.recon_params["output_size"][0],
                                config.recon_params["output_size"][1]),
                        n_iter=args.rltv_iters, tv_lambda=args.rltv_tv,
                        tv_sign=args.rltv_tv_sign,
                        margin=args.rltv_margin,
                        sens_floor=args.rltv_sens_floor,
                        normalize=args.rltv_norm,
                        verbose=(idx == 0 and photons == budgets[0]))
                    coarse = coarse_np = None
                else:
                    coarse = bp(sensor_img=noisy, psf_stack=psf_stack,
                                dz=model_optics.dz_used)
                    rec_t = model_recon(coarse)
            rec = rec_t.squeeze().detach().cpu().numpy().astype(np.float32)
            rec = np.clip(rec, 0.0, 1.0)
            if coarse is not None:
                coarse_np = coarse.squeeze().detach().cpu().numpy().astype(np.float32)

            vol_rows.append({
                "volume": idx, "group": group, "hi": float(hi), "photons": photons,
                "method": args.method,
                "rho": pearson(rec, gt),
                "rho_lift": np.nan if coarse_np is None else pearson(coarse_np, gt),
                "adjz_rec": adj_z(rec), "adjz_gt": adj_z(gt),
                "adjz_lift": np.nan if coarse_np is None else adj_z(coarse_np),
                "psnr_db": psnr(gt, rec),
                "psnr_db_affine": psnr_affine(gt, rec),
                "ssim": float(np.mean([
                    compare_ssim(gt[d], rec[d],
                                 data_range=max(float(gt[d].max() - gt[d].min()), 1e-8))
                    for d in range(D)])),
            })
            vol_rows[-1]["adjz_gap"] = vol_rows[-1]["adjz_rec"] - vol_rows[-1]["adjz_gt"]

            for d in measure_idx:
                g, r = gt[d], rec[d]
                rng = float(g.max() - g.min())
                fwhm, sig, rse, rsp, at_floor = blur_match(g, r, dx_um)
                floor_hits += int(at_floor); floor_total += 1
                # Axial resolution needs its own sigma search over a 15-slice
                # window, so by default it runs only at the reported depths.
                # --axial-depths all pays that cost at every measured slice,
                # which is what a depth-dependence curve needs.
                if args.axial_depths == "all" or d in table_idx:
                    ax_fwhm, ax_sig, ax_floor = blur_match_axial(
                        gt, rec, d, dz_um)
                    axial_floor_hits += int(ax_floor)
                    axial_floor_total += 1
                else:
                    ax_fwhm, ax_sig, ax_floor = np.nan, np.nan, False
                slice_rows.append({
                    "volume": idx, "group": group, "photons": photons,
                    "depth_um": float(z_um[d]), "z_index": d,
                    "in_table": d in table_idx,
                    "axial_res_um": ax_fwhm, "sigma_z_slices": ax_sig,
                    "axial_at_floor": ax_floor,
                    "lat_res_um": fwhm, "sigma_px": sig, "sigma_at_floor": at_floor,
                    "rse": rse, "rsp": rsp,
                    "psnr_db": psnr(g, r),
                    "psnr_db_affine": psnr_affine(g, r),
                    "ssim": np.nan if rng < 1e-8 else compare_ssim(g, r, data_range=rng),
                    "nrmse": nrmse(g, r),
                })

            # Figures and volumes, for the chosen volumes at the training budget.
            #
            # Wrapped in try/except because this is the only part of the script
            # that writes hundreds of MB, and a full disk here used to kill the
            # whole table -- losing every metric already computed, plus the
            # remaining arms of the queue. The numbers are the deliverable; the
            # TIFFs are a convenience. One failed figure must not cost a 20-hour
            # training run's evaluation.
            _want_fig = (args.figure_photons == "all"
                         or photons == budgets[0])
            if idx in args.figure_volumes and _want_fig:
                try:
                    sub = out / f"vol{idx:04d}_N{int(photons):04d}"
                    sub.mkdir(exist_ok=True)
                    fig_xz_yz(gt, rec, dx_um, z_um, sub, f"(N={int(photons)})")
                    fig_orthoview(gt, rec, dx_um, z_um, sub)
                    fig_volume_3d(gt, rec, dx_um, z_um, sub)
                    if args.tiff != "none":
                        tiff.imwrite(sub / "gt.tif", gt)
                        tiff.imwrite(sub / "reconstruction.tif", rec)
                        if args.tiff == "all" and coarse_np is not None:
                            tiff.imwrite(sub / "coarse_lift.tif", coarse_np)
                    print(f"  figures{'' if args.tiff == 'none' else ' + TIFFs'} "
                          f"for volume {idx} (group {group}) -> {sub}")
                except Exception as e:
                    print(f"  !! figures for volume {idx} FAILED ({type(e).__name__}: "
                          f"{e}) -- metrics are unaffected, continuing")

        del target3d, out_optics, psf_stack, sensor_clean
        torch.cuda.empty_cache()
        print(f"  {idx+1}/{n_vol} (group {group})", end="\r")

    print(" " * 30)
    slice_df = pd.DataFrame(slice_rows)
    vol_df = pd.DataFrame(vol_rows)
    slice_df.to_csv(out / "per_slice.csv", index=False)
    vol_df.to_csv(out / "per_volume.csv", index=False)

    # ---- the depth table, per photon budget
    keys = ["lat_res_um", "axial_res_um", "psnr_db", "psnr_db_affine",
            "ssim", "nrmse", "rsp", "rse"]
    for photons in budgets:
        s = slice_df[slice_df["photons"] == photons]
        tbl = s[s["in_table"]]
        depths = sorted(tbl["depth_um"].unique())
        recs = []
        for depth_um in depths:
            d = tbl[tbl["depth_um"] == depth_um]
            r = {"depth_um": depth_um, "n_volumes": len(d)}
            for k in keys:
                r[f"{k}_mean"] = d[k].mean()
                r[f"{k}_sd"] = d[k].std(ddof=0)
            recs.append(r)
        depth_df = pd.DataFrame(recs)
        depth_df.to_csv(out / f"depth_table_N{int(photons):04d}.csv", index=False)
        write_latex(depth_df, out / f"depth_table_N{int(photons):04d}.tex",
                    args.tag, int(photons))

        print(f"\n=== {args.tag}, N = {int(photons)} photons, "
              f"{len(s['volume'].unique())} volumes, "
              f"{len(depths)} reported depths ===")
        hdr = f"{'metric':12s}" + "".join(f"{d:>+13.0f}" for d in depths)
        print(hdr); print("-" * len(hdr))
        for k in keys:
            line = f"{k:12s}"
            for depth_um in depths:
                row = depth_df[depth_df["depth_um"] == depth_um].iloc[0]
                line += f"{row[f'{k}_mean']:>8.3f}±{row[f'{k}_sd']:<4.2f}"
            print(line)

        # Depth uniformity, which is the claim the manuscript now rests on.
        # Quoted over the REPORTED depths so it matches the table the reader
        # sees, not over all 43 slices.
        lr = depth_df["lat_res_um_mean"]
        print(f"\nlateral resolution: mean {lr.mean():.3f} µm, "
              f"best {lr.min():.3f}, worst {lr.max():.3f}, "
              f"spread {lr.max()/max(lr.min(), 1e-9):.2f}x")

        # The full curve, when every slice was measured: this is the
        # resolution-vs-depth figure, and it is the honest version of the
        # 5-point table -- the table can only show what its columns land on.
        if args.depths == "all":
            curve = (s.groupby("depth_um")[["lat_res_um", "psnr_db", "ssim"]]
                      .agg(["mean", "std"]))
            curve.to_csv(out / f"resolution_vs_depth_N{int(photons):04d}.csv")
            try:
                m = s.groupby("depth_um")["lat_res_um"].mean()
                sd = s.groupby("depth_um")["lat_res_um"].std(ddof=0)
                fig, ax = plt.subplots(figsize=(7, 4.4))
                ax.plot(m.index, m.values, "-o", ms=3, lw=1.8, color="#1a1a1a",
                        label=args.tag)
                ax.fill_between(m.index, m - sd, m + sd, alpha=0.18,
                                color="#1a1a1a", lw=0)
                ax.plot(depth_df["depth_um"], depth_df["lat_res_um_mean"], "s",
                        ms=8, mfc="none", mec="#ea580c", mew=2,
                        label="reported depths")
                ax.set_xlabel("depth z (µm)")
                ax.set_ylabel("effective lateral resolution (µm)")
                ax.set_title(f"{args.tag}, N = {int(photons)} photons, "
                             f"{len(s['volume'].unique())} volumes "
                             f"(mean ± s.d.)")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(out / f"resolution_vs_depth_N{int(photons):04d}.png",
                            dpi=300)
                fig.savefig(out / f"resolution_vs_depth_N{int(photons):04d}.svg")
                plt.close(fig)
            except Exception as e:
                print(f"  !! resolution-vs-depth figure failed ({e})")

        v = vol_df[vol_df["photons"] == photons]
        print(f"\nvolume-level:  rho {v['rho'].mean():.4f}  "
              f"(lift {v['rho_lift'].mean():.4f})   "
              f"PSNR {v['psnr_db'].mean():.2f} / scale-free "
              f"{v['psnr_db_affine'].mean():.2f} dB   "
              f"adj-Z {v['adjz_rec'].mean():.4f} vs GT {v['adjz_gt'].mean():.4f}  "
              f"gap {v['adjz_gap'].mean():+.4f}")
        for g in sorted(v["group"].unique()):
            vg = v[v["group"] == g]
            print(f"  group {g} (n={len(vg)}):  rho {vg['rho'].mean():.4f}   "
                  f"SSIM {vg['ssim'].mean():.4f}   "
                  f"PSNR {vg['psnr_db'].mean():.2f}   "
                  f"adj-Z gap {vg['adjz_gap'].mean():+.4f}")

    # ---- the honesty check on the resolution column
    fa = axial_floor_hits / max(axial_floor_total, 1)
    print(f"axial sigma_at_floor: {axial_floor_hits}/{axial_floor_total} "
          f"({100*fa:.1f}%) hit the axial grid floor "
          f"({SIGMA_Z_MIN} slices = {2.355*SIGMA_Z_MIN*dz_um:.2f} µm FWHM). "
          f"The ground truth is itself sampled at {dz_um:.3f} µm, so an axial "
          f"figure near that floor means 'as sharp as the data allows', not a "
          f"measured resolution.")
    frac = floor_hits / max(floor_total, 1)
    print(f"\nsigma_at_floor: {floor_hits}/{floor_total} ({100*frac:.1f}%) "
          f"of blur-matching fits hit the sigma grid floor ({SIGMA_MIN} px).")
    if frac > 0.02:
        print("  !! The reconstruction is sharper than the blurred ground truth in")
        print("     those cases, so the lateral-resolution number is not")
        print("     interpretable there. Do not quote that column without saying so.")

    print(f"\nWrote {out}/  (per_slice.csv, per_volume.csv, depth_table_*.csv/.tex)")
    print("For a publication 3D figure, open the saved gt.tif / reconstruction.tif")
    print("in napari or ImageJ's 3D viewer -- better than the matplotlib scatter.")


if __name__ == "__main__":
    main()
