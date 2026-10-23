"""
random_vs_learned_psf.py -- is the learned focal map better than a random one?

    python random_vs_learned_psf.py --root .. --seeds 12 \
        --learned ../optics_model_f.pth --out random_vs_learned

WHY THIS CONTROL MATTERS

The prior art on Fourier-plane encoding diversity gets its benefit from
RANDOMISING the focal lengths: Fourier DiffuserScope uses randomly positioned
microlenses with varied focal lengths, and the resulting focal diversity is
what extends the axial range. So a reader is entitled to ask whether the
optimization here is doing anything a random draw would not, and the paper
cannot answer that by comparing against the uniform baseline alone -- the
uniform MLA has NO focal diversity, so beating it shows only that diversity
helps, not that LEARNED diversity helps.

This script draws K random focal maps from the same box the optimizer
searched, [f_min, f_max], measures the same two optical quantities on each,
and places the learned map against that distribution as a z-score. The
interpretation is the one random_mask_baseline.py already uses:

    |z| > 2   the learned map is outside the random distribution
    |z| < 1   the gap is seed noise; any map in this range does as well

PURELY OPTICAL, DELIBERATELY

Everything here is measured on the PSF stack, with no reconstruction network
in the loop. That is a limitation -- it is not the end-to-end resolution the
paper reports -- but it is also the only honest cheap comparison available: a
reconstruction-level comparison would need a network trained per random draw,
which is weeks of compute. Measuring the optics alone has the compensating
virtue of isolating the encoder from the reconstruction entirely, so nothing
here can be explained away by the network having adapted to one encoder.

THE TWO METRICS, per depth

  lateral   the effective width of the PSF, sqrt of its equivalent area,
            w = sqrt( (sum h)^2 / sum h^2 ), converted to object space.
            Equivalent area needs no spot segmentation and is monotone in
            blur. The lenslet positions are identical across arms, so
            differences in w reflect how each spot blurs with depth rather
            than how the spots are arranged.

  axial     the depth separation at which the PSF decorrelates to 0.5,
            interpolated between slices, measured in a window around each
            depth so that every depth gets its own value rather than one
            figure for the whole stack. This is the metric
            psf_depth_rank.py uses, made depth-resolved.

From each curve it also reports the SPREAD, max over min across depth. That
is the optical analogue of the depth spread the paper reports on
reconstructions, and it is the quantity the learned design is supposed to
improve, so it is the one to look at first.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from F_Optics_fixed import F_Optics
from dataset import MyDataset


def psf_stack(m, target3d):
    with torch.no_grad():
        _, _, psf = m(target3d)
    return psf.detach().float()


def lateral_width(psf, dx_obj_um, stride, n_views):
    """Effective PSF width per depth, per view, in object-space micrometres.

    The equivalent area of the whole stack is the sum over the n_views
    lenslet spots, so for spots of similar size it scales as n_views times
    the area of one. Dividing the width by sqrt(n_views) therefore reports a
    PER-VIEW width, which is the quantity comparable to a resolution; the
    raw figure is about five times larger here and is not a resolution at
    all. The normalisation is a constant, so it changes none of the
    comparisons below, only the units they are read in."""
    h = psf[..., ::stride, ::stride].double()
    h = h.reshape(h.shape[-3], -1)
    h = h - h.min(dim=1, keepdim=True).values          # kill any pedestal
    s1 = h.sum(dim=1)
    s2 = (h * h).sum(dim=1)
    area_px = (s1 * s1 / s2.clamp_min(1e-30))          # equivalent area
    w = area_px.sqrt().cpu().numpy() * dx_obj_um * stride
    return w / np.sqrt(max(n_views, 1))


def axial_profile(psf, dz_um, stride, half=7, level=0.5):
    """Decorrelation depth per plane, in micrometres."""
    h = psf[..., ::stride, ::stride].double()
    D = h.shape[-3]
    h = h.reshape(D, -1)
    h = h - h.mean(dim=1, keepdim=True)
    h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = (h @ h.T).cpu().numpy()
    del h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = np.full(D, np.nan)
    for i in range(D):
        # look both ways, so a plane near an edge is not penalised for having
        # fewer neighbours on one side
        prof = [1.0]
        for k in range(1, half + 1):
            vals = []
            if i - k >= 0:
                vals.append(G[i, i - k])
            if i + k < D:
                vals.append(G[i, i + k])
            if not vals:
                break
            prof.append(float(np.mean(vals)))
        for k in range(1, len(prof)):
            if prof[k] < level:
                f = (prof[k - 1] - level) / max(prof[k - 1] - prof[k], 1e-12)
                out[i] = (k - 1 + f) * dz_um
                break
    return out


def spread(v):
    v = v[np.isfinite(v)]
    return float(np.nanmax(v) / max(np.nanmin(v), 1e-12)) if v.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--learned", default="../optics_model_f.pth",
                    help="checkpoint of the learned f encoder")
    ap.add_argument("--learned-fD", default=None,
                    help="optional, adds the f,D ablation as a third line")
    ap.add_argument("--seeds", type=int, default=12,
                    help="number of random focal maps to draw")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--sample-idx", type=int, default=8,
                    help="the PSFs do not depend on the sample; this only "
                         "supplies an input of the right shape")
    ap.add_argument("--data-dir",
                    default=config.paths["test_stack_dir"])
    ap.add_argument("--n-views", type=int, default=None,
                    help="lenslets in the illuminated subarray, for the "
                         "per-view normalisation of the lateral width. "
                         "Defaults to mla_size squared.")
    ap.add_argument("--stride", type=int, default=2,
                    help="spatial stride when forming the PSF correlations. "
                         "The stack is several GB at full resolution.")
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"])
    ap.add_argument("--out", default="random_vs_learned")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op = config.optics_params
    dz_um = ((op["volume_depth_max"] - op["volume_depth_min"]) * 1e6
             / (int(op["n_slices"]) - 1))
    dx_obj_um = op["psf_sampling_sp"] * 1e6 / op["M_T"]
    z_um = np.linspace(op["volume_depth_min"] * 1e6,
                       op["volume_depth_max"] * 1e6, int(op["n_slices"]))
    n_views = args.n_views or int(op["microlens"].get("mla_size",
                                  round(op["microlens"]["array_size"]))) ** 2
    seed0 = config.train_params.get("seed", 1234)
    print(f"device {device} | dz {dz_um:.3f} um | object pixel "
          f"{dx_obj_um:.4f} um | {len(z_um)} depths | {n_views} views")

    ds = MyDataset(stack_dir=args.data_dir, config=config,
                   transform_3d=config.transform_3d)
    target3d = ds[args.sample_idx][0].to(device)

    def measure(label, seed=None, ckpt=None, init_f="random", learn_D=False):
        # learn_D has to match the checkpoint: an f,D run stores a D_map that
        # a learn_D=False model has no slot for, and load_state_dict refuses it
        m = F_Optics(op, seed=seed, init_f=init_f, learn_D=learn_D).to(device)
        if ckpt:
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            m.load_state_dict(ck.get("state_dict", ck)
                              if isinstance(ck, dict) else ck, strict=True)
        m.eval()
        psf = psf_stack(m, target3d)
        lat = lateral_width(psf, dx_obj_um, args.stride, n_views)
        ax = axial_profile(psf, dz_um, args.stride)
        del m, psf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {label:<18} lateral {np.nanmean(lat):6.3f} um "
              f"(spread {spread(lat):.2f}) | axial {np.nanmean(ax):6.3f} um "
              f"(spread {spread(ax):.2f})")
        return lat, ax

    print(f"\nrandom draws from [f_min, f_max] ({args.seeds} seeds):")
    rand_lat, rand_ax = [], []
    for s in range(args.seed0, args.seed0 + args.seeds):
        lat, ax = measure(f"random {s}", seed=s)
        rand_lat.append(lat); rand_ax.append(ax)
    rand_lat = np.array(rand_lat); rand_ax = np.array(rand_ax)

    print("\nreference encoders:")
    fix_lat, fix_ax = measure("fixed (nominal)", init_f="nominal")
    lrn_lat, lrn_ax = measure("learned f", seed=seed0, ckpt=args.learned)
    fd = None
    if args.learned_fD:
        fd = measure("learned f,D", seed=seed0, ckpt=args.learned_fD,
                     learn_D=True)

    # ---- the comparison --------------------------------------------------
    print("\n" + "=" * 68)
    print(f"{'quantity':<26}{'random mean':>13}{'sd':>8}{'learned':>10}{'z':>9}")
    rows = []
    for name, rnd, lrn in (
            ("lateral width (um)", rand_lat.mean(axis=1), np.nanmean(lrn_lat)),
            ("lateral spread", np.array([spread(r) for r in rand_lat]),
             spread(lrn_lat)),
            ("axial resolution (um)", np.nanmean(rand_ax, axis=1),
             np.nanmean(lrn_ax)),
            ("axial spread", np.array([spread(r) for r in rand_ax]),
             spread(lrn_ax))):
        mu, sd = float(np.mean(rnd)), float(np.std(rnd, ddof=1))
        z = (mu - lrn) / max(sd, 1e-12)      # lower is better, so positive z favours learned
        rows.append((name, mu, sd, lrn, z))
        print(f"{name:<26}{mu:>13.3f}{sd:>8.3f}{lrn:>10.3f}{z:>+9.2f}")
    print("=" * 68)
    zmax = max(abs(r[4]) for r in rows)
    if zmax > 2:
        print("The learned map lies outside the random distribution on at least\n"
              "one quantity: optimizing the focal lengths does something that\n"
              "drawing them at random does not.")
    elif zmax < 1:
        print("Every gap is within seed noise. On these optical measures a\n"
              "random focal map does as well as the learned one, and that is\n"
              "what the supplementary should say.")
    else:
        print("Marginal (1 < |z| < 2). Raise --seeds before claiming anything.")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax_, data_r, data_f, data_l, ttl, yl in (
            (axes[0], rand_lat, fix_lat, lrn_lat,
             "Lateral", "effective PSF width (µm)"),
            (axes[1], rand_ax, fix_ax, lrn_ax,
             "Axial", "decorrelation depth (µm)")):
        mu = np.nanmean(data_r, axis=0)
        sd = np.nanstd(data_r, axis=0, ddof=1)
        ax_.fill_between(z_um, mu - sd, mu + sd, color="0.75", alpha=0.5,
                         lw=0, label=f"random $f$ (n={args.seeds}, ±1 sd)")
        ax_.plot(z_um, mu, color="0.45", lw=1.2)
        ax_.plot(z_um, data_f, color="#3b76c4", lw=1.6, ls="--",
                 label="fixed optics")
        ax_.plot(z_um, data_l, color="#e0643a", lw=1.9, label="learned $f$")
        if fd is not None:
            ax_.plot(z_um, fd[0] if ttl == "Lateral" else fd[1],
                     color="#2e9e78", lw=1.5, label="learned $f,D$")
        ax_.set_title(ttl, fontsize=10)
        ax_.set_xlabel("depth $z$ (µm)"); ax_.set_ylabel(yl, fontsize=9)
        ax_.set_xlim(z_um[0], z_um[-1])
        for sp in ("top", "right"):
            ax_.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper center")
    fig.tight_layout()
    for e in args.formats:
        fig.savefig(f"{args.out}.{e}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)

    np.savez(f"{args.out}.npz", z_um=z_um, rand_lat=rand_lat, rand_ax=rand_ax,
             fixed_lat=fix_lat, fixed_ax=fix_ax,
             learned_lat=lrn_lat, learned_ax=lrn_ax)
    print(f"\nwrote {args.out}.{'/.'.join(args.formats)} and {args.out}.npz")


if __name__ == "__main__":
    main()
