"""
Depth-resolved Fourier ring correlation (FRC) resolution against the ground truth.

For every model, every test volume and every axial slice, the FRC between the
reconstructed slice and the ground-truth slice is computed, and the lateral
resolution is taken as 1 / (first spatial frequency where the FRC drops below
the threshold). Per-depth values are averaged over the test volumes, and the
depth spread is the ratio of the largest to the smallest mean resolution
across depth (same definition as for the blur matching estimate).

Because the reference is the noiseless ground truth rather than a second noisy
realisation, the 1/2 threshold is the right one; 1/7 belongs to the two-image
FRC where both images carry independent noise.

Expected input
--------------
The folder layout written by make_paper_table.py:

    paper_table_rltv/vol0001_N0100/gt.tif, reconstruction.tif
    paper_table_fixed/vol0001_N0100/...
    paper_table_f/vol0001_N0100/...
    paper_table_fD/vol0001_N0100/...

Each .tif is a (Nz, H, W) stack; .npy files with the same names also work.
Run make_paper_table.py with --tiff-volumes all so that every test volume is
written, not only the two that get figures.

The voxel size is read from meta.json in each model folder when it is there,
so --dx and --dz only matter for older runs that predate it. gt.tif is
identical across arms, so a folder missing it falls back to another arm's copy.

Usage (run from the folder that contains the paper_table_* folders)
-----
python frc_depth.py
python frc_depth.py --root D:/results --out frc_out --reference "Fixed optics + CNN"
"""

import argparse
import csv
import json
import os

import numpy as np


def _tukey2d(h, w, alpha=0.125):
    """Separable Tukey window to suppress edge discontinuities before the FFT."""
    def tukey(n):
        x = np.linspace(0.0, 1.0, n)
        win = np.ones(n)
        edge = alpha / 2.0
        lo = x < edge
        hi = x > 1.0 - edge
        win[lo] = 0.5 * (1 + np.cos(np.pi * (2 * x[lo] / alpha - 1)))
        win[hi] = 0.5 * (1 + np.cos(np.pi * (2 * x[hi] / alpha - 2 / alpha + 1)))
        return win
    return np.outer(tukey(h), tukey(w))


def frc_curve(a, b, window=None):
    """FRC between two equally sized 2D images.

    Returns (frequencies in cycles/pixel, FRC values). Rings are one frequency
    bin wide; only rings up to the Nyquist frequency of the shorter side are kept.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if window is not None:
        a = a * window
        b = b * window
    fa = np.fft.fftshift(np.fft.fft2(a))
    fb = np.fft.fftshift(np.fft.fft2(b))

    h, w = a.shape
    yy, xx = np.indices((h, w))
    # Radius in units of frequency bins of the shorter side
    n = min(h, w)
    ry = (yy - h // 2) * (n / h)
    rx = (xx - w // 2) * (n / w)
    r = np.sqrt(rx ** 2 + ry ** 2).astype(np.int64)
    nbins = n // 2

    mask = r < nbins
    rr = r[mask]
    num = np.bincount(rr, weights=np.real(fa * np.conj(fb))[mask], minlength=nbins)
    den_a = np.bincount(rr, weights=(np.abs(fa) ** 2)[mask], minlength=nbins)
    den_b = np.bincount(rr, weights=(np.abs(fb) ** 2)[mask], minlength=nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        frc = num / np.sqrt(den_a * den_b)
    freqs = np.arange(nbins) / n  # cycles per pixel
    return freqs, frc


def _smooth_curve(frc, smooth):
    frc = np.nan_to_num(frc, nan=0.0)
    # The DC ring is undefined after mean removal; replace it so it does not
    # leak into the first bins through the smoothing.
    frc[0] = frc[1]
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        pad = smooth // 2
        frc = np.convolve(np.pad(frc, pad, mode="edge"), kernel, mode="valid")
    return frc


def crossing(freqs, frc, threshold, min_below=3, first_bin=3):
    """First SUSTAINED crossing of the threshold, in cycles/pixel.

    Two guards, both standard practice and both needed here:

    first_bin  the lowest rings hold almost no pixels (ring 1 has about six),
               so their correlation is dominated by sampling noise, and any
               large-scale intensity mismatch between the reconstruction and
               the ground truth drags them down. Taking the first bin below
               the threshold then reports a resolution of tens of microns for
               a slice whose mid frequencies are perfectly well correlated.
    min_below  the crossing must hold for this many consecutive rings, so a
               transient dip does not end the curve early. A genuine
               resolution limit stays below the threshold.

    Returns NaN when the curve never crosses (the caller pins it at Nyquist).
    """
    below = frc < threshold
    below[:first_bin] = False
    run = 0
    idx = None
    for i in range(first_bin, below.size):
        run = run + 1 if below[i] else 0
        if run >= min_below:
            idx = i - min_below + 1
            break
    if idx is None:
        return np.nan
    f0, f1 = freqs[idx - 1], freqs[idx]
    c0, c1 = frc[idx - 1], frc[idx]
    if c0 == c1:
        return f1
    f_cross = f0 + (c0 - threshold) * (f1 - f0) / (c0 - c1)
    return float(np.clip(f_cross, f0, f1))        # never extrapolate


def frc_resolution(a, b, dx, threshold=0.5, smooth=3, window=None,
                   min_below=3, first_bin=3):
    """Lateral resolution (same unit as dx) from the first sustained crossing.

    Returns (resolution, censored). censored is True when the FRC never falls
    below the threshold, in which case the resolution is pinned at the Nyquist
    limit 2*dx and the true value is somewhere below it; those slices are
    counted and reported so the means are not quietly biased upwards.
    Resolution is NaN for empty slices.
    """
    if a.std() == 0 or b.std() == 0:
        return np.nan, False
    freqs, frc = frc_curve(a, b, window)
    frc = _smooth_curve(frc, smooth)
    f_cross = crossing(freqs, frc, threshold, min_below, first_bin)
    if np.isnan(f_cross):
        return 2.0 * dx, True
    if f_cross <= 0:
        return np.nan, False
    return dx / f_cross, False


def _load_volume(path):
    """Load a (Nz, H, W) volume from .tif/.tiff or .npy."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        vol = tifffile.imread(path)
    else:
        vol = np.load(path)
    vol = np.squeeze(vol).astype(np.float64)
    if vol.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {vol.shape}")
    # Put the axial axis first if it was saved last (H, W, Nz)
    if vol.shape[-1] < vol.shape[0] and vol.shape[-1] < vol.shape[1]:
        vol = np.moveaxis(vol, -1, 0)
    return vol


def _find_file(folder, stem):
    for ext in (".tif", ".tiff", ".npy"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None


def _find_gt(root, models, vname):
    """gt.tif for one volume, from its own folder or from any other arm's."""
    for _, folder in models:
        p = _find_file(os.path.join(root, folder, vname), "gt")
        if p is not None:
            return p
    raise FileNotFoundError(f"No gt.tif/.npy for {vname} in any model folder")


def _read_meta(root, models, dx_cli, dz_cli):
    """Voxel size from meta.json, falling back to the command line."""
    found = {}
    for label, folder in models:
        p = os.path.join(root, folder, "meta.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            found[label] = (float(m["dx_um"]), float(m["dz_um"]))
    if not found:
        print(f"no meta.json found, using --dx {dx_cli} --dz {dz_cli}")
        return dx_cli, dz_cli
    vals = set(found.values())
    if len(vals) > 1:
        raise ValueError(f"model folders disagree on the voxel size: {found}")
    dx, dz = vals.pop()
    print(f"voxel size from meta.json: dx = {dx:.4f} um, dz = {dz:.4f} um")
    return dx, dz


def evaluate(root, models, dx, threshold, smooth, photon_tag,
             min_below=3, first_bin=3):
    """models: list of (label, model folder name under root).

    Each model folder contains one subfolder per test volume
    (e.g. vol0001_N0100) holding gt.tif and reconstruction.tif.
    """
    results = {}
    censored = {}
    vol_names = None
    window = None
    gt_cache = {}
    for label, folder in models:
        mdir = os.path.join(root, folder)
        vols = sorted(d for d in os.listdir(mdir)
                      if os.path.isdir(os.path.join(mdir, d)) and d.endswith(photon_tag))
        if not vols:
            raise FileNotFoundError(f"No volume folders ending in '{photon_tag}' in {mdir}")
        if vol_names is None:
            vol_names = vols
        elif vols != vol_names:
            raise ValueError(f"{label}: volume folders {vols} differ from {vol_names}")

        per_vol = None
        n_cens = 0
        for v, vname in enumerate(vols):
            vdir = os.path.join(mdir, vname)
            if vname not in gt_cache:
                gt_cache[vname] = _load_volume(_find_gt(root, models, vname))
            gt = gt_cache[vname]
            rec_path = _find_file(vdir, "reconstruction")
            if rec_path is None:
                raise FileNotFoundError(f"No reconstruction.tif/.npy in {vdir}")
            rec = _load_volume(rec_path)
            if rec.shape != gt.shape:
                raise ValueError(f"{label}/{vname}: shape {rec.shape} != ground truth {gt.shape}")
            nz, h, w = gt.shape
            if window is None:
                window = _tukey2d(h, w)
            if per_vol is None:
                per_vol = np.full((len(vols), nz), np.nan)
            for k in range(nz):
                per_vol[v, k], cens = frc_resolution(
                    rec[k], gt[k], dx, threshold, smooth, window,
                    min_below, first_bin)
                n_cens += int(cens)
        results[label] = per_vol
        censored[label] = n_cens
        note = "" if n_cens == 0 else f", {n_cens} slices pinned at Nyquist"
        print(f"done: {label} ({len(vols)} volumes{note})")
    return vol_names, results, censored


def summarize(results, dz):
    summary = {}
    for label, per_vol in results.items():
        mean = np.nanmean(per_vol, axis=0)
        sem = np.nanstd(per_vol, axis=0) / np.sqrt(np.sum(~np.isnan(per_vol), axis=0))
        nz = mean.size
        z = (np.arange(nz) - (nz - 1) / 2) * dz
        per_volume_mean = np.nanmean(per_vol, axis=1)     # one number per volume
        summary[label] = {
            "z": z,
            "mean": mean,
            "sem": sem,
            "per_vol": per_vol,
            "per_volume_mean": per_volume_mean,
            "depth_spread": np.nanmax(mean) / np.nanmin(mean),
            "volume_mean": np.nanmean(mean),
            "volume_sd": np.nanstd(per_volume_mean, ddof=1) if per_volume_mean.size > 1 else np.nan,
            "center": mean[nz // 2],
            "edge_neg": mean[0],
            "edge_pos": mean[-1],
        }
    return summary


def paired_tests(summary, reference):
    """Learned arms against one reference arm, paired over the test volumes.

    The volumes differ enormously in content, so a paired test over volumes is
    the fair comparison; an unpaired one is dominated by that spread.
    """
    if reference not in summary:
        print(f"\nreference '{reference}' not among the models, skipping paired tests")
        return []
    ref = summary[reference]["per_volume_mean"]
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    rows = []
    for label, s in summary.items():
        if label == reference:
            continue
        d = s["per_volume_mean"] - ref           # negative = finer than reference
        ok = ~np.isnan(d)
        n = int(ok.sum())
        wins = int(np.sum(d[ok] < 0))
        if wilcoxon is not None and n >= 6:
            try:
                p = float(wilcoxon(d[ok]).pvalue)
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        rows.append({"model": label, "n": n, "finer_on": wins,
                     "median_delta_um": float(np.nanmedian(d[ok])),
                     "mean_delta_um": float(np.nanmean(d[ok])), "p": p})
    return rows


def save_outputs(summary, stats_rows, vol_names, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "frc_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "depth_spread", "volume_mean_um", "volume_sd_um",
                     "z_min_um", "center_um", "z_max_um"])
        for label, s in summary.items():
            wr.writerow([label, f"{s['depth_spread']:.2f}", f"{s['volume_mean']:.2f}",
                         f"{s['volume_sd']:.2f}", f"{s['edge_neg']:.2f}",
                         f"{s['center']:.2f}", f"{s['edge_pos']:.2f}"])

    with open(os.path.join(out_dir, "frc_vs_depth.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        labels = list(summary)
        wr.writerow(["z_um"] + [f"{l} mean" for l in labels] + [f"{l} sem" for l in labels])
        z = summary[labels[0]]["z"]
        for k in range(z.size):
            wr.writerow([f"{z[k]:.2f}"] + [f"{summary[l]['mean'][k]:.3f}" for l in labels]
                        + [f"{summary[l]['sem'][k]:.3f}" for l in labels])

    # Per volume, so the across-volume spread can be reported and the paired
    # comparison re-done without re-running the FRC.
    with open(os.path.join(out_dir, "frc_per_volume.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        labels = list(summary)
        wr.writerow(["volume"] + [f"{l} mean_um" for l in labels]
                    + [f"{l} spread" for l in labels])
        for v, vname in enumerate(vol_names):
            means, spreads = [], []
            for l in labels:
                row = summary[l]["per_vol"][v]
                means.append(f"{np.nanmean(row):.3f}")
                spreads.append(f"{np.nanmax(row) / np.nanmin(row):.3f}")
            wr.writerow([vname] + means + spreads)

    if stats_rows:
        with open(os.path.join(out_dir, "frc_paired.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(stats_rows[0]))
            wr.writeheader()
            wr.writerows(stats_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for label, s in summary.items():
        style = "--" if "RL-TV" in label else "-"
        ax.plot(s["z"], s["mean"], style, label=label)
        ax.fill_between(s["z"], s["mean"] - s["sem"], s["mean"] + s["sem"], alpha=0.2)
    ax.set_xlabel(r"Depth $z$ ($\mu$m)")
    ax.set_ylabel(rf"FRC resolution ($\mu$m), threshold {threshold:g}")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frc_vs_depth.pdf"))
    fig.savefig(os.path.join(out_dir, "frc_vs_depth.png"), dpi=200)


def dump_worst(root, models, results, dx, dz, threshold, smooth, photon_tag,
               vol_names, out_dir, n_worst, min_below, first_bin):
    """Write the FRC curves of the N worst slices, with both crossing rules.

    This is the check on the estimator itself. A slice that is genuinely
    unresolved has a curve that falls early and stays down. A slice that is
    only an artefact of the crossing rule has a curve that dips below the
    threshold in the first few rings and comes straight back up, and the two
    rules disagree by an order of magnitude.
    """
    worst = []
    for label, folder in models:
        per_vol = results[label]
        for v in range(per_vol.shape[0]):
            for k in range(per_vol.shape[1]):
                if not np.isnan(per_vol[v, k]):
                    worst.append((per_vol[v, k], label, folder, v, k))
    worst.sort(reverse=True)
    worst = worst[:n_worst]
    if not worst:
        return

    nz = results[models[0][0]].shape[1]
    window = None
    curves = []
    for res, label, folder, v, k in worst:
        vname = vol_names[v]
        gt = _load_volume(_find_gt(root, models, vname))[k]
        rec = _load_volume(_find_file(os.path.join(root, folder, vname),
                                      "reconstruction"))[k]
        if window is None:
            window = _tukey2d(*gt.shape)
        freqs, frc = frc_curve(rec, gt, window)
        frc = _smooth_curve(frc, smooth)
        f_naive = crossing(freqs, frc, threshold, min_below=1, first_bin=1)
        z = (k - (nz - 1) / 2) * dz
        curves.append({"label": label, "vol": vname, "z_um": z, "freqs": freqs,
                       "frc": frc, "res": res,
                       "res_naive": np.nan if np.isnan(f_naive) or f_naive <= 0
                       else dx / f_naive})

    with open(os.path.join(out_dir, "frc_worst_curves.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "volume", "z_um", "res_um", "res_um_first_bin_rule"]
                    + [f"frc_bin{i}" for i in range(curves[0]["freqs"].size)])
        for c in curves:
            wr.writerow([c["label"], c["vol"], f"{c['z_um']:.2f}",
                         f"{c['res']:.3f}", f"{c['res_naive']:.3f}"]
                        + [f"{x:.4f}" for x in c["frc"]])

    print(f"\nWorst {len(curves)} slices, both crossing rules (um):")
    print(f"{'model':<22}{'volume':<16}{'z':>7}{'sustained':>11}{'first bin':>11}")
    for c in curves:
        print(f"{c['label']:<22}{c['vol']:<16}{c['z_um']:>7.1f}"
              f"{c['res']:>11.2f}{c['res_naive']:>11.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for c in curves:
        ax.plot(c["freqs"], c["frc"], lw=1.0,
                label=f"{c['label']}, {c['vol'][:7]}, z={c['z_um']:+.0f}")
    ax.axhline(threshold, color="k", lw=0.8, ls=":")
    ax.axvline(first_bin / (2 * (curves[0]["freqs"].size)), color="k", lw=0.5)
    ax.set_xlabel("spatial frequency (cycles/pixel)")
    ax.set_ylabel("FRC")
    ax.set_xlim(0, curves[0]["freqs"].max())
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frc_worst_curves.png"), dpi=200)


DEFAULT_MODELS = [
    ("Fixed optics + RL-TV", "paper_table_rltv"),
    ("Fixed optics + CNN", "paper_table_fixed"),
    ("Learned f + CNN", "paper_table_f"),
    ("Learned f,D + CNN", "paper_table_fD"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="folder that contains the paper_table_* folders")
    p.add_argument("--model", action="append",
                   help='"Label=folder" to override the default four models; repeat for each')
    p.add_argument("--photon-tag", default="N0100", help="suffix of the volume folders to use")
    p.add_argument("--dx", type=float, default=0.3769,
                   help="lateral voxel size in um; only used if meta.json is absent")
    p.add_argument("--dz", type=float, default=1.4286,
                   help="axial step in um; only used if meta.json is absent")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="FRC threshold (0.5 against GT, 1/7 for two noisy reconstructions)")
    p.add_argument("--smooth", type=int, default=3, help="moving-average width applied to the FRC curve")
    p.add_argument("--min-below", type=int, default=3,
                   help="consecutive rings that must sit below the threshold "
                        "before the crossing is accepted. 1 reproduces the "
                        "first-bin rule, which a transient dip can trigger")
    p.add_argument("--first-bin", type=int, default=3,
                   help="lowest ring considered. The first few rings hold only "
                        "a handful of pixels each, so their correlation is "
                        "sampling noise plus any large-scale intensity mismatch")
    p.add_argument("--dump-worst", type=int, default=6,
                   help="write the FRC curves of this many worst slices, under "
                        "both crossing rules, for checking the estimator. 0 = off")
    p.add_argument("--reference", default="Fixed optics + CNN",
                   help="arm the others are compared against, paired over volumes")
    p.add_argument("--out", default="frc_out")
    args = p.parse_args()

    if args.model:
        models = [tuple(s.strip() for s in m.split("=", 1)) for m in args.model]
    else:
        models = [m for m in DEFAULT_MODELS
                  if os.path.isdir(os.path.join(args.root, m[1]))]
        missing = [m[1] for m in DEFAULT_MODELS if m not in models]
        if missing:
            print(f"skipping absent folders: {', '.join(missing)}")
        if not models:
            raise FileNotFoundError(f"No paper_table_* folders in {args.root}")

    dx, dz = _read_meta(args.root, models, args.dx, args.dz)
    vol_names, results, censored = evaluate(
        args.root, models, dx, args.threshold, args.smooth, args.photon_tag,
        args.min_below, args.first_bin)
    summary = summarize(results, dz)
    stats_rows = paired_tests(summary, args.reference)
    save_outputs(summary, stats_rows, vol_names, args.out, args.threshold)
    if args.dump_worst:
        dump_worst(args.root, models, results, dx, dz, args.threshold,
                   args.smooth, args.photon_tag, vol_names, args.out,
                   args.dump_worst, args.min_below, args.first_bin)

    print(f"\n{len(vol_names)} volumes: {', '.join(vol_names)}")
    print(f"FRC resolution (threshold {args.threshold:g}), um. "
          f"mean +/- sd is over volumes, of the per-volume depth average.")
    print(f"{'model':<24}{'spread':>8}{'mean':>8}{'sd':>7}{'z_min':>8}{'center':>8}{'z_max':>8}")
    for label, s in summary.items():
        print(f"{label:<24}{s['depth_spread']:>8.2f}{s['volume_mean']:>8.2f}"
              f"{s['volume_sd']:>7.2f}{s['edge_neg']:>8.2f}{s['center']:>8.2f}"
              f"{s['edge_pos']:>8.2f}")

    if stats_rows:
        print(f"\nPaired over volumes against '{args.reference}' "
              f"(negative delta = finer resolution)")
        print(f"{'model':<24}{'finer on':>10}{'median':>9}{'mean':>8}{'p':>8}")
        for r in stats_rows:
            pstr = "     n/a" if np.isnan(r["p"]) else f"{r['p']:>8.3f}"
            frac = f"{r['finer_on']}/{r['n']}"
            print(f"{r['model']:<24}{frac:>10}"
                  f"{r['median_delta_um']:>9.3f}{r['mean_delta_um']:>8.3f}{pstr}")


if __name__ == "__main__":
    main()
