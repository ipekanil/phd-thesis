"""
make_table2.py -- the overall comparison table, with reconstruction cost.

Table 1 is depth x model and has no room for a number that is constant per
model. This is that table: one row per arm, aggregated over all 43 slices and
all 12 test volumes, with the wall-clock cost of recovering one volume beside
the quality metrics. It is the table that answers "RL-TV has the better axial
resolution, so why would I use the network?".

Reads each arm's paper_table_<tag>/ directory and timing.csv, so nothing is
retyped. Depth-uniformity is quoted as the ratio of the worst to the best
per-depth mean over the 43 measured slices -- which needs the arm to have been
run with --depths all; arms run with --depths table get a NaN there rather
than a spread computed from 7 points and silently compared against one
computed from 43.

    python make_table2.py --arms rltv fixed f fD
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {
    "rltv":  r"Fixed optics + RL-TV",
    "fixed": r"Fixed optics + CNN",
    "f":     r"Learned $f$ + CNN",
    "fD":    r"Learned $f,D$ + CNN",
}


def collect(tag, root, timing):
    d = Path(root) / f"paper_table_{tag}"
    if not (d / "per_slice.csv").exists():
        found = sorted(p.name for p in Path(root).glob("paper_table_*")
                       if (p / "per_slice.csv").exists())
        raise SystemExit(
            f"\n!! {d / 'per_slice.csv'} not found.\n"
            f"   Arm directories with metrics under {Path(root).resolve()}: "
            f"{found if found else 'none'}\n"
            f"   Either pass --root <dir containing them>, or drop '{tag}' "
            f"from --arms, or point --arms at the tags above.\n")
    sl = pd.read_csv(d / "per_slice.csv")
    vol = pd.read_csv(d / "per_volume.csv")

    per_depth = sl.groupby("depth_um")["lat_res_um"].mean()
    # only meaningful when every slice was measured; 7 points understate the
    # worst case because the table depths need not land on it
    spread = (per_depth.max() / per_depth.min()) if len(per_depth) > 20 else np.nan

    ax = sl["axial_res_um"].dropna()
    t = timing[timing["tag"] == tag]
    return {
        "tag": tag,
        "lat_res_um": sl["lat_res_um"].mean(),
        "lat_spread": spread,
        "axial_res_um": ax.mean(),
        "psnr_db": vol["psnr_db"].mean(),
        "ssim": vol["ssim"].mean(),
        "rho": vol["rho"].mean(),
        "n_depths": len(per_depth),
        "n_volumes": vol["volume"].nunique(),
        "total_s": float(t["total_s_mean"].iloc[0]) if len(t) else np.nan,
        "total_sd": float(t["total_s_sd"].iloc[0]) if len(t) else np.nan,
        "vps": float(t["volumes_per_s"].iloc[0]) if len(t) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["rltv", "fixed", "f", "fD"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--timing", default="timing.csv")
    ap.add_argument("--out", default="table2_overall.tex")
    args = ap.parse_args()

    tp = Path(args.timing)
    timing = pd.read_csv(tp) if tp.exists() else pd.DataFrame(columns=["tag"])
    if not tp.exists():
        print(f"!! {tp} not found -- the runtime column will be empty. "
              f"Run time_reconstruction.py for each arm first.")

    rows = [collect(a, args.root, timing) for a in args.arms]
    df = pd.DataFrame(rows)

    if df["n_volumes"].nunique() != 1 or df["n_depths"].nunique() != 1:
        print("!! arms differ in coverage -- the rows are not comparable:")
        print(df[["tag", "n_volumes", "n_depths"]].to_string(index=False))

    # (column, lower-is-better, format)
    COLS = [("lat_res_um", True, "{:.2f}"), ("lat_spread", True, "{:.2f}"),
            ("axial_res_um", True, "{:.2f}"), ("psnr_db", False, "{:.2f}"),
            ("ssim", False, "{:.3f}"), ("rho", False, "{:.3f}")]
    best = {}
    for c, lower, _ in COLS:
        v = df[c]
        best[c] = None if v.isna().all() else (v.idxmin() if lower else v.idxmax())

    body = []
    for i, r in df.iterrows():
        cells = []
        for c, _, fmt in COLS:
            s = "--" if pd.isna(r[c]) else fmt.format(r[c])
            cells.append(rf"\textbf{{{s}}}" if best[c] == i and s != "--" else s)
        t = "--" if pd.isna(r["total_s"]) else \
            f"{r['total_s']:.2f} $\\pm$ {r['total_sd']:.2f}"
        v = "--" if pd.isna(r["vps"]) else f"{r['vps']:.2f}"
        body.append(f"{LABEL.get(r['tag'], r['tag'])} & "
                    + " & ".join(cells) + f" & {t} & {v} " + r"\\")

    nv = int(df["n_volumes"].iloc[0])
    nd = int(df["n_depths"].iloc[0])
    caption = (
        rf"Overall comparison, averaged over all {nd} reconstructed depths and "
        rf"{nv} held-out test volumes at $N=100$ photons. Depth spread is the "
        r"ratio of the worst to the best per-depth mean lateral resolution, so "
        r"1.00 would be a reconstruction of uniform quality through the volume. "
        r"Reconstruction time is sensor image to volume on a single GPU, "
        r"excluding the one-off PSF calibration; it is the mean over five "
        r"volumes after a discarded warm-up. Best value per column in bold.")

    Path(args.out).write_text("\n".join([
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", r"\label{tab:overall_comparison}",
        r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{threeparttable}", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccccrc}", r"\toprule",
        r"Model & Lat. res. & Depth & Ax. res. & PSNR & SSIM & $\rho$ & "
        r"Recon. time & Volumes \\",
        r" & ($\mu$m) & spread & ($\mu$m) & (dB) & & & (s) & s$^{-1}$ \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}%", r"}",
        r"\end{threeparttable}", r"\end{table}", ""]))

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
