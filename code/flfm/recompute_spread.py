"""
recompute_spread.py -- depth spread for every arm, on the same footing.

The spread printed by make_paper_table.py is taken over the REPORTED depths,
so a run with --depth-count 7 and a run with --depth-count 9 are not
comparable, and neither matches a caption that says "all 43 reconstructed
depths". Every arm measured all 43 slices (--depths all), so the spread can be
recomputed from per_slice.csv at whatever depth count the manuscript claims,
identically for all arms.

Usage, from the folder holding the paper_table_* directories:

    python recompute_spread.py --root fig1_data
    python recompute_spread.py --root fig1_data --metric axial_res_um
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def pick_depths(available, k):
    """The k evenly spaced reporting depths, snapped to measured slices.

    Same rule make_paper_table.py uses to choose its table columns.
    """
    targets = np.linspace(available.min(), available.max(), k)
    return sorted({float(available[np.argmin(np.abs(available - t))]) for t in targets})


def spread(by_depth, depths=None):
    s = by_depth if depths is None else by_depth.loc[by_depth.index.isin(depths)]
    if s.empty or s.min() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    return s.max() / s.min(), s.mean(), s.min(), s.max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--metric", default="lat_res_um")
    p.add_argument("--photons", type=float, default=None,
                   help="budget to use; default = the only one present")
    p.add_argument("--counts", type=int, nargs="*", default=[7, 9],
                   help="reported-depth counts to show alongside all slices")
    p.add_argument("--out", default="depth_spread_recomputed.csv")
    args = p.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "paper_table_*"))):
        f = os.path.join(d, "per_slice.csv")
        if not os.path.exists(f):
            print(f"skipping {os.path.basename(d)}: no per_slice.csv")
            continue
        df = pd.read_csv(f)
        budgets = sorted(df["photons"].unique())
        ph = args.photons if args.photons is not None else budgets[0]
        if len(budgets) > 1 and args.photons is None:
            print(f"{os.path.basename(d)}: budgets {budgets}, using {ph}")
        df = df[df["photons"] == ph]
        if df[args.metric].isna().all():
            print(f"skipping {os.path.basename(d)}: {args.metric} is all NaN")
            continue

        by_depth = df.groupby("depth_um")[args.metric].mean().dropna()
        avail = np.array(by_depth.index, dtype=float)
        r = {"arm": os.path.basename(d).replace("paper_table_", ""),
             "n_volumes": int(df["volume"].nunique()),
             "n_depths_measured": len(avail)}
        sp, mean, lo, hi = spread(by_depth)
        r["spread_all"] = sp
        r["mean_all_um"] = mean
        r["best_um"] = lo
        r["worst_um"] = hi
        for k in args.counts:
            if k <= len(avail):
                r[f"spread_{k}"] = spread(by_depth, pick_depths(avail, k))[0]
        rows.append(r)

    if not rows:
        raise SystemExit(f"no per_slice.csv found under {args.root}")

    out = pd.DataFrame(rows).sort_values("spread_all")
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(f"\n{args.metric}, N = {args.photons or 'first'} photons\n")
        print(out.to_string(index=False))
    print(f"\nwrote {args.out}")
    print("Quote spread_all if the caption says all 43 depths; otherwise quote "
          "one of the spread_<k> columns, the SAME k for every arm.")


if __name__ == "__main__":
    main()
