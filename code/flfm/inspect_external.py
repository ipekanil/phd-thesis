"""
inspect_external.py -- check the external files before running the evaluation.

Prints, per dataset: what loaded, its shape and dtype, how many channels it
has, what the declared voxel size resamples it to, and how many tiles that
yields. No GPU, no models, a few seconds.

Run this first. The voxel sizes in DATASETS set the resampling factors, and a
wrong one does not crash -- it quietly produces a volume at the wrong physical
scale and every number downstream is wrong in a way no error message will tell
you about. This is the cheap way to catch it.

    python inspect_external.py --root .
"""
import argparse
from pathlib import Path

import numpy as np

from config import config
from eval_external import DATASETS, load_zyx, tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--stride-frac", type=float, default=1.0)
    ap.add_argument("--min-foreground", type=float, default=0.02)
    args = ap.parse_args()

    root = Path(args.root)
    nz = int(config.recon_params["n_slices"])
    ny = int(config.recon_params["output_size"][0])
    nx = int(config.recon_params["output_size"][1])
    dxy = float(config.optics_params["object_pixel_size"]) * 1e6
    zmin = float(config.optics_params["volume_depth_min"]) * 1e6
    zmax = float(config.optics_params["volume_depth_max"]) * 1e6
    dz = (zmax - zmin) / (nz - 1)
    shape = (nz, ny, nx)
    stride = tuple(max(1, int(round(s * args.stride_frac))) for s in shape)

    print(f"model grid {shape}, {dxy:.4f} µm lateral, {dz:.4f} µm axial "
          f"({zmax - zmin:.0f} µm of depth)\n")

    total = 0
    for dname in args.datasets:
        spec = DATASETS[dname]
        print(f"=== {spec['label']}  [{dname}]")
        paths = []
        for pat in spec["files"]:
            paths += sorted(root.glob(pat)) if any(c in pat for c in "*?[") \
                else ([root / pat] if (root / pat).exists() else [])
        if not paths:
            print(f"  !! no file matched {spec['files']} under {root.resolve()}")
            # help them find it rather than just failing
            stem = Path(spec["files"][0]).stem.split("*")[0][:18]
            near = [p for p in root.rglob("*.tif*") if stem.lower() in p.name.lower()][:5]
            near += [p for p in root.rglob("*.oir") if stem.lower() in p.name.lower()][:5]
            if near:
                print("     did you mean:")
                for p in near:
                    print(f"       {p.relative_to(root)}")
            print()
            continue

        for p in paths:
            try:
                v = load_zyx(p, spec.get("channel"))
            except SystemExit as e:
                print(f"  {p.name}: {e}")
                continue
            except Exception as e:
                print(f"  !! {p.name}: {type(e).__name__}: {e}")
                continue

            fz, fy = spec["z_um"] / dz, spec["xy_um"] / dxy
            rz = max(1, int(round(v.shape[0] * fz)))
            ry = max(1, int(round(v.shape[1] * fy)))
            rx = max(1, int(round(v.shape[2] * fy)))

            class _S:
                pass
            dummy = _S(); dummy.shape = (rz, ry, rx)
            n_tiles = len(tiles(dummy, shape, stride))

            span_z = v.shape[0] * spec["z_um"]
            print(f"  {p.name}")
            print(f"    loaded      {v.shape}  "
                  f"min {v.min():.4g} max {v.max():.4g} mean {v.mean():.4g}")
            print(f"    declared    {spec['xy_um']:.4f} µm/px, "
                  f"{spec['z_um']:.4f} µm/slice  -> spans "
                  f"{v.shape[2]*spec['xy_um']:.0f} x "
                  f"{v.shape[1]*spec['xy_um']:.0f} x {span_z:.0f} µm")
            print(f"    resampled   {(rz, ry, rx)}  "
                  f"(x{fy:.3f} lateral, x{fz:.3f} axial)")
            print(f"    tiles       {n_tiles} of {shape} before the "
                  f"{args.min_foreground:.0%} foreground filter")
            if rz < nz or ry < ny or rx < nx:
                print(f"    !! smaller than one tile in at least one axis; it "
                      f"will be zero-padded, and the padding is not data")
            if span_z < (zmax - zmin) * 0.8:
                print(f"    !! only {span_z:.0f} µm of depth against the "
                      f"model's {zmax - zmin:.0f} µm: most of the "
                      f"reconstructed volume will be empty")
            total += n_tiles
        print()

    print(f"{total} tiles before filtering, across {len(args.datasets)} datasets")
    print("If a 'declared' span looks wrong for the specimen, fix xy_um/z_um in")
    print("DATASETS before running eval_external.py -- nothing downstream will")
    print("warn you about it.")


if __name__ == "__main__":
    main()
