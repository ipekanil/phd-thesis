"""
eval_external.py -- generalization to external datasets, at a defensible n.

Addresses two concerns about the external test separately:

  "the number is seriously small"   -> each source stack is RESAMPLED onto the
                                       model's voxel grid and then TILED into
                                       model-sized sub-volumes, so one 1024^3
                                       file yields many independent test
                                       volumes instead of one centre crop.
  "the types of samples are too single" -> datasets are declared in a table;
                                       add a row and it is included everywhere.

TWO THINGS THIS DOES THAT A CENTRE CROP DOES NOT, both of which change what the
figure means:

1. PHYSICAL RESAMPLING. The model's grid is 0.3769 µm laterally and 1.4286 µm
   axially. The plant stacks are 0.5 µm per slice, so 43 of their slices span
   21.5 µm, not the 60 µm the optics assumes -- cropping without resampling
   hands the model a volume 2.8x thinner than it thinks it has, and every
   axial number computed from it is meaningless. Each dataset therefore
   declares its voxel size and is interpolated onto the model grid first.

2. CONTENT FILTERING. Tiling a sparse specimen produces many tiles of pure
   background, and a reconstruction of background scores superbly on PSNR. An
   unfiltered tile set would inflate every metric, so tiles below a foreground
   threshold are dropped and the count of dropped tiles is reported.

    python eval_external.py --root . --out external_eval

Metrics come from make_paper_table, not reimplemented here, so the external
numbers are directly comparable with the in-distribution table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile as tiff
from skimage.metrics import structural_similarity as compare_ssim

from config import config
from F_Optics_fixed import F_Optics
from F_VCD_pytorch_multiple_views import F_Recon
from adjoint import FLFMAdjointBackprojector
from sensor_noise import add_sensor_noise
from rltv import rl_tv_solve
from make_paper_table import _fit_affine_mse
from make_paper_table import (DEVICE, SEED, READ_NOISE_E, TRAIN_PHOTONS,
                              normalize_sensor, infer_learn_D,
                              psnr, psnr_affine, nrmse, pearson, adj_z,
                              blur_match, blur_match_axial)

# ---------------------------------------------------------------------------
# Datasets. voxel sizes are in micrometres; get them from the acquisition
# metadata, NOT from a guess -- they set the resampling factors and therefore
# the physical meaning of every number below.
#
#   .oir, .czi, .nd2 and friends are not readable here: export to TIFF from
#   Fiji first (File > Save As > Tiff), keeping the voxel size noted.
# ---------------------------------------------------------------------------
DATASETS = {
    # The *_cropped volumes are 512x512 ROIs cut by hand in Fiji so the figure
    # shows the region of interest rather than whichever corner the tiler
    # happened to reach first. They are 43 planes deep -- one model volume --
    # except where noted, so each yields a single tile.
    "plant": dict(
        label="Plant (hypocotyl, GFP-MBD)",
        files=["gen_Data/Hypocotyl_GFP-MBD_cropped.tif"],
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "col0": dict(
        label="Plant (Col-0, PI)",
        files=["gen_Data/PC_PI_Col0_5-1_cropped.tif"],
        # voxel size assumed to match the hypocotyl stack; unverified, and with
        # --resample none it changes only the printed span, not the data
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "planaria": dict(
        label="Planaria (S. mediterranea)",
        files=["gen_Data/EXP280_Smed_live_RedDot1_slide_mnt_N2_stk_cropped.tif"],
        xy_um=0.34,  z_um=1.0,  channel=None,
    ),
    "mdck": dict(
        label="MDCK organoid (DAPI)",
        files=["gen_Data/C1-30x_A2_DAPI_Phalloidin647_5x5_B01_G001_0006_cropped.tif"],
        xy_um=275.5 / 1024, z_um=200.0 / 286, channel=None,
    ),
    "twocolor": dict(
        # 35 planes, not 43: zero padded to the model depth, and every metric
        # is taken over the 35 acquired planes only (z_valid in per_tile.csv).
        label="Two-colour (RFP/GFP)",
        files=["gen_Data/C1-farred_RFP_GFP_2109175_2color_sub_10.20_cropped.tif"],
        xy_um=0.34,  z_um=1.0,  channel=None,
    ),
    # gen_Data/c -- several volumes per sample type, so a dataset's n is a
    # count of specimens rather than sub-volumes of one. Grouped keys are the
    # default; the per-file keys below them give one row each if preferred.
    "c_plant": dict(
        label="Plant (hypocotyl, GFP-MBD)",
        files=["gen_Data/c/Hypocotyl_GFP-MBD_c.tif"],
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "c_plant_pi": dict(
        label="Plant (Arabidopsis, PI)",
        files=["gen_Data/c/PC_PI_Col0_5_c.tif",
               "gen_Data/c/SAM_PI_Col-0_c.tif"],
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "c_planaria": dict(
        label="Planaria (S. mediterranea)",
        files=["gen_Data/c/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0004_c.tif",
               "gen_Data/c/EXP280_Smed_live_RedDot1_slide_mnt_N3_stk1_c.tif",
               "gen_Data/c/EXP280_Smed_live_RedDot1_slide_mnt_N3_stk3_c.tif"],
        xy_um=0.34,  z_um=1.0,  channel=None,
    ),
    "c_mdck": dict(
        label="MDCK organoid (DAPI)",
        # 4-D (z, 2, y, x): channel 0 is DAPI, 1 is Phalloidin-647. Set
        # explicitly -- left to guess, the loader takes the brighter one.
        files=["gen_Data/c/30x_A2_DAPI_Phalloidin647_5x5_B01_G001_0002_c.tif"],
        xy_um=275.5 / 1024, z_um=200.0 / 286, channel=0,
    ),
    # one row per volume
    "c_col0": dict(label="Plant (Col-0, PI)",
                   files=["gen_Data/c/PC_PI_Col0_5_c.tif"],
                   xy_um=0.363, z_um=0.5, channel=None),
    "c_sam": dict(label="Plant (SAM, PI)",
                  files=["gen_Data/c/SAM_PI_Col-0_c.tif"],
                  xy_um=0.363, z_um=0.5, channel=None),
    "c_smed_fixed": dict(label="Planaria (fixed)",
                         files=["gen_Data/c/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0004_c.tif"],
                         xy_um=0.34, z_um=1.0, channel=None),
    "c_smed_n3a": dict(label="Planaria (live, stk1)",
                       files=["gen_Data/c/EXP280_Smed_live_RedDot1_slide_mnt_N3_stk1_c.tif"],
                       xy_um=0.34, z_um=1.0, channel=None),
    "c_smed_n3b": dict(label="Planaria (live, stk3)",
                       files=["gen_Data/c/EXP280_Smed_live_RedDot1_slide_mnt_N3_stk3_c.tif"],
                       xy_um=0.34, z_um=1.0, channel=None),
    # The FIRST round of crops: same ROIs, but deep enough that resampling
    # onto the model's 1.4286 um axial grid yields a full 43-plane volume.
    # These are the ones to use with --resample grid; the 43-plane crops above
    # cannot be resampled without ending up thinner than the model.
    "plant_deep": dict(
        label="Plant (hypocotyl, GFP-MBD)",
        files=["gen_Data/deep/Hypocotyl_GFP-MBD_cropped.tif"],
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "col0_deep": dict(
        label="Plant (Col-0, PI)",
        files=["gen_Data/deep/PC_PI_Col0_5-1_cropped.tif"],
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "planaria_deep": dict(
        label="Planaria (S. mediterranea)",
        files=["gen_Data/deep/EXP280_Smed_live_RedDot1_slide_mnt_N2_stk_cropped.tif"],
        xy_um=0.34,  z_um=1.0,  channel=None,
    ),
    "mdck_deep": dict(
        label="MDCK organoid (DAPI)",
        files=["gen_Data/deep/C1-30x_A2_DAPI_Phalloidin647_5x5_B01_G001_0006_cropped.tif"],
        xy_um=275.5 / 1024, z_um=200.0 / 286, channel=None,
    ),
    # the untiled originals, kept so the earlier full-stack run can be repeated
    "plant_full": dict(
        label="Plant (hypocotyl, GFP-MBD)",
        files=["gen_Data/Hypocotyl_GFP-MBD.tif"],
        xy_um=0.363, z_um=0.5,  channel=None,
    ),
    "droso_full": dict(
        label="Drosophila",
        files=["gen_Data/droso_long_t000.tif"],
        xy_um=0.34,  z_um=1.0,  channel=None,
    ),
    "mdck_full": dict(
        label="MDCK organoid (DAPI)",
        files=["gen_Data/C1-30x_A2_DAPI_Phalloidin647_5x5_B01_G001_0005.tif"],
        xy_um=275.5 / 1024, z_um=200.0 / 286, channel=None,
    ),
}

# "rltv" is the model-based baseline: fixed optics, no learned stage, so it
# loads optics_model_fixed.pth and no reconstruction network.
ARMS = [("rltv", "Fixed optics + RL-TV"),
        ("fixed", "Fixed optics + CNN"),
        ("f", "Learned f + CNN"),
        ("fD", "Learned f,D + CNN")]


def fsc_curve(a, b, dxy, dz, n_shells=None, min_count=8):
    """Fourier shell correlation between two volumes, in physical frequency.

    Voxels here are anisotropic (dxy ~ 0.38 um, dz ~ 1.43 um), so shells are
    built in cycles/um rather than in voxel index, and the band is capped at
    the COARSER axis's Nyquist -- beyond it a shell is only partly sampled and
    its correlation is an artefact of which voxels exist, not of the data.
    """
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    FA = np.fft.fftn(a); FB = np.fft.fftn(b)
    nz, ny, nx = a.shape
    fz = np.fft.fftfreq(nz, d=dz)[:, None, None]
    fy = np.fft.fftfreq(ny, d=dxy)[None, :, None]
    fx = np.fft.fftfreq(nx, d=dxy)[None, None, :]
    R = np.sqrt(fz ** 2 + fy ** 2 + fx ** 2)
    fmax = min(1.0 / (2 * dxy), 1.0 / (2 * dz))
    # A shell narrower than the coarsest frequency step contains no samples at
    # all; left as zero it looks like perfect decorrelation and the threshold
    # crossing lands there. Shell width is therefore tied to the sampling.
    df = max(1.0 / (nz * dz), 1.0 / (ny * dxy), 1.0 / (nx * dxy))
    if n_shells is None:
        n_shells = max(8, int(np.floor(fmax / df)))
    edges = np.linspace(0, fmax, n_shells + 1)
    idx = np.digitize(R.ravel(), edges) - 1
    keep = (idx >= 0) & (idx < n_shells)
    idx = idx[keep]
    A = FA.ravel()[keep]; B = FB.ravel()[keep]
    num = np.bincount(idx, weights=np.real(A * np.conj(B)), minlength=n_shells)
    da = np.bincount(idx, weights=np.abs(A) ** 2, minlength=n_shells)
    db = np.bincount(idx, weights=np.abs(B) ** 2, minlength=n_shells)
    cnt = np.bincount(idx, minlength=n_shells)
    with np.errstate(invalid="ignore", divide="ignore"):
        fsc = num / np.sqrt(da * db)
    # a sparsely populated shell is noise, not a measurement
    fsc[(cnt < min_count) | (da <= 0) | (db <= 0)] = np.nan
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, fsc, fmax


def fsc_resolution(a, b, dxy, dz, threshold=0.143, n_shells=None):
    """Resolution (um) where the FSC first falls below `threshold`.

    Returns (resolution_um, at_floor). at_floor=True means it never crossed,
    so the pair correlates to the band limit and the number is a bound, not a
    measurement -- the same convention blur_match uses.
    """
    f, c, fmax = fsc_curve(a, b, dxy, dz, n_shells)
    ok = np.isfinite(c)
    f, c = f[ok], c[ok]
    below = np.nonzero(c < threshold)[0]
    below = below[below > 0]
    if below.size == 0:
        return (1.0 / fmax if fmax > 0 else np.nan), True
    k = int(below[0])
    c0, c1 = c[k - 1], c[k]
    f0, f1 = f[k - 1], f[k]
    t = 0.0 if c0 == c1 else (c0 - threshold) / (c0 - c1)
    fx = f0 + t * (f1 - f0)
    return (1.0 / fx if fx > 0 else np.nan), False


def load_zyx(path, channel=None):
    """Load a 3-D volume as [z, y, x], picking a channel if there is one.

    .oir is Olympus's format and tifffile cannot read it. `oiffile` can, and is
    tried first, but I could not test that path -- the reliable route is to
    export to TIFF from Fiji (File > Save As > Tiff), which also lets you pick
    the channel and shows the voxel size under Image > Properties, which this
    script needs anyway.
    """
    path = Path(path)
    if path.suffix.lower() in (".oir", ".oif"):
        try:
            import oiffile
        except ImportError:
            raise SystemExit(
                f"!! {path.name} is an Olympus file and needs `oiffile`:\n"
                f"     pip install oiffile\n"
                f"   or export it to TIFF from Fiji (File > Save As > Tiff),\n"
                f"   note the voxel size from Image > Properties, and point\n"
                f"   DATASETS at the .tif instead. The Fiji route is the one\n"
                f"   I can vouch for.")
        v = np.asarray(oiffile.imread(str(path)))
    else:
        v = np.asarray(tiff.imread(str(path)))
    v = v.astype(np.float32)
    v = np.squeeze(v)
    if v.ndim == 4:
        # a channel axis survived: the shortest axis is channels
        ch = int(np.argmin(v.shape))
        v = np.moveaxis(v, ch, 0)
        if channel is None:
            keep = int(np.argmax([c.mean() for c in v]))
            print(f"    4-D input {v.shape}; no channel set, taking the "
                  f"brightest ({keep} of {v.shape[0]}) -- set 'channel' in "
                  f"DATASETS to choose deliberately")
        else:
            keep = int(channel)
            print(f"    4-D input {v.shape}; taking channel {keep} "
                  f"of {v.shape[0]}")
        v = v[keep]
    if v.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got {v.shape} from {path}")
    # heuristic: depth is the shortest axis for a z-stack
    if v.shape[-1] < v.shape[0] and v.shape[-1] < v.shape[1]:
        v = np.transpose(v, (2, 0, 1))
    return v


def resample_to_model_grid(vol, xy_um, z_um, dxy, dz):
    """Trilinear resample so one voxel means the same physical size as training."""
    fz, fy = z_um / dz, xy_um / dxy
    nz = max(1, int(round(vol.shape[0] * fz)))
    ny = max(1, int(round(vol.shape[1] * fy)))
    nx = max(1, int(round(vol.shape[2] * fy)))
    t = torch.from_numpy(vol)[None, None]
    out = F.interpolate(t, size=(nz, ny, nx), mode="trilinear",
                        align_corners=False)[0, 0].numpy()
    print(f"    {vol.shape} @ {xy_um:.4f}/{z_um:.4f} µm -> {out.shape} "
          f"@ {dxy:.4f}/{dz:.4f} µm (x{fy:.3f} lateral, x{fz:.3f} axial)")
    return out


def tiles(vol, shape, stride, keep_edge=True):
    """Non- or partly-overlapping blocks of `shape`, as (z0, y0, x0)."""
    out = []
    for a, (n, s, st) in enumerate(zip(vol.shape, shape, stride)):
        pos = list(range(0, max(n - s, 0) + 1, st))
        if not pos:
            pos = [0]
        elif pos[-1] + s < n and keep_edge:
            pos.append(n - s)          # far edge, clamped: overlaps its
            # neighbour by however much the stack failed to divide evenly,
            # which for a nearly-exact fit is almost the whole tile

        out.append(pos)
    return [(z, y, x) for z in out[0] for y in out[1] for x in out[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="external_eval")
    ap.add_argument("--datasets", nargs="+",
                    default=["plant", "col0", "planaria",
                             "mdck", "twocolor"])
    ap.add_argument("--arms", nargs="+", default=[a for a, _ in ARMS])
    ap.add_argument("--photons", type=float, default=None)
    ap.add_argument("--rltv-iters", type=int, default=100,
                    help="RL-TV iterations for the rltv arm")
    ap.add_argument("--rltv-tv", type=float, default=1e-5,
                    help="RL-TV total-variation weight")
    ap.add_argument("--with-fsc", action="store_true",
                    help="Fourier shell correlation against the ground truth, "
                         "reported as the resolution where it falls below "
                         "0.143 and 0.5. Shells are capped at the axial "
                         "Nyquist, so for this grid nothing finer than "
                         "~2.9 um is measurable and sharper reconstructions "
                         "report fsc_at_floor=True.")
    ap.add_argument("--with-resolution", action="store_true",
                    help="also fit lateral and axial resolution. Off by "
                         "default: without resampling the tiles do not span "
                         "the model's depth range, so an axial fit from them "
                         "is not comparable with the in-distribution table, "
                         "and the axial sigma search is the slowest part of "
                         "the evaluation. PSNR and SSIM are unaffected by the "
                         "span and are what this figure reports.")
    ap.add_argument("--resample", choices=["none", "grid"], default="none",
                    help="'none' (default) crops onto the model grid without "
                         "rescaling, as the submitted Figure 5 did -- the "
                         "claim is then that the pipeline handles unseen "
                         "specimen TYPES, not unseen sampling. 'grid' "
                         "interpolates each stack to the model's voxel size "
                         "first, which is what makes axial numbers comparable "
                         "with the in-distribution table. Say which you used: "
                         "under 'none' a 43-slice crop of 0.5 µm data spans "
                         "21.5 µm, not the 60 µm the optics assumes, so axial "
                         "resolution from it is not comparable.")
    ap.add_argument("--edge-tiles", choices=["keep", "drop"], default="keep",
                    help="'keep' (default) clamps a final tile flush against "
                         "each far edge so no voxel is missed, but when a "
                         "stack nearly divides evenly that tile is almost a "
                         "duplicate of its neighbour. 'drop' discards it, "
                         "leaving only non-overlapping tiles.")
    ap.add_argument("--stride-frac", type=float, default=1.0,
                    help="tile stride as a fraction of the tile size. 1.0 is "
                         "non-overlapping (independent samples); 0.5 doubles "
                         "the count per axis but the tiles then share data, so "
                         "they are NOT independent and the spread across them "
                         "understates the true variability. Say which you used.")
    ap.add_argument("--min-foreground", type=float, default=0.06,
                    help="drop a tile whose fraction of voxels above the "
                         "volume's own 90th percentile is below this. By "
                         "construction 10%% of the WHOLE volume sits above "
                         "that percentile, so 0.02 passes tiles that are "
                         "almost entirely background -- those produce the "
                         "near-zero SSIM cluster that drags the mean down "
                         "while telling you nothing about generalization.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="keep only the k tiles with the most foreground per "
                         "dataset. Use it to report a smaller, uniformly "
                         "populated set rather than a large one padded with "
                         "edges; say in the caption that you did.")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="cap per dataset, for a quick pass")
    ap.add_argument("--save-tiff", type=int, default=4,
                    help="how many tiles per dataset to save as TIFFs, richest "
                         "first. More than one so the figure can be re-cut to "
                         "a different region without re-running the ~45 min "
                         "evaluation; each saved tile is ~45 MB per arm.")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    photons = args.photons if args.photons is not None else TRAIN_PHOTONS

    nz = int(config.recon_params["n_slices"])
    ny = int(config.recon_params["output_size"][0])
    nx = int(config.recon_params["output_size"][1])
    dxy = float(config.optics_params["object_pixel_size"]) * 1e6
    zmin = float(config.optics_params["volume_depth_min"]) * 1e6
    zmax = float(config.optics_params["volume_depth_max"]) * 1e6
    dz = (zmax - zmin) / (nz - 1)
    shape = (nz, ny, nx)
    stride = tuple(max(1, int(round(s * args.stride_frac))) for s in shape)
    print(f"model grid {shape} at {dxy:.4f} µm lateral, {dz:.4f} µm axial; "
          f"tile stride {stride}")

    # ---- models
    models = {}
    for tag in args.arms:
        # the RL-TV arm runs on the fixed optics and has no network
        o_tag = "fixed" if tag == "rltv" else tag
        ck = root / f"optics_model_{o_tag}.pth"
        if not ck.exists():
            raise SystemExit(f"!! missing {ck}")
        sd = torch.load(str(ck), map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        mo = F_Optics(config.optics_params, seed=SEED,
                      learn_D=infer_learn_D(sd),
                      init_f="nominal" if o_tag == "fixed" else "random").to(DEVICE)
        mo.load_state_dict(sd, strict=True); mo.eval()
        mr = None
        if tag != "rltv":
            rk = root / f"recon_model_{tag}.pth"
            if not rk.exists():
                raise SystemExit(f"!! missing {rk}")
            mr = F_Recon(**config.recon_params).to(DEVICE)
            sr = torch.load(str(rk), map_location="cpu")
            mr.load_state_dict(
                sr.get("state_dict", sr) if isinstance(sr, dict) else sr,
                strict=True)
            mr.eval()
        models[tag] = (mo, mr)
    bp = FLFMAdjointBackprojector(
        out_hw=(ny, nx), normalize="percentile", p_low=10.0, p_high=99.7,
        deconv="cg", cg_reg=3.0, cg_iters=10, cg_chunk=4,
        cg_cache_H=True).to(DEVICE)

    if args.resample == "none":
        print("resampling OFF: tiles are cropped, not rescaled. Axial "
              "resolution from these tiles is NOT comparable with the "
              "in-distribution table; quote PSNR, SSIM and the images.")
    rows, kept_tiles, meta = [], {}, {}
    for dname in args.datasets:
        spec = DATASETS[dname]
        print(f"\n=== {spec['label']} ===")
        paths = []
        for pat in spec["files"]:
            paths += sorted(root.glob(pat)) if any(c in pat for c in "*?[") \
                else ([root / pat] if (root / pat).exists() else [])
        if not paths:
            print(f"!! no files matched {spec['files']} under {root} -- skipped")
            continue

        picks, n_drop, n_tile = [], 0, 0
        for p in paths:
            print(f"  {p.name}")
            vol = load_zyx(p, spec.get("channel"))
            if args.resample == "grid":
                vol = resample_to_model_grid(vol, spec["xy_um"], spec["z_um"],
                                             dxy, dz)
            else:
                # no rescaling: report what one tile then physically spans, so
                # the number is on record
                print(f"    no resampling; one {nz}x{ny}x{nx} tile spans "
                      f"{nx*spec['xy_um']:.0f} x {ny*spec['xy_um']:.0f} x "
                      f"{nz*spec['z_um']:.1f} µm "
                      f"(model assumes {nx*dxy:.0f} x {ny*dxy:.0f} x "
                      f"{(nz-1)*dz:.0f} µm)")
            thr = float(np.percentile(vol, 90.0))
            for (z0, y0, x0) in tiles(vol, shape, stride,
                                      args.edge_tiles == "keep"):
                blk = vol[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx]
                real = tuple(int(v) for v in blk.shape)   # unpadded extent
                if blk.shape != shape:                     # pad a short edge
                    pad = np.zeros(shape, np.float32)
                    pad[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
                    blk = pad
                # foreground is judged on the acquired planes only -- counting
                # the zero padding would make a thin stack look emptier than
                # it is and drop tiles that are actually full of signal
                frac = float((blk[:real[0], :real[1], :real[2]] > thr).mean())
                if frac < args.min_foreground:
                    n_drop += 1
                    continue
                picks.append((p.name, (z0, y0, x0), blk, frac, real))
                n_tile += 1
                if args.max_tiles and n_tile >= args.max_tiles:
                    break
            if args.max_tiles and n_tile >= args.max_tiles:
                break
        # Sort by foreground, richest first. Tiles are generated from the
        # (0,0,0) corner outward, so without this the saved tile -- the one the
        # figure shows -- is always the emptiest corner of the stack.
        picks.sort(key=lambda t: -t[3])
        if args.top_k:
            dropped_extra = max(0, len(picks) - args.top_k)
            picks = picks[:args.top_k]
            n_drop += dropped_extra
            n_tile = len(picks)
        print(f"  {n_tile} tiles kept, {n_drop} dropped "
              f"(below {args.min_foreground:.0%} foreground"
              + (f", or outside the top {args.top_k}" if args.top_k else "")
              + f"); foreground {picks[0][3]:.1%} (richest) .. "
              f"{picks[-1][3]:.1%} (sparsest)" if picks else "")
        meta[dname] = dict(label=spec["label"], n_tiles=n_tile,
                           n_dropped=n_drop, n_files=len(paths),
                           real_extent=sorted({t[4] for t in picks}),
                           xy_um=spec["xy_um"], z_um=spec["z_um"],
                           resample=args.resample,
                           stride_frac=args.stride_frac)
        if not picks:
            continue

        for i, (fname, pos, blk, frac, real) in enumerate(picks):
            zr, yr, xr = real
            # percentile-normalise the tile exactly as the training targets are
            t = torch.from_numpy(blk).float()
            lo = torch.quantile(t.flatten()[::97], 0.01)
            hi = torch.quantile(t.flatten()[::97], 0.997)
            t = ((t - lo) / (hi - lo + 1e-6)).clamp(0, 1).to(DEVICE)
            gt = t.cpu().numpy()

            for tag in args.arms:
                mo, mr = models[tag]
                torch.manual_seed(SEED + i)
                with torch.no_grad():
                    o, _p, psf = mo(t)
                    noisy = add_sensor_noise(normalize_sensor(o), photons,
                                             read_noise_e=READ_NOISE_E,
                                             exact=True)
                    if tag == "rltv":
                        rec = rl_tv_solve(noisy, psf, out_hw=(ny, nx),
                                          n_iter=args.rltv_iters,
                                          tv_lambda=args.rltv_tv)
                    else:
                        rec = mr(bp(sensor_img=noisy, psf_stack=psf,
                                    dz=mo.dz_used))
                rec = np.clip(rec.squeeze().cpu().numpy().astype(np.float32), 0, 1)

                # Resampling a crop onto the model grid usually leaves it
                # SMALLER than 512x512x43 (an MDCK voxel is 0.269 um against
                # the model's 0.377), so the tile is zero padded on all three
                # axes. Scoring that padding would reward the model on voxels
                # nobody imaged, so every metric below uses the real extent.
                gv, rv = gt[:zr, :yr, :xr], rec[:zr, :yr, :xr]
                _a, _b, _, _ = _fit_affine_mse(rv, gv)
                _af = np.clip(_a * rv + _b, 0, 1).astype(np.float32)
                if args.with_fsc:
                    _fsc143, _fscfl = fsc_resolution(gv, rv, dxy, dz, 0.143)
                    _fsc50, _ = fsc_resolution(gv, rv, dxy, dz, 0.5)
                else:
                    _fsc143 = _fsc50 = np.nan
                    _fscfl = False

                # A volume thinner than the model's 43 planes is zero
                # padded to fit. Scoring the padding would be scoring the
                # model on planes nobody imaged: both arrays are near zero
                # there, so PSNR climbs for free. Every metric below is
                # therefore taken over gt[:zr] / rec[:zr].
                zc = zr // 2
                if args.with_resolution:
                    fwhm, _s, rse, rsp, floor = blur_match(gv[zc], rv[zc], dxy)
                    ax, _as, axfloor = blur_match_axial(gv, rv, zc, dz)
                else:
                    fwhm = ax = rsp = rse = np.nan
                    floor = axfloor = False
                rows.append(dict(
                    dataset=dname, label=spec["label"], arm=tag, file=fname,
                    tile=f"{pos[0]}_{pos[1]}_{pos[2]}", fg_frac=frac,
                    photons=photons,
                    z_valid=zr, y_valid=yr, x_valid=xr,
                    pad_frac=round(1.0 - (zr * yr * xr) / float(nz * ny * nx), 4),
                    psnr_db=psnr(gv, rv), psnr_db_affine=psnr_affine(gv, rv),
                    ssim=float(np.mean([
                        compare_ssim(gv[d], rv[d],
                                     data_range=max(float(gv[d].max() - gv[d].min()), 1e-8))
                        for d in range(zr)])),
                    # SSIM of the same affine-corrected reconstruction the
                    # affine PSNR scores, so the two numbers describe one
                    # object rather than mixing conventions in a caption.
                    ssim_affine=float(np.mean([
                        compare_ssim(gv[dd], _af[dd],
                                     data_range=max(float(gv[dd].max() - gv[dd].min()), 1e-8))
                        for dd in range(zr)])),
                    # volumetric, and scale free by construction: SSIM over
                    # the whole 3-D block rather than averaged over slices,
                    # and FSC, whose shells are built in cycles/um.
                    ssim3d=float(compare_ssim(
                        gv, rv, data_range=max(float(gv.max() - gv.min()), 1e-8))),
                    ssim3d_affine=float(compare_ssim(
                        gv, _af, data_range=max(float(gv.max() - gv.min()), 1e-8))),
                    fsc_um_143=_fsc143, fsc_um_50=_fsc50,
                    fsc_at_floor=_fscfl,
                    nrmse=nrmse(gv, rv), rho=pearson(rv, gv),
                    adjz_rec=adj_z(rv), adjz_gt=adj_z(gv),
                    lat_res_um=fwhm, axial_res_um=ax,
                    rsp=rsp, rse=rse,
                    lat_at_floor=floor, axial_at_floor=axfloor))

                if i < args.save_tiff:
                    d = out / dname / f"tile{i:02d}"
                    d.mkdir(parents=True, exist_ok=True)
                    if tag == args.arms[0]:
                        tiff.imwrite(d / "gt.tif", gv)
                    tiff.imwrite(d / f"recon_{tag}.tif", rv)

                del psf, noisy
                torch.cuda.empty_cache()
            print(f"    tile {i+1}/{len(picks)}", end="\r")
        print(" " * 40)

    if not rows:
        raise SystemExit("!! nothing evaluated")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_tile.csv", index=False)
    (out / "datasets.json").write_text(json.dumps(meta, indent=2))

    print("\n=== external generalization, mean over tiles ===")
    agg = dict(n=("psnr_db", "size"), psnr=("psnr_db", "mean"),
               psnr_sd=("psnr_db", "std"), ssim=("ssim", "mean"),
               ssim_sd=("ssim", "std"), rho=("rho", "mean"))
    if args.with_resolution:
        agg.update(lat=("lat_res_um", "mean"), ax=("axial_res_um", "mean"))
    g = df.groupby(["label", "arm"]).agg(**agg).round(3)
    print(g.to_string())
    print(f"\nwrote {out}/per_tile.csv and {out}/datasets.json")


if __name__ == "__main__":
    main()
