"""
finetune_external.py -- how much does a modest fine-tune of the POST-PROCESSING
network buy, on a specimen type the model was never trained on, and does it buy
the same amount for the conventional and the learned encoder?

THE QUESTION
------------
The learned encoder is ahead in distribution. A natural question is
whether that lead is real or whether the conventional array would catch up once
its network is adapted to the new specimen. This script answers that directly:
the optics are FROZEN, only F_Recon is fine-tuned, and both arms get an
identical budget, schedule, seed and tile order.

WHY THERE IS A STEP SWEEP AND NOT A SINGLE BUDGET
-------------------------------------------------
A single budget reports one point with no evidence that it was the right one.
The first run of this script showed held-out PSNR FALLING after fine-tuning on
the same specimen type while SSIM rose -- the signature of a network learning
to smooth, and of overfitting a handful of tiles. So cheap metrics (PSNR, SSIM,
rho) are now evaluated on the held-out split every --eval-every steps, the whole
curve is written out, and the checkpoint that maximises --select-by is kept.
Either the curve peaks and you report the fine-tune at a defensible stopping
point, or it declines from the start and you have honestly shown that this
network does not benefit from adaptation at this data scale. Both are results;
one point in the middle of an unknown curve is not.

The expensive metrics (blur-matched lateral resolution and the depth spread)
run only at the start and at the end, on the selected checkpoint.

WHAT MAKES IT A FAIR TEST, AND WHAT WOULD BREAK IT
--------------------------------------------------
* The split is BY FILE, not by tile. Tiles from one stack share illumination,
  focus and specimen, so splitting tiles would leak the evaluation specimen
  into the fine-tuning set and the gain would be inflated.
* Evaluation noise is seeded per tile, so every evaluation of a given tile sees
  the SAME sensor realisation and differences are the network, not the draw.
* Both arms start from their own checkpoint and are trained with the same seed,
  so the tile order and the noise sequence match step for step.
* The optics are frozen in eval() with requires_grad False. If they were left
  trainable this would stop being an ablation of the network.
* Selecting the checkpoint on the held-out split means that split is no longer
  untouched. The control files are never used for selection, so they remain a
  clean test -- read the control numbers, not the held-out ones, as the
  unbiased estimate of what adaptation bought.

Preprocessing (resampling to the model voxel grid, tiling, foreground
filtering, percentile normalisation) is imported from eval_external.py so the
numbers stay comparable with the generalisation table.

    python finetune_external.py --root ./data/external \
        --arms fixed f fD --steps 300 --eval-every 25

Add --dry-run to list the split and the tile counts without touching a GPU.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from skimage.metrics import structural_similarity as compare_ssim

from config import config
from F_Optics_fixed import F_Optics
from F_VCD_pytorch_multiple_views import F_Recon
from adjoint import FLFMAdjointBackprojector
from sensor_noise import add_sensor_noise
from custom_losses import (mae_loss, EdgeLoss, WrappedSSIMLoss,
                           axial_diff_matching_loss)
from eval_external import load_zyx, resample_to_model_grid, tiles
from make_paper_table import (DEVICE, SEED, READ_NOISE_E, TRAIN_PHOTONS,
                              normalize_sensor, infer_learn_D,
                              psnr, pearson, blur_match)

# Voxel sizes read from the ImageJ/Leica metadata of the stacks themselves
# (XResolution 2.75 px/µm, spacing 0.5002 µm), not assumed.
XY_UM, Z_UM = 1.0 / 2.75, 0.5002

# SAM_PI_Col-0 is deliberately absent: 114 slices at 0.5 µm is 57 µm, short of
# the 60 µm the optics assumes, so it would have to be padded and every axial
# number taken on it would mean something different from the others.
FINETUNE_FILES = [f"PC_PI_Col0_{i}.tif" for i in (1, 2, 3, 4, 5)]
EVAL_FILES = [f"PC_PI_Col0_{i}.tif" for i in (6, 7, 8)]
# Not fine-tuned on, not used for checkpoint selection, and not even the same
# label -- the one unbiased read on what the fine-tune actually bought.
CONTROL_FILES = ["Hypocotyl_GFP-MBD.tif"]


# ---------------------------------------------------------------- tiles
def shape_um(shape):
    dxy = float(config.optics_params["object_pixel_size"]) * 1e6
    zmin = float(config.optics_params["volume_depth_min"]) * 1e6
    zmax = float(config.optics_params["volume_depth_max"]) * 1e6
    dz = (zmax - zmin) / (shape[0] - 1)
    return dxy, dz


def build_tiles(root, files, shape, stride, min_fg, max_per_file, resample):
    """Normalised model-sized tiles, tagged with the file they came from."""
    out = []
    for name in files:
        p = Path(root) / name
        if not p.exists():
            print(f"  !! missing, skipped: {p}")
            continue
        vol = load_zyx(p)
        if resample:
            vol = resample_to_model_grid(vol, XY_UM, Z_UM, *shape_um(shape))
        thr = float(np.percentile(vol, 90.0))
        kept = 0
        for (z0, y0, x0) in tiles(vol, shape, stride, keep_edge=True):
            blk = vol[z0:z0 + shape[0], y0:y0 + shape[1], x0:x0 + shape[2]]
            if blk.shape != tuple(shape):
                continue                       # drop short edges rather than pad
            frac = float((blk > thr).mean())
            if frac < min_fg:
                continue
            t = torch.from_numpy(np.ascontiguousarray(blk)).float()
            # percentile normalisation, identical to eval_external.py
            lo = torch.quantile(t.flatten()[::97], 0.01)
            hi = torch.quantile(t.flatten()[::97], 0.997)
            t = ((t - lo) / (hi - lo + 1e-6)).clamp(0, 1)
            out.append(dict(file=name, pos=(z0, y0, x0), vol=t, fg=frac))
            kept += 1
            if max_per_file and kept >= max_per_file:
                break
        print(f"  {name}: {kept} tiles kept")
    return out


# ---------------------------------------------------------------- models
def load_arm(tag, optics_path, recon_path):
    sd_o = torch.load(optics_path, map_location="cpu")
    sd_o = sd_o.get("state_dict", sd_o) if isinstance(sd_o, dict) else sd_o
    learn_D = infer_learn_D(sd_o)
    mo = F_Optics(config.optics_params, seed=SEED, learn_D=learn_D,
                  init_f="nominal" if tag == "fixed" else "random").to(DEVICE)
    mo.load_state_dict(sd_o, strict=True)
    mo.eval()
    for p in mo.parameters():            # the whole point of the experiment
        p.requires_grad_(False)

    mr = F_Recon(**config.recon_params).to(DEVICE)
    sd_r = torch.load(recon_path, map_location="cpu")
    mr.load_state_dict(sd_r.get("state_dict", sd_r)
                       if isinstance(sd_r, dict) else sd_r, strict=True)
    print(f"[{tag}] learn_D={learn_D}, optics frozen "
          f"({sum(p.numel() for p in mr.parameters()):,} trainable in F_Recon)")
    return mo, mr


def lift(mo, bp, vol, photons, seed):
    """Sensor formation, noise and the CG lift. No parameters, so no grad."""
    torch.manual_seed(seed)
    with torch.no_grad():
        o, _pitch, psf = mo(vol)
        noisy = add_sensor_noise(normalize_sensor(o), photons,
                                 read_noise_e=READ_NOISE_E, exact=True)
        return bp(sensor_img=noisy, psf_stack=psf, dz=mo.dz_used)


# ---------------------------------------------------------------- metrics
def recon_tile(mo, mr, bp, t, photons, i):
    """Reconstruction and ground truth for one tile, on a seed fixed per tile."""
    gt_t = t["vol"].to(DEVICE)
    coarse = lift(mo, bp, gt_t, photons, SEED + 1000 + i)
    with torch.no_grad():
        rec = mr(coarse)
    return (np.clip(rec.squeeze().cpu().numpy().astype(np.float32), 0, 1),
            gt_t.cpu().numpy())


def evaluate_fast(mo, mr, bp, tile_set, photons):
    """PSNR, SSIM and rho only. No blur matching, so this is cheap enough to
    run on a schedule during training."""
    mr.eval()
    rows = []
    for i, t in enumerate(tile_set):
        rec, gt = recon_tile(mo, mr, bp, t, photons, i)
        rows.append(dict(
            file=t["file"], tile="_".join(map(str, t["pos"])),
            psnr_db=psnr(gt, rec),
            ssim=float(np.mean([
                compare_ssim(gt[d], rec[d],
                             data_range=max(float(gt[d].max() - gt[d].min()), 1e-8))
                for d in range(gt.shape[0])])),
            rho=pearson(rec, gt)))
    mr.train()
    return pd.DataFrame(rows)


def evaluate_full(mo, mr, bp, tile_set, photons, dxy, depth_idx):
    """Everything, including the blur-matched resolution and the depth spread."""
    mr.eval()
    rows = []
    for i, t in enumerate(tile_set):
        rec, gt = recon_tile(mo, mr, bp, t, photons, i)
        lat = np.array([blur_match(gt[d], rec[d], dxy)[0] for d in depth_idx],
                       dtype=float)
        rows.append(dict(
            file=t["file"], tile="_".join(map(str, t["pos"])), fg=t["fg"],
            psnr_db=psnr(gt, rec),
            ssim=float(np.mean([
                compare_ssim(gt[d], rec[d],
                             data_range=max(float(gt[d].max() - gt[d].min()), 1e-8))
                for d in range(gt.shape[0])])),
            rho=pearson(rec, gt),
            lat_res_um=float(np.nanmean(lat)),
            depth_spread=float(np.nanmax(lat) / np.nanmin(lat))))
    return pd.DataFrame(rows)


FULL_KEYS = ("psnr_db", "ssim", "rho", "lat_res_um", "depth_spread")


def summarize(df):
    return {k: float(df[k].mean()) for k in FULL_KEYS if k in df}


# ---------------------------------------------------------------- plot
def plot_sweep(sweep, out, select_by):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = ["psnr_db", "ssim", "rho"]
    labels = {"psnr_db": "PSNR (dB)", "ssim": "SSIM", "rho": r"$\rho$"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 2.7),
                             squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for tag, g in sweep.groupby("arm", sort=False):
            ax.plot(g["step"], g[m], marker="o", ms=3, lw=1.3, label=tag)
        ax.set_xlabel("fine-tuning step")
        ax.set_ylabel(labels[m])
        if m == select_by:
            ax.set_title("selection metric", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"sweep.{ext}", dpi=300)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/external",
                    help="folder holding the plant stacks")
    ap.add_argument("--arms", nargs="+", default=["fixed", "f", "fD"])
    ap.add_argument("--optics-fmt", default="optics_model_{}.pth")
    ap.add_argument("--recon-fmt", default="recon_model_{}.pth")
    ap.add_argument("--steps", type=int, default=300,
                    help="optimiser steps; identical for every arm")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="lower than training from scratch: this is a fine-tune")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="cheap-metric evaluation interval, in steps. 0 = off, "
                         "which reduces this to a single-budget run")
    ap.add_argument("--select-by", default="psnr_db",
                    choices=["psnr_db", "ssim", "rho"],
                    help="held-out metric the kept checkpoint maximises")
    ap.add_argument("--eval-control-during", action="store_true",
                    help="also sweep the control split; costs more and the "
                         "control is only read at the end anyway")
    ap.add_argument("--photons", type=float, default=None)
    ap.add_argument("--min-foreground", type=float, default=0.06)
    ap.add_argument("--max-tiles-per-file", type=int, default=4)
    ap.add_argument("--stride-frac", type=float, default=1.0,
                    help="tile stride as a fraction of the tile size; 1.0 = "
                         "non-overlapping")
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--depth-count", type=int, default=9,
                    help="depths at which lateral resolution and the depth "
                         "spread are measured. Reported with the numbers, "
                         "because the spread depends on it.")
    ap.add_argument("--out", default="finetune_external")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    photons = args.photons or TRAIN_PHOTONS
    nz = int(config.optics_params["n_slices"])
    ny, nx = (config.recon_params["output_size"][0],
              config.recon_params["output_size"][1])
    shape = (nz, ny, nx)
    stride = tuple(max(1, int(round(s * args.stride_frac))) for s in shape)
    dxy, dz = shape_um(shape)
    depth_idx = sorted({int(round(v)) for v in
                        np.linspace(0, nz - 1, max(2, args.depth_count))})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"model grid {shape} at {dxy:.4f} / {dz:.4f} µm, N = {photons:.0f} "
          f"photons, read noise {READ_NOISE_E} e-")
    print(f"resolution measured at {len(depth_idx)} depths\n")

    print("fine-tuning files:")
    ft = build_tiles(args.root, FINETUNE_FILES, shape, stride,
                     args.min_foreground, args.max_tiles_per_file,
                     not args.no_resample)
    print("held-out evaluation files:")
    ev = build_tiles(args.root, EVAL_FILES, shape, stride,
                     args.min_foreground, args.max_tiles_per_file,
                     not args.no_resample)
    print("control files (different label, never fine-tuned on, never selected on):")
    ct = build_tiles(args.root, CONTROL_FILES, shape, stride,
                     args.min_foreground, args.max_tiles_per_file,
                     not args.no_resample)

    overlap = {t["file"] for t in ft} & {t["file"] for t in ev + ct}
    if overlap:
        raise SystemExit(f"!! same file in both splits: {overlap}")
    print(f"\n{len(ft)} fine-tuning tiles, {len(ev)} held-out tiles, "
          f"{len(ct)} control tiles")
    if args.dry_run:
        return
    if not ft or not ev:
        raise SystemExit("!! empty split -- check --root and the file names")

    bp = FLFMAdjointBackprojector(
        out_hw=(ny, nx), normalize="percentile", p_low=10.0, p_high=99.7,
        deconv="cg", cg_reg=3.0, cg_iters=10, cg_chunk=4, cg_cache_H=True,
    ).to(DEVICE)
    edge = EdgeLoss(); ssim_l = WrappedSSIMLoss().to(DEVICE)
    w = dict(config.loss["Recon_loss"])
    print(f"loss weights {w}, lr {args.lr}, {args.steps} steps per arm, "
          f"selecting on held-out {args.select_by}\n")

    records, sweep_rows = [], []
    for tag in args.arms:
        mo, mr = load_arm(tag, args.optics_fmt.format(tag),
                          args.recon_fmt.format(tag))
        splits = [("heldout", ev)] + ([("control", ct)] if ct else [])
        before = {n: evaluate_full(mo, mr, bp, s, photons, dxy, depth_idx)
                  for n, s in splits}

        # step 0 of the sweep is the zero-shot model, so the curve starts where
        # the generalisation table does
        z = summarize(evaluate_fast(mo, mr, bp, ev, photons))
        sweep_rows.append(dict(arm=tag, step=0, split="heldout", **z))
        best = dict(score=z[args.select_by], step=0,
                    state=copy.deepcopy({k: v.detach().cpu()
                                         for k, v in mr.state_dict().items()}))
        print(f"  [{tag}] step   0  {args.select_by} {best['score']:.4f} (zero-shot)")

        # Identical schedule for every arm: same seed, so the tile order and
        # the training noise sequence are the same sequence of draws.
        torch.manual_seed(SEED); np.random.seed(SEED)
        opt = Adam(mr.parameters(), lr=args.lr, weight_decay=1e-4)
        mr.train()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            t = ft[np.random.randint(len(ft))]
            gt_t = t["vol"].to(DEVICE)
            coarse = lift(mo, bp, gt_t, photons, SEED + 10_000 + step)
            opt.zero_grad()
            o = mr(coarse)
            loss = (w["mae_loss"] * mae_loss(o, gt_t)
                    + w["edge_loss"] * edge(o, gt_t)
                    + w["ssim_loss"] * (1.0 - ssim_l(o.clamp(0, 1), gt_t))
                    + w["axial_diff_matching_loss"] * axial_diff_matching_loss(o, gt_t))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mr.parameters(), 1.0)
            opt.step()

            if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
                s = summarize(evaluate_fast(mo, mr, bp, ev, photons))
                sweep_rows.append(dict(arm=tag, step=step, split="heldout", **s))
                if args.eval_control_during and ct:
                    sc = summarize(evaluate_fast(mo, mr, bp, ct, photons))
                    sweep_rows.append(dict(arm=tag, step=step, split="control", **sc))
                mark = ""
                if s[args.select_by] > best["score"]:
                    best = dict(score=s[args.select_by], step=step,
                                state=copy.deepcopy({k: v.detach().cpu()
                                                     for k, v in mr.state_dict().items()}))
                    mark = "  <- best"
                print(f"  [{tag}] step {step:3d}  loss {loss.item():.4f}  "
                      f"PSNR {s['psnr_db']:.3f}  SSIM {s['ssim']:.4f}  "
                      f"rho {s['rho']:.4f}{mark}")
        mins = (time.time() - t0) / 60
        print(f"  [{tag}] {args.steps} steps in {mins:.1f} min; "
              f"best held-out {args.select_by} at step {best['step']}")

        # final-step model, then the selected one
        after_final = {n: evaluate_full(mo, mr, bp, s, photons, dxy, depth_idx)
                       for n, s in splits}
        torch.save(mr.state_dict(), out / f"recon_model_{tag}_ft_final.pth")
        mr.load_state_dict(best["state"])
        mr.to(DEVICE)
        after_best = {n: evaluate_full(mo, mr, bp, s, photons, dxy, depth_idx)
                      for n, s in splits}
        torch.save(mr.state_dict(), out / f"recon_model_{tag}_ft_best.pth")

        for split, _ in splits:
            for when, d in (("before", before), ("after_final", after_final),
                            ("after_best", after_best)):
                d[split].to_csv(out / f"per_tile_{tag}_{split}_{when}.csv",
                                index=False)
                records.append(dict(
                    arm=tag, split=split, when=when, n_tiles=len(d[split]),
                    step=(0 if when == "before"
                          else args.steps if when == "after_final"
                          else best["step"]),
                    minutes=(0.0 if when == "before" else mins),
                    **summarize(d[split])))
        del mo, mr
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(out / "summary.csv", index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out / "sweep.csv", index=False)
    plot_sweep(sweep[sweep.split == "heldout"], out, args.select_by)
    json.dump(dict(steps=args.steps, lr=args.lr, photons=photons,
                   eval_every=args.eval_every, select_by=args.select_by,
                   depth_count=len(depth_idx),
                   finetune_files=FINETUNE_FILES, eval_files=EVAL_FILES,
                   control_files=CONTROL_FILES,
                   n_finetune_tiles=len(ft), n_eval_tiles=len(ev),
                   n_control_tiles=len(ct)),
              open(out / "meta.json", "w"), indent=2)

    with pd.option_context("display.width", 220,
                           "display.float_format", lambda v: f"{v:.3f}"):
        for split in df["split"].unique():
            print(f"\n=== {split} ===")
            print(df[df.split == split].drop(columns=["split"]).to_string(index=False))
        print("\ndelta against zero-shot (selected checkpoint):")
        for split in df["split"].unique():
            s = df[df.split == split]
            for tag in args.arms:
                b = s[(s.arm == tag) & (s.when == "before")]
                a = s[(s.arm == tag) & (s.when == "after_best")]
                if b.empty or a.empty:
                    continue
                print(f"  {split:8s} {tag:6s} step {int(a.step.iloc[0]):3d}  "
                      f"PSNR {float(a.psnr_db.iloc[0]-b.psnr_db.iloc[0]):+.2f} dB   "
                      f"SSIM {float(a.ssim.iloc[0]-b.ssim.iloc[0]):+.3f}   "
                      f"lat {float(a.lat_res_um.iloc[0]-b.lat_res_um.iloc[0]):+.3f} µm   "
                      f"spread {float(a.depth_spread.iloc[0]-b.depth_spread.iloc[0]):+.3f}")
    print(f"\nwrote {out}/summary.csv, {out}/sweep.csv, {out}/sweep.pdf")
    print("Read the CONTROL rows as the unbiased result: the held-out split was "
          "used to choose the checkpoint.")


if __name__ == "__main__":
    main()
