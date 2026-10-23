"""
time_reconstruction.py -- wall-clock cost of recovering one volume, per arm.

The number that belongs beside PSNR when a learned reconstruction is compared
against an iterative one. RL-TV wins axial resolution in Table 1; this says
what it costs to win it.

WHAT IS TIMED, and why it is drawn where it is: sensor image -> volume, and
nothing else. The optics forward pass is excluded because in a real instrument
it does not happen -- the camera produces the sensor image. The PSF stack is
excluded for the same reason: it is a one-off calibration, measured or
simulated once per instrument, not per acquisition. So the timer covers the CG
lift plus the network for the learned arms, and the RL iterations for RL-TV.
Including either excluded piece would flatter the iterative method, since both
arms would then pay the same large constant.

CUDA is asynchronous, so every interval is bracketed by torch.cuda.synchronize()
-- without it the timer measures how fast Python can enqueue kernels, which for
the learned arms is roughly instant and completely wrong. The first volume is a
warm-up and is discarded: cuFFT plans, cuDNN algorithm selection and the
allocator's first-touch cost all land on it, and on one GPU here the warm-up
volume ran 3x the steady-state time.

Usage:
    python time_reconstruction.py --tag f --optics optics_model_f.pth \
                                  --recon recon_model_f.pth
    python time_reconstruction.py --tag rltv --method rltv --fixed-optics \
                                  --optics optics_model_fixed.pth

Appends one row per arm to timing.csv, which make_table2.py turns into the
summary table.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from config import config
from F_Optics_fixed import F_Optics
from F_VCD_pytorch_multiple_views import F_Recon
from dataset import MyDataset
from adjoint import FLFMAdjointBackprojector
from sensor_noise import add_sensor_noise
from rltv import rl_tv_solve
from make_paper_table import (DEVICE, TEST_DATA_DIR, READ_NOISE_E,
                              TRAIN_PHOTONS, SEED, normalize_sensor,
                              infer_learn_D)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--optics", default="optics_model.pth")
    ap.add_argument("--recon", default="recon_model.pth")
    ap.add_argument("--method", choices=["cnn", "rltv"], default="cnn")
    ap.add_argument("--fixed-optics", action="store_true")
    ap.add_argument("--n-volumes", type=int, default=6,
                    help="first is a warm-up and is discarded")
    ap.add_argument("--photons", type=float, default=None)
    ap.add_argument("--rltv-iters", type=int, default=100)
    ap.add_argument("--rltv-tv", type=float, default=1e-5)
    ap.add_argument("--out", default="timing.csv")
    args = ap.parse_args()

    photons = args.photons if args.photons is not None else TRAIN_PHOTONS
    torch.manual_seed(SEED); np.random.seed(SEED)

    sd_o = torch.load(args.optics, map_location="cpu")
    sd_o = sd_o.get("state_dict", sd_o) if isinstance(sd_o, dict) else sd_o
    learn_D = infer_learn_D(sd_o)
    model_optics = F_Optics(config.optics_params, seed=SEED, learn_D=learn_D,
                            init_f="nominal" if args.fixed_optics else "random").to(DEVICE)
    model_optics.load_state_dict(sd_o, strict=True)
    model_optics.eval()

    out_hw = (config.recon_params["output_size"][0],
              config.recon_params["output_size"][1])

    model_recon = None
    bp = None
    if args.method == "cnn":
        model_recon = F_Recon(**config.recon_params).to(DEVICE)
        sd_r = torch.load(args.recon, map_location="cpu")
        model_recon.load_state_dict(
            sd_r.get("state_dict", sd_r) if isinstance(sd_r, dict) else sd_r,
            strict=True)
        model_recon.eval()
        bp = FLFMAdjointBackprojector(
            out_hw=out_hw, normalize="percentile", p_low=10.0, p_high=99.7,
            deconv="cg", cg_reg=3.0, cg_iters=10, cg_chunk=4,
            cg_cache_H=True).to(DEVICE)

    ds = MyDataset(stack_dir=TEST_DATA_DIR, config=config,
                   transform_3d=config.transform_3d)
    n = min(len(ds), args.n_volumes)
    print(f"[{args.tag}] method={args.method}  {n} volumes "
          f"(1 warm-up)  N={photons:.0f} photons  device={DEVICE}")

    lift_t, net_t, total_t = [], [], []
    for i in range(n):
        target3d, _lo, _hi = ds[i]
        target3d = target3d.to(DEVICE)
        with torch.no_grad():
            out_optics, _p, psf_stack = model_optics(target3d)
            sensor = normalize_sensor(out_optics)
            torch.manual_seed(SEED + i)
            noisy = add_sensor_noise(sensor, photons,
                                     read_noise_e=READ_NOISE_E, exact=True)
        sync()

        with torch.no_grad():
            if args.method == "rltv":
                t0 = time.perf_counter()
                _ = rl_tv_solve(noisy, psf_stack, out_hw=out_hw,
                                n_iter=args.rltv_iters, tv_lambda=args.rltv_tv)
                sync()
                t3 = time.perf_counter()
                tl = tn = float("nan")
            else:
                t0 = time.perf_counter()
                coarse = bp(sensor_img=noisy, psf_stack=psf_stack,
                            dz=model_optics.dz_used)
                sync()
                t1 = time.perf_counter()
                _ = model_recon(coarse)
                sync()
                t3 = time.perf_counter()
                tl, tn = t1 - t0, t3 - t1

        tag = "warm-up, discarded" if i == 0 else ""
        if i > 0:
            lift_t.append(tl); net_t.append(tn); total_t.append(t3 - t0)
        print(f"  vol {i}: total {t3-t0:7.3f} s"
              + (f"   (lift {tl:.3f} + net {tn:.3f})" if args.method == "cnn" else "")
              + (f"   {tag}" if tag else ""))

        del target3d, out_optics, psf_stack, sensor, noisy
        torch.cuda.empty_cache()

    tot = np.array(total_t)
    row = {
        "tag": args.tag, "method": args.method, "photons": int(photons),
        "n_timed": len(tot),
        "total_s_mean": tot.mean(), "total_s_sd": tot.std(ddof=0),
        "volumes_per_s": 1.0 / tot.mean(),
        "lift_s_mean": float(np.mean(lift_t)) if args.method == "cnn" else float("nan"),
        "net_s_mean": float(np.mean(net_t)) if args.method == "cnn" else float("nan"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    print(f"\n[{args.tag}] {tot.mean():.3f} +/- {tot.std(ddof=0):.3f} s per volume "
          f"({row['volumes_per_s']:.3f} volumes/s) on {row['device']}")

    p = Path(args.out)
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended to {p}")


if __name__ == "__main__":
    main()
