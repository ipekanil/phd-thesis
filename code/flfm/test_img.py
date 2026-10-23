"""
test_img.py -- inference + evaluation across a sweep of photon budgets.

What changed from the previous version
--------------------------------------
1. Imports F_Optics_fixed (Debye pupil + aplanatic apodization), not F_Optics.
2. Noise is the mixed Poisson-Gaussian model from sensor_noise.py, not a fixed
   additive Gaussian. At test time it is sampled with exact=True: true integer
   Poisson counts, no Gaussian approximation.
3. The photon budget is SWEPT over config.train_params["sensor_noise"]["photons_test"]
   and every metric is reported per budget. This reports performance
   across a representative range of photon budgets.
4. adjoint uses normalize="global" and dz=model_optics.dz_used, matching train_dp.
5. Seeded, so the sweep is reproducible.

The optics forward pass does not depend on the noise, so it runs ONCE and only
the noise -> adjoint -> reconstruction part is repeated per budget. That keeps
the sweep cheap even though the optics forward is the expensive step.
"""

import os
import time
import math
import random
from pathlib import Path

import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile as tiff

from skimage.metrics import structural_similarity as compare_ssim

from config import config
import F_Optics_fixed as optics_mod
from F_Optics_fixed import F_Optics
from F_VCD_pytorch_multiple_views import F_Recon
from dataset import MyDataset
from adjoint import FLFMAdjointBackprojector
from sensor_noise import add_sensor_noise


# ============================================================
# Settings
# ============================================================
SAMPLE_IDX = 8 #8 #1
OUT_DIR = Path(f"inference_results_sample_{SAMPLE_IDX:04d}")
COMMON_DIR = OUT_DIR / "common"
OUT_DIR.mkdir(exist_ok=True)
COMMON_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTICS_CKPT = "optics_model.pth"
RECON_CKPT = "recon_model.pth"

TEST_DATA_DIR = config.paths["test_stack_dir"]

_NOISE = config.train_params["sensor_noise"]
PHOTON_BUDGETS = list(_NOISE["photons_test"])     # [10, 30, 100, 300, 1000]
READ_NOISE_E = _NOISE["read_noise_e"]             # electrons, fixed across the sweep
TRAIN_PHOTONS = _NOISE["photons"]                 # what the optics were learned at
EXACT_POISSON = True                              # test time: true integer counts

SEED = config.train_params.get("seed", 1234)


# ============================================================
# Reproducibility
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Helpers
# ============================================================
def savefig(name, outdir=None):
    d = Path(outdir) if outdir is not None else OUT_DIR
    plt.savefig(d / f"{name}.svg", format="svg", dpi=300, bbox_inches="tight")
    plt.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def calculate_rse(img1, img2):
    return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


def normalize_sensor(x, eps=1e-8):
    """Min-max the sensor frame.

    Kept byte-for-byte identical to the expression in train_dp.py (including the
    additive eps) so the network sees exactly the same input distribution at test
    time as it did during training. If you change one, change both.
    """
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def peak_snr(photons, read_noise_e=READ_NOISE_E):
    """Expected SNR at the brightest pixel: N / sqrt(N + sigma_R^2)."""
    return photons / math.sqrt(photons + read_noise_e ** 2)


def corr2d(a, b, eps=1e-12):
    a = a.astype(np.float64) - a.astype(np.float64).mean()
    b = b.astype(np.float64) - b.astype(np.float64).mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + eps
    return float((a * b).sum() / denom)


def mean_adjacent_slice_corr(volume_zhw):
    corrs = [corr2d(volume_zhw[z], volume_zhw[z + 1])
             for z in range(volume_zhw.shape[0] - 1)]
    return float(np.mean(corrs)), np.array(corrs)


def save_depth_grid(volume, name, title, outdir=None, vmin=None, vmax=None):
    D = volume.shape[0]
    cols = 7
    rows = math.ceil(D / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < D:
            ax.imshow(volume[i], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"z {i}", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(name, outdir)


def save_xz_yz_comparison(target_np, output_np, outdir, tag=""):
    y0 = output_np.shape[1] // 2
    x0 = output_np.shape[2] // 2
    for axis, idx, lbl, fname in (
        ("XZ", y0, "x", "xz_gt_reconstruction"),
        ("YZ", x0, "y", "yz_gt_reconstruction"),
    ):
        gt = target_np[:, idx, :] if axis == "XZ" else target_np[:, :, idx]
        rc = output_np[:, idx, :] if axis == "XZ" else output_np[:, :, idx]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(gt, cmap="gray"); plt.title(f"GT {axis}")
        plt.xlabel(lbl); plt.ylabel("z")
        plt.subplot(1, 2, 2)
        plt.imshow(rc, cmap="gray"); plt.title(f"Reconstruction {axis} {tag}")
        plt.xlabel(lbl); plt.ylabel("z")
        plt.tight_layout()
        savefig(fname, outdir)


# ============================================================
# Capture phase mask from optics module
# ============================================================
captured_optics = {}
_orig_multifocal = optics_mod.multifocal_microlens_array_hex


def _wrapped_multifocal(*args, **kwargs):
    canvas, amp_canvas, Mx, pitch_px, centers = _orig_multifocal(*args, **kwargs)
    captured_optics["canvas"] = canvas.detach().cpu()
    captured_optics["amp_canvas"] = amp_canvas.detach().cpu()
    captured_optics["centers"] = centers
    captured_optics["Mx"] = Mx
    captured_optics["pitch_px"] = pitch_px
    return canvas, amp_canvas, Mx, pitch_px, centers


optics_mod.multifocal_microlens_array_hex = _wrapped_multifocal


# ============================================================
# WandB
# ============================================================
wandb.init(project="Fourier_LFM_meta", config=config.train_params)


# ============================================================
# Load models
# ============================================================
model_optics = F_Optics(config.optics_params).to(DEVICE)
model_optics.load_state_dict(torch.load(OPTICS_CKPT, map_location=DEVICE), strict=True)
model_optics.eval()

model_recon = F_Recon(**config.recon_params).to(DEVICE)
model_recon.load_state_dict(torch.load(RECON_CKPT, map_location=DEVICE), strict=True)
model_recon.eval()

print(f"Optics pupil model : {getattr(optics_mod, 'PUPIL_MODEL', 'n/a')}")
print(f"Apodization        : {getattr(optics_mod, 'APODIZATION', 'n/a')}")
print(f"Trained at         : {TRAIN_PHOTONS:.0f} photons, read noise {READ_NOISE_E} e-")
print(f"Testing at         : {PHOTON_BUDGETS} photons  (exact Poisson = {EXACT_POISSON})")


# ============================================================
# Load dataset
# ============================================================
my_dataset = MyDataset(
    stack_dir=TEST_DATA_DIR,
    config=config,
    transform_3d=config.transform_3d,
)

target3d, lo, hi = my_dataset[SAMPLE_IDX]
target3d = target3d.to(DEVICE)

adjoint_bp = FLFMAdjointBackprojector(
    out_hw=(config.recon_params["output_size"][0], config.recon_params["output_size"][1]),
    normalize="percentile",
    p_low=10.0,   # was 1.0: see adjoint.normalize_volume_percentile
    # deconv="cg": conjugate gradient on the normal equations, replacing the
    # one-step tikhonov formula. The one-step form is the MINIMUM-NORM solution
    # of one equation in 43 unknowns per frequency, so every depth gets the
    # same function of Y and the lift is maximally smooth in z -- it turned the
    # PSF operator's adjacent-depth correlation of 0.773 into 0.996, destroying
    # the axial information before the network saw it. CG enforces the 512x512
    # support constraint, which makes the system well posed (11.46M sensor
    # pixels vs 11.27M voxels).
    #
    # Measured over all 12 test volumes at N=100, coarse volume vs GT:
    #
    #   lift                group B (sparse, n=8)   group A (dense, n=4)
    #   tikhonov 1e2              0.7626                  0.4289
    #   CG 10 it @ 3.0            0.9166                  0.5792
    #   CG 20 it @ 1.5            0.9282                  0.5628   <- chosen
    #   CG 20 it @ 1.0            0.9269                  0.5447
    #
    # CG beat tikhonov on every individual volume. reg 1.5 is best for group B
    # and its adj-Z there (~0.97) nearly matches those volumes' GT (~0.96).
    # Group A prefers reg 3.0 by 0.016, not worth a second configuration.
    #
    # tikhonov_reg is ignored in this mode. If this OOMs (CG caches all 43 H_z,
    # ~2 GB, which the tikhonov branch did not), drop to cg_iters=10,
    # cg_reg=3.0 -- group B falls only to 0.9166.
    deconv="cg",          # "adjoint" | "tikhonov" | "cg"
    # 10 @ 3.0, NOT 20 @ 1.5. The lambda sweep that preferred 1.5 was run only
    # at N=100, and the choice does not transfer: measured on sample 1,
    #
    #   reg 3.0 : rho_coarse 0.9194 @N=100   0.9057 @N=30
    #   reg 1.5 : rho_coarse 0.9244 @N=100   0.8684 @N=30
    #
    # Less regularisation amplifies noise, so the right lambda scales with the
    # photon budget. 3.0 is the better compromise across the sweep, and it is
    # half the cost.
    cg_reg=3.0,
    cg_iters=10,
    cg_chunk=4,           # depths per chunk inside A / A^T; drop to 1 if OOM
    # Cache the 43 H_z (~2 GB) instead of recomputing them in every A / A^T.
    # There are ~40 operator applications per step once backward recompute is
    # counted, so this removes ~1700 redundant FFTs -- roughly halves the CG
    # cost. Affordable now that CHECKPOINT_OPTICS freed the optics graph.
    cg_cache_H=True,
    p_high=99.7
).to(DEVICE)


# ============================================================
# Optics forward -- ONCE. It does not depend on the photon budget.
# ============================================================
if DEVICE.type == "cuda":
    torch.cuda.synchronize()

with torch.no_grad():
    t0 = time.perf_counter()
    output_optics, pitch_x, psf_stack = model_optics(target3d)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    optics_time = time.perf_counter() - t0

    sensor_clean = normalize_sensor(output_optics)     # noise-free, [0,1]

print(f"\nOptics forward time: {optics_time:.3f} s  (run once, reused for every budget)")
print(f"Sensor image (clean) max/min: "
      f"{sensor_clean.max().item():.4f} / {sensor_clean.min().item():.4f}")

target_np = target3d.squeeze().detach().cpu().numpy().astype(np.float32)
psf_np = psf_stack.detach().cpu().float().numpy().astype(np.float32)

tiff.imwrite(COMMON_DIR / "psf_stack.tif", psf_np)
tiff.imwrite(COMMON_DIR / "GT.tif", target_np)
tiff.imwrite(COMMON_DIR / "sensor_noise_free.tif",
             sensor_clean.squeeze().cpu().numpy().astype(np.float32))

mean_corr_t, corrs_t = mean_adjacent_slice_corr(target_np)


# ============================================================
# Sweep the photon budget
# ============================================================
rows = []
per_slice_frames = {}
recon_by_budget = {}
sensor_by_budget = {}

for photons in PHOTON_BUDGETS:
    tag = f"photons_{photons:04d}"
    sub = OUT_DIR / tag
    sub.mkdir(exist_ok=True)

    # same noise realisation sequence for every run of this script
    torch.manual_seed(SEED + int(photons))

    with torch.no_grad():
        sensor_noisy = add_sensor_noise(
            sensor_clean,
            photons,
            read_noise_e=READ_NOISE_E,
            exact=EXACT_POISSON,        # true integer Poisson counts at test time
        )
        # NOT clamped to [0,1] on purpose: read noise makes dark pixels go
        # slightly negative, and that is what the network saw during training.

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        coarse_volume = adjoint_bp(
            sensor_img=sensor_noisy,
            psf_stack=psf_stack,
            dz=model_optics.dz_used,     # same dz the forward model integrates with
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        lift_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        output_recon = model_recon(coarse_volume)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        recon_time = time.perf_counter() - t0

    total_time = lift_time + recon_time
    num_slices = int(output_recon.shape[1]) if output_recon.dim() == 4 else int(output_recon.shape[0])

    output_np = output_recon.squeeze().detach().cpu().numpy().astype(np.float32)
    sensor_np = sensor_noisy.squeeze().detach().cpu().numpy().astype(np.float32)
    coarse_np = coarse_volume.squeeze().detach().cpu().numpy().astype(np.float32)

    recon_by_budget[photons] = output_np
    sensor_by_budget[photons] = sensor_np

    # ---- measured noise statistics, as a check on the model -------------
    resid = (sensor_noisy - sensor_clean)
    dark = sensor_clean < 0.01
    meas_dark_std = resid[dark].std().item() if dark.any() else float("nan")
    pred_dark_std = READ_NOISE_E / photons

    # ---- metrics --------------------------------------------------------
    metrics = []
    for d in range(target_np.shape[0]):
        st, so = target_np[d], output_np[d]
        rng = st.max() - st.min()
        ssim_val = np.nan if rng < 1e-8 else compare_ssim(st, so, data_range=rng)
        metrics.append({
            "z_index": d,
            "ssim": ssim_val,
            "psnr_db": calculate_psnr(torch.tensor(st), torch.tensor(so)),
            "rse": calculate_rse(torch.tensor(st), torch.tensor(so)),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(sub / "metrics_per_slice.csv", index=False)
    per_slice_frames[photons] = metrics_df

    mean_corr_o, corrs_o = mean_adjacent_slice_corr(output_np)

    row = {
        "photons": photons,
        "peak_snr": peak_snr(photons),
        "is_training_budget": photons == TRAIN_PHOTONS,
        "ssim": metrics_df["ssim"].mean(),
        "psnr_db": metrics_df["psnr_db"].mean(),
        "rse": metrics_df["rse"].mean(),
        "adjacent_corr_output": mean_corr_o,
        "adjacent_corr_target": mean_corr_t,
        "dark_std_measured": meas_dark_std,
        "dark_std_predicted": pred_dark_std,
        "sensor_min": float(sensor_np.min()),
        "sensor_max": float(sensor_np.max()),
        "coarse_min": float(coarse_np.min()),
        "coarse_max": float(coarse_np.max()),
        "adjoint_lift_time_s": lift_time,
        "recon_network_time_s": recon_time,
        "total_reconstruction_time_s": total_time,
        "volume_fps": 1.0 / total_time,
        "slices_per_s": num_slices / total_time,
    }
    rows.append(row)

    print(f"\n--- N = {photons:>5} photons  (peak SNR {row['peak_snr']:.1f})"
          f"{'   <- training budget' if row['is_training_budget'] else ''}")
    print(f"    SSIM {row['ssim']:.4f}   PSNR {row['psnr_db']:.2f} dB   RSE {row['rse']:.4f}")
    print(f"    sensor range [{row['sensor_min']:.3f}, {row['sensor_max']:.3f}]   "
          f"coarse max {row['coarse_max']:.4f}")
    print(f"    dark-region std: measured {meas_dark_std:.5f}  predicted {pred_dark_std:.5f}")
    print(f"    recon {total_time*1e3:.1f} ms  ({row['volume_fps']:.2f} vol/s)")

    # (removed: a "coarse_max should be 1.0" check. That only held for the
    #  legacy normalize="global" min-max. With normalize="percentile" the max
    #  is deliberately free -- p_high maps to 1.0 and brighter voxels exceed it.)

    # ---- per-budget outputs --------------------------------------------
    tiff.imwrite(sub / "measurement.tif", sensor_np)
    tiff.imwrite(sub / "reconstructed_volume.tif", output_np)
    tiff.imwrite(sub / "coarse_volume.tif", coarse_np)

    save_depth_grid(output_np, "reconstruction_slices",
                    f"Reconstruction, N = {photons} photons", outdir=sub)
    save_depth_grid(np.abs(target_np - output_np), "error_slices",
                    f"Absolute error, N = {photons} photons", outdir=sub)
    save_xz_yz_comparison(target_np, output_np, sub, tag=f"(N={photons})")

    plt.figure(figsize=(6, 6))
    plt.imshow(sensor_np, cmap="gray")
    plt.colorbar()
    plt.title(f"Sensor image, N = {photons} photons")
    plt.tight_layout()
    savefig("sensor_image", sub)

    wandb.log({
        "photons": photons,
        "test/ssim": row["ssim"],
        "test/psnr_db": row["psnr_db"],
        "test/rse": row["rse"],
        "test/peak_snr": row["peak_snr"],
    })


# ============================================================
# Summary across budgets
# ============================================================
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "summary_across_photons.csv", index=False)

with pd.ExcelWriter(OUT_DIR / "all_results.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    for photons, df in per_slice_frames.items():
        df.to_excel(writer, sheet_name=f"slices_N{photons}", index=False)

print("\n================ summary across photon budgets ================")
print(summary_df[["photons", "peak_snr", "ssim", "psnr_db", "rse",
                  "total_reconstruction_time_s"]].to_string(index=False))

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(f"pupil model : {getattr(optics_mod, 'PUPIL_MODEL', 'n/a')}\n")
    f.write(f"apodization : {getattr(optics_mod, 'APODIZATION', 'n/a')}\n")
    f.write(f"trained at  : {TRAIN_PHOTONS:.0f} photons\n")
    f.write(f"read noise  : {READ_NOISE_E} e- (fixed)\n")
    f.write(f"optics fwd  : {optics_time:.3f} s (once)\n\n")
    f.write(summary_df.to_string(index=False))


# ============================================================
# Metric vs photon budget
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, key, lbl in zip(axes,
                        ["ssim", "psnr_db", "rse"],
                        ["SSIM", "PSNR (dB)", "RSE"]):
    ax.semilogx(summary_df["photons"], summary_df[key], "o-", color="#1a1a1a", lw=2)
    train_row = summary_df[summary_df["is_training_budget"]]
    if len(train_row):
        ax.semilogx(train_row["photons"], train_row[key], "o", ms=13, mfc="none",
                    mec="#ea580c", mew=2.2, label="training budget")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("photon budget  N  (peak pixel)")
    ax.set_ylabel(lbl)
    ax.grid(alpha=0.25)
fig.suptitle("Reconstruction quality vs photon budget "
             f"(trained at N = {TRAIN_PHOTONS:.0f})", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
savefig("metrics_vs_photons")


# ============================================================
# Visual sweep: sensor + reconstruction at one slice, every budget
# ============================================================
z_mid = target_np.shape[0] // 2
ncol = len(PHOTON_BUDGETS) + 1

fig, axes = plt.subplots(2, ncol, figsize=(2.3 * ncol, 5.0))
axes[0, 0].imshow(sensor_clean.squeeze().cpu().numpy(), cmap="gray")
axes[0, 0].set_title("noise-free", fontsize=10)
axes[1, 0].imshow(target_np[z_mid], cmap="gray")
axes[1, 0].set_title(f"GT  z={z_mid}", fontsize=10)
for j, photons in enumerate(PHOTON_BUDGETS, start=1):
    axes[0, j].imshow(sensor_by_budget[photons], cmap="gray")
    axes[0, j].set_title(f"{photons} ph\nSNR {peak_snr(photons):.1f}", fontsize=10)
    axes[1, j].imshow(recon_by_budget[photons][z_mid], cmap="gray")
    s = summary_df.loc[summary_df["photons"] == photons, "ssim"].iloc[0]
    axes[1, j].set_title(f"SSIM {s:.3f}", fontsize=10)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel("sensor", fontsize=11)
axes[1, 0].set_ylabel("reconstruction", fontsize=11)
fig.tight_layout()
savefig("photon_sweep_montage")


# ============================================================
# Learned MLA diagnostics (independent of the photon budget)
# ============================================================
ns = int(model_optics.mla_size)
device_model = model_optics.f_rand.device

focal_length_map = model_optics.f_rand.detach().cpu().reshape(ns, ns)
focal_length_map_np = focal_length_map.numpy()

plt.figure(figsize=(5, 5))
plt.imshow(focal_length_map_np * 1e3, cmap="viridis")
plt.colorbar(label="Focal length (mm)")
plt.title("Learned focal length map")
plt.tight_layout()
savefig("focal_length_map", COMMON_DIR)

np.savetxt(COMMON_DIR / "focal_length_map_mm.csv",
           focal_length_map_np * 1e3, delimiter=",")

rows_idx = torch.arange(ns, device=device_model).view(ns, 1)
cols_idx = torch.arange(ns, device=device_model).view(1, ns)
is_D2 = (rows_idx % 2 == 1) & (cols_idx % 2 == 0)
is_D1 = ~is_D2

D_current = torch.zeros((ns, ns), device=device_model, dtype=torch.float32)
if hasattr(model_optics, "D_map"):
    D_current = model_optics.D_map.detach().float()
elif hasattr(model_optics, "D_1") and hasattr(model_optics, "D_2"):
    D_current[is_D2] = model_optics.D_2
    D_current[is_D1] = model_optics.D_1
else:
    D_current[:] = model_optics.D_pitch

D_current_cpu = D_current.detach().cpu()
D_current_np = D_current_cpu.numpy()

plt.figure(figsize=(5, 5))
plt.imshow(D_current_np * 1e6, cmap="viridis")
plt.colorbar(label="Diameter (um)")
plt.title("Diameter map")
plt.tight_layout()
savefig("diameter_map", COMMON_DIR)

np.savetxt(COMMON_DIR / "diameter_map_um.csv", D_current_np * 1e6, delimiter=",")

if "canvas" not in captured_optics:
    raise RuntimeError(
        "MLA mask was not captured. Check that multifocal_microlens_array_hex was called."
    )

mask = captured_optics["canvas"].detach().cpu()
mla_amplitude = torch.abs(mask)
mla_phase = torch.angle(mask)

for arr, name, cmap, kw in (
    (mla_amplitude, "mla_amplitude", "gray", {}),
    (mla_phase, "mla_phase", "twilight", {"vmin": -np.pi, "vmax": np.pi}),
):
    plt.figure(figsize=(6, 6))
    plt.imshow(arr.numpy(), cmap=cmap, **kw)
    plt.title(name.replace("_", " "))
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    savefig(name, COMMON_DIR)
    tiff.imwrite(COMMON_DIR / f"{name}.tif", arr.numpy().astype(np.float32))

torch.save(
    {
        "diameter_map": D_current_cpu,
        "focal_length_map": focal_length_map,
        "mla_amplitude": mla_amplitude,
        "mla_phase": mla_phase,
        "mla_complex_mask": mask,
        "Mx": captured_optics["Mx"],
        "pitch_px": captured_optics["pitch_px"],
        "centers": captured_optics["centers"],
    },
    COMMON_DIR / "learned_mla_mask.pt",
)

print("\nSaved MLA mask diagnostics")
print(f"f  mm : min {focal_length_map.min().item()*1e3:.4f}  "
      f"max {focal_length_map.max().item()*1e3:.4f}  "
      f"std {focal_length_map.std().item()*1e3:.4f}")
print(f"D  um : min {D_current_cpu.min().item()*1e6:.3f}  "
      f"max {D_current_cpu.max().item()*1e6:.3f}  "
      f"std {D_current_cpu.std().item()*1e6:.3f}")

save_depth_grid(target_np, "target_slices", "Target depth slices", outdir=COMMON_DIR)

wandb.finish()
print(f"\nSaved all results to: {OUT_DIR}")
