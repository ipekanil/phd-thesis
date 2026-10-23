import torch
from torch import nn
from torch.optim import Adam
#from torch.utils.tensorboard import SummaryWriter
from F_VCD_pytorch_multiple_views import F_Recon
from F_Optics_fixed import F_Optics
from config import config  # Import the config object
from dataset import MyDataset  # Assuming you have a custom dataset class
from datetime import datetime
from torch.utils.data import DataLoader
from custom_losses import mse_loss, mae_loss, EdgeLoss, WrappedSSIMLoss, axial_diff_matching_loss
from skimage.metrics import structural_similarity as compare_ssim
from pathlib import Path
import numpy as np
import os
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import time
import wandb  # Import wandb
import torch.nn.functional as F

from adjoint import FLFMAdjointBackprojector
from sensor_noise import add_sensor_noise

os.environ["WANDB_SYMLINK"] = "false"
os.environ["WANDB_START_METHOD"] = "thread"

import gc
gc.collect()
torch.cuda.empty_cache()

print(f"Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")


torch.cuda.ipc_collect()
#torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # Limit to 80% of GPU memory
#torch.cuda.amp.autocast()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


print("Script has started running!")
sys.stdout.flush()  # Forces immediate print output

def setup_distributed():
    """Initialize the Distributed Data Parallel (DDP) process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"])
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # Default to GPU 0 if not found
        torch.cuda.set_device(local_rank)  # Ensure valid GPU selection
        print(f" Initialized DDP: RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}, LOCAL_RANK={local_rank}")
    else:
        print("Distributed environment variables not set! Running in single-GPU mode.")

setup_distributed()

# Create a log file and write to it
with open("debug_log.txt", "w") as f:
    f.write("Script has started running!\n")
    f.flush()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ---------------------------------------------------------------------------
# VERBOSE: the per-step trace (tensor ranges, requires_grad flags, memory,
# per-parameter gradients and weight deltas, sub-timings). Useful when
# something is broken, pure noise across a 40-hour run. Epoch summaries,
# validation, the coarse-volume diagnostics and all warnings print either way.
# ---------------------------------------------------------------------------
VERBOSE = False

# Recompute the optics forward in backward instead of storing it. Costs one
# extra forward (~8.7 s) per step, frees most of ~29 GB. Set False if memory
# is not tight.
CHECKPOINT_OPTICS = True

def vprint(*a, **k):
    if VERBOSE:
        print(*a, **k)


def print_memory_usage(stage):
    vprint(f"🔍 {stage} - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def get_recon_loss_weights(step):
    """
    Two stage warmup for reconstruction losses.

    Stage 1:
        Learn appearance first.
        Strong MAE, moderate SSIM, weak edge and axial.

    Stage 2:
        Gradually introduce stronger edge and axial supervision.

    Stage 3:
        Full training weights.
    """

    # Stage 1: early warmup
    if step < 0: #800: #96: #800:
        return {
            "mae": 1.0,
            "edge": 0.03,
            "ssim": 0.15,
            "axial": 0.08,
        }

    # Stage 2: transition
    elif step < 0: #2500: #312: #2500:
        # linear interpolation from stage 1 to stage 3
        t = (step - 800) / (2500 - 800)

        mae_w   = (1.0 - t) * 1.0  + t * 1.0
        edge_w  = (1.0 - t) * 0.03 + t * 0.10
        ssim_w  = (1.0 - t) * 0.15 + t * 0.20
        axial_w = (1.0 - t) * 0.08 + t * 0.35

        return {
            "mae": mae_w,
            "edge": edge_w,
            "ssim": ssim_w,
            "axial": axial_w,
        }

    # Stage 3: full weights -- read from config.loss["Recon_loss"] so there is a
    # single place to tune them. (These used to be hard-coded here while
    # config.loss sat unused with DIFFERENT numbers, which made editing the
    # config silently do nothing.)
    else:
        _w = config.loss["Recon_loss"]
        return {
            "mae":   _w["mae_loss"],
            "edge":  _w["edge_loss"],
            "ssim":  _w["ssim_loss"],
            "axial": _w["axial_diff_matching_loss"],
        }
    
def preprocess_lf(x, eps=1e-6):
    mu = x.mean(dim=(-2, -1), keepdim=True).detach()
    sigma = x.std(dim=(-2, -1), keepdim=True, unbiased=False).detach().clamp_min(eps)
    return (x - mu) / sigma

def min_max_normalize_all_views(x, eps=1e-8):
    min_val = x.amin(dim=(-3, -2, -1), keepdim=True)
    max_val = x.amax(dim=(-3, -2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + eps)

_NOISE = config.train_params["sensor_noise"]

# ---------------------------------------------------------------------------
# FREEZE_OPTICS: while the reconstruction network re-adapts to the
# unclamped training output, keep the phase mask fixed so that any change in the
# reconstructions can be attributed to F_Recon alone. Gradients are still
# computed (the pre-clip optics grad norm keeps being logged) -- only the
# optics optimizer step is skipped. Set to False to resume joint training.
# ---------------------------------------------------------------------------
FREEZE_OPTICS = False

# ---------------------------------------------------------------------------
# USE_PRETRAINED:
#   False -> train from scratch (random recon weights, initial phase masks);
#            the checkpoint paths in config are ignored.
#   True  -> load step_1 optics/recon checkpoints from config; a missing or
#            broken file is a hard error, so a fine-tune never silently
#            restarts from scratch.
# Typical combos:  from scratch  -> USE_PRETRAINED=False, FREEZE_OPTICS=False
#                  fine-tune     -> USE_PRETRAINED=True,  FREEZE_OPTICS=True
#
# Default False: every arm of the optics comparison trains from scratch, and
# the previous default of True meant the `f` arm tried to load
# ./pretrained/optics_model.pth and raised when that file was absent. Set the
# ENVIRONMENT variable USE_PRETRAINED=1 for a deliberate fine-tune, or edit
# this line.
# ---------------------------------------------------------------------------
USE_PRETRAINED = False
if os.environ.get("USE_PRETRAINED") is not None:
    USE_PRETRAINED = os.environ["USE_PRETRAINED"].strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# GROUP_HI_THRESHOLD: the dataset holds two populations, separated by `hi`
# (the raw 99.7 percentile each volume is normalised by, which the dataloader
# already returns):
#
#   A  hi <  2500 : dense, diffuse, textured.   density 0.151, gt_adjz 0.586
#   B  hi >= 2500 : sparse bright objects.      density 0.065, gt_adjz 0.942
#
# 22 of 63 volumes are group A. They are much harder -- FLFM is compressive,
# and diffuse texture offers no sparsity to exploit -- so a single average
# over the test set hides a rho that runs from ~0.93 (B) to ~0.50 (A). The
# test phase below reports both groups separately for that reason.
#
# Test-set membership: group A = volumes 3, 5, 8, 11; group B = the other 8.
# ---------------------------------------------------------------------------
GROUP_HI_THRESHOLD = 2500.0

# ---------------------------------------------------------------------------
# LEARN_D -- the (f, D) design.
#
#   False  focal length only. Diameters stay at the constant (min_D+max_D)/2.
#          This is the configuration every reported result used.
#   True   per-lenslet diameters are learned as well, initialised at that same
#          constant, so epoch 0 is identical to the f-only run and any
#          difference afterwards is training, not a different starting point.
#
# Run the two back to back with the same SEED and the same epoch count, then
# compare. Everything else in this file is untouched by the flag.
#
# Note on what the old (f, D) script measured: it imported F_Optics.py, which
# still had the paraxial spherical-wave pupil. That writes the defocus phase
# n^2 = 1.77x too strong, making the PSF 1.48x too blurry at z = 30 um -- so
# the focal lengths and diameters it learned were fitted to a defocus response
# roughly 40% more pessimistic than the Debye model says is real. Those numbers
# cannot be compared against anything produced here. This is the re-run.
#
# lr_D: the diameter gradient arrives through the straight-through estimator in
# make_binary_aperture_ste, not through a true derivative of a hard aperture
# edge, so its scale has no reason to match the focal-length gradient's. It gets
# its own learning rate, read from config if present.
# ---------------------------------------------------------------------------
LEARN_D = False

# ---------------------------------------------------------------------------
# OPTICS_MODE -- which arm of the comparison this run is. Set it with an
# ENVIRONMENT VARIABLE so a queue of runs needs no file edits between them:
#
#   OPTICS_MODE=fixed ./run.sh     conventional periodic MLA, optics frozen.
#                                  One focal length (43.1 mm nominal), one
#                                  diameter. Only the network learns.
#                                  = the paper's "fixed optics + CNN".
#   OPTICS_MODE=f ./run.sh         per-lenslet focal lengths learned. (default)
#   OPTICS_MODE=fD ./run.sh        focal lengths AND diameters learned.
#
# All three share the seed, the epoch count, the lift, the noise model and the
# network, so the optics are the only difference. RUN_TAG names the output
# checkpoints, so no arm overwrites another.
# ---------------------------------------------------------------------------
OPTICS_MODE = os.environ.get("OPTICS_MODE", "f").strip().lower()
if OPTICS_MODE not in ("fixed", "f", "fD".lower(), "fd"):
    raise SystemExit(f"OPTICS_MODE must be fixed | f | fD, got {OPTICS_MODE!r}")
if OPTICS_MODE == "fd":
    OPTICS_MODE = "fD"

LEARN_D = (OPTICS_MODE == "fD")
INIT_F = "nominal" if OPTICS_MODE == "fixed" else "random"

# USE_PRETRAINED is resolved where it is declared above, so there is nothing to
# override here. Worth stating why from-scratch is right for all three arms: a
# warm start from a learned mask would defeat the point of the fixed arm, and
# the f-only checkpoint has no D_map against a strict=True load, so the (f, D)
# arm could not warm start from it anyway.
#
# The fixed arm must not move the optics. A manually set FREEZE_OPTICS=True
# above is preserved; `fixed` forces it on regardless.
FREEZE_OPTICS = FREEZE_OPTICS or (OPTICS_MODE == "fixed")

# Epoch count and output names, both overridable from the environment so the
# queue in run.sh can set them per arm.
if os.environ.get("NUM_EPOCHS"):
    config.train_params["num_epochs"] = int(os.environ["NUM_EPOCHS"])
RUN_TAG = os.environ.get("RUN_TAG", OPTICS_MODE)
OPTICS_OUT = f"optics_model_{RUN_TAG}.pth"
RECON_OUT  = f"recon_model_{RUN_TAG}.pth"

print("=" * 70)
print(f"  OPTICS_MODE = {OPTICS_MODE}   (learn_D={LEARN_D}, init_f={INIT_F}, "
      f"freeze_optics={FREEZE_OPTICS})")
print(f"  epochs      = {config.train_params['num_epochs']}")
print(f"  checkpoints -> {OPTICS_OUT} / {RECON_OUT}")
print("=" * 70)

# Diversity regulariser on the learned parameters (-var(f) - var(D)), as in the
# old (f, D) script. OFF by default and deliberately so: it changes the
# objective, so switching it on for the (f, D) arm and not the f-only arm would
# confound the comparison with a different loss. Turn it on for BOTH arms or
# neither.
DIVERSITY_REG = 0.0      # old script used 1e-4


def apply_sensor_noise(x, photons=None, exact=False):
    """Mixed Poisson-Gaussian noise. photons defaults to the training value."""
    return add_sensor_noise(
        x,
        photons if photons is not None else _NOISE["photons"],
        read_noise_e=_NOISE["read_noise_e"],
        exact=exact,
    )


# ---------------------------------------------------------------------------
# RANDOMISE_PHOTONS: draw the photon budget log-uniformly from
# TRAIN_PHOTON_RANGE at every training step instead of fixing it at
# _NOISE["photons"].
#
# Why: trained at a fixed N=100, the network does not generalise to other
# budgets and actively HURTS at low light -- measured on the dense test volume
# at N=10, the CG lift gives rho 0.3008 and the network turns it into 0.2506,
# a net -0.050. The photon sweep is a headline figure for the paper, so its
# weakest column should not be the network damaging a better lift.
#
# Log-uniform, not uniform: the sweep spans 10-1000, two decades, and uniform
# sampling would put ~90% of the steps above N=100 and almost never visit the
# low-light regime that needs the help.
#
# Validation and test keep their FIXED budgets so those curves stay comparable
# with every earlier run.
# ---------------------------------------------------------------------------
# Set False: measured WORSE at every budget. Trained across 10-1000 the
# network's contribution went from +0.0037 to -0.0027 at N=100 (sample 1) and
# SSIM at N=10 fell 0.273 -> 0.148. It learned an average behaviour that fits
# no budget, including the low end it was meant to help.
RANDOMISE_PHOTONS  = False
TRAIN_PHOTON_RANGE = (10.0, 1000.0)


def sample_train_photons():
    if not RANDOMISE_PHOTONS:
        return _NOISE["photons"]
    lo, hi = TRAIN_PHOTON_RANGE
    u = torch.rand(1).item()
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


def calculate_nrmse(target, output, reduction='mean', eps=1e-8):
    """
    target, output: [D, H, W] or [B, D, H, W]
    NRMSE per depth: RMSE(slice) / (std(target_slice) + eps)
    """
    if target.shape != output.shape:
        raise ValueError(f"Shape mismatch: {target.shape} vs {output.shape}")

    # Ensure float
    target = target.float()
    output = output.float()

    # Add batch dim if missing
    if target.dim() == 3:
        target = target.unsqueeze(0)
        output = output.unsqueeze(0)

    B, D, H, W = target.shape

    diff2 = (target - output).pow(2).reshape(B, D, -1)
    rmse = diff2.mean(dim=-1).sqrt()  # [B, D]

    std = target.reshape(B, D, -1).std(dim=-1, unbiased=False)  # [B, D]
    nrmse = rmse / (std + eps)

    if reduction == "mean":
        return nrmse.mean().item()
    if reduction == "none":
        return nrmse  # tensor [B, D]
    raise ValueError("reduction must be 'mean' or 'none'")


# Initialize wandb
wandb.init(project="Fourier_LFM_meta", config=config.train_params)
global_step = 0



# Get local rank and set the correct device
#local_rank = int(os.environ["LOCAL_RANK"])
#device = torch.device(f"cuda:{local_rank}")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Reproducibility: seed everything before any model is constructed, so that
# f_rand, the recon weight init, the sensor noise and the dataloader shuffle
# are all repeatable.  Set config.train_params["seed"] = None to opt out.
# Note DDP: each rank offsets the seed so ranks do not draw identical noise.
# ---------------------------------------------------------------------------
import random

SEED = config.train_params.get("seed", None)
if SEED is not None:
    seed_r = int(SEED) + local_rank
    random.seed(seed_r)
    np.random.seed(seed_r)
    torch.manual_seed(seed_r)
    torch.cuda.manual_seed_all(seed_r)
    print(f"🎲 Seeded RNGs with {seed_r} (base {SEED} + local_rank {local_rank})")
else:
    print("🎲 No seed set — this run is not reproducible")

# Move model to device before applying mixed precision conversion
# seed is passed explicitly so f_rand is reproducible regardless of how much
# RNG anything imported earlier happened to consume.
model_optics = F_Optics(config.optics_params, seed=SEED,
                        learn_D=LEARN_D, init_f=INIT_F).to(device)
model_recons = F_Recon(**config.recon_params).to(device)

#model_optics = torch.nn.DataParallel(model_optics).to(device)
#model_recons = torch.nn.DataParallel(model_recons).to(device)

#print("Models converted to FP16 and wrapped in DDP")
# Print GPU memory usage after moving models to GPUs
print_memory_usage("After Model Init")

# If you're using DataParallel, unwrap with `.module`
#optics_total, optics_trainable = count_parameters(model_optics.module)
#recons_total, recons_trainable = count_parameters(model_recons.module)
optics_total, optics_trainable = count_parameters(model_optics)
recons_total, recons_trainable = count_parameters(model_recons)

if RANDOMISE_PHOTONS:
    print(f"📷 Photon budget randomised per step: log-uniform "
          f"{TRAIN_PHOTON_RANGE[0]:.0f}-{TRAIN_PHOTON_RANGE[1]:.0f} "
          f"(val/test stay fixed)")
else:
    print(f"📷 Photon budget fixed at {_NOISE['photons']:.0f}")
print(f"🧠 Optics Model: Total = {optics_total:,}, Trainable = {optics_trainable:,}")
print(f"🧠 Reconstruction Model: Total = {recons_total:,}, Trainable = {recons_trainable:,}")

# Define optimizers with dummy LR (will be updated in training loop)
initial_lr = config.train_params.get("step_1", {}).get("lr_digital", 1e-4)  # fallback
initial_lr_optics = config.train_params.get("step_1", {}).get("lr_optical", 1e-5)

optimizer_recon = Adam(model_recons.parameters(), lr=initial_lr, weight_decay=1e-4)

# f_rand and D_map get separate param groups so the diameters can take their
# own learning rate (see the LEARN_D note above). With LEARN_D=False there is
# no D_map and this reduces to the single group it was before.
_lr_D = config.train_params.get("step_1", {}).get("lr_D", initial_lr_optics)
_optics_groups = [{"params": [model_optics.f_rand],
                   "lr": initial_lr_optics, "name": "f_rand"}]
if LEARN_D:
    _optics_groups.append({"params": [model_optics.D_map],
                           "lr": _lr_D, "name": "D_map"})
optimizer_optics = Adam(_optics_groups)
print(f"🔧 LEARN_D={LEARN_D}  optics groups: "
      f"{[g['name'] for g in _optics_groups]}  "
      f"lr_f={initial_lr_optics:g}" + (f"  lr_D={_lr_D:g}" if LEARN_D else ""))
if DIVERSITY_REG:
    print(f"⚠️  diversity regulariser ON at {DIVERSITY_REG:g} -- it must be on "
          f"for BOTH the f-only and the (f, D) run or the comparison is invalid")

for param_group in optimizer_recon.param_groups:
    print(f"🔍 Initial LR: {param_group['lr']}")
    
for param_group in optimizer_optics.param_groups:
    print(f"🔍 Initial LR Optics: {param_group['lr']}")

# Define losses individually
edge_loss = EdgeLoss()
ssim_loss = WrappedSSIMLoss().to(device)

loss_functions = {
    'mae_loss': mae_loss,
    'edge_loss': edge_loss,
    'ssim_loss': ssim_loss,
    'axial_diff_matching_loss': axial_diff_matching_loss
}


# dataset
train_dataset = MyDataset(
    stack_dir = config.paths['stack_dir'],
    config=config,
    transform_3d=config.transform_3d
)

val_dataset = MyDataset(
    stack_dir=config.paths['val_stack_dir'],
    config=config,
    transform_3d=config.transform_3d
)

test_dataset = MyDataset(
    stack_dir=config.paths['test_stack_dir'],
    config=config,
    transform_3d=config.transform_3d
)

# Load datasets
train_loader = DataLoader(
    train_dataset,
    batch_size=config.train_params['batch_size'],
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.train_params['batch_size'],
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.train_params['batch_size'],
    shuffle=False,
)

# Load pretrained model at Step 1 (if a checkpoint path is provided)

step1 = config.train_params.get("step_1", {})

optics_path = step1.get("optics_ckpt_path", "")
recon_path  = step1.get("recon_ckpt_path", "")

device = "cuda" if torch.cuda.is_available() else "cpu"

if not USE_PRETRAINED:
    print("USE_PRETRAINED=False -> training from scratch (checkpoint paths ignored)")
elif optics_path or recon_path:
    print("Loading pretrained weights for Step 1")

    if optics_path:
        try:
            print(f"Loading optics from {optics_path}")
            ckpt = torch.load(optics_path, map_location=device)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model_optics.load_state_dict(sd, strict=True)
        except Exception as e:
            # Fail loudly: a missing checkpoint would otherwise silently start
            # the fine-tune from scratch.
            raise RuntimeError(f"Could not load optics checkpoint from {optics_path}: {e}")

    if recon_path:
        try:
            print(f"Loading recon from {recon_path}")
            ckpt = torch.load(recon_path, map_location=device)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model_recons.load_state_dict(sd, strict=True)
        except Exception as e:
            raise RuntimeError(f"Could not load recon checkpoint from {recon_path}: {e}")


torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print_memory_usage("Before Training Loop")

prev_optics_weights = {
    name: p.detach().clone()
    for name, p in model_optics.named_parameters()
    if p.requires_grad
}

prev_recon_weights = {
    name: p.detach().clone()
    for name, p in model_recons.named_parameters()
    if p.requires_grad
}


# Training loop
#############################################      
# Training phase
#############################################

#torch.cuda.set_per_process_memory_fraction(0.9)  # Limit memory usage to avoid OOM
accumulation_steps = 1  # Adjust based on memory availability

# Keep normalization layers in FP32 to prevent instability
for module in model_recons.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        module.float()

# create once, before training loop
# adjoint operator
#physics aware first step for the network to see a coarse depth resolved representation instead of one raw multiplexed sensor image.
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
).to(device)


# running medians for the coarse-volume scale check (see the diagnostic below)
_cv_med_hist = []
_gt_med_hist = []

for epoch in range(config.train_params['num_epochs']):
    init_start = time.time()

    model_optics.train()
    model_recons.train()

    epoch_recon_loss = 0.0
    epoch_total_loss = 0.0  
    total_steps = len(train_loader)
    
    # **Step 2: Fine-tuning Optics**
    if epoch == config.train_params["step_2"]["epoch"]:  
        print(f"🔥 Switching to Step 2 at epoch {epoch}")

        if optimizer_optics is not None:
            for param_group in optimizer_optics.param_groups:
                param_group["lr"] = config.train_params["step_2"]["lr_optical"]
                print(f"🔄 Setting learning rate of optimizer_optics to {param_group['lr']}")
        else:
            print("⚠️ Optics optimizer is None — skipping optical fine-tuning.")

        for param_group in optimizer_recon.param_groups:
            param_group["lr"] = config.train_params["step_2"]["lr_digital"]
            print(f"🔄 Setting learning rate of optimizer_recon to {param_group['lr']}")
    init_finish = time.time()
    print(f"""🕒 Init timing:{init_finish - init_start:.2f}s""")

    # Accumulators for averaging individual losses
    epoch_loss_accumulators = {
        "mae_loss": 0.0,
        "edge_loss": 0.0,
        "ssim_loss": 0.0,
        "axial_diff_matching_loss": 0.0,
    }

    for i, data in enumerate(train_loader):
        data_load_start = time.time()
        step_start = time.time()

        target3d, lo, hi = data
        target3d = target3d.squeeze(dim=0).to(device, non_blocking=True)
        lo = lo.to(device)
        hi = hi.to(device)

        vprint("Target 3d max/min:", target3d.max().item(), target3d.min().item())
        vprint("Target 3d shape:", target3d.shape)

        data_load_end = time.time()

        # Reset gradients
        if optimizer_optics is not None:
            optimizer_optics.zero_grad()
        optimizer_recon.zero_grad()

        if (i + 1) % 1 == 0:
            vprint(f"🔁 Iteration {i+1} | Current LR: {optimizer_recon.param_groups[0]['lr']}")
            vprint(f"🔁 Iteration {i+1} | Current LR Optics: {optimizer_optics.param_groups[0]['lr']}"
                  f"{'  (FROZEN, step skipped)' if FREEZE_OPTICS else ''}")

        #############################################
        # **Forward Pass**
        #############################################
        #torch.autograd.set_detect_anomaly(True)

        forward_start = time.time()

        #OPTICS FORWARD
        f0 = time.time()

        # Checkpointed: the optics forward retains ~29 GB of graph (43 PSFs at
        # 3385^2 plus every intermediate), which left no headroom for the CG
        # solve's backward and OOMed at 31.66/31.73 GB. Checkpointing stores
        # only the inputs and recomputes the forward during backward, at the
        # cost of one extra optics forward (~8.7 s) per step.
        #
        # This keeps the mask gradient through the CG solve fully intact --
        # unlike cg_detach_psf, which would buy memory by making the solve
        # blind to the mask, defeating the reason for using CG at all.
        if CHECKPOINT_OPTICS and torch.is_grad_enabled():
            output_optics, pitch_x, psf_stack = checkpoint(
                model_optics, target3d, use_reentrant=False)
        else:
            output_optics, pitch_x, psf_stack = model_optics(target3d)
        
        f1 = time.time() 
        print_memory_usage(f"After Optics Forward Pass - Batch {i}")
        vprint("Output optics requires grad:", output_optics.requires_grad)
        vprint("Output optics max/min:", output_optics.max().item(), output_optics.min().item())
        #normalize optics output
        output_optics_norm = (output_optics - output_optics.amin(dim=(-2, -1), keepdim=True)) / (
            output_optics.amax(dim=(-2, -1), keepdim=True) - output_optics.amin(dim=(-2, -1), keepdim=True) + 1e-8
        )

        # add sensor noise here (mixed Poisson-Gaussian), at a photon budget
        # drawn per step -- see RANDOMISE_PHOTONS
        step_photons = sample_train_photons()
        output_optics_norm = apply_sensor_noise(output_optics_norm,
                                                photons=step_photons)
        vprint("Output optics norm requires grad:", output_optics_norm.requires_grad)
        vprint("Output optics norm max/min:", output_optics_norm.max().item(), output_optics_norm.min().item())
        vprint("Output optics norm shape:", output_optics_norm.shape)

        # adjoint backprojection
        # psf_stack is deliberately NOT detached: optics gradients are meant to
        # flow through the backprojector as well as through the forward model.
        coarse_volume = adjoint_bp(
            sensor_img=output_optics_norm,
            psf_stack=psf_stack,
            dz=model_optics.dz_used      # same dz the forward model integrates with
        )   # [B, D, 512, 512]

        del psf_stack

        vprint("coarse_volume shape:", coarse_volume.shape)
        vprint("coarse_volume requires grad:", coarse_volume.requires_grad)
        # --- scale diagnostic -------------------------------------------------
        # F_Recon does  out = coarse_volume + res_scale*conv(...)  -- UNCLAMPED in
        # train mode (losses see the true values), clamped to [0,1] in eval mode.
        #
        # What matters is NOT that max == 1 (that was only true for the legacy
        # normalize="global" min-max). With normalize="percentile" the max is
        # deliberately free, and what has to line up is the MEDIAN: the coarse
        # volume and the target should sit on the same intensity scale, or the
        # residual branch has to spend itself producing a constant offset.
        # `hf` is high-frequency energy; with deconv="adjoint" it comes out at a
        # few percent of the target's, which is the sharpness ceiling the
        # network then has to work against. deconv="tikhonov" should raise it.
        with torch.no_grad():
            cv_min = coarse_volume.min().item()
            cv_max = coarse_volume.max().item()
            cv_mean = coarse_volume.mean().item()
            cv_med = coarse_volume.median().item()
            gt_med = target3d.median().item()
            cv_neg = (coarse_volume < 0).float().mean().item() * 100.0
            # high-frequency energy of the middle slice, coarse vs target
            _z = coarse_volume.shape[-3] // 2
            _cv = coarse_volume.reshape(-1, *coarse_volume.shape[-2:])[_z][None, None]
            _gt = target3d.reshape(-1, *target3d.shape[-2:])[_z][None, None]
            _k = 9
            _blur = lambda t: torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(t, (_k // 2,) * 4, mode="replicate"), _k, stride=1)
            hf_cv = (_cv - _blur(_cv)).std().item()
            hf_gt = (_gt - _blur(_gt)).std().item()
            # Correlation decides whether extra sharpness is signal or noise:
            # more high-frequency energy AND higher corr = real detail recovered;
            # more energy but LOWER corr = tikhonov_reg is too small and you are
            # amplifying noise. Sweep tikhonov_reg for the best corr, not the
            # highest hf%.
            _a = (_cv - _cv.mean()).flatten()
            _b = (_gt - _gt.mean()).flatten()
            cv_corr = (_a @ _b / (_a.norm() * _b.norm()).clamp_min(1e-12)).item()
        if i % 20 == 0:
         print(f"coarse_volume  min={cv_min:.4e}  max={cv_max:.4e}  mean={cv_mean:.4e}")
         print(f"coarse_volume  median={cv_med:.4f} vs target median={gt_med:.4f}"
               f"   (ratio {cv_med / max(gt_med, 1e-8):.2f}x)   negatives {cv_neg:.2f}%")
         print(f"coarse_volume  high-freq energy: coarse {hf_cv:.4f}  target {hf_gt:.4f}"
               f"   ({100 * hf_cv / max(hf_gt, 1e-8):.1f}% of target)")
         print(f"coarse_volume  corr with target (mid slice) = {cv_corr:.3f}"
               f"    <- maximise THIS when sweeping cg_reg")
        # Per-sample ratios are meaningless here: PercentileNormalize3D scales each
        # target by its OWN 1/99.7 percentiles, so a sparse volume has a median of
        # ~0.015 and a dense one ~0.19, while the coarse median stays ~0.11. Only
        # the DATASET AVERAGE says whether p_low is set right, so accumulate and
        # judge on the running means.
        _cv_med_hist.append(cv_med)
        _gt_med_hist.append(gt_med)
        if len(_cv_med_hist) >= 20:
            _cvm = sum(_cv_med_hist) / len(_cv_med_hist)
            _gtm = sum(_gt_med_hist) / len(_gt_med_hist)
            if len(_cv_med_hist) % 20 == 0:
                print(f"coarse_volume  running median: coarse {_cvm:.4f}  target {_gtm:.4f}"
                      f"   (ratio {_cvm / max(_gtm, 1e-8):.2f}x, over {len(_cv_med_hist)} steps)")
            if _gtm > 1e-6 and not (0.5 < _cvm / _gtm < 2.0) and len(_cv_med_hist) % 20 == 0:
                print(f"!!  running coarse median is {_cvm / _gtm:.2f}x the target's -- the "
                      f"residual branch will have to learn a constant offset. Adjust p_low "
                      f"in FLFMAdjointBackprojector (higher p_low -> lower median).")
        if cv_max < 1e-3:
            print(f"!!  coarse_volume max = {cv_max:.3e}: the volume is ~0, so x_in is "
                  f"being ignored in F_Recon. Check `normalize` is not 'none'.")
        #torch.save(coarse_volume, "coarse_volume.pt")

        #RECONSTRUCTION FORWARD            
        output_recon = model_recons(coarse_volume)
        vprint("Output recon requires grad:", output_recon.requires_grad)
        print_memory_usage(f"After Recons Forward Pass - Batch {i}")
        vprint("Output recon max/min:", output_recon.max().item(), output_recon.min().item())
        vprint("Output recon shape:", output_recon.shape)
        f2 = time.time()

        # -----------------------------
        # COMPUTE LOSS (with axial warmup)
        # -----------------------------
        # Scheduled loss weights
        iteration = global_step
        loss_w = get_recon_loss_weights(iteration)

        mae_w   = loss_w["mae"]
        edge_w  = loss_w["edge"]
        ssim_w  = loss_w["ssim"]
        axial_w = loss_w["axial"]

        vprint(f"Loss weights at iter {iteration}: "
            f"MAE={mae_w:.3f}, EDGE={edge_w:.3f}, SSIM={ssim_w:.3f}, AXIAL={axial_w:.3f}")

        # individual losses
        mae_loss_val = mae_loss(output_recon, target3d)
        vprint("MAE Loss:", mae_loss_val.item(), mae_loss_val.requires_grad)

        edge_loss_val = edge_loss(output_recon, target3d)
        vprint("L1 Edge Loss:", edge_loss_val.item(), edge_loss_val.requires_grad)

        # Only SSIM gets a clamped copy: piqa assumes inputs in [0, 1].
        # MAE / edge / axial keep the unclamped output so they penalise overshoot.
        avg_ssim = ssim_loss(output_recon.clamp(0.0, 1.0), target3d)
        vprint("Average SSIM over D slices:", avg_ssim.item(), avg_ssim.requires_grad)

        ssim_loss_val = 1.0 - avg_ssim
        vprint("SSIM Loss (1-SSIM):", ssim_loss_val.item(), ssim_loss_val.requires_grad)

        axial_loss_val = axial_diff_matching_loss(output_recon, target3d)
        vprint("Axial Huber Loss:", axial_loss_val.item(), axial_loss_val.requires_grad)

        # weighted losses
        mae_term   = mae_w   * mae_loss_val
        edge_term  = edge_w  * edge_loss_val
        ssim_term  = ssim_w  * ssim_loss_val
        axial_term = axial_w * axial_loss_val

        vprint("mae weighted loss:", mae_term.item())
        vprint("edge weighted loss:", edge_term.item())
        vprint("ssim weighted loss:", ssim_term.item())
        vprint("axial weighted loss:", axial_term.item())

        # (focal diversity regulariser removed -- it was 1e-4 * var(f) with f in
        #  metres, i.e. ~1e-9 against an MAE of ~1e-1, so it contributed nothing)

        # final loss used for backward
        loss = mae_term + edge_term + ssim_term + axial_term
        if DIVERSITY_REG:
            # -var() pushes the learned parameters apart. See the DIVERSITY_REG
            # note at the top: on for both arms or neither.
            _o = model_optics.module if hasattr(model_optics, "module") else model_optics
            div_loss = -_o.f_rand.var()
            if LEARN_D:
                div_loss = div_loss - _o.D_map.view(-1).var()
            loss = loss + DIVERSITY_REG * div_loss
            vprint("diversity weighted loss:", (DIVERSITY_REG * div_loss).item())
        total_recon_loss = loss / accumulation_steps

        vprint("Loss:", total_recon_loss)

        # epoch accumulators
        epoch_loss_accumulators["mae_loss"] += mae_term.item()
        epoch_loss_accumulators["edge_loss"] += edge_term.item()
        epoch_loss_accumulators["ssim_loss"] += ssim_term.item()
        epoch_loss_accumulators["axial_diff_matching_loss"] += axial_term.item()

        epoch_recon_loss += total_recon_loss.item()


        f3 = time.time()
        vprint("Loss:", total_recon_loss)
        print(f"""
        🔬 Forward sub-timing:
        - Optics forward:  {f1 - f0:.2f}s
        - Recon forward:   {f2 - f1:.2f}s
        - Loss computation:{f3 - f2:.2f}s
        """)
        forward_end = time.time()

        #############################################

        #############################################
        # **Backward Pass**
        #############################################
        #torch.autograd.set_detect_anomaly(True)
        backward_start = time.time()
        vprint("Loss value:", total_recon_loss)
        vprint("Has grad:", total_recon_loss.requires_grad)
        vprint("Grad fn:", total_recon_loss.grad_fn)

        if torch.isnan(output_recon).any() or torch.isinf(output_recon).any():
            print("? NaN/Inf in output_recon. Skipping step.")
            continue
        
        if torch.isnan(total_recon_loss).any() or torch.isinf(total_recon_loss).any():
            print("? NaN/Inf in loss. Skipping step.")
            continue

        #Backward prop
        total_recon_loss.backward()

        # Clip gradients BEFORE checking for NaNs
        # clip_grad_norm_ returns the norm BEFORE clipping -- log that, since the
        # per-parameter print below only ever sees clipped gradients.
        g_recon = torch.nn.utils.clip_grad_norm_(model_recons.parameters(), max_norm=1.0)
        g_optics = torch.nn.utils.clip_grad_norm_(model_optics.parameters(), max_norm=1.0)
        vprint(f"grad norm (pre-clip): recon {g_recon.item():.3e}  optics {g_optics.item():.3e}")

        skip_step = False

        # FIX: the optics model is checked too. Previously only model_recons was
        # scanned, so a NaN in an optics gradient passed straight through:
        # clip_grad_norm_ PROPAGATES NaN rather than removing it, the optimizer
        # then wrote NaN into f_rand, and clamp_(NaN) stays NaN — after which
        # every PSF is silently garbage for the rest of the run.
        for tag, mdl in (("RECON", model_recons), ("OPTICS", model_optics)):
            for name, param in mdl.named_parameters():
                if param.grad is not None and (
                        torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"❌ NaN/Inf in {tag} gradient for {name}. Skipping optimizer step.")
                    skip_step = True
                    break
            if skip_step:
                break

        if VERBOSE:
            for n, p in model_optics.named_parameters():
                if p.requires_grad:
                    print(n, None if p.grad is None else f"{p.grad.abs().mean():.3e}")

        # Optimizer step
        if not skip_step:
            if optimizer_optics is not None and not FREEZE_OPTICS:
                optimizer_optics.step()

                with torch.no_grad():
                    _o = model_optics.module if hasattr(model_optics, "module") else model_optics
                    _o.f_rand.clamp_(_o.fmin, _o.fmax)
                    # Clamped on the PARAMETER after the step, not inside the
                    # forward pass: clamping in forward would zero the gradient
                    # at the bounds and the diameters would stick there.
                    if LEARN_D:
                        _o.D_map.clamp_(_o.min_D, _o.max_D)

            optimizer_recon.step()
        else:
            if optimizer_optics is not None:
                optimizer_optics.zero_grad()
            optimizer_recon.zero_grad()

            for param_group in optimizer_recon.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 0.5
                param_group['lr'] = max(new_lr, 1e-7)
                print(f"🔻 Reduced learning rate from {old_lr:.2e} to {new_lr:.2e} due to NaN in gradients!")

        vprint(f"🔁 Optimizer stepped at iteration {i+1}")

        with torch.no_grad():
            for name, param in model_optics.named_parameters():
                if name in prev_optics_weights and param.requires_grad:
                    delta = (param.detach() - prev_optics_weights[name]).abs().max().item()
                    vprint(f"OPTICS {name} max abs update: {delta:.4e}")
                    if delta == 0:
                        print(f"❌ OPTICS {name} NOT updated!")

            for name, param in model_recons.named_parameters():
                if name in prev_recon_weights and param.requires_grad:
                    delta = (param.detach() - prev_recon_weights[name]).abs().max().item()
                    vprint(f"RECON {name} max abs update: {delta:.4e}")
                    if delta == 0:
                        print(f"❌ RECON {name} NOT updated!")

            prev_optics_weights = {
                name: p.detach().clone()
                for name, p in model_optics.named_parameters()
                if p.requires_grad
            }
            prev_recon_weights = {
                name: p.detach().clone()
                for name, p in model_recons.named_parameters()
                if p.requires_grad
            }


        
        step_end = time.time()
        print(f"""
        🕒 Step {i+1} timing:
        - Data loading:       {data_load_end - data_load_start:.2f}s
        - Forward pass:       {forward_end - forward_start:.2f}s
        - Backward + step:    {step_end - backward_start:.2f}s
        - Total per step:     {step_end - step_start:.2f}s
        """)
        del coarse_volume, output_optics_norm, output_optics, output_recon
        log_time_start = time.time()
        #torch.cuda.synchronize()  # Ensure all operations are completed before logging
        torch.cuda.empty_cache()  

        epoch_avg_loss = epoch_recon_loss / len(train_loader)  # or total_iters_in_epoch


        # **Log Losses**
        log_dict = {
            "train/epoch": epoch,
            "train/iter": global_step,
            "train/batch_in_epoch": i,
            "train/lr": optimizer_recon.param_groups[0]["lr"],
            "train/lr_optical": optimizer_optics.param_groups[0]["lr"],
            "train/photons": step_photons,
            "train/loss_total": total_recon_loss.item(),
            "train/loss_mae_weighted": mae_term.item(),
            "train/loss_edge_weighted": edge_term.item(),
            "train/loss_ssim_weighted": ssim_term.item(),
            "train/loss_axial_weighted": axial_term.item(),
        }

        wandb.log(log_dict, step=global_step)
        global_step += 1

        log_time_end = time.time()
        print(f"""
        🕒 Step {i+1} timing:
        - Wandb Logging:       {log_time_end - log_time_start:.2f}s
        """)

    # **Epoch-Level Logging**
    avg_recon_loss = epoch_recon_loss / total_steps
    print(f"Epoch {epoch+1} Completed - Avg Recon Loss: {avg_recon_loss:.4f}")

    avg_losses_epoch = {
        f"epoch_avg/{key}": epoch_loss_accumulators[key] / total_steps
        for key in epoch_loss_accumulators
    }

    epoch_log_step = max(global_step - 1, 0)

    epoch_log_dict = {
        **avg_losses_epoch,
        "epoch": epoch + 1,
        "epoch/recon_loss": avg_recon_loss,
    }

    wandb.log(epoch_log_dict, step=epoch_log_step)


        
    #############################################      
    # Validation phase
    #############################################

    model_optics.eval()
    model_recons.eval()
    
    with torch.no_grad():  # Disable gradient calculation for validation
        val_total_loss = 0
        val_mae_total = 0.0
        val_edge_total = 0.0
        val_ssim_total = 0.0
        val_axial_total = 0.0
        # saturation / offset diagnostics (targets: sat_frac -> GT's ~0.003,
        # bias -> 0, bg_median -> GT's bg_median)
        val_sat_total = 0.0
        val_sat_gt_total = 0.0
        val_bias_total = 0.0
        val_bg_recon_total = 0.0
        val_bg_gt_total = 0.0
        # Correlation with the target. This is THE number: once the intensity
        # offset is right, both NRMSE and PSNR are bounded by it --
        #       NRMSE_min = sqrt(1 - rho^2)
        # and that bound is tight (measured rho=0.365 -> predicted 0.9310,
        # actual affine-corrected 0.9311). Nothing improves until rho does.
        val_corr_total = 0.0
        # Correlation of the LIFT alone, before F_Recon touches it. The
        # difference val/corr - val/corr_coarse is the network's contribution,
        # visible every epoch instead of only after unzipping test volumes.
        # It has been near zero or negative in every configuration so far
        # (+0.004 at N=100, -0.050 at N=10), so this is the number that says
        # whether the network earns its place.
        val_corr_coarse_total = 0.0

        # use the current training step for the scheduled validation loss
        val_loss_w = get_recon_loss_weights(global_step)

        mae_w   = val_loss_w["mae"]
        edge_w  = val_loss_w["edge"]
        ssim_w  = val_loss_w["ssim"]
        axial_w = val_loss_w["axial"]

        print(f"[VAL] Loss weights at global_step {global_step}: "
            f"MAE={mae_w:.3f}, EDGE={edge_w:.3f}, SSIM={ssim_w:.3f}, AXIAL={axial_w:.3f}")
    


        for i, data in enumerate(val_loader):
            
            target3d, lo, hi = data
            target3d = target3d.squeeze(dim=0).to(device, non_blocking=True)
            lo = lo.to(device)
            hi = hi.to(device)

            vprint("Target 3d max/min:", target3d.max().item(), target3d.min().item())
            vprint("lo/hi:", lo.item(), hi.item())

            # Forward pass
            #optics
            output_optics, pitch_x, psf_stack = model_optics(target3d) 
            output_optics_norm = (output_optics - output_optics.amin(dim=(-2, -1), keepdim=True)) / (
                output_optics.amax(dim=(-2, -1), keepdim=True) - output_optics.amin(dim=(-2, -1), keepdim=True) + 1e-8
            )
            # add sensor noise
            output_optics_norm = apply_sensor_noise(output_optics_norm)

            # adjoint backprojection (validation runs under no_grad, so there is
            # nothing to detach; psf_stack is passed through unchanged, matching
            # training where optics gradients DO flow through the backprojector)
            coarse_volume = adjoint_bp(
                sensor_img=output_optics_norm,
                psf_stack=psf_stack,
                dz=model_optics.dz_used
            )   # [B, D, 512, 512]

            _c = (coarse_volume.float() - coarse_volume.float().mean()).flatten()
            _t = (target3d.float() - target3d.float().mean()).flatten()
            val_corr_coarse_total += (
                _c @ _t / (_c.norm() * _t.norm()).clamp_min(1e-12)).item()

            #3d recons       
            output_recon = model_recons(coarse_volume)        

            # individual validation losses
            mae_loss_val = mae_loss(output_recon, target3d)
            edge_loss_val = edge_loss(output_recon, target3d)

            avg_ssim = ssim_loss(output_recon, target3d)
            ssim_loss_val = 1.0 - avg_ssim

            axial_loss_val = axial_diff_matching_loss(output_recon, target3d)

            # weighted validation loss using current training schedule
            val_loss = (
                mae_w   * mae_loss_val +
                edge_w  * edge_loss_val +
                ssim_w  * ssim_loss_val +
                axial_w * axial_loss_val
            )

            val_total_loss += val_loss.item()
            val_mae_total += mae_loss_val.item()
            val_edge_total += edge_loss_val.item()
            val_ssim_total += avg_ssim.item()              # log actual SSIM here
            val_axial_total += axial_loss_val.item()

            val_sat_total      += (output_recon >= 1.0).float().mean().item()
            val_sat_gt_total   += (target3d >= 1.0).float().mean().item()
            val_bias_total     += (output_recon - target3d).mean().item()
            val_bg_recon_total += output_recon.float().median().item()
            val_bg_gt_total    += target3d.float().median().item()

            _a = (output_recon.float() - output_recon.float().mean()).flatten()
            _b = (target3d.float() - target3d.float().mean()).flatten()
            val_corr_total += (_a @ _b / (_a.norm() * _b.norm()).clamp_min(1e-12)).item()

       
        n_val = max(1, len(val_loader))
        avg_val_loss = val_total_loss / n_val
        avg_val_mae = val_mae_total / n_val
        avg_val_edge = val_edge_total / n_val
        avg_val_ssim = val_ssim_total / n_val
        avg_val_axial = val_axial_total / n_val
        avg_val_sat = val_sat_total / n_val
        avg_val_sat_gt = val_sat_gt_total / n_val
        avg_val_bias = val_bias_total / n_val
        avg_val_bg_recon = val_bg_recon_total / n_val
        avg_val_bg_gt = val_bg_gt_total / n_val
        avg_val_corr = val_corr_total / n_val
        avg_val_corr_coarse = val_corr_coarse_total / n_val
        avg_val_net = avg_val_corr - avg_val_corr_coarse

        print(f"Validation Loss at Epoch {epoch}: {avg_val_loss}")
        print(f"Validation MAE at Epoch {epoch}: {avg_val_mae}")
        print(f"Validation Edge at Epoch {epoch}: {avg_val_edge}")
        print(f"Validation SSIM at Epoch {epoch}: {avg_val_ssim}")
        print(f"Validation Axial at Epoch {epoch}: {avg_val_axial}")
        print(f"Validation saturation at Epoch {epoch}: recon {avg_val_sat*100:.3f}%  "
              f"GT {avg_val_sat_gt*100:.3f}%")
        print(f"Validation bias (recon-GT) at Epoch {epoch}: {avg_val_bias:+.4f}   "
              f"bg median recon {avg_val_bg_recon:.4f} / GT {avg_val_bg_gt:.4f}")
        print(f"Validation corr: lift {avg_val_corr_coarse:.4f} -> recon "
              f"{avg_val_corr:.4f}   NETWORK CONTRIBUTION {avg_val_net:+.4f}"
              f"{'   <- network is HURTING' if avg_val_net < -0.002 else ''}")
        _nrmse_floor = (1.0 - min(avg_val_corr, 1.0) ** 2) ** 0.5
        print(f"Validation corr with target at Epoch {epoch}: {avg_val_corr:.4f}"
              f"   -> best achievable NRMSE = sqrt(1-rho^2) = {_nrmse_floor:.4f}"
              f"   (NOTE: judge the network on SSIM and PSNR, not on rho --"
              f" rho is affine-invariant and cannot see the calibration it fixes)")

    # log once per epoch
    epoch_log_step = max(global_step - 1, 0)

    wandb.log({
        "val/loss_epoch": avg_val_loss,
        "val/mae_epoch": avg_val_mae,
        "val/edge_epoch": avg_val_edge,
        "val/ssim_epoch": avg_val_ssim,
        "val/axial_epoch": avg_val_axial,
        "val/sat_frac": avg_val_sat,
        "val/sat_frac_gt": avg_val_sat_gt,
        "val/bias": avg_val_bias,
        "val/bg_median_recon": avg_val_bg_recon,
        "val/bg_median_gt": avg_val_bg_gt,
        "val/corr": avg_val_corr,
        "val/corr_coarse": avg_val_corr_coarse,
        "val/net_contribution": avg_val_net,
        "val/nrmse_floor": _nrmse_floor,
        "epoch": epoch + 1,
    }, step=epoch_log_step)


    # Save models every n epochs as artifacts
    if (epoch + 1) % 5 == 0:   # every epoch while monitoring the fine-tune
        # Save into the run directory to avoid permission and path issues on Windows
        optics_path = os.path.join(wandb.run.dir,
                                   f"optics_model_{RUN_TAG}_epoch_{epoch+1}.pth")
        recon_path  = os.path.join(wandb.run.dir,
                                   f"recon_model_{RUN_TAG}_epoch_{epoch+1}.pth")

        os.makedirs(os.path.dirname(optics_path), exist_ok=True)
        os.makedirs(os.path.dirname(recon_path), exist_ok=True)

        torch.save(model_optics.state_dict(), optics_path)
        torch.save(model_recons.state_dict(), recon_path)

        # Create and log artifacts
        optics_artifact = wandb.Artifact(f"optics_model_epoch_{epoch+1}", type="model")
        optics_artifact.add_file(optics_path)
        wandb.log_artifact(optics_artifact)

        recon_artifact = wandb.Artifact(f"recon_model_epoch_{epoch+1}", type="model")
        recon_artifact.add_file(recon_path)
        wandb.log_artifact(recon_artifact)

        print(f"Saved models and logged artifacts for epoch {epoch+1}")
        
        
#############################################      
# Test phase
#############################################
model_optics.eval()
model_recons.eval()
# Sweep the sensor photon budget at test time, so performance is reported across a representative range of photon
# counts rather than at one unstated noise level.
all_photon_results = {}
for photons_test in config.train_params["sensor_noise"]["photons_test"]:
    print(f"\n{'='*60}\nTEST at {photons_test} photons/pixel (peak)\n{'='*60}")
    with torch.no_grad():  # No gradients needed for testing
        ssim_values_all  = []
        rmse_values_all  = []
        nrmse_values_all = []
        groups_all       = []   # "A" / "B" per volume; see GROUP_HI_THRESHOLD

        test_loss_values_all  = []
        test_mae_all   = []
        test_edge_all  = []
        test_ssim_all  = []
        test_axial_all = []

        # use the latest training step for reporting
        test_log_step = max(global_step - 1, 0)

        # use the same scheduled weights style as training/validation
        test_loss_w = get_recon_loss_weights(global_step)

        mae_w   = test_loss_w["mae"]
        edge_w  = test_loss_w["edge"]
        ssim_w  = test_loss_w["ssim"]
        axial_w = test_loss_w["axial"]

        print(f"[TEST] Loss weights at global_step {global_step}: "
              f"MAE={mae_w:.3f}, EDGE={edge_w:.3f}, SSIM={ssim_w:.3f}, AXIAL={axial_w:.3f}")

        for i, data in enumerate(test_loader):
        
            target3d, lo, hi = data
            # tag BEFORE moving to device -- hi is the raw percentile
            vol_group = "A" if float(hi.item()) < GROUP_HI_THRESHOLD else "B"
            target3d = target3d.squeeze(dim=0).to(device, non_blocking=True)
            lo = lo.to(device)
            hi = hi.to(device)

            vprint("Target 3d max/min:", target3d.max().item(), target3d.min().item())
            vprint("lo/hi:", lo.item(), hi.item())
        
            # Forward pass
            #optics
            output_optics, pitch_x, psf_stack = model_optics(target3d) 
            output_optics_norm = (output_optics - output_optics.amin(dim=(-2, -1), keepdim=True)) / (
                output_optics.amax(dim=(-2, -1), keepdim=True) - output_optics.amin(dim=(-2, -1), keepdim=True) + 1e-8
            )
            # add sensor noise at the photon count being swept (exact Poisson:
            # no gradients needed at test time)
            output_optics_norm = apply_sensor_noise(
                output_optics_norm, photons=photons_test, exact=True)

            # adjoint backprojection
            coarse_volume = adjoint_bp(
                sensor_img=output_optics_norm,
                psf_stack=psf_stack,
                dz=model_optics.dz_used
            )   # [B, D, 512, 512]

            #3d recons       
            output_recon = model_recons(coarse_volume)  

            # component losses
            mae_loss_val = mae_loss(output_recon, target3d)
            edge_loss_val = edge_loss(output_recon, target3d)

            avg_ssim = ssim_loss(output_recon, target3d)   # actual SSIM
            ssim_loss_val = 1.0 - avg_ssim

            axial_loss_val = axial_diff_matching_loss(output_recon, target3d)

            # weighted test loss
            total_test_recon_loss = (
                mae_w   * mae_loss_val +
                edge_w  * edge_loss_val +
                ssim_w  * ssim_loss_val +
                axial_w * axial_loss_val
            )

            ###### volume metrics ######
            target3d_np = target3d.detach().cpu().numpy().squeeze()
            output_recon_np = output_recon.detach().cpu().numpy().squeeze()

            assert target3d_np.shape == output_recon_np.shape, "Input shapes must match!"

            ssim_slices = []
            rmse_slices = []
            for d in range(target3d_np.shape[0]):
                slice_target = target3d_np[d]
                slice_recon = output_recon_np[d]

                data_range = max(float(slice_target.max() - slice_target.min()), 1e-8)

                ssim_slices.append(compare_ssim(slice_target, slice_recon, data_range=data_range))
                rmse_slices.append(np.sqrt(np.mean((slice_target - slice_recon) ** 2)))

            ssim_value_volume = float(np.mean(ssim_slices))
            rmse_value_volume = float(np.mean(rmse_slices))
            nrmse_value_volume = float(calculate_nrmse(target3d, output_recon))

            # accumulate
            test_loss_values_all.append(float(total_test_recon_loss.item()))
            test_mae_all.append(float(mae_loss_val.item()))
            test_edge_all.append(float(edge_loss_val.item()))
            test_ssim_all.append(float(avg_ssim.item()))
            test_axial_all.append(float(axial_loss_val.item()))

            rmse_values_all.append(rmse_value_volume)
            ssim_values_all.append(ssim_value_volume)
            nrmse_values_all.append(nrmse_value_volume)
            groups_all.append(vol_group)


        # averages
        avg_loss  = float(np.mean(test_loss_values_all))
        avg_mae   = float(np.mean(test_mae_all))
        avg_edge  = float(np.mean(test_edge_all))
        avg_ssim_loss_metric = float(np.mean(test_ssim_all))   # this is actual SSIM
        avg_axial = float(np.mean(test_axial_all))

        avg_rmse  = float(np.mean(rmse_values_all))
        avg_ssim  = float(np.mean(ssim_values_all))
        avg_nrmse = float(np.mean(nrmse_values_all))

        # ---- per-group breakdown -------------------------------------
        # A single mean over the test set mixes two populations that differ
        # by ~0.4 in rho, so it is not a meaningful number on its own.
        g = np.array(groups_all)
        grp_stats = {}
        for gname in ("A", "B"):
            sel = g == gname
            if sel.sum() == 0:
                continue
            grp_stats[gname] = {
                "n": int(sel.sum()),
                "ssim": float(np.mean(np.array(ssim_values_all)[sel])),
                "rmse": float(np.mean(np.array(rmse_values_all)[sel])),
                "nrmse": float(np.mean(np.array(nrmse_values_all)[sel])),
            }

        wandb.log(
            {
                f"test_p{photons_test}/loss": avg_loss,
                f"test_p{photons_test}/mae": avg_mae,
                f"test_p{photons_test}/edge": avg_edge,
                f"test_p{photons_test}/ssim_loss_metric": avg_ssim_loss_metric,
                f"test_p{photons_test}/axial": avg_axial,
                f"test_p{photons_test}/avg_rmse_volume": avg_rmse,
                f"test_p{photons_test}/avg_ssim_volume": avg_ssim,
                f"test_p{photons_test}/avg_nrmse_volume": avg_nrmse,
                **{f"test_p{photons_test}/group{gn}_{k}": v
                   for gn, s in grp_stats.items() for k, v in s.items()},
            },
            step=test_log_step,
            commit=True
        )

        print(
            "Test Results: "
            f"Average Test Loss: {avg_loss}, "
            f"Average MAE: {avg_mae}, "
            f"Average Edge: {avg_edge}, "
            f"Average SSIM Loss Metric: {avg_ssim_loss_metric}, "
            f"Average Axial: {avg_axial}, "
            f"Average RMSE Volume: {avg_rmse}, "
            f"Average SSIM Volume: {avg_ssim}, "
            f"Average NRMSE Volume: {avg_nrmse}"
        )

        for gname, s in sorted(grp_stats.items()):
            tag = ("dense / diffuse" if gname == "A" else "sparse / bright")
            print(f"    group {gname} ({tag}, n={s['n']}): "
                  f"SSIM {s['ssim']:.4f}  RMSE {s['rmse']:.4f}  "
                  f"NRMSE {s['nrmse']:.4f}")

    all_photon_results[photons_test] = {
        "ssim": avg_ssim, "rmse": avg_rmse, "nrmse": avg_nrmse, "loss": avg_loss,
        "groups": grp_stats,
    }


# ---- summary table across the photon sweep --------------------------------
print("\n" + "=" * 78)
print("TEST SUMMARY vs photon budget   (A = dense/diffuse, B = sparse/bright)")
print(f"{'photons':>9}{'SSIM':>9}{'RMSE':>10}{'NRMSE':>9}{'loss':>10}"
      f"{'SSIM A':>10}{'SSIM B':>10}{'NRMSE A':>10}{'NRMSE B':>10}")
for ph, r in all_photon_results.items():
    gs = r.get("groups", {})
    ga, gb = gs.get("A"), gs.get("B")
    f = lambda s, k: f"{s[k]:>10.4f}" if s else f"{'--':>10}"
    print(f"{ph:>9}{r['ssim']:>9.4f}{r['rmse']:>10.4f}{r['nrmse']:>9.4f}"
          f"{r['loss']:>10.4f}{f(ga,'ssim')}{f(gb,'ssim')}"
          f"{f(ga,'nrmse')}{f(gb,'nrmse')}")
print("\nReport the two groups separately. The combined mean averages over two\n"
      "populations that differ by ~0.4 in rho, so on its own it says little.")
wandb.log({f"test_sweep/ssim_p{ph}": r["ssim"] for ph, r in all_photon_results.items()},
          step=test_log_step)
    
   
   
# Close the TensorBoard writer
#writer.close()

# Finish wandb logging session
wandb.finish()

# Save models.
# Tagged by arm (OPTICS_OUT / RECON_OUT) so queued runs cannot overwrite each
# other -- the previous version always wrote config.paths['model_optics_path'],
# so a second arm silently destroyed the first one's reported model.
torch.save(model_optics.state_dict(), OPTICS_OUT)
torch.save(model_recons.state_dict(), RECON_OUT)
print(f"saved {OPTICS_OUT} and {RECON_OUT}")

# Also written to the legacy paths, so test_img.py and anything else that reads
# config.paths keeps working on the most recent run.
torch.save(model_optics.state_dict(), config.paths['model_optics_path'])
torch.save(model_recons.state_dict(), config.paths['model_recon_path'])
