#!/usr/bin/env bash
# Train the three reconstruction arms of the FLFM comparison one after another.
#
#   fixed : conventional periodic MLA (all lenslets at 43.1 mm), optics frozen
#   f     : per-lenslet focal lengths learned (main design)
#   fD    : focal lengths and aperture diameters learned (ablation)
#
# All three use the same seed, epoch count, conjugate-gradient lift, noise
# model and reconstruction network, so the encoder is the only difference.
# Checkpoints are written to the current folder as optics_model_<arm>.pth and
# recon_model_<arm>.pth.
#
# Usage:
#   export FLFM_DATA_DIR=/path/to/data      # contains train/ val/ test/
#   WANDB_MODE=offline ./train_all_arms.sh  # or log in to Weights & Biases
set -euo pipefail

for arm in fixed f fD; do
  echo "=== training arm: ${arm}"
  OPTICS_MODE="${arm}" RUN_TAG="${arm}" python train_dp.py
done
