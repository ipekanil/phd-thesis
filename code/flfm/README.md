# FLFM with learned phase mask arrays: code

End-to-end training framework for *Fourier light field microscopy with learned phase mask arrays* (Atalay Appak, Singh, Sahin, Guillemot, Caglayan). It contains the differentiable wave-optics forward model, the conjugate-gradient lift, the reconstruction network, and the scripts that produce the tables and figures.

Documentation: <https://ipekanil.github.io/phd-thesis/flfm/>

## Contents

| File | Role |
|---|---|
| `config.py` | All optical, network, noise and training parameters. Derived quantities (pupil size, sampling, focal-length bounds) are computed here. |
| `F_Optics_fixed.py` | Differentiable forward model: scalar Debye pupil with aplanatic apodization, relay to the Fourier plane, hexagonal lenslet array with learnable focal lengths (and diameters), angular-spectrum propagation to the sensor. Class `F_Optics`. |
| `adjoint.py` | Encoder-dependent operators \(A_\Theta\), \(A_\Theta^\top\) and the unrolled conjugate-gradient lift (`FLFMAdjointBackprojector`, `deconv="cg"`). |
| `F_VCD_pytorch_multiple_views.py` | Reconstruction network `F_Recon`: convolutional stem, atrous block, multi-resolution U-Net, residual output. |
| `custom_losses.py` | MAE, Sobel edge, SSIM and axial-consistency losses. |
| `sensor_noise.py` | Mixed Poisson–Gaussian sCMOS noise (differentiable Gaussian limit during training, exact Poisson at test). |
| `dataset.py` | Loads HDF5 volumes and crops the central 43 planes. |
| `rltv.py` | Richardson–Lucy with total-variation regularisation on the same forward operator (training-free baseline). |
| `train_dp.py` | Joint training of encoder and network. The arm is set with `OPTICS_MODE`. |
| `train_all_arms.sh` | Trains the `fixed`, `f` and `fD` arms in sequence. |
| `make_paper_table.py` | Per-depth metrics (PSNR, SSIM, ρ, blur-matched lateral/axial resolution) and figures for one arm. |
| `make_table2.py` | Overall comparison table across arms. |
| `time_reconstruction.py` | Wall-clock reconstruction time per volume. |
| `frc_depth.py` | Depth-resolved Fourier ring correlation resolution. |
| `random_vs_learned_psf.py` | Learned versus random focal-length maps, from the PSF stacks alone. |
| `make_figure2_optics.py` | Learned parameter maps and xz/yz PSF profiles. |
| `test_img.py` | Single-volume diagnostic across photon budgets. |
| `inspect_external.py`, `eval_external.py`, `finetune_external.py` | Evaluation on external specimens, with optional fine-tuning of the network. |
| `recompute_spread.py` | Recomputes the depth spread from saved per-slice results. |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install -r requirements.txt
```

A GPU with 32 GB of memory was used (NVIDIA Tesla V100). If memory runs out, lower `cg_chunk` in `train_dp.py`.

## Data

Volumes are HDF5 files with one dataset named `volume`, shape Z × Y × X, lateral size 512 × 512 and at least 43 planes. Voxel size: 0.377 µm laterally, 1.43 µm axially. The central 43 planes are used.

```text
data/
├── train/   *.h5
├── val/     *.h5
└── test/    *.h5
```

Point the code at it with `export FLFM_DATA_DIR=/path/to/data`. The confocal training volumes used in the paper are available from the authors on reasonable request.

## Training

```bash
export FLFM_DATA_DIR=/path/to/data
export WANDB_MODE=offline          # or `wandb login` to log online

OPTICS_MODE=fixed python train_dp.py   # conventional MLA, optics frozen
OPTICS_MODE=f     python train_dp.py   # learned focal lengths (main design)
OPTICS_MODE=fD    python train_dp.py   # learned focal lengths and diameters (ablation)
# or: ./train_all_arms.sh
```

Each run writes `optics_model_<arm>.pth` and `recon_model_<arm>.pth` to the current folder and evaluates the test set across the photon budgets in `config.train_params["sensor_noise"]["photons_test"]`.

## Evaluation

Run from the folder holding the checkpoints:

```bash
python make_paper_table.py --tag fixed --optics optics_model_fixed.pth --recon recon_model_fixed.pth --fixed-optics
python make_paper_table.py --tag f     --optics optics_model_f.pth     --recon recon_model_f.pth
python make_paper_table.py --tag fD    --optics optics_model_fD.pth    --recon recon_model_fD.pth
python make_paper_table.py --tag rltv  --optics optics_model_fixed.pth --method rltv --fixed-optics

for arm in fixed f fD; do
  python time_reconstruction.py --tag $arm --optics optics_model_$arm.pth --recon recon_model_$arm.pth
done
python time_reconstruction.py --tag rltv --optics optics_model_fixed.pth --method rltv

python make_table2.py                    # overall table (depth spread, PSNR, SSIM, ρ, time)
python make_figure2_optics.py --root .   # learned parameter maps and PSF profiles
python random_vs_learned_psf.py --root . --learned optics_model_f.pth --learned-fD optics_model_fD.pth --seeds 40
python frc_depth.py --root .             # needs gt/reconstruction TIFFs for each test volume
```

Add `--photons 10 30 100 300 1000` to `make_paper_table.py` for the photon-budget sweep. `frc_depth.py` reads the TIFFs written by `make_paper_table.py`. Pass every test-volume index to `--figure-volumes` so that TIFFs are saved for all of them.

### External specimens

```bash
python inspect_external.py --root ./data/external      # check files and voxel sizes first
python eval_external.py   --root ./data/external --resample grid
```

`--resample grid` resamples each stack onto the model's voxel grid (0.377 µm × 1.43 µm) before forward projection, as in the paper. Dataset definitions and voxel sizes are in `DATASETS` at the top of `eval_external.py`.

## Citation

Please cite the paper and the thesis; BibTeX is at <https://ipekanil.github.io/phd-thesis/resources/publications/>.
