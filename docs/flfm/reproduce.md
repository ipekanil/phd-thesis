# 5. Reproduce

The complete end-to-end framework, including the differentiable optical model, is in the [`code/flfm`](https://github.com/ipekanil/phd-thesis/tree/main/code/flfm) folder of this site's repository.

| What | Where |
|---|---|
| Forward model, encoder, CG lift, network, losses, noise | `code/flfm/` |
| Training of the three arms | `train_dp.py`, `train_all_arms.sh` |
| Tables, figures, FRC, random-vs-learned control, timing | `make_paper_table.py`, `make_table2.py`, `make_figure2_optics.py`, `frc_depth.py`, `random_vs_learned_psf.py`, `time_reconstruction.py` |
| External specimens | `inspect_external.py`, `eval_external.py`, `finetune_external.py` |
| Training data (confocal mouse-tissue volumes) | Available from the authors on reasonable request |

## 1. Set up

```bash
git clone https://github.com/ipekanil/phd-thesis.git
cd phd-thesis/code/flfm

python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # choose the build for your CUDA version at pytorch.org
pip install -r requirements.txt
```

Training used one NVIDIA Tesla V100 with 32 GB. If memory runs out, lower `cg_chunk` (depths per chunk inside \(A\) and \(A^\top\)) in `train_dp.py`.

## 2. Prepare the data

Each volume is an HDF5 file with one dataset named `volume`, shape Z × Y × X, 512 × 512 laterally with at least 43 planes, at 0.377 µm × 0.377 µm × 1.43 µm. The central 43 planes are used, normalised by their 1st and 99.7th percentiles.

```text
data/
├── train/   *.h5
├── val/     *.h5
└── test/    *.h5
```

```bash
export FLFM_DATA_DIR=/path/to/data
```

To use your own confocal stacks, resample them to this voxel grid first, so that 43 planes span 60 µm as the optics assume.

## 3. Train

```bash
export WANDB_MODE=offline      # or run `wandb login` to log to Weights & Biases

OPTICS_MODE=fixed python train_dp.py   # conventional MLA, optics frozen
OPTICS_MODE=f     python train_dp.py   # learned focal lengths
OPTICS_MODE=fD    python train_dp.py   # learned focal lengths + diameters
```

`./train_all_arms.sh` runs the three in sequence. Each run writes `optics_model_<arm>.pth` and `recon_model_<arm>.pth`, then evaluates the test set at N = 10, 30, 100, 300 and 1000 photons.

**Defaults in `config.py`:**

| Setting | Value |
|---|---|
| Seed | 1234 (focal-length initialisation, network weights, noise, data order) |
| Epochs | 20, batch size 1 |
| Learning rates | optics 1e-4, network 2e-4 (Adam; network weight decay 1e-4) |
| Noise | N = 100 photons, 1.0 e⁻ read noise during training |
| CG lift | 10 iterations, κ = 3.0 |
| Focal-length bounds | computed from the ±30 µm volume: 41.49–44.84 mm |
| Diameter bounds (f, D arm) | 0.25–1.75 × pitch |

## 4. Evaluate

From the folder holding the checkpoints:

```bash
# per-depth metrics and figures for each arm (writes paper_table_<tag>/)
python make_paper_table.py --tag fixed --optics optics_model_fixed.pth --recon recon_model_fixed.pth --fixed-optics
python make_paper_table.py --tag f     --optics optics_model_f.pth     --recon recon_model_f.pth
python make_paper_table.py --tag fD    --optics optics_model_fD.pth    --recon recon_model_fD.pth
python make_paper_table.py --tag rltv  --optics optics_model_fixed.pth --method rltv --fixed-optics

# reconstruction time, then the overall table
for arm in fixed f fD; do
  python time_reconstruction.py --tag $arm --optics optics_model_$arm.pth --recon recon_model_$arm.pth
done
python time_reconstruction.py --tag rltv --optics optics_model_fixed.pth --method rltv
python make_table2.py

# learned optics figure and the random-focal-length control
python make_figure2_optics.py --root .
python random_vs_learned_psf.py --root . --learned optics_model_f.pth --learned-fD optics_model_fD.pth --seeds 40
```

| To reproduce | Run |
|---|---|
| Overall table (depth spread, PSNR, SSIM, ρ, time) | `make_paper_table.py` for the four arms, `time_reconstruction.py`, `make_table2.py` |
| Resolution against depth | `paper_table_<tag>/per_slice.csv`. Use `--axial-depths all` for axial resolution at every slice |
| Photon-budget sweep | add `--photons 10 30 100 300 1000` to `make_paper_table.py` |
| Fourier ring correlation | `frc_depth.py --root .`, after `make_paper_table.py` has saved TIFFs for every test volume (list all indices in `--figure-volumes`) |
| RL-TV on the learned encoders | `make_paper_table.py --method rltv --optics optics_model_f.pth` (and `_fD`) |

## 5. External specimens

```bash
python inspect_external.py --root ./data/external                  # check files, shapes, voxel sizes
python eval_external.py   --root ./data/external --resample grid   # resample to the model grid, then evaluate
```

File names and voxel sizes are set in `DATASETS` at the top of `eval_external.py`. The plant volumes are from Erguvan et al. (BMC Biology, 2019), and the planaria from the CARE collection of Weigert et al.

## No GPU?

The physics can be explored without the full pipeline. `python examples/flfm_figures.py` (NumPy and Matplotlib only) recreates the figures on these pages: the focal length ↔ depth relation, the phase masks, the toy view-blur model and the result charts. The [toy simulation](../background/toy-simulation.md) scripts reproduce the Fourier-optics behaviour of the EDoF part, and `config.py` prints every derived optical quantity when imported.
