# 6. Limitations and outlook

## Simulation only

All results are simulated. An experimental system will need:

- the fabricated encoder to be **calibrated**, with an **experimental PSF stack** measured;
- the reconstruction network to be **fine-tuned** for model mismatch: aberrations, alignment errors and real sensor noise.

## Model assumptions

- **Scalar** Debye model: polarisation is not represented.
- **Single wavelength** (633 nm), and **lateral shift invariance** across the field of view.
- In the \(f, D\) ablation, overlapping apertures are resolved by **sequential overwriting** on one grid. This is one well-defined transmittance, but a fabricated element would have to realise exactly this composited pattern.

## Reading the metrics

- **Blur-matched resolution** assumes the reconstruction differs from the ground truth mainly by blur. Noise, artefacts and intensity errors are partly absorbed into the fitted width, so the number is an *effective* reconstruction resolution, not an optical one. Near focus it can favour a grainier reconstruction (RL-TV), so the Fourier ring correlation and the line profiles (Publication III, Fig. S3) are reported alongside.
- **Global metrics** (PSNR, SSIM, ρ) are dominated by the bulk of the volume, so they move less than the depth-resolved resolution.

## Generalisation

On unseen specimens the learned encoders keep the morphology and their speed advantage, but the fidelity gain from optical learning shrinks, and RL-TV is more accurate on average. The results do not separate whether this comes from the network or from the jointly optimised optics. For new specimen types, labelling densities or channels, the network, and possibly the optics, can be fine-tuned on representative data (`finetune_external.py`).

## Towards hardware

- The learned design keeps the **positions and pitch** of the commercial MLA. A fused-silica version only needs the lenslet **radii of curvature changed by about ±4 %**, for example by grayscale lithography or diamond turning.
- The phase can also be realised as a **metasurface**. The simulation grid is already a whole number of 225 nm TiO₂ unit cells (thesis Section 4.6).

## Outlook

- Larger axial ranges, by adjusting the training depth range, the PSF sampling and the focal-length bounds. The balance between axial range, lateral resolution and photon efficiency then has to be re-evaluated.
- The focal-length interval was deliberately limited to the reconstructed volume. Several lenslets reach its limits, so a wider interval would likely be used by the optimiser if the volume were extended.
