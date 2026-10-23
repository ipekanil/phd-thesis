# 3. Reconstruction and training

Recovering 43 planes from one image is ill-posed: many volumes produce the same measurement. The reconstruction therefore combines a **physics-based regularised solve** with a **learned prior**. Both are differentiable, so the encoder is trained through them.

\[
\Theta \;\to\; \{h_{\Theta,k}\} \;\to\; y_\Theta \;\to\; b = A_\Theta^\top y_\Theta \;\to\; o_{\mathrm{cg}} \;\to\; \hat o = R_\psi(o_{\mathrm{cg}}) \;\to\; \mathcal{L}
\]

## Step 1: conjugate-gradient lift

The first volume estimate solves a Tikhonov-regularised least-squares problem against the **same** forward operator that produced the measurement:

\[
o_{\mathrm{cg}} = \arg\min_o \;\lVert y_\Theta - A_\Theta o\rVert_2^2 + \tau\lVert o\rVert_2^2
\quad\Longleftrightarrow\quad
\big(A_\Theta^\top A_\Theta + \tau I\big)\, o_{\mathrm{cg}} = A_\Theta^\top y_\Theta .
\]

The system matrix is symmetric positive definite for \(\tau > 0\), so **conjugate gradients** apply. Each iteration needs one application of \(A_\Theta\) and one of \(A_\Theta^\top\), both per-plane FFT convolutions and correlations. The normal matrix is never assembled, and memory stays at one volume.

Two regularisers act together:

- **explicit:** \(\tau\) bounds the solution norm, which is needed because \(A_\Theta^\top A_\Theta\) is rank-deficient. It is set relative to the PSF energy, \(\tau = \kappa \big\langle \sum_k \lvert H_k(f)\rvert^2 \big\rangle_f\) with \(\kappa = 3.0\), so one value works for every encoder;
- **implicit:** stopping after \(J = 10\) iterations. CG recovers the best-conditioned components first, so the iteration count acts as a regularisation parameter.

The iteration count is fixed and every step is differentiable, so the recursion **unrolls into the computational graph** and gradients reach \(\Theta\) through the measurement, the right-hand side \(A_\Theta^\top y_\Theta\) and the operator applied in each iteration. \(o_{\mathrm{cg}}\) is normalised by its 10th and 99.7th percentiles before the network.

## Step 2: reconstruction network

The network \(R_\psi\) treats the 43 planes as channels of a 2D convolutional network and predicts a **residual correction** to the CG volume:

\[
\hat o = \mathrm{clamp}\big(o_{\mathrm{cg}} + 0.2\, r_\psi(o_{\mathrm{cg}}),\,0,\,1\big).
\]

| Part | Configuration |
|---|---|
| Stem | 2 × (3×3 conv, LeakyReLU 0.01, InstanceNorm): 43 → 64 channels |
| Context | parallel atrous convolutions, dilation 1, 2, 4, concatenated and projected (1×1) to 64 channels |
| Encoder | 3 multi-resolution blocks of depthwise-separable convolutions, widths 64, 96, 128, max-pool downsampling |
| Decoder | bilinear upsampling, skip concatenation, convolutional refinement |
| Output | 43-channel residual, scale 0.2, clamped to [0, 1]; 512 × 512 per plane |

The residual form means the network refines a physically consistent estimate rather than learning the full inverse from the multiplexed image.

## Loss

\[
\mathcal{L}_{\mathrm{total}} = \lambda_{\mathrm{MAE}}\,\mathcal{L}_{\mathrm{MAE}} + \lambda_{\mathrm{edge}}\,\mathcal{L}_{\mathrm{edge}} + \lambda_{\mathrm{SSIM}}\,(1-\mathrm{SSIM}) + \lambda_{\mathrm{axial}}\,\mathcal{L}_{\mathrm{axial}}
\]

- **MAE:** voxel-wise agreement.
- **Edge:** L1 between Sobel edge magnitudes, slice by slice, for fine lateral structure.
- **SSIM:** mean 2D SSIM over the axial slices.
- **Axial:** matches differences between adjacent slices after removing each slice's mean, with anchor terms on the first, central and last slices, for consistency in depth.

The weights are the same for all three arms, so the **encoder is the only difference** between them. There is no PSF-shaping term: the encoder is optimised only through the quality of the final volume.

## Sensor noise

Mixed Poisson–Gaussian sCMOS noise, with \(x\) the noise-free image normalised to [0, 1] and \(N\) the expected photon count in its brightest pixel:

\[
y = \frac{1}{N}\Big[\mathrm{Pois}(N x) + \sigma_{\mathrm{read}}\,\varepsilon\Big],
\qquad \varepsilon \sim \mathcal{N}(0,1).
\]

- Read noise is specified **in electrons** (\(\sigma_{\mathrm{read}} = 1.0\,e^-\)), so it stays a property of the camera when \(N\) is swept. In normalised units the detector would become artificially quiet at low \(N\) and noisy at high \(N\).
- **Training:** the Poisson term uses its Gaussian limit \(\mu + \sqrt{\mu}\,\varepsilon\) with the amplitude \(\sqrt{\mu}\) **detached** from the graph. Otherwise \(\partial y/\partial x\) contains \(\varepsilon/(2\sqrt{Nx})\), which diverges on the dark pixels that fill most of an FLFM frame.
- **Test:** exact Poisson sampling.
- Default budget \(N = 100\) photons; test sweep \(N = 10\) to \(1000\).

## Data

Ground-truth volumes come from confocal stacks of fixed mouse tissue sections (FluoTissue slides: kidney, intestine, brain, liver), recorded with a 30×/1.05 NA oil objective in two emission channels (≈ 525 and 570 nm). Each channel is used as an independent grayscale volume.

| | |
|---|---|
| Volume | 512 × 512 × 43 voxels, 0.377 µm × 0.377 µm × 1.43 µm |
| Split | 91 train / 10 validation / 12 test volumes |
| PSFs | computed at 633 nm, independent of the confocal channel |

## Baseline without learning: RL-TV

Richardson–Lucy deconvolution with total-variation regularisation on the same forward operator, 100 iterations, \(\lambda_{\mathrm{TV}} = 10^{-5}\):

\[
\hat o = \arg\min_{o \ge 0}\; \tfrac12\Big\lVert y - \sum_k o_k * h_k\Big\rVert_2^2 + \lambda_{\mathrm{TV}}\lVert\nabla o\rVert_1 .
\]

It needs no training and solves a new optimisation for every volume.

!!! success "Next"
    [Results](results.md)
