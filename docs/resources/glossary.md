# Symbols and glossary

## Symbols

| Symbol | Meaning |
|---|---|
| \(\lambda\), \(\lambda_j\) | Wavelength; emission channel \(j\) (433, 521, 601, 681 nm in MANTIS) |
| \(\mathrm{NA}\), \(\mathrm{NA}_{\mathrm{eff}}\) | Numerical aperture; effective NA of the modelled pupil (Publication I) |
| \(f_1, f_2\) | Objective and tube-lens focal lengths |
| \(f_3, f_4\) | Relay (Fourier) lens focal lengths |
| \(M, M_p\) | Total magnification; pupil-to-relayed-Fourier-plane magnification |
| \(r\), \(D\) | Pupil radius, \(r = \mathrm{NA}\,f_1\); pupil diameter \(D = 2r\) |
| \(r_{\mathrm{OCE}}\) | Radius of the coding element at the relayed Fourier plane, \(\lvert M_p\rvert f_1 \mathrm{NA}\) |
| \((x_0,y_0)\), \((x_1,y_1)\), \((x_2,y_2)\), \((x_3,y_3)\), \((x_4,y_4)\) | Object, objective pupil, intermediate image, relayed Fourier plane, sensor |
| \(z\), \(z_0\), \(\Delta z\) | Object depth, nominal focus, axial offset |
| \(P = A e^{i\phi}\) | Generalised pupil function (aperture \(A\), phase \(\phi\)) |
| \(\phi^{\mathrm{DF}}_z\), \(\phi_{\mathrm{defocus}}\) | Defocus phase \(\psi_z \rho^2/r^2\) |
| \(\psi_z\), \(\psi_{\max}\) | Defocus coefficient \((\pi/\lambda)(1/z - 1/z_0)\,r^2\) [rad]; its maximum over the target range |
| \(\phi^M\), \(\phi_{\mathrm{OCE}}\) | Phase added by the coding element (mask, DOE or meta-optic) |
| \(h\), PSF | Intensity point spread function \(\lvert\mathcal{F}\{P\}\rvert^2\) |
| OTF, MTF | Optical transfer function \(\mathcal{F}\{h\}\); its normalised magnitude |
| \(\Delta s\), \(\Delta s_{\mathrm{crit}}\) | Fourier-plane grid pitch; phase-sampling bound \(\pi r / (8\psi_{\max})\) |
| SBP, \(\alpha\) | Space-bandwidth product; oversampling factor |
| \(h\) (DOE), \(n\) | DOE height map; refractive index |
| \(w\), \(\gamma\), \(\Gamma_{\mathrm{full}}\) | Nanopillar width map; radial width vector; its rotational replication |
| \(U_w\), \(U_{\phi,\lambda}\) | Unit-cell library: simulated widths and phases |
| \(x\), \(GT\) | Ground-truth (sharp) image |
| \(y\), \(SI\) | Sensor image |
| \(\hat{x}\), \(OI\) | Reconstructed (output) image |
| \(f_\theta\), \(\theta\) | Reconstruction network and its weights |
| \(\mathcal{H}_\phi\) | Forward (image-formation) operator for optical parameters \(\phi\) |
| \(K\) (or \(M\)), \(C\) | Number of training depths; number of spectral channels |
| \(\eta\), \(n\), \(\sigma\) | Sensor noise and its standard deviation |
| \(o\), \(\hat o\), \(o_{\mathrm{cg}}\) | FLFM: ground-truth volume, reconstruction, conjugate-gradient estimate |
| \(\Theta\), \(f_{m,n}\), \(D_{m,n}\) | FLFM encoder parameters: focal length and aperture diameter of lenslet \([m,n]\) |
| \(f_{\mathrm{MLA}}\), \(D_{\mathrm{pitch}}\) | Baseline lenslet focal length (43.1 mm) and pitch (2.12 mm) |
| \(A_\Theta\), \(h_{\Theta,k}\) | FLFM forward operator and the PSF of depth plane \(k\) |
| \(\tau\), \(J\) | Tikhonov weight and number of conjugate-gradient iterations |
| \(R_\psi\), \(\psi\) | FLFM reconstruction network and its weights |
| \(N\), \(\sigma_{\mathrm{read}}\) | Photon budget (expected photons in the brightest pixel); read noise in electrons |

## Terms

4f system
:   Two lenses separated by \(f_a + f_b\). The first produces the Fourier transform of the input at the middle plane, and the second transforms back to an image.

Fourier plane / pupil plane
:   The plane where each position corresponds to a spatial frequency (a propagation angle) of the image. A mask here acts identically on all field points.

Depth of field (DoF)
:   The axial range over which an object appears acceptably sharp. It scales as \(\lambda/\mathrm{NA}^2\).

Extended depth of field (EDoF)
:   Any technique that yields sharp images over a range larger than the native DoF.

Wavefront coding
:   Adding a pupil phase so that the PSF becomes insensitive to an aberration (here, defocus), then recovering the image computationally.

End-to-end optimisation / co-design
:   Optimising optical and computational parameters together, by back-propagating an image loss through a differentiable model of the optics.

DOE
:   Diffractive optical element. A surface-relief phase plate whose local height sets the phase delay.

Metasurface / meta-optic / metalens
:   A flat array of sub-wavelength nanostructures (here GaN or TiO₂ pillars) whose geometry sets the local phase (and possibly amplitude and polarisation). A metalens is a metasurface used for focusing or imaging.

Unit-cell library
:   Simulated phase and transmission of a periodic array of identical pillars for each geometry and wavelength, used to map phase to geometry and back.

D-CNN
:   Deblurring convolutional neural network, the reconstruction module.

PSNR / SSIM
:   Peak signal-to-noise ratio (dB) and structural similarity index (0–1): image-quality metrics against a ground truth.

Depth spread
:   FLFM: ratio of the worst to the best per-depth mean effective lateral resolution. 1 means uniform quality through the volume.

Blur matching
:   Effective resolution estimate: the Gaussian blur of the ground truth that best matches the reconstruction, reported as a FWHM.

Conjugate gradient (CG)
:   Iterative solver for symmetric positive-definite systems. Here it solves the regularised normal equations of the FLFM forward model.

FLFM
:   Fourier light field microscopy. A lenslet array at the Fourier plane records several perspective views at once, for snapshot 3D imaging.
