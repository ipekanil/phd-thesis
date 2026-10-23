# 2. Sampling criteria

The Fourier-plane grid pitch \(\Delta s\) is the single most important numerical design choice. It decides whether the simulated PSF is **correct** (no aliasing), how many **degrees of freedom** the optical element has, and therefore whether the element can be a **DOE** or must be a **metasurface**.

Two conditions set \(\Delta s\).

## Condition 1: phase-sampling bound (no aliasing)

Write the defocus phase in the radial pupil coordinate \(\rho\): \(\phi_{\mathrm{defocus}} = \psi_z\,\rho^2/r^2\). Its local spatial angular frequency is the gradient, largest at the pupil edge \(\rho = r\):

\[
\omega_{\lambda,z} = \frac{\partial\phi}{\partial\rho}\Big|_{\rho=r} = \frac{2\lvert\psi_z(\lambda)\rvert}{r}.
\]

Two effects widen the spectrum further: the sharp aperture edge, and multiplication by the coding phase (a convolution of spectra). Following Akpinar et al. (IEEE TIP 2021), the coded pupil must be represented up to \(\omega_{\mathrm{coded,max}} = 4\,\omega_{\lambda,z}\). Imposing Nyquist, \(\omega_{\mathrm{coded,max}} \le \pi/\Delta s\), at the worst-case depth gives

\[
\boxed{\;\Delta s \;\le\; \Delta s_{\mathrm{crit}}(\lambda) = \frac{\pi\, r}{8\,\psi_{\max}(\lambda)}\;},
\qquad
\psi_{\max}(\lambda) = \max\big(\lvert\psi_{z^-}(\lambda)\rvert,\,\lvert\psi_{z^+}(\lambda)\rvert\big).
\]

\(\psi_{\max}\) grows with NA, with the depth range, and as the wavelength gets shorter. So **the shortest wavelength and the largest DoF set the bound**.

!!! tip "Rule of thumb"
    In samples across the pupil radius: \( r/\Delta s \ge 8\psi_{\max}/\pi \approx 2.5\,\psi_{\max} \).
    For \(\psi_{\max} = 221\) (MANTIS, 50 µm), that is at least about 560 samples across the radius, or about 1,100 across the pupil. See the [toy example](../background/toy-simulation.md#3-why-sampling-matters) of what goes wrong otherwise.

## Condition 2: space-bandwidth product (enough degrees of freedom)

Avoiding aliasing does not guarantee that the grid carries enough degrees of freedom for the object information the system passes. At each wavelength, the passed bandwidth is the smaller of the objective cutoff and the camera Nyquist frequency mapped to the object:

\[
f_{\mathrm{pass}}(\lambda) = \min\Big\{\frac{2\,\mathrm{NA}}{\lambda},\; \frac{M}{2p}\Big\},
\qquad
\mathrm{SBP}(\lambda) \approx \big(2L_x f_{\mathrm{pass}}\big)\big(2L_y f_{\mathrm{pass}}\big),
\qquad L_{x,y} = \frac{N_{x,y}\,p}{M},
\]

where \(p\) is the pixel pitch, \(M\) the magnification and \(N_{x,y}\) the pixel counts. For the MANTIS system:

| λ (nm) | 433 | 521 | 601 | 681 | **Total** |
|---|---|---|---|---|---|
| SBP | 4.2 M | 3.3 M | 2.6 M | 2.05 M | **12.3 M** |

The critical grid is then **oversampled** by a factor \(\alpha\):

\[
\Delta s = \frac{\Delta s_{\mathrm{crit}}}{\alpha},
\qquad \alpha \;\gtrsim\; \frac{\mathrm{SBP}_{\mathrm{total}}}{\mathrm{SBP}(\lambda_{\min})} \approx 2.9 .
\]

The effect of \(\alpha\) was tested directly for the 50 µm design (Publication II, Supplementary Table 1; thesis Table 3.3):

| \(\alpha\) | 1 | 2 | **4** | 6 |
|---|---|---|---|---|
| PSNR (dB) | 22.19 | 23.41 | **23.53** | 23.17 |
| SSIM | 0.67 | 0.73 | **0.74** | 0.72 |

\(\alpha = 4\) was adopted. It clears the SBP ratio with a margin of more than 3×, and the gain saturates beyond it.

!!! note "Physical pitch"
    The element sits at the *relayed* Fourier plane, so a grid pitch \(\Delta s\) at the objective pupil corresponds to a physical pitch \(\Delta s_{\mathrm{phys}} = \lvert M_p\rvert\,\Delta s\) on the device, where \(M_p\) is the pupil-to-Fourier-plane magnification. The relayed pupil radius is \(r_{\mathrm{OCE}} = \lvert M_p\rvert f_1\,\mathrm{NA}\).

## Resulting design parameters

### Publication I: single wavelength, λ = 550 nm

| Objective | NA<sub>eff</sub> | Target DoF | \(\psi_{\max}\) | \(\Delta s\) (µm) | Realised as |
|---|---|---|---|---|---|
| RMS4X-PF | 0.041 | ±100 µm | 0.97 | 740 | DOE ✱ |
| RMS4X-PF | 0.041 | ±1 mm | 9.88 | 73 | DOE |
| RMS20X-PF | 0.159 | ±100 µm | 14.69 | 38 | DOE |
| Mitutoyo 50X | 0.134 | ±100 µm | 10.48 | 20 | DOE |
| Mitutoyo 50X | 0.134 | ±1 mm | 136.24 | 1.54 | GaN metalens ✱ |
| Mitutoyo 50X | 0.134 | ±1.5 mm | 245.54 | 0.858 | GaN metalens ✱ |

✱ = design retained for the 200 µm, 2 mm and 3 mm DoF targets. NA<sub>eff</sub> is explained on the [Limitations](limitations.md#effective-na-in-publication-i) page. Run `python examples/design_calculator.py` to reproduce the \(\psi_{\max}\) and \(\Delta s\) columns.

### Publication II: MANTIS, NA 1.1, four wavelengths

| Target DoF | \(\psi_{\max}\) | \(\Delta s\) (µm) | Training depths K |
|---|---|---|---|
| 25 µm | 110.20 | 2.25 | 5 |
| 50 µm | 221.32 | 1.125 | 5 |
| 75 µm | 333.38 | 0.675 | 7 |

These \(\Delta s\) values are pitches **on the meta-optic itself**, rounded down to a whole number of 225 nm TiO₂ unit cells (10, 5 and 3 cells). Each grid sample is a small "supercell" of identical pillars. The \(\psi_{\max}\) column is reproduced exactly by `python examples/design_calculator.py --f1 3 --na 1.1 --dz 0.025 --wl 433` (and `--dz 0.0125`, `--dz 0.0375`).

!!! success "Why this matters beyond numerics"
    \(\Delta s\) also sets the **search space** of the optimiser: the number of free parameters in the optical element. Matching it to the difficulty of the problem helped convergence in Publication I. Too few parameters (for example, a small Zernike basis) under-represent the needed phase. Too many only slow training down. See the DeepDOF comparison in [Simulation results](simulation-results.md).
