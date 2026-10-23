# 9. Limitations and notes

## What MANTIS is and is not

- **It is** a single-shot, extended-depth-of-field 2D image: structures from the whole target range appear sharp together.
- **It is not** optical sectioning or 3D reconstruction. It does not tell you *where* in depth a structure is. That is the goal of the [FLFM part](../flfm/index.md).

## When to expect it to generalise

MANTIS should transfer to new specimens when the **microscope configuration is unchanged** and the **fluorophores emit in the same four bands**. The unseen mouse-kidney tissue is an example. Performance may drop for:

- strongly **scattering** or **autofluorescent** samples;
- **very different spatial statistics** from the training data;
- **emission outside the trained bands**.

**Remedies:** for a new sample type in the same bands, fine-tune the network on a small calibration set recorded through the meta-optic. For new dyes or filter sets, the PSFs themselves change, so end-to-end retraining (optics included) may be needed.

## Model assumptions

- **Shift-invariant PSF** across the field of view (no field-dependent aberrations).
- **Local periodic approximation** for the nanopillar library (no coupling between neighbouring pillars of different sizes).
- **Uniform pupil amplitude.** Transmission is used only to choose a high-transmission geometry range.
- **Planar objects at discrete depths** during training. Real thick samples superimpose many depths.
- **Gaussian noise.** Poisson noise can be substituted in the same framework.
- **Fabrication deviations** can make the real PSF differ from the model. This was not a problem here, because the measured PSFs followed the design. If needed, calibrate the forward model with measured PSFs and fine-tune the network.

## How to compare with other EDoF methods

EDoF methods sit at different points of a trade-off between **depth range**, **lateral resolution** and **spectral bandwidth**. Some aim for a near-diffraction-limited, depth-uniform PSF over a short range. Others give up resolution for a long range, and many work at one wavelength. MANTIS deliberately accepts a lower peak sharpness at any single plane in exchange for **stable reconstruction across strong defocus and four channels**. Compare methods at a matched \(\psi_{\max}\), NA and number of channels, not by DoF alone.

## Effective NA in Publication I

In Publication I, the objective pupil radius was computed from a spatial-frequency expression (Eq. 7 of the paper) that **under-fills the aperture** by a constant factor. The modelled pupil is therefore smaller than \(r = \mathrm{NA}\,f_1\). The effective numerical aperture, \(\mathrm{NA}_{\mathrm{eff}} = r_{\mathrm{model}}/f_1\), is what the simulations actually represent:

| Objective | Nominal NA | NA<sub>eff</sub> used in the simulations |
|---|---|---|
| RMS4X-PF | 0.13 | 0.041 |
| RMS20X-PF | 0.40 | 0.159 |
| Mitutoyo 50X | 0.42 | 0.134 |

**Consequences:**

- All Publication I comparisons (cubic mask, DeepDOF, learned design) used the **same** simulated aperture, so the *relative* conclusions hold, at the stated NA<sub>eff</sub>.
- The absolute numbers should **not** be read as performance at the nominal NA.
- The ±100 µm, NA<sub>eff</sub> = 0.041 case is within the native DoF of that aperture and serves as a low-difficulty reference.
- The \(\psi_{\max}\) and \(\Delta s\) values in Table 1 are consistent with NA<sub>eff</sub>. `examples/design_calculator.py` reproduces them.
- Publication II and the thesis define the pupil through \(r_{\mathrm{OCE}} = \lvert M_p\rvert f_1\,\mathrm{NA}\), so **Publication II is not affected**.

This clarification is also given in Section 3.6 of the thesis.
