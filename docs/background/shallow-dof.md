# Why high-NA microscopes have a shallow depth of field

## Resolution and depth of field are linked

The **numerical aperture** \( \mathrm{NA} = n \sin\theta \) measures the cone of light an objective collects. A larger cone gives finer lateral resolution, following the Rayleigh criterion:

\[
\delta_{xy} \approx \frac{0.61\,\lambda}{\mathrm{NA}}.
\]

The same wide cone makes the image change quickly when the object moves along the optical axis. The wave-optical **depth of field** (DoF) scales as

\[
\mathrm{DoF} \;\propto\; \frac{\lambda}{\mathrm{NA}^2} \;\propto\; \frac{\text{resolution}^2}{\lambda}.
\]

Lateral resolution improves as \(1/\mathrm{NA}\), but the DoF shrinks as \(1/\mathrm{NA}^2\). Doubling the NA halves the resolvable distance and cuts the DoF to a quarter.

| Objective (example) | NA | Approx. DoF, \( n\lambda/\mathrm{NA}^2 \) at 520 nm |
|---|---|---|
| 4× air | 0.13 | ≈ 31 µm |
| 50× air | 0.42 | ≈ 3 µm |
| 60× water immersion (used for MANTIS) | 1.1 | ≈ 0.6 µm |

*Only the wave-optical term is shown. A detector-sampling term adds a little on top.*

A cell monolayer is a few µm thick. A spheroid or tissue section is tens of µm thick. At NA 1.1, most of such a sample is out of focus in any single image.

## Why not just scan?

The standard answer is **axial scanning**: record a z-stack and fuse it. This works, but:

- acquisition is **slower** by the number of planes, so fast dynamics are missed;
- every plane **re-illuminates the whole sample**, which increases photobleaching and phototoxicity;
- planes are recorded at **different times**, so moving samples give motion artefacts.

Light-sheet microscopy, remote focusing and multi-plane detection reduce some of these costs. They usually need extra, more complex hardware.

## The multispectral complication

Fluorescence studies often label several structures with different dyes (nuclei, actin, and so on) and image several **emission wavelengths**. The PSF depends on wavelength, so residual chromatic aberration moves the best-focus plane from channel to channel. In thick samples, one channel can be sharp while another is blurred at the same stage position. This breaks co-localisation and quantitative analysis.

!!! success "What this means for the method"
    We want a system that stays useful over a depth range many times its native DoF, **in a single exposure**, and **equally well in every colour channel**. The next page shows how defocus appears in the optics, which is the first step to undoing it.
