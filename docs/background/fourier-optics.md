# Fourier optics of a microscope

This page links four quantities: the **pupil function**, the **point spread function (PSF)**, the **optical transfer function (OTF / MTF)** and the **defocus coefficient** \( \psi \). The whole method builds on these relations.

## The 4f system and the Fourier plane

A lens forms the Fourier transform of the field in its front focal plane at its back focal plane. Two lenses separated by the sum of their focal lengths (a **4f system**) therefore:

1. take the object field to the **Fourier plane** (spatial frequencies ↔ positions), and
2. transform it back into an image on the sensor.

In an infinity-corrected microscope, the objective and tube lens form such a system. The Fourier plane is the objective's **pupil**. An element placed there multiplies *every* spatial frequency of *every* point in the field in the same way. This is why the pupil is the natural place for an encoder.

<figure markdown="span">
  ![EDoF microscope with relay](../assets/figures/papers/th_fig3_2_relay.png){ loading=lazy }
  <figcaption>The objective (f₁) and tube lens (f₂) form an intermediate image. A second 4f relay (f₃, f₄) creates an accessible Fourier plane where the optical coding element (blue) is placed. Thesis Fig. 3.2, adapted from Publication II (© 2026 the authors, CC BY 4.0).</figcaption>
</figure>

The pupil of a real objective sits inside the objective housing. This is why the experimental system adds a **relay 4f** that re-images the pupil outside the objective (see [Forward model](../edof/forward-model.md)).

## Pupil function → PSF

The **generalised pupil function** describes how the pupil changes the amplitude and phase of the wavefront:

\[
P(x_1,y_1) = A(x_1,y_1)\, e^{\,i\phi(x_1,y_1)},
\]

where \(A\) is the aperture (1 inside a circle of radius \(r\), 0 outside) and \(\phi\) is any phase added there.

Fluorescent emitters are **mutually incoherent**, so their *intensities* add at the sensor. The intensity PSF is the squared magnitude of the Fourier transform of the pupil:

\[
h(x_2,y_2) = \big\lvert \mathcal{F}\{P(x_1,y_1)\} \big\rvert^2 .
\]

If the PSF is the same across the field of view (**shift invariance**), the camera image is a convolution of the object with the PSF, plus noise:

\[
y = x * h + n .
\]

## PSF → OTF and MTF

The **optical transfer function** is the Fourier transform of the PSF. Equivalently, it is the autocorrelation of the pupil. Its magnitude, the **modulation transfer function (MTF)**, tells how much contrast survives at each spatial frequency:

\[
\mathrm{OTF}(f) = \mathcal{F}\{h\},\qquad \mathrm{MTF}(f) = \frac{\lvert \mathrm{OTF}(f) \rvert}{\lvert \mathrm{OTF}(0) \rvert}.
\]

For incoherent imaging the MTF is zero beyond the cutoff \( f_c = 2\,\mathrm{NA}/\lambda \).

!!! note "Why the MTF matters for deblurring"
    Where the MTF is **zero**, that frequency is gone from the measurement and no algorithm can bring it back. Where it is **small but non-zero**, a good decoder can amplify it again, at the cost of amplifying noise. A good encoder therefore keeps the MTF *away from zero* over the band of interest.

## Defocus is a quadratic pupil phase

A point at axial position \(z\), away from the nominal focus \(z_0\), produces a spherical wavefront that no longer matches the lens. In the pupil this appears as a **quadratic phase**:

\[
\phi^{\mathrm{DF}}_z(x_1,y_1) = \psi_z \,\frac{x_1^2+y_1^2}{r^2},
\qquad
\psi_z = \frac{\pi}{\lambda}\left(\frac{1}{z}-\frac{1}{z_0}\right) r^2 .
\]

\( \psi_z \) is the **defocus coefficient**: the phase error, in radians, at the pupil edge. \( \psi/2\pi \) is the number of waves of defocus. For a microscope with the object near the front focal plane (\(z_0 = f_1\), \(r = \mathrm{NA}\,f_1\)) and a small axial offset \(\Delta z \ll f_1\):

\[
\lvert\psi\rvert \;\approx\; \frac{\pi\,\mathrm{NA}^2\,\Delta z}{\lambda}.
\]

!!! example "Worked example (Publication II setting)"
    NA = 1.1, shortest emission λ = 433 nm, target DoF 50 µm (so Δz = ±25 µm):

    \( \psi_{\max} \approx \pi \cdot 1.1^2 \cdot 25\,\mu\text{m} / 0.433\,\mu\text{m} \approx 220 \) rad, which is about **35 waves** of defocus at the pupil edge. The paper reports \( \psi_{\max} = 221.32 \) using the exact expression.

    For comparison, an ordinary clear-pupil microscope already shows MTF zeros at a few radians of defocus. Earlier EDoF studies worked at \( \psi_{\max} \lesssim 82 \).

The same expression explains the \( \lambda/\mathrm{NA}^2 \) law of the [previous page](shallow-dof.md). Keeping \(\psi\) below a fixed value allows a \(\Delta z\) proportional to \( \lambda/\mathrm{NA}^2 \).

!!! success "What this means for the method"
    Defocus enters the physics only as an **additive phase in the pupil**. We can add another phase \( \phi^{M} \) in the same plane, so the total pupil phase is \( \psi_z\rho^2 + \phi^{M} \). The design question is which \( \phi^{M} \) makes \(h_z\) informative for *every* \(z\) in the target range. This is **wavefront coding**.
