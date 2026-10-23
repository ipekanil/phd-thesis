# 1. Forward model

The forward model maps a fluorescence volume to the FLFM sensor image. It is fully differentiable with respect to the encoder parameters \(\Theta\), so the reconstruction loss can update the lenslets.

The optical path is objective → tube lens (180 mm) → Fourier lens (300 mm) → encoder at the relayed Fourier plane → sensor at \(f_{\mathrm{MLA}} = 43.1\) mm behind it (schematic on the [overview page](index.md)).

## From a volume to one sensor image

The volume \(o(x,y,z)\) is treated as \(N_z\) planes. Under the laterally shift-invariant FLFM model each plane is blurred by its own depth-dependent PSF and the intensities add, because fluorescence is incoherent:

\[
y_\Theta(x,y) = \sum_{k=1}^{N_z} o(x,y,z_k) * h_{\Theta,k}(x,y) + \eta(x,y).
\]

Stacking the slices into a vector, this is one **linear operator**:

\[
y_\Theta = A_\Theta\, o + \eta,
\qquad
A_\Theta = \big[\,C_{\Theta,1}\;\cdots\;C_{\Theta,N_z}\,\big],
\]

where \(C_{\Theta,k}\) is convolution with \(h_{\Theta,k}\). \(A_\Theta\) maps \(N_z = 43\) planes onto one image, so it has 43 times more columns than rows. The inverse problem is **underdetermined**. Its transpose acts plane by plane as a **correlation**, \((A_\Theta^\top y)_k = y \star h_{\Theta,k}\). Both operators are evaluated with FFTs and never formed as matrices.

## Computing one PSF \(h_{\Theta,k}\)

**(1) Pupil field of a point emitter, without a paraxial approximation.** At NA 1.1 in water, the marginal ray leaves the sample at \(\theta_{\max} = \arcsin(\mathrm{NA}/n) \approx 56^\circ\), where the quadratic defocus model is no longer accurate. The model uses the **scalar Debye** form. In pupil spatial-frequency coordinates \(R_f\):

\[
\cos\theta = \sqrt{1 - \big(\lambda R_f / n\big)^2},
\qquad
\Phi(R_f; z) = k_0\, n\, z \cos\theta,
\qquad
u_1 \propto e^{\,i\Phi},
\qquad R_f \le \mathrm{NA}/\lambda .
\]

**(2) Aplanatic apodization.** For an integral over the flat pupil, the amplitude weight is \(A(\theta) = 1/\sqrt{\cos\theta}\). One factor of \(\cos\theta\) is already supplied by the change of variables from angle to flat-pupil coordinates, so using \(\sqrt{\cos\theta}\) here would count it twice. The pupil field is then normalised to the same collected power at every depth.

**(3) Relay and encoder.** The pupil is relayed to the Fourier plane with magnification \(M_R = 1.67\) (6.6 mm → 11.0 mm). The encoder multiplies the field by its complex transmittance:

\[
u_3(x_3,y_3) = S_\Theta(x_3,y_3)\;\tilde u_2\!\Big(\tfrac{x_3}{M_R}, \tfrac{y_3}{M_R}\Big),
\qquad
S_\Theta = A_S\, e^{\,i\varphi_\Theta}.
\]

\(S_\Theta\) is the lenslet array described on the [next page](encoder.md).

**(4) Propagation to the sensor** over \(z_{\mathrm{cam}} = 43.1\) mm by the **angular spectrum** method, discarding evanescent components:

\[
U_4(f_x,f_y) = \mathcal{F}\{u_3\}\; e^{\,i k_z z_{\mathrm{cam}}},
\qquad
k_z = k\sqrt{1-\lambda^2 f_x^2 - \lambda^2 f_y^2},
\qquad
h_{\Theta,k} = \big\lvert \mathcal{F}^{-1}\{U_4\}\big\rvert^2 .
\]

Lateral emitter positions add a linear phase tilt in the pupil. Under shift invariance, one on-axis PSF per depth is enough.

!!! note "Implementation details that matter"
    - \(\cos\theta\) does not depend on \(z\), so it is computed once for all 43 depths. Each depth costs one scalar multiplication in the pupil.
    - A constant piston \(k_0 n z\) is removed before casting to single precision. It cannot move the PSF, and removing it keeps the residual phase small enough to represent accurately.
    - The model is scalar, so polarisation is not represented.

!!! success "Next"
    The encoder \(S_\Theta\) is where the learning happens. See [Learned encoder](encoder.md).
