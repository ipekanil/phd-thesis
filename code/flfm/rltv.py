"""
rltv.py -- Richardson-Lucy with total-variation regularisation on the FLFM
forward operator. The model-based baseline the submitted paper reports as
"Fixed optics + RL-TV".

This is a port of the submitted solver.py, kept deliberately faithful so the
row stays continuous with the numbers already in the manuscript. Everything
that affects the fixed point is copied: the per-depth PSF sum normalisation,
the clamp at zero inside BOTH the forward and the adjoint, the adjoint-of-the-
measurement initialisation scaled to max 1, the A^T(1) sensitivity normaliser,
the TV sign convention and the `1 - lam*div + EPS` denominator, EPS = 1e-8, and
the min-max normalisation of the returned volume.

ONE DEVIATION, and it is deliberate. solver.py reconstructs on the SENSOR grid
(PSF size == measurement size, 3385^2) and centre-crops to 512^2 at the end.
Here the unknown lives on the 512^2 volume grid and A centre-pads it to the
sensor before multiplying by H_z. Two reasons:

  1. It is the same operator the CG lift uses for the learned arms, so the
     comparison is of SOLVERS on one forward model rather than of two
     different forward models.
  2. The sensor-grid version needs ~2 GB for x, ~2 GB for the cached OTF and
     eight more volumes inside the TV term -- about 16 GB of transients, which
     OOMs. On the volume grid a volume is 45 MB and the whole solve fits in
     well under a gigabyte.

The fixed point differs only in what happens to signal outside the 512^2
field, which the other arms cannot represent either.

A SIGN FINDING IN THE SUBMITTED TV TERM, which I kept rather than silently
corrected. solver.py differences as `dz[:-1] = x[:-1] - x[1:]`, i.e. the
NEGATIVE of a forward difference, and then takes the divergence of that. The
net effect flips the sign of the regulariser: on a smooth bump the divergence
comes out POSITIVE at the peak, so `x / (1 - lam*div)` multiplies the peak by
1 + lam*div > 1 and the term SHARPENS instead of smoothing. Dey et al. intend
the opposite. The saving grace is the magnitude: at lam = 1e-5 the per-
iteration factor is 1.00003, so over 100 iterations the TV term moves the
solution by a fraction of a percent and the submitted "RL-TV" is numerically
indistinguishable from plain RL. The default here reproduces that, because
reproducing the submitted baseline is the point; pass tv_sign=-1.0 for the
intended sign if you want to see whether a correctly signed TV term at a
usable lam actually buys anything.

A NOTE ON SCALE, which matters when reading the table. RL-TV converges to its
own intensity scale; the network's output is trained to match the target's.
Both are min-max / percentile normalised before scoring, so all arms reach the
metrics on one scale -- but that is a CHOICE, not a measurement. Lateral and
axial resolution, RSP and RSE are affine-invariant and so are untouched by it.
PSNR and SSIM are not. make_paper_table.py therefore also reports
psnr_db_affine, which fits the one global (alpha, beta) minimising MSE against
the ground truth before computing PSNR; that column is the scale-free one and
is the fair cross-arm comparison. Say which you quote in the caption.
"""
import torch

EPS = 1e-8


@torch.no_grad()
def rl_tv_solve(y, psf_stack, out_hw, n_iter=100, tv_lambda=1e-5,
                tv_sign=1.0, chunk=4, margin=0, sens_floor=0.0,
                normalize="percentile", p_low=10.0, p_high=99.7,
                verbose=False):
    """Richardson-Lucy + TV on y = sum_z h_z * x_z.

    Parameters
    ----------
    y          : sensor image, [Hs, Ws] (or [1,1,Hs,Ws] / [1,Hs,Ws]).
    psf_stack  : [D, Hs, Ws], the depth-dependent PSFs, un-normalised.
    out_hw     : (h, w) of the reconstructed volume.
    n_iter     : RL iterations. solver.py uses 100.
    tv_lambda  : TV weight. solver.py uses 1e-5; 0 gives plain RL.
    tv_sign    : +1.0 reproduces solver.py (see the sign finding above);
                 -1.0 gives the intended smoothing sign.
    chunk      : depths per chunk inside A / A^T, to bound peak memory.
    margin     : extra pixels solved for on every side, discarded at the end.
                 The faithful analogue of solver.py's solve-big-then-crop.
    sens_floor : floor on A^T(1), RELATIVE to its max. See the spike note.
    normalize  : "percentile" (default, matches the CG lift and so the learned
                 arms) or "minmax" (reproduces solver.py).

    Returns
    -------
    x : [D, out_h, out_w] float32 in [0, 1].
    """
    y = y.squeeze()
    assert y.dim() == 2, f"expected a 2-D sensor image, got {tuple(y.shape)}"

    h = psf_stack.to(y.device).float()
    D, Hs, Ws = h.shape
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    m = int(margin)
    sol_h, sol_w = min(out_h + 2 * m, Hs), min(out_w + 2 * m, Ws)
    r0, c0 = (Hs - sol_h) // 2, (Ws - sol_w) // 2

    # per-depth sum normalisation, as in solver.py. RL's multiplicative update
    # assumes each PSF conserves energy; without this the depths with the most
    # total PSF energy are systematically over-weighted.
    h = h / (h.sum(dim=(-2, -1), keepdim=True) + EPS)

    y = y.float().clamp_min(0.0)        # RL needs non-negative Poisson data;
                                        # read noise pushes dark pixels below 0
    H = torch.fft.rfft2(h, dim=(-2, -1))
    del h

    def A(x):
        """volume [D,sh,sw] -> sensor [Hs,Ws], clamped at 0 as in solver.py"""
        Yf = None
        for a in range(0, D, chunk):
            b = min(a + chunk, D)
            xp = x.new_zeros((b - a, Hs, Ws))
            xp[:, r0:r0 + sol_h, c0:c0 + sol_w] = x[a:b]
            part = (torch.fft.rfft2(xp, dim=(-2, -1)) * H[a:b]).sum(0)
            Yf = part if Yf is None else Yf + part
            del xp, part
        out = torch.fft.irfft2(Yf, s=(Hs, Ws), dim=(-2, -1))
        return out.clamp_min(0.0)

    def AT(r):
        """sensor [Hs,Ws] -> volume [D,sh,sw], clamped at 0 as in solver.py"""
        Rf = torch.fft.rfft2(r, dim=(-2, -1))
        outs = []
        for a in range(0, D, chunk):
            b = min(a + chunk, D)
            xf = torch.fft.irfft2(Rf[None] * torch.conj(H[a:b]),
                                  s=(Hs, Ws), dim=(-2, -1))
            outs.append(xf[:, r0:r0 + sol_h, c0:c0 + sol_w].clamp_min(0.0))
            del xf
        return torch.cat(outs, dim=0)

    def tv_div(x):
        """div(grad x / |grad x|), solver.py's sign and differencing exactly."""
        dz = torch.zeros_like(x)
        dy = torch.zeros_like(x)
        dx = torch.zeros_like(x)
        dz[:-1] = x[:-1] - x[1:]
        dy[:, :-1] = x[:, :-1] - x[:, 1:]
        dx[:, :, :-1] = x[:, :, :-1] - x[:, :, 1:]
        mag = torch.sqrt(dx * dx + dy * dy + dz * dz + EPS)
        dx /= mag; dy /= mag; dz /= mag
        del mag
        div = torch.zeros_like(x)
        div[:, :, :-1] += dx[:, :, :-1]
        div[:, :, 1:] -= dx[:, :, :-1]
        div[:, :-1, :] += dy[:, :-1, :]
        div[:, 1:, :] -= dy[:, :-1, :]
        div[:-1, :, :] += dz[:-1, :, :]
        div[1:, :, :] -= dz[:-1, :, :]
        return div

    # Sensitivity A^T(1), floored as in solver.py.
    #
    # sens_floor applies a floor RELATIVE to the max instead, and defaults to
    # off. It is here because the first real run produced four voxels at the
    # corners of the outermost depth slice (z = 42) sitting 60x above the 99.9th
    # percentile, and a near-zero sensitivity was my first guess at why. It is
    # not: A^T(1) for sum-normalised PSFs under circular convolution is
    # essentially constant, and a toy reproduction showed the floor changes
    # max/p99.9 by 1%. A second guess -- ratio = y/A(x) blowing up where the
    # model cannot explain out-of-field emitters -- also failed to reproduce.
    # So the mechanism of those four voxels is NOT established, and this knob is
    # left off rather than shipped as a fix for something it may not fix.
    #
    # What does fix the symptom is the normalisation at the bottom of this
    # function: four voxels out of 11.3M cannot move a percentile, where they
    # move min-max by 60x. That was the whole of the black-figure problem.
    sens = AT(torch.ones_like(y)).clamp_min(EPS)
    if sens_floor > 0:
        sens = sens.clamp_min(sens_floor * float(sens.max()))

    # initialise from the adjoint of the measurement, scaled to max 1
    x = AT(y)
    x = x / (x.max() + EPS)
    x = x.clamp_min(EPS)

    for it in range(n_iter):
        pred = A(x)
        ratio = y / (pred + EPS)
        x = x * (AT(ratio) / sens)
        if tv_lambda:
            x = x / (1.0 - tv_sign * tv_lambda * tv_div(x) + EPS)
        x = x.clamp_min(EPS)
        if verbose and it % 10 == 0:
            err = torch.mean(torch.abs(pred - y)).item()
            print(f"    RL-TV iter {it:04d}  mean abs error {err:.6e}")
        del pred, ratio

    # discard the margin, then normalise
    if (sol_h, sol_w) != (out_h, out_w):
        a0, b0 = (sol_h - out_h) // 2, (sol_w - out_w) // 2
        x = x[:, a0:a0 + out_h, b0:b0 + out_w]

    if normalize == "minmax":
        x = x - x.min()
        return x / (x.max() + EPS)

    # Percentile, via the SAME function the CG lift uses -- imported rather
    # than reimplemented so the two cannot drift apart. This is the scale the
    # learned arms' reconstructions live on, so using it here is what makes
    # PSNR and SSIM comparable across arms at all. A single outlier voxel moves
    # min-max normalisation by 60x and the percentile version not at all.
    # (It subsamples for the quantile because torch.quantile caps its input
    # size and a volume is 11M+ voxels.)
    from adjoint import normalize_volume_percentile
    return normalize_volume_percentile(
        x, p_low=p_low, p_high=p_high, eps=EPS).clamp(0.0, 1.0)
