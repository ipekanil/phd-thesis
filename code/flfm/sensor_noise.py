"""
sensor_noise.py -- mixed Poisson-Gaussian sensor noise.

This is the old additive-Gaussian model with shot noise added on top:

    old:  y = x + sigma * N(0,1)
    new:  y = [ Pois(x * photons) + read_noise_e * N(0,1) ] / photons
                ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
                shot (signal-dep.)  read noise, in ELECTRONS

Both terms live in electrons and the whole thing is divided by `photons` at the
end. That matters for the test sweep: read noise is a property of the camera
(~1.6 e-) and must NOT change when you vary the photon budget. Specifying it in
normalised units instead would make the camera silently quiet at 10 photons and
absurdly noisy at 1000.

`photons` is the expected photon count in the brightest pixel. Lower photons =
more shot noise. At photons -> infinity the first term becomes x again and you
recover the old model exactly.

Reference: Mandracchia et al., Nat. Commun. 10, 5222 (2019), Eq. 1
(gain and offset are dropped here -- they cancel once the image is normalised,
and only affect quantisation).
"""

import torch


def add_sensor_noise(x, photons, read_noise_e=1.6, exact=False, eps=1e-8):
    """Mixed Poisson-Gaussian noise on an image scaled to roughly [0, 1].

    Parameters
    ----------
    x          : sensor image, [0,1]-ish, non-negative.
    photons    : expected photons in the brightest pixel (e.g. 10 ... 1000).
    read_noise_e : Gaussian read noise in ELECTRONS (sCMOS is typically
                 1-2 e-). Constant across the photon sweep, as a real camera is.
    exact      : False (default) samples the Poisson via its Gaussian limit,
                 which is DIFFERENTIABLE -- needed while f_rand is being
                 learned. True uses torch.poisson for exact integer counts,
                 which has NO gradient: use it at test time only.
    """
    lam = x.clamp_min(0.0) * photons                  # expected photons/pixel

    if exact:
        shot = torch.poisson(lam)                     # exact, not differentiable
    else:
        # Pois(lam) ~ N(lam, lam). Error <1% above ~5 photons.
        #
        # .detach() on the noise AMPLITUDE is deliberate. Without it,
        #     dy/dx = 1 + e1 / (2*sqrt(x*photons))
        # and that second term has standard deviation 5000 at x = 0. Roughly
        # two thirds of an FLFM sensor frame is dark, so most of the image
        # would contribute enormous random gradients. Its expected value is
        # zero (e1 is zero-mean), so it adds variance to the gradient estimate
        # without adding signal. Detaching gives dy/dx = 1 exactly, while the
        # SAMPLED VALUES are completely unchanged -- same noise, clean gradient.
        shot = lam + torch.sqrt(lam.detach() + eps) * torch.randn_like(lam)

    read = read_noise_e * torch.randn_like(x)      # electrons, photon-independent
    return (shot + read) / photons


if __name__ == "__main__":
    # sanity: variance should be (x*photons + read_noise_e^2) / photons^2
    torch.manual_seed(0)
    rn = 1.6
    print(f"{'photons':>9}{'x':>6}{'var measured':>15}{'var expected':>15}")
    for p in (10, 100, 1000):
        for xv in (0.1, 1.0):
            x = torch.full((500, 500), xv)
            y = add_sensor_noise(x, p, read_noise_e=rn, exact=True)
            print(f"{p:>9}{xv:>6}{y.var().item():>15.3e}"
                  f"{(xv * p + rn ** 2) / p ** 2:>15.3e}")
