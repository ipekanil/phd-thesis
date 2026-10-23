import torch
import torch.utils.checkpoint
from torch import nn
from typing import Union

def center_crop_4d(x, out_h, out_w):
    B, D, H, W = x.shape
    y0 = (H - out_h) // 2
    x0 = (W - out_w) // 2
    return x[:, :, y0:y0 + out_h, x0:x0 + out_w]

def normalize_volume_global(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.amin(dim=(-3, -2, -1), keepdim=True)
    x_max = x.amax(dim=(-3, -2, -1), keepdim=True)
    # clamp_min, NOT "+ eps". The adjoint volume carries a dz factor in metres,
    # so its range is ~7e-6 and an additive 1e-8 would eat 0.14% of it (max came
    # out 0.99863 instead of 1). Clamping only guards the degenerate all-constant
    # case, where the numerator is exactly 0 anyway, so the result is still 0.
    return (x - x_min) / (x_max - x_min).clamp_min(eps)

def normalize_volume_percentile(x: torch.Tensor, p_low: float = 10.0, p_high: float = 99.7,
                                eps: float = 1e-8, max_samples: int = 1_000_000) -> torch.Tensor:
    """Percentile normalisation, matching config.PercentileNormalize3D on the GT.

    Why not min-max: min-max sets the scale from ONE voxel (the brightest), so a
    volume with a single bright soma comes out compressed while the GT, scaled by
    its 99.7th percentile, does not. The coarse volume then lands at a median of
    ~0.31 while the target's is ~0.09, and the residual branch has to spend itself
    producing a constant offset before it reconstructs anything. Using the same
    percentiles as the target puts both on one scale.

    p_low is NOT 1.0 here even though the GT uses 1.0. The lifted volume has no
    dark background the way a sparse target does: its 1st percentile sits far
    below its own median, so scaling by it leaves the median far above the
    target's. A higher p_low puts the background floor at 0 instead, and the
    clamp_min below rectifies what falls under it, so the network is handed a
    volume on roughly the target's scale rather than having to synthesise a large
    constant offset through the res_scale*conv residual branch.

    The right value depends on the OPERATOR, because it changes the
    distribution: 25 suited deconv="adjoint"; with deconv="tikhonov" at
    tikhonov_reg=1e2 it left the coarse median at 0.067 against the target's
    0.188, hence 10. Tune it against the "median vs target median" ratio the
    training loop prints (aim for ~1.0 on the RUNNING average, not per sample --
    each target is scaled by its own percentiles, so single-sample ratios swing
    by 10x). Note p_low also fixes the fraction of voxels clamped to exactly
    zero: p_low=25 -> 25% are 0, p_low=10 -> 10%. That fraction is the quickest
    way to confirm which value a saved volume was made with.

    Differences from the GT transform, both deliberate:
      * quantiles are estimated on a random subsample (torch.quantile caps the
        input size, and a full volume is 11M+ voxels);
      * values above 1 are NOT clipped -- the network's output is what gets
        clamped, and clipping the input would flatten the brightest structures
        exactly where the old artefact lived.
    Gradients flow: torch.quantile is differentiable w.r.t. the sampled values.
    """
    flat = x.reshape(-1)
    if flat.numel() > max_samples:
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
        sample = flat[idx]
    else:
        sample = flat
    lo = torch.quantile(sample, p_low / 100.0)
    hi = torch.quantile(sample, p_high / 100.0)
    # Diagnostic. hi is a quantile OF x, so it must satisfy lo <= hi <= x.max(),
    # and the normalised volume must then have p_high -> 1.0 exactly and a max
    # above 1. A saved tikhonov coarse volume violated this (its 99.7th
    # percentile came out 0.46 and its max 0.81, i.e. hi exceeded x.max()),
    # which is also why it was too dim. If the line below ever shows hi > x.max,
    # the quantile is not coming from this tensor and that is the bug -- not
    # p_low.
    if hi > x.max():
        print(f"!!  normalize_volume_percentile: hi ({hi.item():.4e}) > x.max "
              f"({x.max().item():.4e}) -- quantile is inconsistent with the volume")
    out = (x - lo) / (hi - lo).clamp_min(eps)
    return out.clamp_min(0.0)   # negatives are below-background, as in the GT


class FLFMAdjointBackprojector(nn.Module):
    def __init__(self, out_hw=(512, 512), chunk_size: int = 1,
                 normalize: str = "global", clamp_min: bool = True, eps: float = 1e-8,
                 p_low: float = 10.0, p_high: float = 99.7,
                 deconv: str = "tikhonov", tikhonov_reg: float = 1e2,
                 cg_reg: float = 1.5, cg_iters: int = 20,
                 cg_checkpoint: bool = True, cg_chunk: int = 1,
                 cg_cache_H: bool = False, cg_detach_psf: bool = False):
        """normalize: "percentile" (recommended -- same scale as the GT),
        "global" (min-max, legacy), "per_depth", or anything else for none.
        p_high should match the PercentileNormalize3D setting used on the
        targets; p_low should NOT (see normalize_volume_percentile)."""
        super().__init__()
        self.out_hw = out_hw
        self.chunk_size = chunk_size
        self.normalize = normalize
        # In percentile mode the per-chunk rectification is skipped: p_low has to
        # see the real below-background values to locate the background, and the
        # final clamp_min(0) does the rectifying instead.
        self.clamp_min = clamp_min and normalize != "percentile"
        self.eps = eps
        self.p_low = p_low
        self.p_high = p_high
        # deconv:
        #   "adjoint"  -- x = F^-1{ Y conj(H_z) }.  The TRANSPOSE of the forward
        #                 operator, not its inverse: it convolves the measurement
        #                 with the PSF a second time, so the coarse volume carries
        #                 only a few percent of the target's high-frequency energy.
        #   "tikhonov" -- x = F^-1{ Y conj(H_z) / (sum_z' |H_z'|^2 + lam) }.
        #                 The least-squares solution of y = sum_z h_z * x_z.
        #                 The denominator sums over ALL depths, which is what makes
        #                 it stable: where depth z's own OTF is weak the other
        #                 depths hold the denominator up. (Dividing by |H_z|^2 for
        #                 that depth alone explodes there -- measured 125-209% of
        #                 the target's high-frequency energy, i.e. pure noise, and
        #                 optics gradient norms of 82-203 against the adjoint's 3.)
        # tikhonov_reg is relative to mean(sum_z |H_z|^2), so it is dimensionless.
        # Smaller = sharper and noisier. Swept against the "corr with target"
        # line the training loop prints, on two matched samples:
        #
        #   reg        corr           hf % of target   optics grad norm
        #   adjoint    0.576 / 0.647      13 /   6         3.1 / 2.8
        #   1e-2       ~0.07 (test vol)  ~970              --
        #   1e0        0.703 / 0.689     156 / 114        22.6 / 22.8
        #   1e1        0.765 / 0.735      64 /  44         7.8 / 9.5
        #   1e2        0.780 / 0.742      28 /  13         1.8 / 2.5   <- peak
        #   1e3        0.721 / 0.717       4 /   2         2.7 / 1.7
        #
        # Clean unimodal curve, so 1e2 is the default. Note the hf figures at 1e2
        # sit near the ~25% of the target's high-frequency energy that is inside
        # this system's optical passband (cutoff 0.252 cyc/px at 0.377 um pixels,
        # 1.82 um lateral resolution) -- above that there is nothing to recover,
        # which is why the smaller regs only added noise.
        self.deconv = deconv
        self.tikhonov_reg = tikhonov_reg
        # ------------------------------------------------------------------
        # deconv="cg": conjugate gradient on the normal equations
        #
        #       (A^T A + lam I) x = A^T y
        #
        # WHY, over "tikhonov". The one-step formula is the MINIMUM-NORM
        # solution of one equation in D unknowns per frequency: every depth
        # gets the same function of Y, scaled by conj(H_z). That is maximally
        # smooth in z by construction. Measured: the PSF operator has an
        # adjacent-depth correlation of 0.773, and the one-step lift turns it
        # into 0.996 -- the axial information is destroyed at the lift, before
        # the network ever sees it.
        #
        # The system is not actually underdetermined: 3385^2 = 11.46M sensor
        # pixels against 43 x 512^2 = 11.27M voxels. The one-step formula only
        # looks underdetermined because it never enforces that each X_z has
        # 512x512 support. CG applies A and A^T to a 512x512 volume, so that
        # constraint holds by construction and the extra equations get used.
        #
        # Measured across 5 test volumes at N=100, against the trained mask:
        #
        #   lift              rho mean   rho min   adj-Z mean
        #   tikhonov 1e2       0.6881    0.3708     0.9951
        #   CG 10 it @ 3.0     0.8309    0.4968     0.9831
        #   CG 20 it @ 3.0     0.8326    0.4895     0.9761
        #   CG 20 it @ 1.5     0.8354    0.4647     0.9529   <- default
        #   CG 20 it @ 1.0     0.8300    0.4382     0.9333
        #
        # +0.147 in mean rho over the one-step lift, BEFORE any network runs.
        # For reference the best fully trained system ever reached rho 0.419
        # on sample 8, where this lift alone gets 0.465.
        #
        # cg_reg is relative to mean(sum_z |H_z|^2), like tikhonov_reg, but is
        # NOT comparable in magnitude: tikhonov_reg=1e2 had to suppress a
        # min-norm solution, cg_reg only stabilises a well-posed one. Above
        # cg_reg ~3 the result is identical at 20/30/50 iterations, i.e. lam
        # regularises and the iteration count stops mattering; below ~1 CG
        # semiconverges and the iteration count becomes load-bearing.
        #
        # MEMORY. Unlike the tikhonov branch this caches all D of H_z
        # (~2 GB at 43 x 3385 x 1693 complex64) because CG needs them every
        # iteration; recomputing would cost 2*D FFTs per iteration. If this
        # OOMs, drop to cg_iters=10, cg_reg=3.0 first (nearly the same rho,
        # half the retained iterates), and leave cg_checkpoint=True.
        # ------------------------------------------------------------------
        self.cg_reg = cg_reg
        self.cg_iters = cg_iters
        self.cg_checkpoint = cg_checkpoint
        # cg_chunk: depths processed at once inside A / A^T. The naive version
        # built the whole [B, D, Hs, Ws] stack (1.97 GB real + 3.9 GB complex
        # at D=43, 3385^2) before cropping to 512x512, and OOMed with 1.6 GB
        # free. Chunking caps the transient at roughly cg_chunk/D of that.
        # cg_cache_H: caching all D of H_z costs another ~2 GB, which does not
        # fit alongside the retained optics graph, so it is off and H_z is
        # recomputed per chunk. That costs 2*D FFTs per CG iteration -- at
        # ~3.6 ms each (measured: the tikhonov lift does ~86 in 0.31 s) it is
        # ~3 s per lift at cg_iters=10, doubled by checkpointing. Turn it on
        # only if there is spare memory.
        # cg_detach_psf: treat H_z as a CONSTANT inside the solve. The optics
        # still receive gradient through the sensor image (y depends on the
        # mask), but the solve itself stops being differentiated w.r.t. the
        # mask. Last resort for memory -- it weakens the whole point of CG,
        # which is that the mask determines the fixed point rather than
        # reweighting a fixed solution.
        self.cg_chunk = cg_chunk
        self.cg_cache_H = cg_cache_H
        self.cg_detach_psf = cg_detach_psf

    def forward(self, sensor_img: torch.Tensor, psf_stack: torch.Tensor,
                dz: Union[float, torch.Tensor]) -> torch.Tensor:
        y = sensor_img[:, 0].float().contiguous()
        h = psf_stack.to(y.device).float().contiguous()

        B, Hs, Ws = y.shape
        D = h.shape[0]
        out_h, out_w = self.out_hw

        out = torch.empty((B, D, out_h, out_w), device=y.device, dtype=y.dtype)
        Y = torch.fft.rfft2(y, dim=(-2, -1))

        # ------------------------------------------------------------------
        # deconv="tikhonov": one extra pass over the PSFs to accumulate
        # sum_z |H_z|^2 BEFORE any depth is divided by it. Only the real
        # accumulator is kept (~23 MB); the H_z are recomputed in the main loop
        # rather than cached, which would be ~2 GB for 43 depths.
        # Cost: 43 extra FFTs on top of the adjoint's 87, i.e. ~1.5x -- a couple
        # of percent of an iteration dominated by the ~9 s optics forward.
        # ------------------------------------------------------------------
        denom = None
        if self.deconv == "tikhonov":
            # no_grad is what makes this affordable. Under grad, each of the 43
            # rfft2 calls here is kept alive for the backward pass (~92 MB per
            # depth, ~4 GB total) and the run OOMs. Treating the denominator as
            # a constant drops all of it: the gradient to the phase mask still
            # flows through conj(H_z) in the NUMERATOR below, which is the main
            # path; what is dropped is the second-order term through the
            # normaliser. Same trade as the .detach() on the noise amplitude in
            # sensor_noise.py -- the sampled VALUES are identical either way.
            with torch.no_grad():
                Hsum = torch.zeros(Y.shape[-2:], device=y.device, dtype=y.dtype)
                for z0 in range(0, D, self.chunk_size):
                    Hc = torch.fft.rfft2(h[z0:min(z0 + self.chunk_size, D)].detach(),
                                         dim=(-2, -1))
                    Hsum += (Hc.real ** 2 + Hc.imag ** 2).sum(0)
                    del Hc
                lam = self.tikhonov_reg * Hsum.mean().clamp_min(self.eps)
                denom = Hsum + lam

        # ------------------------------------------------------------------
        # deconv="cg": solve rather than apply a closed form. Returns early --
        # the chunked adjoint loop below is for "adjoint" / "tikhonov" only.
        # ------------------------------------------------------------------
        if self.deconv == "cg":
            out = self._cg_solve(y, h, out_h, out_w)
            return self._normalize(out)

        for z0 in range(0, D, self.chunk_size):
            z1 = min(z0 + self.chunk_size, D)
            h_chunk = h[z0:z1]
            H = torch.fft.rfft2(h_chunk, dim=(-2, -1))

            num = Y[:, None] * torch.conj(H)[None]
            if denom is not None:
                num = num / denom
            x_full = torch.fft.irfft2(num, s=(Hs, Ws), dim=(-2, -1))

            x_full = x_full * dz
            if self.clamp_min:
                x_full = x_full.clamp_min(0.0)

            out[:, z0:z1] = center_crop_4d(x_full, out_h, out_w)

        return self._normalize(out)

    # ----------------------------------------------------------------------
    def _normalize(self, out):
        if self.normalize == "percentile":
            return normalize_volume_percentile(out, self.p_low, self.p_high, eps=self.eps)
        if self.normalize == "global":
            return normalize_volume_global(out, eps=self.eps)
        if self.normalize == "per_depth":
            mn = out.amin(dim=(-2, -1), keepdim=True)
            mx = out.amax(dim=(-2, -1), keepdim=True)
            return (out - mn) / (mx - mn).clamp_min(self.eps)
        return out

    # ----------------------------------------------------------------------
    # The forward operator A and its transpose A^T.
    #
    # A^T is exactly what the "adjoint" branch computes: multiply by
    # conj(H_z), inverse FFT, centre-crop. A is its mirror: centre-PAD each
    # depth back to sensor size, multiply by H_z, sum over z. The crop and the
    # pad use the same offsets, which is what makes them a true adjoint pair --
    # if you change center_crop_4d, change _pad here to match.
    #
    # dz is dropped from both. It is a constant scale on A, so it cancels in
    # the normal equations up to the meaning of lam, and _normalize removes
    # what is left.
    #
    # Nothing here crops the sensor into per-lenslet sub-images, so this is
    # unaffected by a learnable lenslet diameter -- sub-images may overlap
    # freely.
    # ----------------------------------------------------------------------
    def _cg_solve(self, y, h, out_h, out_w):
        B, Hs, Ws = y.shape
        D = h.shape[0]
        y0 = (Hs - out_h) // 2
        x0 = (Ws - out_w) // 2
        C = max(1, int(self.cg_chunk))
        if self.cg_detach_psf:
            h = h.detach()

        Hc_all = torch.fft.rfft2(h, dim=(-2, -1)) if self.cg_cache_H else None

        def Hz(z0, z1):
            return Hc_all[z0:z1] if Hc_all is not None \
                else torch.fft.rfft2(h[z0:z1], dim=(-2, -1))

        def A(x):                                    # [B,D,oh,ow] -> [B,Hs,Ws]
            Yf = None
            for z0 in range(0, D, C):
                z1 = min(z0 + C, D)
                xp = x.new_zeros((x.shape[0], z1 - z0, Hs, Ws))
                xp[:, :, y0:y0 + out_h, x0:x0 + out_w] = x[:, z0:z1]
                part = (torch.fft.rfft2(xp, dim=(-2, -1)) * Hz(z0, z1)[None]).sum(1)
                Yf = part if Yf is None else Yf + part
                del xp, part
            return torch.fft.irfft2(Yf, s=(Hs, Ws), dim=(-2, -1))

        def AT(yy):                                  # [B,Hs,Ws] -> [B,D,oh,ow]
            Yf = torch.fft.rfft2(yy, dim=(-2, -1))
            outs = []
            for z0 in range(0, D, C):
                z1 = min(z0 + C, D)
                Xf = Yf[:, None] * torch.conj(Hz(z0, z1))[None]
                xf = torch.fft.irfft2(Xf, s=(Hs, Ws), dim=(-2, -1))
                outs.append(xf[:, :, y0:y0 + out_h, x0:x0 + out_w])
                del Xf, xf
            return torch.cat(outs, dim=1)

        with torch.no_grad():
            Hsum = None
            for z0 in range(0, D, C):
                Hc = torch.fft.rfft2(h[z0:min(z0 + C, D)].detach(), dim=(-2, -1))
                part = (Hc.real ** 2 + Hc.imag ** 2).sum(0)
                Hsum = part if Hsum is None else Hsum + part
                del Hc, part
            lam = self.cg_reg * Hsum.mean().clamp_min(self.eps)
            del Hsum

        def step(x, r, p, rs):
            q = AT(A(p)) + lam * p
            alpha = rs / (p * q).flatten(1).sum(1).clamp_min(1e-30).view(-1, 1, 1, 1)
            x = x + alpha * p
            r = r - alpha * q
            rs_new = (r * r).flatten(1).sum(1).view(-1, 1, 1, 1)
            p = r + (rs_new / rs.clamp_min(1e-30)) * p
            return x, r, p, rs_new

        b = AT(y)
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = (r * r).flatten(1).sum(1).view(-1, 1, 1, 1)

        for _ in range(self.cg_iters):
            if self.cg_checkpoint and torch.is_grad_enabled():
                x, r, p, rs = torch.utils.checkpoint.checkpoint(
                    step, x, r, p, rs, use_reentrant=False)
            else:
                x, r, p, rs = step(x, r, p, rs)

        return x
