from multiprocessing.util import debug
from matplotlib import scale
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.fft import fft2, ifft2, fftshift
from config import config  # Import global config
import torchvision.transforms as transforms
import h5py
import matplotlib.animation as animation
import torch.nn as nn
import time
import torch.nn.functional as F
import os
import math

import gc
gc.collect()
torch.cuda.empty_cache()

import torch
################################################
#debug helpers
################################################

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _save_img(t, path, title="", log10=False):
    a = t.detach().float().cpu()
    if log10:
        a = torch.log10(a.clamp_min(1e-12))
    plt.figure()
    plt.imshow(a.numpy(), cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, dpi=300)
    plt.close()

def _save_phase(t_complex, path, title=""):
    a = torch.angle(t_complex.detach()).float().cpu()
    plt.figure()
    plt.imshow(a.numpy())
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, dpi=200)
    plt.close()
################################################
################################################

def crop_to_tiled(psf, ns, Mx):
    H, W = psf.shape
    S = ns * Mx
    y0 = (H - S) // 2
    x0 = (W - S) // 2
    return psf[y0:y0+S, x0:x0+S]

def peak_in_lenslet2(psf, ns, Mx, u, v):
    psf = crop_to_tiled(psf, ns, Mx)
    tile = psf[u*Mx:(u+1)*Mx, v*Mx:(v+1)*Mx]
    k = torch.argmax(tile)
    y = int(k // Mx)
    x = int(k % Mx)
    return x, y  # inside that lenslet only

def center_pad_or_crop(img2d: torch.Tensor, target_hw: int):
    H, W = img2d.shape
    if H == target_hw and W == target_hw:
        return img2d
    if H > target_hw:
        y0 = (H - target_hw) // 2
        x0 = (W - target_hw) // 2
        return img2d[y0:y0 + target_hw, x0:x0 + target_hw]
    pad_h = target_hw - H
    pad_w = target_hw - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(img2d, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

def fwhm_1d(profile: torch.Tensor, dx_m: float):
    p = profile.detach().float().cpu()
    p = p - p.min()
    if p.max() <= 0:
        return np.nan
    p = p / p.max()

    half = 0.5
    idx = torch.where(p >= half)[0]
    if idx.numel() < 2:
        return np.nan

    left = idx[0].item()
    right = idx[-1].item()

    # linear interpolation for sub pixel edges
    def interp_edge(i0, i1):
        y0 = p[i0].item()
        y1 = p[i1].item()
        if y1 == y0:
            return float(i0)
        return i0 + (half - y0) / (y1 - y0)

    left_f = interp_edge(max(left - 1, 0), left)
    right_f = interp_edge(right, min(right + 1, p.numel() - 1))

    return (right_f - left_f) * dx_m

def rms_width_1d(profile, dx_m):
    p = profile.clamp_min(0)
    s = p.sum() + 1e-12
    x = (torch.arange(p.numel(), device=p.device) - p.numel()/2) * dx_m
    mu = (p * x).sum() / s
    var = (p * (x - mu)**2).sum() / s
    return torch.sqrt(var)

def dof_from_fwhm(psf_stack_crop: torch.Tensor, z_vals_m: np.ndarray, dx_m: float,  M_T, lateral_resolution): # factor: float = np.sqrt(2)
    # psf_stack_crop: [D, tile, tile] intensity
    D = psf_stack_crop.shape[0]
    widths = np.zeros(D, dtype=np.float64)

    for i in range(D):
        psf = psf_stack_crop[i]
        # use profile through the peak inside the lenslet crop
        y_peak, x_peak = torch.nonzero(psf == psf.max(), as_tuple=True)
        y0 = int(y_peak[0])
        prof = psf[y0, :]
        #widths[i] = fwhm_1d(prof, dx_m)
        widths[i] = 2.355 * rms_width_1d(prof, dx_m) #equivalent FWHM from RMS width


    w0 = np.nanmin(widths)
    lateral_resolution_sensor_limit = lateral_resolution* M_T* 2 #np.sqrt(2)  # scaled to sensor plane
    ok = widths <= (lateral_resolution_sensor_limit)

    if not np.any(ok):
        return widths, w0, None

    z_ok = z_vals_m[ok]
    dof_m = z_ok.max() - z_ok.min()
    return widths, w0, dof_m

def plot_xz_for_lenslet(psf_list, depth_list, ns, tile, u, v, pixel_size_xy_m, save_path):
    psf_stack = torch.stack([p.float().cpu() for p in psf_list], dim=0)  # [D,H,W]
    D, H, W = psf_stack.shape

    S = ns * tile
    y0 = (H - S) // 2
    x0 = (W - S) // 2
    psf_crop = psf_stack[:, y0:y0+S, x0:x0+S]                            # [D,S,S]

    tile_stack = psf_crop[:, u*tile:(u+1)*tile, v*tile:(v+1)*tile]       # [D,tile,tile]
    xz = tile_stack[:, tile//2, :]                                       # [D,tile]
    xz = xz / (xz.max() + 1e-12)

    z_um = np.array(depth_list) * 1e6
    x_um = (np.arange(tile) - tile/2) * pixel_size_xy_m * 1e6
    order = np.argsort(z_um)

    plt.figure(figsize=(6,4))
    plt.imshow(
        xz[order].numpy(),
        extent=[x_um[0], x_um[-1], z_um[order][0], z_um[order][-1]],
        origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1
    )
    plt.xlabel("x in lenslet (µm)")
    plt.ylabel("z (µm)")
    plt.title(f"XZ inside lenslet (u={u}, v={v})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_xz_yz_psf(psf_list, depth_list, pixel_size_xy_m, save_prefix="PSF"):
    """
    psf_list: list of 2D tensors [H, W] (intensity PSF per depth)
    depth_list: list of floats (z positions in meters), same length as psf_list
    pixel_size_xy_m: pixel size at the PSF plane for x and y axes in meters
    """

    # Stack into [D, H, W]
    psf_stack = torch.stack([p.float().cpu() for p in psf_list], dim=0)
    D, H, W = psf_stack.shape

    z_vals = np.array(depth_list, dtype=np.float64)  # meters

    # Find global peak location across all depths
    #flat_idx = torch.argmax(psf_stack)
    #d0 = int(flat_idx // (H * W))
    #rem = int(flat_idx % (H * W))
    #y0 = rem // W
    #x0 = rem % W
    y0 = H //2
    x0 = W //2
    d0 = D //2


    # Extract slices
    xz = psf_stack[:, y0, :]   # [D, W]
    yz = psf_stack[:, :, x0]   # [D, H]

    # Normalize for display
    xz = xz / (xz.max() + 1e-12)
    yz = yz / (yz.max() + 1e-12)

    # Axes in microns
    x_um = (np.arange(W) - W / 2) * pixel_size_xy_m * 1e6
    y_um = (np.arange(H) - H / 2) * pixel_size_xy_m * 1e6
    z_um = z_vals * 1e6

    # If z is not sorted, sort for a clean axis
    order = np.argsort(z_um)
    z_um_sorted = z_um[order]
    xz_sorted = xz[order].numpy()
    yz_sorted = yz[order].numpy()


    eps = 1e-12
    log_floor = -10  # shows down to 1e-6 of peak

    xz_log = torch.log10(xz + eps).clamp_min(log_floor)
    yz_log = torch.log10(yz + eps).clamp_min(log_floor)

    xz_log_sorted = xz_log[order].numpy()
    yz_log_sorted = yz_log[order].numpy()


    # Plot XZ
    plt.figure(figsize=(7, 4))
    plt.imshow(
        xz_sorted,
        extent=[x_um[0], x_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("X (µm)")
    plt.ylabel("Z (µm)")
    plt.title("XZ PSF")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_XZ.png", dpi=300)
    plt.close()

    # Plot YZ
    plt.figure(figsize=(7, 4))
    plt.imshow(
        yz_sorted,
        extent=[y_um[0], y_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Y (µm)")
    plt.ylabel("Z (µm)")
    plt.title("YZ PSF")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_YZ.png", dpi=300)
    plt.close()

    # Plot XZ (log10)
    plt.figure(figsize=(7, 4))
    plt.imshow(
        xz_log_sorted,
        extent=[x_um[0], x_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=log_floor,
        vmax=0,
    )
    plt.xlabel("X (µm)")
    plt.ylabel("Z (µm)")
    plt.title("XZ PSF (log10, normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_XZ_log10.png", dpi=300)
    plt.close()

    # Plot YZ (log10)
    plt.figure(figsize=(7, 4))
    plt.imshow(
        yz_log_sorted,
        extent=[y_um[0], y_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=log_floor,
        vmax=0,
    )
    plt.xlabel("Y (µm)")
    plt.ylabel("Z (µm)")
    plt.title("YZ PSF (log10, normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_YZ_log10.png", dpi=300)
    plt.close()


    print(f"Saved {save_prefix}_XZ.png and {save_prefix}_YZ.png")
    print(f"Saved {save_prefix}_XZ_log10.png and {save_prefix}_YZ_log10.png")
    print(f"Slice taken at x0={x0}, y0={y0}, depth index d0={d0}")


def plot_xz_yz_psf_onelenslet(psf_list, depth_list, pixel_size_xy_m, Mx, lateral_resolution, M_T, save_prefix="PSF"):
    """
    psf_list: list of 2D tensors [H, W] (intensity PSF per depth)
    depth_list: list of floats (z positions in meters), same length as psf_list
    pixel_size_xy_m: pixel size at the PSF plane for x and y axes in meters
    Mx: pixels per lenslet
    """

    print(f"Plotting one lenslet cross-section with Mx={Mx}")
    # Stack into [D, H, W]
    psf_stack = torch.stack([p.float().cpu() for p in psf_list], dim=0)
    D, H, W = psf_stack.shape

    z_vals = np.array(depth_list, dtype=np.float64)  # meters

    # Center MLA lenslet coordinates
    y0 = H //2
    x0 = W //2
    d0 = D //2

    tile = Mx - 101  # pixels per lenslet
    half = tile // 2

    # crop one tile
    x_start = max(0, x0 - half)
    x_end   = min(W, x0 + half) +1
    y_start = max(0, y0 - half)
    y_end   = min(H, y0 + half) +1

    psf_stack_crop = psf_stack[:, y_start:y_end, x_start:x_end]  # [D, tile, tile]
    xz = psf_stack_crop[:, psf_stack_crop.shape[1]//2, :]  # [D, tile]
    yz = psf_stack_crop[:, :, psf_stack_crop.shape[2]//2]  # [D, tile]


    # “per lenslet DOF” based on how a lenslet localized spot broadens with defocus.
    z_vals_m = np.array(depth_list, dtype=np.float64)
    widths, w0, dof_m = dof_from_fwhm(psf_stack_crop, z_vals_m, dx_m= pixel_size_xy_m, M_T=M_T, lateral_resolution=lateral_resolution)
    widths_um = widths * 1e6
    z_um = z_vals_m * 1e6

    plt.figure()
    plt.plot(z_um, widths_um, marker="o")
    plt.axhline(w0 * 1e6, linestyle=":")
    plt.axhline(lateral_resolution*M_T*2 * 1e6, linestyle=":")
    plt.xlabel("z (µm)")
    plt.ylabel("FWHM (µm)")
    plt.title("Lenslet PSF FWHM vs z")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lenslet_fwhm_vs_z.png", dpi=300)
    plt.close()

    widths_um_obj = widths_um / M_T
    w0_obj = w0 / M_T
    plt.figure()
    plt.plot(z_um, widths_um_obj, marker="o")
    plt.axhline(w0_obj * 1e6, linestyle=":")
    plt.axhline(lateral_resolution*2 * 1e6, linestyle=":")
    plt.xlabel("z (µm)")
    plt.ylabel("FWHM (µm)")
    plt.title("Lenslet PSF FWHM vs z")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lenslet_fwhm_vs_z_object.png", dpi=300)
    plt.close()

    print("Best focus FWHM (um):", w0 * 1e6)
    print("DOF (um):", None if dof_m is None else dof_m * 1e6)

    # Normalize for display
    xz = xz / (xz.max() + 1e-12)
    yz = yz / (yz.max() + 1e-12)

    # Axes in microns
    D, H, W = psf_stack_crop.shape

    x_um = (np.arange(W) - W / 2) * pixel_size_xy_m * 1e6
    y_um = (np.arange(H) - H / 2) * pixel_size_xy_m * 1e6
    z_um = z_vals * 1e6

    # If z is not sorted, sort for a clean axis
    order = np.argsort(z_um)
    z_um_sorted = z_um[order]
    xz_sorted = xz[order].numpy()
    yz_sorted = yz[order].numpy()


    eps = 1e-12
    log_floor = -10  # shows down to 1e-6 of peak

    xz_log = torch.log10(xz + eps).clamp_min(log_floor)
    yz_log = torch.log10(yz + eps).clamp_min(log_floor)

    xz_log_sorted = xz_log[order].numpy()
    yz_log_sorted = yz_log[order].numpy()


    # Plot XZ
    plt.figure(figsize=(7, 4))
    plt.imshow(
        xz_sorted,
        extent=[x_um[0], x_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("X (µm)")
    plt.ylabel("Z (µm)")
    plt.title("XZ PSF")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_XZ_onelenslet.png", dpi=300)
    plt.close()

    # Plot YZ
    plt.figure(figsize=(7, 4))
    plt.imshow(
        yz_sorted,
        extent=[y_um[0], y_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Y (µm)")
    plt.ylabel("Z (µm)")
    plt.title("YZ PSF")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_YZ_onelenslet.png", dpi=300)
    plt.close()

    # Plot XZ (log10)
    plt.figure(figsize=(7, 4))
    plt.imshow(
        xz_log_sorted,
        extent=[x_um[0], x_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=log_floor,
        vmax=0,
    )
    plt.xlabel("X (µm)")
    plt.ylabel("Z (µm)")
    plt.title("XZ PSF (log10, normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_XZ_log10_onelenslet.png", dpi=300)
    plt.close()

    # Plot YZ (log10)
    plt.figure(figsize=(7, 4))
    plt.imshow(
        yz_log_sorted,
        extent=[y_um[0], y_um[-1], z_um_sorted[0], z_um_sorted[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=log_floor,
        vmax=0,
    )
    plt.xlabel("Y (µm)")
    plt.ylabel("Z (µm)")
    plt.title("YZ PSF (log10, normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_YZ_log10_onelenslet.png", dpi=300)
    plt.close()

    #plot one lenslet cross-sections
    print(f"Saved {save_prefix}_XZ.png and {save_prefix}_YZ.png")
    print(f"Saved {save_prefix}_XZ_log10.png and {save_prefix}_YZ_log10.png")
    print(f"Slice taken at x0={x0}, y0={y0}, depth index d0={d0}")


def calculate_zi(focal_length, z):
    if z == 0:
        # Handling division by zero
        return None  # Or return None, or raise an exception, depending on your use case
    else:
        return focal_length**2 / (z)

def tilt(uin, L, lambda_, alpha, theta, use_gpu=True, chunk_size=128):
    """
    Apply a tilt to the phase front of an optical field with memory optimization.

    Parameters:
        uin (torch.Tensor): Input optical field (complex tensor).
        L (float): Side length of the simulation area (meters).
        lambda_ (float): Wavelength of light (meters).
        alpha (float): Tilt angle (radians).
        theta (float): Rotation angle, from the x-axis (radians).
        use_gpu (bool): Use GPU for calculations if available and True.
        chunk_size (int): Number of rows to process at a time to reduce memory usage.

    Returns:
        torch.Tensor: Tilted optical field (complex tensor).
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    uin = uin.to(device)
    M, N = uin.shape
    dx = L / M

    # Generate coordinate arrays
    x = torch.linspace(-L/2 + dx/2, L/2 - dx/2, M, device=device)
    X, Y = torch.meshgrid(x, x)

    # Wave number
    k = 2 * np.pi / lambda_

    # Compute phase shift in chunks to prevent OOM
    uout = torch.empty_like(uin, dtype=torch.complex64, device=device)

    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        #phase_shift_chunk = k * (X[i:end_i] * torch.cos(theta) + Y[i:end_i] * torch.sin(theta)) * torch.tan(alpha)
        phase_shift_chunk = k * (X[i:end_i] * torch.cos(theta) + Y[i:end_i] * torch.sin(theta)) * torch.sin(alpha)

        uout[i:end_i] = uin[i:end_i] * torch.exp(1j * phase_shift_chunk)

        # Explicitly delete large intermediate tensor and clear GPU cache
        del phase_shift_chunk
        torch.cuda.empty_cache()

    return uout




def propagate_angularspec_fast(uin, blur_kernel_size_fp, lambda_, psf_sampling_fp, z, use_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    uin = uin.to(device)
    M = blur_kernel_size_fp
    dx = psf_sampling_fp #L / M
    k = 2 * np.pi / lambda_

    # Compute frequency grid
    #df = 1 / (dx * M)
    #fx = (torch.arange(-M//2, M//2, device=device) + 0.5) * df
    #FX, FY = torch.meshgrid(fx, fx, indexing='ij')


    fx = torch.fft.fftfreq(M, d=dx, device=device)          # cycles per meter
    fx = torch.fft.fftshift(fx)                             # match fftshift(fft2)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")

    #H = torch.exp(1j * k * z * torch.sqrt(1 - (lambda_ * FX)**2 - (lambda_ * FY)**2 + 0j)).to(torch.complex64)
    sqrt_arg = 1 - (lambda_ * FX)**2 - (lambda_ * FY)**2 + 0j
    H = torch.exp(1j * k * z * torch.sqrt(sqrt_arg.to(torch.complex64)))


    U1 = torch.fft.fftshift(torch.fft.fft2(uin))
    U2 = U1 * H
    U2 = torch.fft.ifftshift(U2)
    uout = torch.fft.ifft2(U2)

    return uout


def wavefront_spherical(
    A, lambda_, medium_index, L, M, x, y, z, AS_sampling, usegpu, debug=False
):
    
    """
    Simulate a spherical wave originating from point (x, y, z) in 3D space.
    Now using double precision to preserve phase integrity for high k.

    Args:
        A: Amplitude
        lambda_: Wavelength (meters)
        medium_index: Refractive index of medium
        L: Simulation size (meters)
        M: Grid size (pixels)
        x, y, z: Source position (meters)
        usegpu: Bool for GPU usage
        debug: Bool to plot and analyze phase behavior

    Returns:
        Complex tensor (M, M) with spherical wavefront
    """
    device = torch.device("cuda" if torch.cuda.is_available() and usegpu else "cpu")
    dx = AS_sampling
    #L = M * dx  # enforce consistency if you want dx to be the real sampling
    #print(f"Wavefront spherical parameters: L={L:.2e} m, M={M}, dx={dx:.2e} m")

    # Use float64 for better precision
    coords = (-L/2 + dx/2) + torch.arange(M, device=device, dtype=torch.float64) * dx
    X, Y = torch.meshgrid(coords - x, coords - y, indexing="ij")

    z = torch.as_tensor(z, device=device, dtype=torch.float32)
    r = torch.sqrt(X**2 + Y**2 + z**2).clamp_min(1e-30)  # always positive
    k = 2 * np.pi * medium_index / lambda_

    #need the sign of z to affect the direction of curvature. 
    # #The simplest way, consistent with the Fresnel limit where the quadratic phase has 1/z, 
    # #is to flip the sign of the phase when z flips sign
    # IMPORTANT:
    # r = sqrt(x^2 + y^2 + z^2) is even in z
    # sign(z) must be applied to phase to preserve propagation direction

    sgn = torch.sign(z)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)  # avoid zero
    phase = k * r * sgn

    u = (A / r) * torch.exp(1j * phase)

    # Convert back to complex64 for consistency with rest of pipeline
    u = u.to(torch.complex64)

    if debug:
        # Plot amplitude
        plt.figure()
        plt.imshow(torch.abs(u).cpu(), cmap='gray')
        plt.title("Amplitude of spherical wave")
        plt.colorbar()
        plt.savefig("spherical_wave_amplitude.png", dpi=300)
        plt.close()

        # Plot phase
        plt.figure()
        phase_rel = phase - phase[M//2, M//2]   # remove global phase offset
        phase_wrapped = torch.remainder(phase_rel + torch.pi, 2*torch.pi) 
        plt.imshow(phase_wrapped.cpu())
        plt.title("Phase of spherical wave")
        plt.colorbar()
        plt.savefig("spherical_wave_phase.png", dpi=300)
        plt.close()

        # Phase profile across center row
        center_row = M // 2
        plt.figure()
        plt.plot(torch.angle(u[center_row, :]).cpu().numpy())
        plt.title("Phase along center row")
        plt.xlabel("Pixel")
        plt.ylabel("Phase (rad)")
        plt.grid(True)
        plt.savefig("spherical_wave_phase_profile.png", dpi=300)
        plt.close()

    return u

# =============================================================================
# FIX 1 of 2 -- non-paraxial defocus pupil (scalar Debye)
# =============================================================================
# The previous model mapped object defocus z to an image-side radius
# zi = f_obj^2 / z and laid a spherical wave sqrt(rho^2 + zi^2) across the pupil.
# Because zi is huge (0.3 m at z = 30 um) while rho is only 3.3 mm, that square
# root collapses to a purely quadratic -- i.e. paraxial -- pupil phase, and its
# coefficient carries n instead of 1/n.  Net effect: each pupil zone is sent to
# n^2*sin(theta) per unit defocus instead of the correct tan(theta).
#
# The scalar Debye pupil below writes the defocus exactly in direction-cosine
# space, with no expansion of cos(theta) anywhere.
#
# Flip these back to "spherical" / "legacy" to reproduce the old behaviour.
# NOTE: the two fixes partially cancel in blur radius at NA 1.1 / n 1.33.
# Applying only ONE of them makes the model WORSE than applying neither.
# Change both together, or neither.
# =============================================================================
PUPIL_MODEL = "debye"        # "debye" (correct) | "spherical" (previous)
APODIZATION = "aplanatic"    # "aplanatic" (correct) | "legacy" (previous)
DEBYE_SIGN = -1              # chosen so that +z keeps the meaning it had before


def wavefront_debye(A, lambda_, medium_index, objective_na,
                    objective_pupil_diameter, M, dx, z, device,
                    sign=DEBYE_SIGN, remove_piston=True):
    """Non-paraxial scalar Debye pupil field for an object defocus `z`.

        P(f) = A * exp( i * sign * k0 * n * z * cos(theta) )
        cos(theta) = sqrt(1 - (lambda * f_r / n)^2)

    Returned WITHOUT the apodization -- PSF_lf_Fourier_relay applies that on the
    next line, exactly as it did for wavefront_spherical.

    The frequency grid is built with the same `coeff` expression the apodization
    block uses, so the two are guaranteed to land on identical samples.

    float64 is used for the phase: k0*n*z is ~4e6 rad at z = 30 um, and
    subtracting the piston in float32 would lose all the surviving digits.
    """
    coeff = (2.0 * objective_na / lambda_) / objective_pupil_diameter   # = 1/(lam*f_obj)
    fx_ticks = torch.linspace(-M * dx / 2 + dx / 2, M * dx / 2 - dx / 2, M,
                              device=device, dtype=torch.float64) * coeff
    fX, fY = torch.meshgrid(fx_ticks, fx_ticks, indexing="ij")
    R_f = torch.sqrt(fX ** 2 + fY ** 2)

    sin_th = torch.clamp(R_f * lambda_ / medium_index, max=1.0)
    cos_th = torch.sqrt(torch.clamp(1.0 - sin_th ** 2, min=0.0))

    k0 = 2.0 * np.pi / lambda_
    z_t = torch.as_tensor(float(z), device=device, dtype=torch.float64)
    phase = sign * k0 * medium_index * z_t * cos_th
    if remove_piston:
        # a constant pupil phase cannot move the PSF; removing it keeps the
        # remaining phase small enough to survive the cast to complex64
        phase = phase - sign * k0 * medium_index * z_t

    u = torch.exp(1j * phase).to(torch.complex64)
    return A * u


def wavefront_fresnel(
    A, lambda_, medium_index, L, M, x, y, z, usegpu, debug=False
):
    """
    Fresnel approximation to a spherical wave.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and usegpu else "cpu")
    dx = L / M
    coords = torch.linspace(-L/2 + dx/2, L/2 - dx/2, M, device=device).double()
    X, Y = torch.meshgrid(coords - x, coords - y, indexing='ij')

    k = 2 * np.pi * medium_index / lambda_

    # Fresnel approximation of spherical wave
    r_quad = z + (X**2 + Y**2) / (2 * z)
    phase = torch.exp(-1j * k * r_quad)

    # You could include 1/z falloff here if needed:
    # amplitude = A / z
    u = A * phase

    if debug:
        # Plot amplitude
        plt.figure()
        plt.imshow(torch.abs(u).cpu(), cmap='gray')
        plt.title("Amplitude of spherical wave, Fresnel approximation")
        plt.colorbar()
        plt.show()

        # Plot phase
        plt.figure()
        plt.imshow(torch.angle(u).cpu(), cmap='twilight', vmin=-np.pi, vmax=np.pi)
        plt.title("Phase of spherical wave, Fresnel approximation")
        plt.colorbar()
        plt.show()

        # Phase profile across center row
        center_row = M // 2
        plt.figure()
        plt.plot(torch.angle(u[center_row, :]).cpu().numpy())
        plt.title("Phase along center row")
        plt.xlabel("Pixel")
        plt.ylabel("Phase (rad)")
        plt.grid(True)
        plt.show()


    return u.to(torch.complex64)



def check_fresnel_validity(L, z, lambda_, n=1.0):
    """
    Checks if the Fresnel approximation is valid for a given field size and propagation depth.
    L: Field size (meters)
    z: Propagation distance (meters)
    lambda_: Wavelength (meters)
    n: Medium index
    """
    theta_max = np.arctan((L / 2) / z)
    NA = n * np.sin(theta_max)
    print(f"Maximum angle: {np.degrees(theta_max):.2f}°")
    print(f"Estimated NA: {NA:.3f}")

    if NA <= 0.5:
        print("✅ Fresnel approximation is valid (NA ≤ 0.5).")
    else:
        print("⚠️  Fresnel may break down — use full spherical wave.")



def hex_centers_px(nx, ny, pitch_px, device, center_xy=None):
    """
    Returns lenslet centers (cx, cy) in pixel coordinates for a staggered hex grid.

    nx: number of lenslets per row (columns)
    ny: number of rows
    pitch_px: horizontal center to center spacing in pixels
    """
    dy = pitch_px * math.sqrt(3) / 2.0

    if center_xy is None:
        # center the lattice roughly around (0,0) then you can shift later
        cx0 = 0.0
        cy0 = 0.0
    else:
        cx0, cy0 = center_xy

    centers = []
    for r in range(ny):
        row_shift = 0.5 * pitch_px if (r % 2 == 1) else 0.0
        for c in range(nx):
            cx = cx0 + (c - (nx - 1) / 2.0) * pitch_px + row_shift
            cy = cy0 + (r - (ny - 1) / 2.0) * dy
            centers.append((cx, cy))
    return centers, dy

"""
def make_soft_circular_mask(Mx, diameter, psf_sampling_fp, device, tau=1.5):
    #diameter: scalar tensor, still connected to autograd
    #tau: softness in pixel units, larger means smoother boundary
    coords = torch.arange(Mx, device=device, dtype=torch.float32) - (Mx - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)

    radius_px = diameter / psf_sampling_fp / 2.0

    # Soft circular aperture
    mask = torch.sigmoid((radius_px - rr) / tau)
    return mask
""" 

def make_binary_aperture_ste(Mx, diameter, psf_sampling_fp, device, tau=0.1):
    """
    Hard binary aperture in the forward pass,
    soft surrogate gradient in the backward pass.

    Mx                : patch size
    diameter          : scalar tensor, learnable through autograd
    psf_sampling_fp   : physical size per pixel
    tau               : softness for backward surrogate
    """
    coords = torch.arange(Mx, device=device, dtype=torch.float32) - (Mx - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)

    radius_px = diameter / psf_sampling_fp / 2.0

    # Soft surrogate used only for gradient
    soft = torch.sigmoid((radius_px - rr) / tau)

    # True binary aperture used in forward
    hard = (rr <= radius_px).float()

    # Straight through estimator
    #PyTorch will handle it automatically because of this line

    #hard.detach() has no gradient.
    #soft.detach() also has no gradient.
    #Only soft still belongs to the computation graph.

    mask = hard.detach() + soft - soft.detach() 
    return mask

def multifocal_microlens_array_hex(
    blur_kernel_size_fp,
    lambda_,
    nx, ny, D_pitch,
    D,
    f_rand,
    psf_sampling_fp,
    device=None,
    max_D=None
):
    if device is None:
        device = f_rand.device

    D_map = D
    D_max = max_D if max_D is not None else float(D_map.max().item())
    Mx = int(np.floor(D_max / psf_sampling_fp))
    pitch_px = int(np.floor(D_pitch / psf_sampling_fp))

    x = (-D_max / 2 + psf_sampling_fp / 2) + torch.arange(
        Mx, device=device, dtype=torch.float32
    ) * psf_sampling_fp
    X, Y = torch.meshgrid(x, x, indexing="ij")

    Xg = X.unsqueeze(0).unsqueeze(0)
    Yg = Y.unsqueeze(0).unsqueeze(0)

    k = 2 * torch.pi / lambda_
    f_rand = f_rand.reshape(ny, nx).to(device)
    denom = (2.0 * f_rand).unsqueeze(-1).unsqueeze(-1)

    # Phase only, no soft aperture here
    phi = (-k / denom) * (Xg**2 + Yg**2)
    lens_phase = torch.exp(1j * phi).to(torch.complex64)
    """

    
    # quadratic lens phase
    phi_quad = (-k / denom) * (Xg**2 + Yg**2)

    # cubic phase
    alpha_cubic = 100.0
    R = (D_map.reshape(ny, nx).to(device) / 2.0).unsqueeze(-1).unsqueeze(-1)
    R = torch.clamp(R, min=1e-12)
    Xn = Xg / R
    Yn = Yg / R
    phi_cubic = alpha_cubic * (Xn**3 + Yn**3)

    # total phase
    phi = phi_quad + phi_cubic
    lens_phase = torch.exp(1j * phi).to(torch.complex64)
    """



    canvas = torch.zeros(
        (blur_kernel_size_fp, blur_kernel_size_fp),
        dtype=torch.complex64,
        device=device
    )

    amp_canvas = torch.zeros(
        (blur_kernel_size_fp, blur_kernel_size_fp),
        dtype=torch.float32,
        device=device
    )

    center_xy = (blur_kernel_size_fp / 2.0, blur_kernel_size_fp / 2.0)
    centers, _ = hex_centers_px(nx, ny, pitch_px=pitch_px, device=device, center_xy=center_xy)

    idx = 0
    for r in range(ny):
        for c in range(nx):
            cx, cy = centers[idx]
            idx += 1

            D_rc = D_map[r, c]

            aperture_mask = make_binary_aperture_ste(
                Mx=Mx,
                diameter=D_rc,
                psf_sampling_fp=psf_sampling_fp,
                device=device,
                tau=0.5
            )

            x0 = int(round(cx)) - Mx // 2
            x1 = x0 + Mx
            y0 = int(round(cy)) - Mx // 2
            y1 = y0 + Mx

            xs0 = max(0, x0)
            ys0 = max(0, y0)
            xs1 = min(blur_kernel_size_fp, x1)
            ys1 = min(blur_kernel_size_fp, y1)

            px0 = xs0 - x0
            py0 = ys0 - y0
            px1 = px0 + (xs1 - xs0)
            py1 = py0 + (ys1 - ys0)

            mask_crop = aperture_mask[py0:py1, px0:px1]
            phase_crop = lens_phase[r, c, py0:py1, px0:px1]

            patch_crop = mask_crop.to(torch.complex64) * phase_crop

            region = canvas[ys0:ys1, xs0:xs1].clone()

            updated_region = region * (1.0 - mask_crop.to(region.dtype)) + patch_crop

            new_canvas = canvas.clone()
            new_canvas[ys0:ys1, xs0:xs1] = updated_region
            canvas = new_canvas

            amp_canvas[ys0:ys1, xs0:xs1] = mask_crop

    return canvas, amp_canvas, Mx, pitch_px, centers


def resample_complex_field_to_camera(u_old, dx_old, dx_new):
    """
    u_old: 2D complex tensor [H, W]
    dx_old: old sampling in meters
    dx_new: new sampling in meters
    """
    H_old, W_old = u_old.shape

    scale = dx_old / dx_new
    H_new = int(round(H_old * scale))
    W_new = int(round(W_old * scale))

    real_in = u_old.real[None, None, :, :]
    imag_in = u_old.imag[None, None, :, :]

    real_new = F.interpolate(
        real_in,
        size=(H_new, W_new),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    imag_new = F.interpolate(
        imag_in,
        size=(H_new, W_new),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    return torch.complex(real_new, imag_new)

def resample_complex_field(u, grid):
    """Resample complex field with bilinear interpolation using grid_sample"""
    if torch.is_complex(u):
        real = F.grid_sample(u.real.unsqueeze(0).unsqueeze(0),
                             grid, mode="bilinear",
                             padding_mode="zeros",
                             align_corners=False)[0, 0]
        imag = F.grid_sample(u.imag.unsqueeze(0).unsqueeze(0),
                             grid, mode="bilinear",
                             padding_mode="zeros",
                             align_corners=False)[0, 0]
        return real + 1j * imag
    else:
        return F.grid_sample(u.unsqueeze(0).unsqueeze(0),
                             grid, mode="bilinear",
                             padding_mode="zeros",
                             align_corners=False)[0, 0]

def PSF_lf_Fourier_relay(x, y, z, display, phasemask, lambda_, objective_focal_length, objective_medium_index, objective_pupil_diameter, objective_demagnification, camera_distance_from_phasemask, objective_na, blur_kernel_size_sp, psf_sampling_sp, blur_kernel_size_fp, psf_sampling_fp, blur_kernel_size_as, sampling_AS, use_fresnel=False):
  
    # PSF_lf_Fourier_relay Generate PSF for a Fourier space light field microscope,
    # by calculating the sperical wavefront at back pupil plane, demagnify by a relay system,
    # and propagate to the sensor.
    # x,y,z, - point source locations(meters)
    # recipe - optical recipe file

    #print(f'Calculating Fourier LFM PSF of x={x:.2e}, y={y:.2e}, z={z:.6e}')
    usegpu = True

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() and usegpu else "cpu")

    # Simulation parameters    
    zi = calculate_zi(objective_focal_length, z) # z is object defocus relative to best focus (meters)
    I = 1  # Point source intensity

    #print(f"zi: {zi*1e3:.6e} mm")


    #if zi is None:
    #    print("zi: None (in focus, plane wave)")
    #else:
    #    print(f"zi: {zi*1e3:.6e} mm")

   
    dx1 = sampling_AS #% pupil plane I (Apeture stop) pixel size (m)
    M_AS = blur_kernel_size_as #Number of pixels in pupil plane I (Apeture stop)

    # Wavefront generation
    #t0 = time.time()
    if display:
        print(f"[Wavefront] Using {'Fresnel' if use_fresnel else 'Full Spherical'} model")

    A0 = torch.sqrt(torch.as_tensor(I, device=device, dtype=torch.float32))

    if PUPIL_MODEL == "debye":
        # FIX 1 of 2: exact non-paraxial defocus phase. Handles z == 0 on its
        # own (cos term vanishes -> flat pupil), so no special case is needed.
        u1 = wavefront_debye(
            A=A0,
            lambda_=lambda_,
            medium_index=objective_medium_index,
            objective_na=objective_na,
            objective_pupil_diameter=objective_pupil_diameter,
            M=M_AS,
            dx=dx1,
            z=(z.item() if torch.is_tensor(z) else z),
            device=device,
        )
    elif zi is None:
        # in-focus: plane wave over the pupil support
        u1 = A0 * torch.ones((M_AS, M_AS), device=device, dtype=torch.complex64)
    else:
        # out-of-focus: spherical wave with radius zi
        u1 = wavefront_spherical(
            A=A0,
            lambda_=lambda_,
            medium_index=objective_medium_index,
            L=objective_pupil_diameter,
            M=M_AS,
            x=0.0,
            y=0.0,
            z=zi,
            AS_sampling=dx1,
            usegpu=usegpu,
            debug=False,
        )
    #print("zi used", zi)

    # "Simulate aperture and Debye apodization:" 
    fx_cutoff_pupil = 2 * objective_na / lambda_ #back aperture size in frequency space, largest spatial frequency that an objective can transmit 
    coeff = fx_cutoff_pupil / objective_pupil_diameter
    fx_ticks = torch.linspace(-M_AS * dx1 / 2 + dx1 / 2, M_AS * dx1 / 2 - dx1 / 2, M_AS).to(device) * coeff  #after multiplication by coeff it becomes the frequency coordinate fx
    fX, fY = torch.meshgrid(fx_ticks, fx_ticks)
    R_f = torch.sqrt(fX**2 + fY**2)
    # =====================================================================
    # FIX 2 of 2 -- aplanatic apodization for a FLAT-PUPIL transform.
    #
    # Textbooks (Born & Wolf 8.8, Broxton 2013 Eq. 3) quote sqrt(cos t) for the
    # 1-D Debye integral over ANGLE, whose measure is sin t dt.  This code
    # transforms over the flat pupil instead, and
    #       2 pi f_r df_r  =  (2 pi n^2 / lam^2) cos t sin t dt
    # already carries a factor cos t.  Matching the two integrands gives
    #       A(f_x,f_y) = P(theta) / cos t = 1 / sqrt(cos t)
    # so writing sqrt(cos t) here double-counts cos t (it is equivalent to using
    # P = cos^{3/2} t in the textbook integral).  Verified numerically against
    # direct quadrature of the 1-D integral -- see check_apodization.py.
    # =====================================================================
    inside = (R_f <= (fx_cutoff_pupil / 2)).to(torch.float32)
    cos_theta = torch.sqrt(torch.clamp(
        1.0 - (R_f * lambda_ / objective_medium_index) ** 2, min=0.0))
    # clamp only guards NA -> n; for NA 1.1 in water cos(theta_max) = 0.562
    cos_safe = torch.clamp(cos_theta, min=1e-6)

    if APODIZATION == "aplanatic":
        apodization = (1.0 / torch.sqrt(cos_safe)) * inside
    else:                                    # "legacy" -- previous behaviour
        apodization = torch.sqrt(cos_safe) * inside
    apodization = apodization.type(torch.float32)

    u2 = u1 * apodization

    # Normalize intensity
    I_u2 = torch.abs(u2) ** 2
    I_frac = (1 - (objective_focal_length / np.sqrt(objective_focal_length**2 + objective_pupil_diameter**2 / 4))) / 2 #this comes from 3D geometry, cone solid angle fraction of the objective (sigma_col over 4pi)
    #normalize the intensity at pupil so that every z layer has the same total enerygy
    u2_normalize =  u2 / (torch.sqrt(I_u2.sum() + 1e-6) * torch.sqrt(I *  torch.tensor(I_frac, dtype=torch.float32, device=u2.device)) ) # Avoid division by zero 
    #t2 = time.time()
    #print(f"[F_Optics] Norm u2: {t2 - t1:.2f}s")

    # Apply tilt to off-axis point
    x = torch.tensor(x)
    y = torch.tensor(y)
    alpha = torch.atan(torch.sqrt(x**2 + y**2) / objective_focal_length)
    theta = torch.atan2(y, x)
    u2_tilt = tilt(u2_normalize, objective_pupil_diameter, lambda_, alpha, theta, use_gpu=True, chunk_size=64)
      
    #t3 = time.time()
    #print(f"[F_Optics] Tilt: {t3 - t2:.2f}s")

    # De-magnify the relayed pupil
    L = objective_pupil_diameter #% Simulation size in the back pupil space(meters).
    L = L / objective_demagnification #% Simulation size in the back pupil space(meters). 
    M_FP = blur_kernel_size_fp #% Number of pixels in Fourier plane (phasemask plane) after demagnification
    delta_x3 = psf_sampling_fp #% pixel size at Fourier plane (phasemask plane) after demagnification(meters)
    dx3 = dx1 / objective_demagnification
    #print(f"dx1: {dx1:.2e}, dx3: {dx3:.2e}, delta_x3: {delta_x3:.2e}")

    #Resample u₂_tilted onto the demagnified grid and multiply by the mask there
    #M should be scaled by the demagnification factor and the interpolation grid must be built with the smaller size.
    # grid for plane 1 (aperture stop plane)
    x1 = (torch.arange(M_AS, device=device) - M_AS / 2) * dx1
    y1 = (torch.arange(M_AS, device=device) - M_AS / 2) * dx1
    Y1, X1 = torch.meshgrid(y1, x1, indexing="ij")

    # grid for plane 3 (fourier plane)
    x3 = (torch.arange(M_FP, device=device) - M_FP / 2) * dx3
    y3 = (torch.arange(M_FP, device=device) - M_FP / 2) * dx3
    Y3, X3 = torch.meshgrid(y3, x3, indexing="ij")

    # mapping from plane 3 coordinates back to plane 1
    X1_from_X3 = objective_demagnification * X3
    Y1_from_Y3 = objective_demagnification * Y3

    
    # convert physical coordinates at plane 1 into pixel indices
    ix = X1_from_X3 / dx1 + M_AS / 2
    iy = Y1_from_Y3 / dx1 + M_AS / 2

    # grid_sample expects normalized coordinates in [-1, 1]
    grid_x = 2.0 * ix / (M_AS - 1) - 1.0
    grid_y = 2.0 * iy / (M_AS- 1) - 1.0

    # grid has shape [1, H, W, 2]
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    # prepare field for grid_sample [N, C, H, W]
    u2_tilt_batched = u2_tilt.unsqueeze(0).unsqueeze(0)

    # resample u2_tilt onto plane 3 grid
    u2_on_plane3 = resample_complex_field(u2_tilt, grid)

    # phase_mask is already defined on (X3, Y3) grid with shape [H, W]
    # Apply phase of phasemask
    u3 = u2_on_plane3.cuda() * phasemask.cuda()
    #t4 = time.time()
    #print(f"[F_Optics] Apply phase mask: {t4 - t3:.2f}s")
    
    #Apply padding to u3 to match sensor pixel size
    pad_size = (blur_kernel_size_sp - blur_kernel_size_fp) // 2
    #u3 = F.pad(u3, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
    #print(f'u3 shape after padding: {u3.shape}')
    #print(f'PSF sampling at sensor: {psf_sampling_sp:.2e} m')

    blur_kernel_size_fp_padded = u3.shape[0]

    #Propagate to the sensor plane
    #sampling is still determined by the Fourier plane sampling
    u4 = propagate_angularspec_fast(u3, blur_kernel_size_fp_padded, lambda_, psf_sampling_fp, camera_distance_from_phasemask)
    u4_cam = resample_complex_field_to_camera(u4, psf_sampling_fp, psf_sampling_sp)

    #t5 = time.time()
    #print(f"[F_Optics] propagate_angularspec: {t5 - t4:.2f}s")

    # Calculate PSF
    Isensor = torch.abs(u4_cam)**2
  
    #

    if display:    
        # Image wavewront
        I_wv = torch.abs(u1)**2
        plt.figure()
        plt.imshow(I_wv.cpu().numpy())
        plt.title("Spherical wavefront magnitude")
        plt.colorbar()
        plt.savefig("Field_u1_magnitude.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.angle(u1.cpu()))
        plt.title("Spherical wavefront phase")
        plt.colorbar()
        plt.savefig("Field_u1_magnitude_phase.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.abs(u2.cpu())**2)
        plt.title("After apodization at back pupil plane")
        plt.colorbar()
        plt.savefig("Field_u2_after_apodization_magnitude.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.angle(u2.cpu()))
        plt.title("After apodization at back pupil plane phase")
        plt.colorbar()
        plt.savefig("Field_u2_after_apodization_phase.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.abs(u2_tilt.cpu())**2)
        plt.title("Apply tilt to off-axis point")
        plt.colorbar()
        plt.savefig("Field_u2_after_tilt_and_normalization_magnitude.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.angle(u2_tilt.cpu()))
        plt.title("Apply tilt to off-axis point phase")
        plt.colorbar()
        plt.savefig("Field_u2_after_tilt_and_normalization_phase.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.abs(u3.cpu())**2)
        plt.title("Apply phase mask")
        plt.colorbar()
        plt.savefig("Field_u3_after_phasemask_magnitude.png")
        plt.close()

        amp = u3.abs()
        phase_u3 = torch.angle(u3)
        phase_plot = phase_u3.clone()
        phase_plot[amp < 1e-6 * amp.max()] = float("nan")
        plt.figure()
        plt.imshow(phase_plot.cpu())
        plt.title("Apply phase mask phase")
        plt.colorbar()
        plt.savefig("Field_u3_after_phasemask_phase.png")
        plt.close()

        plt.figure()
        plt.imshow(Isensor.cpu().numpy(), cmap='binary')
        plt.title(f"Spherical wavefront at sensor plane, PSF (z={z.item()*1000:.6f}) Magnitude")  
        plt.colorbar()
        plt.savefig(f"PSF_at_sensor_(z={z.item()*1000:.6f})_magnitude.png")
        plt.close()

        plt.figure()
        plt.imshow(torch.angle(u4.cpu()) )
        plt.title(f"Spherical wavefront at sensor plane, PSF (z={z.item()*1000:.6f}) Phase")
        plt.colorbar()
        plt.savefig(f"PSF_at_sensor_(z={z.item()*1000:.6f})_phase.png")
        plt.close()
    
    del u4, u3, u1, u2_tilt, u2_normalize
    gc.collect()
    torch.cuda.empty_cache()

    return pad_size, dx1, Isensor 


def animate_LF_views(LF_views, save_path="LF_views.mp4", fps=2, dpi=300):
    batch_size, NN, H, W = LF_views.shape
    N = int(np.sqrt(NN))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.axis("off")

    def update(frame):
        u, v = divmod(frame, N)
        idx = u * N + v
        ax.clear()
        ax.axis("off")
        ax.imshow(LF_views[0, idx].detach().cpu().numpy(), cmap="gray", interpolation="nearest")
        ax.set_title(f"View ({u}, {v})")

    ani = animation.FuncAnimation(fig, update, frames=NN, interval=1000 // fps)

    if save_path.endswith(".gif"):
        ani.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    elif save_path.endswith(".mp4"):
        ani.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi, bitrate=12000)
    else:
        raise ValueError("Use .gif or .mp4")

    plt.show()



def animate_LF_views_hex(LF_views, row_counts, save_path="LF_views.mp4", fps=2, dpi=300):
    """
    Animate LF views one by one.

    Parameters
    ----------
    LF_views : torch.Tensor
        Shape [B, NV, H, W]
    row_counts : list[int]
        Hex row layout, does not need to sum to NV
    save_path : str
        Output path, must end with .gif or .mp4
    fps : int
        Frames per second
    dpi : int
        Output DPI
    """
    B, NV, H, W = LF_views.shape

    if B > 1:
        print("Warning: only animating the first sample in batch")

    # map idx -> (row, col_in_row), based on available layout slots
    idx2rc = []
    for r, n in enumerate(row_counts):
        for c in range(n):
            idx2rc.append((r, c))

    n_frames = min(NV, len(idx2rc))

    if n_frames == 0:
        raise ValueError("No frames to animate. Check LF_views or row_counts.")

    if len(idx2rc) < NV:
        print(f"Warning: row_counts provides only {len(idx2rc)} slots, so only first {n_frames} of {NV} views will be animated.")
    elif len(idx2rc) > NV:
        print(f"Info: {NV} views available, extra row_counts slots will be ignored.")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.axis("off")

    def update(frame):
        r, c = idx2rc[frame]
        ax.clear()
        ax.axis("off")
        ax.imshow(
            LF_views[0, frame].detach().cpu().numpy(),
            cmap="gray",
            interpolation="nearest"
        )
        ax.set_title(f"View idx={frame}  row={r}  col={c}", fontsize=10)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=int(1000 / fps),
        repeat=True
    )

    if save_path.endswith(".gif"):
        ani.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    elif save_path.endswith(".mp4"):
        ani.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi, bitrate=12000)
    else:
        raise ValueError("Use .gif or .mp4")

    plt.show()

def plot_psf_grid(psf_list):
    """
    Plots all PSFs.
    """
    D = len(psf_list)
    rows = 6
    cols = int(np.ceil(D / rows))  # Calculate columns to fit all slices

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < D:
            ax.imshow(psf_list[i] / psf_list[i].max(), cmap='gray')
            ax.set_title(f"Slice {i+1}")
            ax.axis("off")  # Remove axis labels
    plt.tight_layout()
    plt.show()

def plot_LF_views_grid(LF_views):
    """
    Plots all Light Field (LF) views in an N x N grid.

    Args:
        LF_views (torch.Tensor): Light field views of shape (batch_size, N×N, H, W)
    """
    batch_size, NN, H, W = LF_views.shape
    N = int(np.sqrt(NN))  # Compute N assuming N×N = NN

    if batch_size > 1:
        print("Warning: Only visualizing the first sample in batch!")

    fig, axes = plt.subplots(N, N, figsize=(10, 10))

    for idx in range(NN):  # Iterate over the flattened views
        u, v = divmod(idx, N)  # Convert index into 2D grid position
        ax = axes[u, v]
        ax.imshow(LF_views[0, idx].cpu().numpy(), cmap='gray')  # First sample in batch
        ax.set_title(f"View ({u},{v})")
        ax.axis("off")  # Hide axes for cleaner visualization

    plt.tight_layout()
    plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt

def plot_hex_1d_profiles(profiles, row_counts, save_path="profiles_hex.png", title_prefix="Profile"):
    """
    profiles: tensor or numpy array of shape [Nviews, Nr]
    row_counts: e.g. [5,4,5,4,5]
    """
    if hasattr(profiles, "detach"):
        profiles = profiles.detach().cpu().numpy()

    Nviews, Nr = profiles.shape
    if sum(row_counts) != Nviews:
        raise ValueError(f"sum(row_counts)={sum(row_counts)} must equal Nviews={Nviews}")

    nrows = len(row_counts)
    ncols = max(row_counts)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows), squeeze=False)

    # turn all off first
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    idx = 0
    for r, count in enumerate(row_counts):
        col_offset = (ncols - count) / 2.0
        for c in range(count):
            ax = axes[r, int(np.floor(c + col_offset))]
            ax.axis("on")

            y = profiles[idx]
            x = np.arange(len(y))
            ax.plot(x, y, linewidth=1.2)
            ax.set_title(f"{title_prefix} {idx}", fontsize=8)
            ax.set_xlim(0, len(y) - 1)

            # optional: keep y-axis consistent
            ax.set_ylim(0, max(1e-8, profiles.max()))

            ax.tick_params(labelsize=6)
            idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_LF_views_hex(LF_views, row_counts, cmap="gray", title_prefix="View", save_path="LF_views_hex.png"):
    """
    Plot LF views in a staggered hex layout.

    Parameters
    ----------
    LF_views : torch.Tensor
        Shape [B, NV, H, W]
    row_counts : list[int]
        Example: [5,4,5,4,5]
        Does NOT need to sum to NV anymore.
    cmap : str
        Matplotlib colormap
    title_prefix : str
        Title prefix for each view
    save_path : str or None
        If not None, saves the figure

    Behavior
    --------
    - If sum(row_counts) > NV, extra slots stay empty
    - If sum(row_counts) < NV, only the first sum(row_counts) views are shown
    """

    B, NV, H, W = LF_views.shape
    if B > 1:
        print("Warning: only visualizing the first sample in batch")

    n_rows = len(row_counts)
    max_cols = max(row_counts)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(2.2 * max_cols, 2.2 * n_rows))

    # Make axes always 2D
    if n_rows == 1 and max_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif max_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Turn all axes off first
    for r in range(n_rows):
        for c in range(max_cols):
            axes[r, c].axis("off")

    idx = 0
    for r, n_in_row in enumerate(row_counts):
        # Center each row
        base_offset = (max_cols - n_in_row) // 2

        # Optional extra shift for odd rows to mimic hex staggering
        odd_extra = 1 if (r % 2 == 1 and (base_offset + n_in_row) < max_cols) else 0
        offset = base_offset + odd_extra

        for j in range(n_in_row):
            # Stop if no more views are available
            if idx >= NV:
                break

            col = offset + j
            if col >= max_cols:
                continue

            ax = axes[r, col]
            ax.imshow(LF_views[0, idx].detach().cpu().numpy(), cmap=cmap)
            ax.set_title(f"{title_prefix} {idx}", fontsize=8)
            ax.axis("off")
            idx += 1

    if sum(row_counts) < NV:
        print(f"Warning: only plotted {sum(row_counts)} of {NV} views because row_counts has fewer slots.")
    elif sum(row_counts) > NV:
        print(f"Info: plotted all {NV} available views, leaving extra layout slots empty.")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()
    
def next_power_of_2(n):
    return 2 ** (n - 1).bit_length()



def extract_hex_views(sensor_img, nx, ny, pitch_px, patch_px):
    """
    sensor_img: [H, W] tensor
    Returns: [1, ny*nx, patch_px, patch_px]
    """
    H, W = sensor_img.shape
    device = sensor_img.device
    center_xy = (W / 2.0, H / 2.0)
    centers, _ = hex_centers_px(nx, ny, pitch_px=pitch_px, device=device, center_xy=center_xy)

    half = patch_px // 2
    views = []
    for (cx, cy) in centers:
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        x1 = x0 + patch_px
        y1 = y0 + patch_px

        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            # pad if you prefer, but skip is simplest
            continue

        views.append(sensor_img[y0:y1, x0:x1])

    views = torch.stack(views, dim=0)  # [Nviews, patch, patch]
    return views.unsqueeze(0)



def plot_sensor_with_centers(debug_dir, sensor_img, centers_rescaled, psf_sampling_fp, dia=None, index=None):
    """
    sensor_img: 2D tensor or numpy array
    centers_rescaled: list of (cx, cy) in sensor pixel units
    dia: tensor of shape [ny, nx], microlens diameters in microns
    psf_sampling_fp: sampling in microns/pixel (used to convert diameter to pixels)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if isinstance(sensor_img, torch.Tensor):
        sensor_img = sensor_img.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sensor_img, cmap='gray')

    if dia is not None:
        ny, nx = dia.shape

    for i, (cx, cy) in enumerate(centers_rescaled):
        cx = float(cx)
        cy = float(cy)
        ax.plot(cx, cy, 'ro', markersize=3)

        if dia is not None:
            r, c = divmod(i, nx)
            radius_px = float(dia[r, c].item()) / psf_sampling_fp / 2
            circ = plt.Circle((cx, cy), radius_px, color='r', fill=False, linewidth=1)
            ax.add_patch(circ)

    ax.set_title("Sensor image with rescaled lenslet centers")
    plt.axis('on')    
    plt.savefig(os.path.join(debug_dir, f"sensor_with_centers_sensor_contrib_z_{index:02d}.png"), dpi=300)
    plt.close()


def check_view_center_alignment(sensor_img, views, centers_rescaled, dia_px_list):
    """
    For each extracted view, plot it and show the expected center point overlaid.

    sensor_img: [H, W] tensor or numpy array (original sensor image)
    views: list of cropped patches (each a [H_i, W_i] tensor)
    centers_rescaled: list of (cx, cy) in pixel units (same order as views)
    dia_px_list: list of diameters in pixels, same length as views
    """
    import matplotlib.pyplot as plt
    import numpy as np

    for i, (view, (cx, cy), D) in enumerate(zip(views, centers_rescaled, dia_px_list)):
        if isinstance(view, torch.Tensor):
            view_np = view.detach().cpu().numpy()
        else:
            view_np = view

        # Calculate where the original center maps into the view coordinates
        # The crop bounds are:
        D_px = int(D)
        offset = D_px // 2
        x0 = int(round(cx)) - offset
        y0 = int(round(cy)) - offset

        # Center of the cropped image relative to original sensor
        local_cx = cx - x0
        local_cy = cy - y0

        plt.figure(figsize=(3, 3))
        plt.imshow(view_np, cmap='gray')
        plt.scatter([local_cx], [local_cy], color='red', s=10, label='Expected center')
        plt.title(f'Lenslet {i} (Patch size: {view_np.shape[1]}x{view_np.shape[0]})')
        plt.legend()
        plt.axis('off')
        plt.show()


def compute_mtf_from_psf(psf: torch.Tensor, eps: float = 1e-12):
    """
    psf: [H, W], real nonnegative PSF
    returns:
        mtf: [H, W], normalized 2D MTF
    """
    psf = psf.float()
    psf = psf / (psf.sum() + eps)

    otf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf)))
    mtf = torch.abs(otf)
    mtf = mtf / (mtf.max() + eps)
    return mtf

def radial_profile(img: torch.Tensor):
    """
    img: [H, W]
    returns radial mean profile
    """
    H, W = img.shape
    cy, cx = H // 2, W // 2

    y, x = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing="ij"
    )
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = r.long()

    max_r = int(r_int.max().item()) + 1
    prof = torch.zeros(max_r, device=img.device)
    cnt = torch.zeros(max_r, device=img.device)

    prof.scatter_add_(0, r_int.flatten(), img.flatten())
    cnt.scatter_add_(0, r_int.flatten(), torch.ones_like(img).flatten())

    prof = prof / (cnt + 1e-12)
    return prof

class F_Optics(torch.nn.Module):
    def __init__(self, optics_params, use_fresnel=False, debug=False, debug_dir="dbg_60um_scalarDebye",
                 seed=None, learn_D=False, init_f="random"):
        super(F_Optics, self).__init__()
        # `seed` makes the f_rand initialisation reproducible. Pass
        # config.train_params["seed"]; None keeps the old random behaviour.
        self.seed = seed
        # learn_D=False  -- focal length only. Lenslet diameters stay at the
        #                   constant (min_D + max_D)/2, exactly as every
        #                   reported result used.
        # learn_D=True   -- the (f, D) design: per-lenslet diameters become a
        #                   learnable 5x5 map, initialised at that same
        #                   constant so epoch 0 is identical to learn_D=False.
        #
        # This replaces the separate F_Optics.py that the old (f, D) script
        # used. That module still had the paraxial spherical-wave pupil, which
        # writes the defocus phase n^2 = 1.77x too strong and makes the PSF
        # 1.48x too blurry at z = 30 um. Any (f, D) result produced with it was
        # optimised against the wrong defocus response, so the comparison has
        # to be re-run here, on the Debye pupil, or it is not a comparison.
        self.learn_D = learn_D
        # init_f:
        #   "random"  -- f_rand drawn uniformly in [f_min, f_max] under `seed`.
        #                The multifocal starting point; what the learned runs use.
        #   "nominal" -- every lenslet set to microlens.focal_length (43.1 mm).
        #                Combined with learn_D=False this is the CONVENTIONAL
        #                periodic MLA: one focal length, one diameter. Freeze the
        #                optics on top of it and you have the paper's
        #                "fixed optics + CNN" baseline.
        self.init_f = init_f
        self.debug = debug
        self.debug_dir = debug_dir
        self.optics_params = optics_params
        self.mla_size = optics_params["microlens"]["array_size"]  # number of lenslets in each linear dimension
        self.pixel_size = optics_params["camera"]["pixel_size"]
        self.sensor_size = optics_params["camera"]["sensor_size"]
        self.sampling = optics_params["microlens"]["sampling_FP"]
        self.sampling_AS = optics_params["microlens"]["sampling_AS"]
        self.focal_length = optics_params["microlens"]["focal_length"]
        self.emission_wavelength = optics_params["emission_wavelength"]
        self.D_pitch = optics_params["microlens"]["pitch"]
        self.min_D = optics_params["microlens"]["min_diameter"]  # min allowed diameter
        self.max_D = optics_params["microlens"]["max_diameter"]  # max allowed diameter
        self.M = optics_params["M"] # Simulation grid
        self.L = optics_params["L"]  #side length of the diffuser(meters)
        self.fmin = optics_params["microlens"]["f_min"] 
        self.fmax = optics_params["microlens"]["f_max"]  
        self.objective_focal_length = optics_params["objective"]["focal_length"]
        self.objective_medium_index = optics_params["objective"]["medium_index"]
        self.objective_pupil_diameter = optics_params["objective"]["pupil_diameter"]
        self.demagnification = optics_params["demagnification"]
        self.camera_distance_from_phasemask = optics_params["camera"]["distance_from_phasemask"]
        self.objective_na = optics_params["objective"]["na"]
        self.use_fresnel = use_fresnel  # <- new flag
        self.blur_kernel_size_as = optics_params["blur_kernel_size_as"]
        self.blur_kernel_size_fs = optics_params["blur_kernel_size_fs"]
        self.blur_kernel_size_fp = optics_params["blur_kernel_size_fp"]
        self.blur_kernel_size_sp = optics_params["blur_kernel_size_sp"]
        self.psf_sampling_fs = optics_params["psf_sampling_fs"]
        self.psf_sampling_fp = optics_params["psf_sampling_fp"] #should be equal to MLA sampling
        self.psf_sampling_sp = optics_params["psf_sampling_sp"]
        self.volume_depth_min = optics_params["volume_depth_min"]
        self.volume_depth_max = optics_params["volume_depth_max"]
        self.object_pixel_size = optics_params["object_pixel_size"]
        self.M_T = optics_params["M_T"]  # Simulation grid at object plane
        self.lateral_resolution = optics_params["lateral_resolution"]  # Lateral resolution at object plane (meters)
        self.depth_resolution = optics_params["depth_resolution"]  # Depth resolution at object plane (meters)
        self.padding_fp = optics_params["padding_fp"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize learnable diameters D for each microlens as a Parameter
        ns = int(self.mla_size)  # number of lenslets per side
        #D_init = torch.full((ns, ns), fill_value=self.D_pitch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        #To avoid overlap define the diameter of each microlens to be slightly smaller than the pitch, and allow it to be learnable within a range that ensures no overlap
        self.gap_px = 30
        #self.D1_u = nn.Parameter(0.2 * torch.randn((), device=device))  # scalar

        # ...
        # (If f_rand is to be learnable, similar approach can be taken, but here we keep f_rand fixed per forward.)

        # Initialize f_rand within the range [f_min, f_max] as a trainable parameter
        #f_rand_init = ((self.fmax + self.fmin) /2 )* torch.ones(self.mla_size * self.mla_size, device=device) #uniform start
        #self.f_rand = nn.Parameter(f_rand_init)  

        """
        #uniform init
        self.f_rand = nn.Parameter(
            torch.full((ns * ns,), (self.fmin + self.fmax) / 2, device=device)
        )# Now it's a PyTorch Parameter!
        """

        # --- reproducible f_rand initialisation -----------------------------
        # Generated on the CPU with an explicit generator, then moved to the
        # device: torch.rand on CUDA uses a different RNG stream, so a CPU
        # generator is what makes this reproducible across machines.
        if self.init_f == "nominal":
            # conventional periodic MLA: every lenslet at the nominal focal length
            f_init = torch.full((ns * ns,), float(self.focal_length))
            print(f"init_f='nominal' -- all {ns*ns} lenslets at "
                  f"{float(self.focal_length)*1e3:.3f} mm (conventional MLA)")
        elif self.seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(self.seed))
            f_init = self.fmin + (self.fmax - self.fmin) * torch.rand(ns * ns, generator=g)
        else:
            f_init = self.fmin + (self.fmax - self.fmin) * torch.rand(ns * ns)
        self.f_rand = nn.Parameter(f_init.to(device))

        # --- learnable diameters, only when learn_D=True ---------------------
        # Initialised at the SAME constant the fixed-D path uses, so the (f, D)
        # run starts from the f-only configuration and any difference is
        # something training did, not a different starting point. Registered as
        # a parameter only in this branch, so an f-only checkpoint and an
        # f-only state_dict are unchanged.
        if self.learn_D:
            Dmid_init = (float(self.min_D) + float(self.max_D)) / 2.0
            # `ns`, not self.mla_size: mla_size is the COMPUTED lenslet count
            # and is a float (5.1887 here), which torch.full rejects as a shape.
            # ns = int(mla_size) is the integer grid the rest of the module uses.
            self.D_map = nn.Parameter(
                torch.full((ns, ns), Dmid_init, device=device, dtype=torch.float32))
            print(f"learn_D=True -- D_map {ns}x{ns} initialised at "
                  f"{Dmid_init*1e3:.4f} mm, clamped to "
                  f"[{float(self.min_D)*1e3:.4f}, {float(self.max_D)*1e3:.4f}] mm")

        # --- depth sampling, computed once so the adjoint can reuse it ------
        # forward() slices the volume to D planes and spreads them over
        # [volume_depth_min, volume_depth_max], so the ACTUAL spacing is
        # (max-min)/(D-1), which is not exactly depth_resolution.  Expose it so
        # train_dp.py can hand the same dz to the backprojector instead of
        # depth_resolution (previously they disagreed by ~2%).
        self.D_slices = int(np.ceil(
            (-self.volume_depth_min + self.volume_depth_max) / self.depth_resolution))
        self.dz_used = (self.volume_depth_max - self.volume_depth_min) / (self.D_slices - 1)


    def forward(self, input_image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.debug:
            _ensure_dir(self.debug_dir)
        D_before, H, W = input_image.shape

        D = (-self.volume_depth_min + self.volume_depth_max ) /self.depth_resolution
        D = int(np.ceil(D))  # Number of depth slices
        input_image = input_image[ D_before//2 - D//2 : D_before//2 + D//2 +1 , :, :]  

        ns = int(self.mla_size)  # Light field view grid size
        D_pitch = self.D_pitch  # Microlens pitch (m)

        #fixed focal lengths for all microlenses
        #self.f_rand = ((self.fmax + self.fmin) / 2) * torch.ones(ns * ns, device=input_image.device)
        #learned focal lengths for all microlenses
        f_rand = self.f_rand
        #f_rand = torch.clamp(f_rand, min=self.fmin, max=self.fmax)
        #print(f"f_rand values after clamp: {f_rand*1e3} mm")
  
        # Prepare diameter parameter 

        Dmin = float(self.min_D)
        Dmax = float(self.max_D)
        L = float(self.L)
        g = float(self.gap_px) * float(self.psf_sampling_fp)

        Dmid = (Dmin + Dmax) / 2

        self.D_2 = Dmid 
        self.D_1 = Dmid 

        rows = torch.arange(ns, device=device).view(ns, 1)   # (ns, 1)
        cols = torch.arange(ns, device=device).view(1, ns)   # (1, ns)

        #pattern
        is_D2 = (rows % 2 == 1) & (cols % 2 == 0)   # odd rows AND odd columns
        is_D1 = ~is_D2                              # opposite everywhere
        D1 = self.D_1
        D2 = self.D_2   



        if self.learn_D:
            # the learnable map, as-is. No clamp here -- clamping in the
            # forward pass would zero the gradient at the bounds; train_dp.py
            # clamps the PARAMETER after each optimizer step instead, which
            # keeps it in range without blocking learning.
            D_current = self.D_map.to(device)
        else:
            D_current = torch.zeros((ns, ns), device=device, dtype=torch.float32)
            D_current[is_D2] = D2
            D_current[is_D1] = D1

        # Generate multi-focal microlens array PSF with varying diameters
        mask, Mx_max, pitch_px, t_ulens, centers = multifocal_microlens_array_hex(
            self.blur_kernel_size_fp,
            self.emission_wavelength,
            ns, ns, D_pitch,
            D_current,               # use the tensor of diameters
            f_rand,
            self.psf_sampling_fp,
            device=device,
            max_D=self.max_D         # ensure tile size covers max diameter
        )
        
        #torch.save(centers, os.path.join(self.debug_dir, "mla_centers.pt"))
        if self.debug:
            _save_img(torch.abs(mask), os.path.join(self.debug_dir, "mla_amp.png"), "MLA amplitude")
            _save_img(torch.angle(mask), os.path.join(self.debug_dir, "mla_phase.png"), "MLA phase")
       
        # Calculate the PSF at the sensor plane
        # Generate depths between fmin and fmax (D depths) D is n_slices from the dataset
        depths = torch.linspace(self.volume_depth_min, self.volume_depth_max, D, device=f_rand.device)         
        dz = self.dz_used   # same value as before, now also readable from outside

        psf_list_dbg = []
        depth_list_dbg = []

        ##################################
        #compute all depths
        ##################################
        sensor_img = None
        psf_stack = []

        for i, depth in enumerate(depths):
            pad_u3, df, h_sensor = PSF_lf_Fourier_relay(
                0, 0, depth, 0, mask, self.emission_wavelength,
                self.objective_focal_length, self.objective_medium_index,
                self.objective_pupil_diameter, self.demagnification,
                self.camera_distance_from_phasemask, self.objective_na,
                self.blur_kernel_size_sp, self.psf_sampling_sp,
                self.blur_kernel_size_fp, self.psf_sampling_fp,
                self.blur_kernel_size_as, self.sampling_AS,
                use_fresnel=self.use_fresnel,
            )            


            if self.debug and (i == 0 or i == D//2 or i == D-1):
                _save_img(h_sensor, os.path.join(self.debug_dir, f"psf_z_{i:02d}.png"), f"PSF z index {i}")
                _save_img(h_sensor / (h_sensor.max() + 1e-12),
                        os.path.join(self.debug_dir, f"psf_log10_z_{i:02d}.png"),
                        f"PSF log10 z index {i}", log10=True)
                


            #h_sensor = h_sensor / (h_sensor.sum() + 1e-12)
            
            Mx = int(np.floor(D_pitch / self.psf_sampling_fp))  # tile size 

            scale_factor = self.psf_sampling_fp / self.psf_sampling_sp #this comes from the Fourier optics scaling, the PSF is sampled at psf_sampling_fp but the sensor samples at psf_sampling_sp, so we need to scale the PSF to match the sensor sampling
            pitch_x, pitch_y = int(np.ceil(scale_factor*Mx)) , int(np.ceil(scale_factor*Mx))
            off_x, off_y = 0, 0

            if self.debug:
                print("Mx", Mx, "ns", ns)
                print("pitch_x pitch_y", pitch_x, pitch_y, "off_x off_y", off_x, off_y)


            if self.debug and (i == 0 or i == D//4 or i == D//2 or i == D or i == 3*(D//4) or i == D-1): # visualize the extracted MTFs for a few depth slices
                psf_views = extract_hex_views(h_sensor, nx=ns, ny=ns, pitch_px=pitch_x, patch_px=512)
                mtf_views = []
                mtf_radial_profiles = []
                for k in range(psf_views.shape[1]):
                    psf_patch = psf_views[0, k]
                    mtf_patch = compute_mtf_from_psf(psf_patch)
                    mtf_views.append(mtf_patch)
                    mtf_radial_patch = radial_profile(mtf_patch)
                    mtf_radial_profiles.append(mtf_radial_patch)

                mtf_views = torch.stack(mtf_views, dim=0)   # [Nviews, H, W]
                plot_LF_views_hex(mtf_views.unsqueeze(0), row_counts=[5,4,5,4,5], title_prefix="MTF", save_path=os.path.join(self.debug_dir, f"mtf_views_z_{i:02d}.png"))
                mtf_radial_profiles = torch.stack(mtf_radial_profiles, dim=0)
                plot_hex_1d_profiles(
                    mtf_radial_profiles,
                    row_counts=[5,4,5,4,5],
                    title_prefix="MTF radial",
                    save_path=os.path.join(self.debug_dir, f"mtf_radial_profiles_z_{i:02d}.png")
                )

            I = input_image[i].float()            
            if I.shape[0] != h_sensor.shape[0]:
                I = center_pad_or_crop(I, h_sensor.shape[0])
                    
            if self.debug and (i == 0 or i == D//4 or i == D//2 or i == D or i == 3*(D//4) or i == D-1):
                _save_img(I, os.path.join(self.debug_dir, f"obj_slice_ap1_z_{i:02d}.png"), f"Object slice with ap1 z index {i}")


            # Move PSF center to (0,0) for FFT convolution
            PSF_H0 = torch.fft.ifftshift(h_sensor).to(device)
            psf_stack.append(PSF_H0)

            if self.debug:

                save_dir = os.path.join(self.debug_dir, "psf_h0_tensors")
                os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    PSF_H0.detach().cpu(),
                    os.path.join(save_dir, f"PSF_H0_z_{i:02d}.pt")
                )


            # FFT convolution
            O = torch.real(torch.fft.ifft2(torch.fft.fft2(I).cuda() * torch.fft.fft2(PSF_H0).cuda()))
            if self.debug:
                print("O shape", tuple(O.shape))

            O =  O.clamp_min(0)  # zero out negative values and weight by diameter

            D_map_D1 = torch.zeros_like(D_current)
            D_map_D2 = torch.zeros_like(D_current)
            D_map_D1[is_D1] = D_current[is_D1]
            D_map_D2[is_D2] = D_current[is_D2]

            
            # Scale centers and diameters before use
            centers_sensor = [(cx * scale_factor, cy * scale_factor) for (cx, cy) in centers]
            diameters_sensor = D_current * scale_factor

            if self.debug:
                plot_sensor_with_centers(self.debug_dir, O, centers_sensor, self.psf_sampling_fp, dia=diameters_sensor, index=i)
            

            if self.debug and (i == 0 or i == D//4 or i == D//2 or i == D or i == 3*(D//4) or i == D-1):
                _save_img(O / (O.max() + 1e-12),
                        os.path.join(self.debug_dir, f"sensor_contrib_z_{i:02d}.png"),
                        f"Sensor contribution z index {i}")


            if sensor_img is None:
                sensor_img = O * dz
            else:
                sensor_img = sensor_img + O * dz
            

            if self.debug:
                with torch.no_grad():
                    psf_list_dbg.append(h_sensor.detach().float().cpu())
                    depth_list_dbg.append(float(depth.item()))



        if self.debug:
            with torch.no_grad():
                psf_list_dbg_n = []
                for p in psf_list_dbg:
                    p = p / (p.max() + 1e-12)
                    psf_list_dbg_n.append(p)

            plot_xz_yz_psf(
                psf_list=psf_list_dbg_n,
                depth_list=depth_list_dbg,
                pixel_size_xy_m=float(self.pixel_size),  # or psf_sampling_sp if you did not bin
                save_prefix=os.path.join(self.debug_dir, f"psf_stack_{self.emission_wavelength*1e9:.0f}nm"),
            )

            plot_xz_yz_psf_onelenslet(
                psf_list=psf_list_dbg_n,
                depth_list=depth_list_dbg,
                pixel_size_xy_m= float(self.pixel_size),
                Mx=pitch_x,
                lateral_resolution=float(self.lateral_resolution),
                M_T=float(self.M_T),
                save_prefix=os.path.join(self.debug_dir, f"psf_stack_onelenslet_{self.emission_wavelength*1e9:.0f}nm"),
            )
       


        gc.collect()
        torch.cuda.empty_cache()

        print(sensor_img.shape)

        psf_stack = torch.stack(psf_stack, dim=0)   # [D,Hs,Ws]

        return sensor_img.unsqueeze(0).unsqueeze(0), pitch_x, psf_stack # [1, 1, Hs,Ws]

   
# Instantiate the optical model using config parameters
model_optics = F_Optics(config.optics_params)

# Example usage
if __name__ == "__main__":
    from dataset import MyDataset  # Assuming you have a custom dataset class
    from config import config  # Import the config object


    dataset = MyDataset(
        stack_dir=config.paths["test_stack_dir"],
        config=config,
        transform_3d=config.transform_3d
    )

    L = config.optics_params["L"]
    z = config.optics_params["objective"]["focal_length"]  # 3 mm axial distance
    lambda_ = config.optics_params["emission_wavelength"]

    check_fresnel_validity(L, z, lambda_, n=config.optics_params["objective"]["medium_index"])


    sample_idx = 0 #204  # Select a sample image (e.g., first image in dataset)
    stack_image, lo, h0 = dataset[sample_idx]
    sensor_img, pitch_x = model_optics(stack_image)
    
    plt.imshow(sensor_img[0,0].cpu().detach().numpy())
    plt.savefig('Sensor Image.png')
    plt.close()
    print("pitch_x:", pitch_x)
