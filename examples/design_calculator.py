"""
EDoF design calculator: defocus coefficient and Fourier-plane sampling pitch.

Implements the two design equations of Publication I (Nanophotonics 2023):

    psi_z  = (pi / lambda) * (1/z - 1/z0) * r^2          (defocus coefficient)
    ds    <= pi * r / (8 * psi_max)                        (phase-sampling bound)

with z0 = f1 (objective focal length, object at the front focal plane) and
r = pupil radius. In the thesis the pupil radius is r = NA * f1 (at the
objective pupil) or |Mp| * f1 * NA at a relayed Fourier plane.

Running the script reproduces the psi_max and Δs columns of Table 1 of
Publication I (Table 3.1 of the thesis), using the effective NA values
reported in the thesis.

Usage:
    python examples/design_calculator.py
    python examples/design_calculator.py --f1 4 --na 0.134 --dz 1.5 --wl 550
"""

from __future__ import annotations

import argparse
import math


def psi(z_mm: float, z0_mm: float, r_mm: float, wl_nm: float) -> float:
    wl_mm = wl_nm * 1e-6
    return math.pi / wl_mm * (1.0 / z_mm - 1.0 / z0_mm) * r_mm**2


def design(f1_mm: float, na: float, dz_mm: float, wl_nm: float):
    r = na * f1_mm
    psi_near = psi(f1_mm - dz_mm, f1_mm, r, wl_nm)
    psi_far = psi(f1_mm + dz_mm, f1_mm, r, wl_nm)
    psi_max = max(abs(psi_near), abs(psi_far))
    ds_um = math.pi * (r * 1e3) / (8 * psi_max)
    return r, psi_max, ds_um


# (objective, f1 [mm], NA_eff, half DoF [mm], reported psi_max, reported ds [um])
PUB1_TABLE = [
    ("RMS4X-PF",    45.0, 0.041, 0.100,   0.97, 740),
    ("RMS4X-PF",    45.0, 0.041, 1.000,   9.877, 73),
    ("RMS20X-PF",    9.0, 0.159, 0.100,  14.69,  38),
    ("Mitutoyo50X",  4.0, 0.134, 0.100,  10.48,  20),
    ("Mitutoyo50X",  4.0, 0.134, 1.000, 136.24,   1.54),
    ("Mitutoyo50X",  4.0, 0.134, 1.500, 245.54,   0.858),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1", type=float, help="objective focal length [mm]")
    ap.add_argument("--na", type=float, help="(effective) numerical aperture")
    ap.add_argument("--dz", type=float, help="half depth range, target DoF = ±dz [mm]")
    ap.add_argument("--wl", type=float, default=550.0, help="wavelength [nm]")
    a = ap.parse_args()

    if a.f1 and a.na and a.dz:
        r, pm, ds = design(a.f1, a.na, a.dz, a.wl)
        print(f"pupil radius r = {r:.4f} mm, psi_max = {pm:.2f} rad, Δs <= {ds:.3f} µm")
        return

    print("Reproducing Publication I, Table 1 (λ = 550 nm)\n")
    print(f"{'objective':<12} {'NA_eff':>6} {'DoF':>9} | {'psi_max':>8} {'(paper)':>8} | {'Δs µm':>8} {'(paper)':>8}")
    for name, f1, na, dz, psi_p, ds_p in PUB1_TABLE:
        _, pm, ds = design(f1, na, dz, 550.0)
        dof = f"±{dz*1e3:.0f} µm" if dz < 1 else f"±{dz:g} mm"
        print(f"{name:<12} {na:>6.3f} {dof:>9} | {pm:>8.2f} {psi_p:>8.2f} | {ds:>8.3f} {ds_p:>8}")
    print("\nSmall differences come from rounding of f1 and NA_eff.")


if __name__ == "__main__":
    main()
