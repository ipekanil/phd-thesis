# Physics background

These pages explain the optics behind the thesis at the level needed to follow the methods. A physics or engineering undergraduate should be able to follow them. Each page is short and ends with what matters for the next one.

| Page | Main question | Key result |
|---|---|---|
| [Shallow depth of field](shallow-dof.md) | Why does a high-NA microscope see only a thin slice? | \( \mathrm{DoF} \propto \lambda / \mathrm{NA}^2 \) |
| [Fourier optics of a microscope](fourier-optics.md) | How does the pupil set the PSF, and how does defocus enter? | \( h = \lvert \mathcal{F}\{P\} \rvert^2 \), defocus = quadratic pupil phase \( \psi\rho^2 \) |
| [Wavefront coding](wavefront-coding.md) | How can a phase mask make the PSF insensitive to defocus? | Keep the MTF non-zero and similar across depth, then decode |
| [Flat optics](flat-optics.md) | How is a phase profile turned into a physical device? | DOE height map or metasurface nanopillar map |
| [Try it: toy simulation](toy-simulation.md) | Can I see this myself? | A 150-line NumPy script that recreates the figures |

!!! tip "Notation"
    Symbols are collected in the [glossary](../resources/glossary.md). Coordinates follow the thesis: \((x_0,y_0)\) object, \((x_1,y_1)\) pupil / Fourier plane, and later \((x_3,y_3)\) relayed Fourier plane and \((x_4,y_4)\) sensor.
