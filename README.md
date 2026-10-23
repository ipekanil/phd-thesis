# Wavefront Coding for Computational Microscopy: documentation site

Source for **https://ipekanil.github.io/phd-thesis/**, the public documentation of the PhD work of Ipek Anil Atalay Appak (Tampere University / Université de Rennes, INRIA). It explains the physics, methods and results of the thesis and shows how to reproduce them.

- **Part 1: Extended depth of field (EDoF)**: Publications I and II (MANTIS). Complete.
- **Part 2: Fourier light field microscopy (FLFM)**: Publication III (in submission). Simulation results and full code in `code/flfm/`.

Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Equations use MathJax.

## Preview locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve            # open http://127.0.0.1:8000
```

## Publish on GitHub Pages (one-time setup)

1. Create a **public** repository named `phd-thesis` under the `ipekanil` account.
2. Push this folder to its `main` branch:
   ```bash
   git init && git add . && git commit -m "Initial documentation site"
   git branch -M main
   git remote add origin https://github.com/ipekanil/phd-thesis.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. The workflow in `.github/workflows/deploy.yml` builds the site on every push to `main`. After the first run (1–2 minutes), the site is live at `https://ipekanil.github.io/phd-thesis/`.

## Repository layout

```text
mkdocs.yml                  site configuration and navigation
docs/
  index.md                  home page
  background/               physics background (DoF, Fourier optics, wavefront coding, flat optics)
  edof/                     EDoF method, results, reproduction, limitations
  flfm/                     FLFM method, results, reproduction, limitations
  resources/                publications, BibTeX, glossary
  assets/figures/papers/    figures from Publications I–II and the thesis (CC BY 4.0)
  assets/figures/generated/ figures produced by examples/toy_forward_model.py
  assets/figures/flfm/      FLFM schematic and figures produced by examples/flfm_figures.py
code/
  flfm/                     FLFM end-to-end training framework (see code/flfm/README.md)
examples/
  toy_forward_model.py      NumPy toy model: PSF/MTF vs defocus, cubic mask, sampling
  design_calculator.py      defocus coefficient and sampling pitch (reproduces Pub. I Table 1)
  flfm_figures.py           FLFM toy-model figures and result charts
```

## Editing

- Each page is a Markdown file in `docs/`. The navigation is in `mkdocs.yml` under `nav:`.
- Inline math: `\( ... \)`. Display math: `\[ ... \]`.
- Figures: put images in `docs/assets/figures/` and use
  `<figure markdown="span">![alt](../assets/figures/...){ loading=lazy }<figcaption>…</figcaption></figure>`.
- Regenerate the toy figures: `python examples/toy_forward_model.py` and `python examples/flfm_figures.py`.

## Adding pages

Add a Markdown file next to the existing ones in `docs/` and list it under `nav:` in `mkdocs.yml`. The site build ignores `code/`, so code changes never break the documentation.

## Licence

Text © Ipek Anil Atalay Appak. Figures from the publications and thesis are © the authors, reused under CC BY 4.0. Code in `examples/` and `code/flfm/`: MIT.
