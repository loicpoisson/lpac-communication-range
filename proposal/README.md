# Project proposal (due 20 October 2026)

NeurIPS'22 style, 2 pages without references, no abstract, as required by the course.

| File | Content |
|---|---|
| `main.tex` | the proposal: section 1 *Application domain* (problem, dataset, task and metric, why this dataset), section 2 *Graph ML techniques* (model with equation and figure, preliminary findings, planned techniques) |
| `refs.bib` | references |
| `neurips_2022.sty` | official NeurIPS 2022 style file |
| `figures/results.pdf` | results panel of Figure 1, generated from the repository's result CSVs |
| `figures/make_results_figure.py` | regenerates `figures/results.pdf` (`python proposal/figures/make_results_figure.py`) |
| `main.pdf` | compiled version |

## Compile

- **Overleaf**: upload the whole `proposal/` folder, set the main document to `main.tex`, compiler pdfLaTeX.
- **Locally**: `tectonic -X compile main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).

## Before submitting

- Check the 2-page limit: the references must start on page 3 at the latest.
- Check the author emails (the SUNet ID `lpoisson` is assumed).
- Numbers in the text come from `results.csv`, `results_2x2.csv`, `results_train/` and `results_variants/`; if a result changes, regenerate the figure and update the text.
