# Publishing with GitHub Pages

The Portuguese entry point is `site/index.html`; the English version is `site/en.html`. Both pages use local files from `site/assets`, including the plotting library, data and downloadable summaries. The language button switches between the editions.

The destination is `eniorubens/enioRubens_dataScienceProjects`, under `CMAPSS_Predictive_Maintenance/`. The English report URL is https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/en.html; the Portuguese entry point omits `en.html`.

## Rebuild and preview

Run `python scripts/build_portfolio_pages.py` from the project root using a Python environment with pandas and Plotly. The command reads saved evidence and does not train a model. Rebuild after changing a README or summary notebook to refresh the downloadable copies.

For a local preview, run `python -m http.server 8765 --directory site --bind 127.0.0.1` and open `http://127.0.0.1:8765/`. Press Ctrl+C in the terminal to stop the server.

## Publish

The active workflow is `.github/workflows/pages.yml` at the portfolio repository root. It packages `CMAPSS_Predictive_Maintenance/site/` and an entry page from `pages-publicacao/`. In Settings → Pages, select GitHub Actions as the source. In Actions, run Publicar portfólio FD001 manually. It excludes personal references and performs no training. The workflow inside the project directory is only a standalone deployment template; GitHub does not execute nested workflows.

Existing projects are preserved. Source code, notebooks, evidence, Optuna studies and the final model are stored in the repository; only the static report is deployed to Pages. Personal references, caches and duplicate dataset ZIP archives are not published.

## Editorial scope

Portuguese remains the canonical edition. The README, HTML report and notebook 27 are bilingual; the other notebooks retain their original Portuguese edition. Translation does not change feature identifiers, command-line arguments or model behavior.

The site offers copies of the README and notebook for download. Their internal relative links target the complete repository structure; read them on GitHub or preserve that structure when downloading the project.

Green denotes timely alerts in the alert chart and the Intermediate reference scenario in the cost chart. Green in the cost chart does not mean that the scenario is always cheapest.

Official benchmark values retain the previous record. Alerts, hypothetical costs and empirical intervals remain exploratory development analyses. Publication performs no further training, model selection or independent validation.
