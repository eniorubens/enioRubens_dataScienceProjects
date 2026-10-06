# NASA C-MAPSS — remaining useful life prediction — FD001

[Português](README.md) · [English HTML report](site/en.html) · [Technical summary](notebooks/en-US/27_conclusion_delivery_fd001.ipynb)

This portfolio project estimates the remaining useful life (RUL) of simulated turbofan engines in operating cycles. The FD001 stage uses Extra Trees and temporal features built exclusively from current and past measurements. The delivery supports reproducible analysis and local point predictions.

## Data source

+The project uses the **Turbofan Engine Degradation Simulation Data Set**, provided by the Prognostics Center of Excellence (PCoE) at NASA Ames Research Center (Saxena and Goebel, 2008). NASA stands for **National Aeronautics and Space Administration**, the United States agency for space and aeronautics. In this project's title, it identifies the data source; the analysis and model are independent portfolio work. [Official NASA repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

The time series were generated with **C-MAPSS — Commercial Modular Aero-Propulsion System Simulation**, which models the operation and degradation of commercial aircraft turbofan engines. These are simulated data: training trajectories continue until failure, while test histories stop earlier and the true RUL at the last cycle is supplied separately. [Official dataset catalog](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

This project uses **FD001**, with 100 training engines and 100 test engines, one operating condition and one failure mode: high-pressure compressor (HPC) degradation. This controlled setting supports remaining-life prediction research; it does not demonstrate performance on a real fleet.

**Dataset reference:** Saxena, A.; Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames, Prognostics Data Repository. [Original source and credit](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

Original local files are preserved in `data/raw/CMAPSSData.zip` and `data/raw/dataset`. The initial audit and the distribution URL used are recorded in `CONTEXT_HANDOFF.md`. That document describes the starting phase, not the final project state.

## Recorded results

| Model | MAE (cycles) | RMSE (cycles) | Mean signed error (cycles) |
| --- | ---: | ---: | ---: |
| Extra Trees | 16.23 | 23.80 | +10.08 |
| XGBoost | 17.29 | 25.15 | +9.31 |
| HistGradientBoosting | 17.77 | 27.01 | +8.83 |

These are the recorded official benchmark results for 100 test engines, using one prediction at the last available cycle of each truncated trajectory. All three candidates had frozen hyperparameters before the benchmark. Extra Trees was selected after comparing these test results; this is a comparative benchmark, not another independent evaluation after winner selection. No further tuning was performed using these results.

Positive signed error indicates overestimation of remaining life. The point prediction is not a safe operating limit.

## Reading path

Start with [notebook 27 — conclusion and delivery](notebooks/en-US/27_conclusion_delivery_fd001.ipynb) or the [English HTML report](site/en.html). The principal evidence is preserved in the canonical Portuguese notebooks: [16 — temporal ablation](notebooks/pt-BR/16_ablation_atributos_temporais_rul_fd001.ipynb), [18 — model screening](notebooks/pt-BR/18_triagem_modelos_atributos_temporais_rul_fd001.ipynb), [19 — optimization](notebooks/pt-BR/19_otimizacao_optuna_atributos_temporais_rul_fd001.ipynb), [21 — benchmark](notebooks/pt-BR/21_benchmark_oficial_finalistas_atributos_temporais_rul_fd001.ipynb), [22 — selection and errors](notebooks/pt-BR/22_decisao_final_analise_erro_extra_trees_fd001.ipynb), [24 — feature importance](notebooks/pt-BR/24_importancia_features_extra_trees_fd001.ipynb), [25 — alerts and costs](notebooks/pt-BR/25_alertas_modelo_final_extra_trees_fd001.ipynb), and [26 — uncertainty](notebooks/pt-BR/26_incerteza_extra_trees_final_fd001.ipynb).

The remaining notebooks preserve experiment history. Running all notebooks is unnecessary for inference; some launch lengthy searches or reproduce experiments preceding the final configuration. English coverage intentionally focuses on the README, HTML report and summary notebook.

## Artifact and input contract

The pipeline is saved at `outputs/models/extra_trees_fd001_v1/model.joblib`. Its `metadata.json` records features, parameters, versions, hashes and checks. Of 205 offered features, 93 were selected. Causal windows span 5, 10, 20, 40 and 80 cycles, with rolling means, population standard deviations and per-engine deltas. Current measurements participate in the windows.

The input CSV must contain `unit`, `cycle`, `setting1`, `setting2`, `s2`, `s3`, `s4`, `s6`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20` and `s21`. Required columns must be numeric and contain no missing or infinite values. Each engine must have consecutive cycles from cycle 1 through the current measurement. Rows may be unordered because histories are sorted by engine and cycle. The default output contains only the most recent prediction per engine.

The included example comes from a training-data prefix and demonstrates interface behavior and pipeline equivalence; it does not measure generalization.

## Run a prediction

From the project root, with a compatible Python environment activated:

```powershell
python scripts/predict_final_extra_trees_fd001.py --input outputs/models/extra_trees_fd001_v1/historico_exemplo.csv --output outputs/reports/prediction_demo_fd001.csv
```

Output columns remain `unit`, `cycle` and `rul_predito` (predicted RUL) to preserve the canonical interface. Add `--all-cycles` to predict every cycle in the supplied history. Existing output files are preserved; choose a different output filename to rerun the example.

The original environment specification is in `environment.yml`; actual delivery versions are recorded in `metadata.json`. The environment file includes a local editable `multilang` path, which must be adjusted on another computer. Joblib artifacts require a compatible environment and should only be loaded from a trusted local source.

## Rebuild the artifact

`scripts/export_final_extra_trees_fd001.py` fits the frozen pipeline once using FD001 training data and the existing Optuna study. It does not start another search. An existing delivery directory is preserved, and exporting again to that directory is refused. The script checks prediction equivalence after serialization, consistency with training feature construction, invariance of past predictions when future history is appended, and rejection of invalid histories.

## Interpretation and limitations

Permutation importance identified observed cycle and rolling standard deviations, particularly over 80 cycles, as influential. Correlated features may share importance, so individual scores are not exclusive measures of information. Explainability is predictive and does not identify physical causes of failure.

Alerts were analyzed on complete development OOF trajectories. An illustrative useful lead-time window of 5–30 cycles and three-cycle confirmation were used. A threshold of 20 cycles produced timely alerts for all 100 engines, with median lead time of 16 cycles. The folds also participated in hyperparameter selection; this is not an industrially validated policy. Costs are relative and hypothetical.

Empirical error intervals were calibrated and evaluated on different engines within OOF folds. Mean within-engine cycle coverage was 85.9%, mean width was 91.3 cycles, and 46% of trajectories were fully covered. The 90% calibration reference is not a coverage guarantee. The exported artifact supplies point predictions only; these intervals were not validated for the model refitted on all training data.

The delivered scope is simulated FD001. Generalization to FD002–FD004 and industrial validation remain extensions. Real maintenance decisions require operational conditions, costs and minimum lead times agreed with the responsible team.

## HTML report and publication

The report is available in [Portuguese](site/index.html) and [English](site/en.html), with local interactive charts and downloadable summaries. Rebuild it with `python scripts/build_portfolio_pages.py`. No model is fitted by this command.

Website files are isolated in `site/`; personal references in `docs/` are excluded from the public repository. The project lives under `CMAPSS_Predictive_Maintenance/` in the portfolio repository. The root workflow, `Publicar portfólio FD001`, publishes only the static report and entry page, without training models. [Publication guide](site/PUBLISHING_GITHUB_PAGES.en.md)

[Published report (English)](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/en.html) · [Português online](https://eniorubens.github.io/enioRubens_dataScienceProjects/CMAPSS_Predictive_Maintenance/)
