# Claim Frequency per Policy

[English version](README.en-US.md) · [Versão em português](README.pt-BR.md)

## 1. Summary and result

This project studies **claim frequency** on the French motor insurance benchmark `freMTPL2freq` (OpenML 41214, v1; 678,013 records). The analysis estimates expected claims per **100 policy-years** across vehicle, driver, and regional profiles. It also shows where predictions track observed totals and where calibration errors remain.

The reserved test set was used for one official evaluation. It contains 101,859 records, 53,839.09 policy-years, and 5,415 claims. The **Poisson model with a fixed exposure offset**, frozen after training, reached a mean Poisson deviance of 0.31902, compared with 0.33079 for the global reference rate. The paired difference was −0.01177 (95% CI −0.01317 to −0.01018; by profile, −0.01338 to −0.01028). The test therefore supports a Poisson gain in mean deviance.

For aggregate calibration, the observed rate was 10.058 claims per 100 policy-years. Poisson predicted 10.108 and the global rate predicted 10.084. The 95% CI for predicted − observed includes zero for both candidates; it ranges from −0.233 to +0.317 for Poisson and from −0.253 to +0.288 for the baseline. These intervals assess each candidate separately; they do not test equivalence between them or compare their point absolute errors, which were 0.050 for Poisson and 0.027 for the global rate.

The main limitation appears in calibration by coverage duration. In the `(0; 0.02]` band, the observed rate was 255.51, while Poisson predicted 14.04 claims per 100 policy-years. Conclusions are limited to this benchmark, and `production_eligible = false`.

## 2. Data and question

The `freMTPL2freq` dataset is available from OpenML under a CC0 license. Each record represents the coverage identified by `IDpol`, with `ClaimNb` for the claim count and `Exposure` for exposure time in years. The evaluation first tests whether the frozen Poisson outperforms the global rate on records that were not used during development. It then identifies the exposure bands, vehicle ages, and regions where predicted rates depart from observed rates.

## 3. Repository layout

| Path | Content |
| --- | --- |
| `src/freq_sinistros/` | Reusable logic (reading, audit, partition, preprocessing, models, metrics, freezing, final evaluation, plots, PT/EN labels) |
| `notebooks/pt_br/`, `notebooks/en_us/` | Four notebooks per language, with the same code: `01_eda_treino`, `02_comparacao_modelos`, `03_diagnostico_zeros`, `04_avaliacao_final` |
| `scripts/` | Notebook generation and execution |
| `tests/` | Synthetic tests (no network or local data) |
| `outputs/etapa_01` … `etapa_05` | **Aggregated, versionable** results, manifests, hashes and execution records |
| `reports/` | Technical reports per stage and the evaluation protocol |
| `data/` | **Local, not versioned**: the raw ARFF (`data/raw`) and the partition assignments (`data/splits`) |

## 4. Reading map

The [executive report](https://eniorubens.github.io/enioRubens_dataScienceProjects/Claim_Frequency_per_Policy/en.html) summarizes the problem, modeling decision, and main results. It is published on GitHub Pages, and its source, `site/en.html`, can also be opened locally in a browser. For the complete analysis, read the notebooks in the order below:

1. [Training EDA](notebooks/en_us/01_eda_treino.ipynb) — presents the variables, partition contract, and exploration conducted on training data only.
2. [Model comparison](notebooks/en_us/02_comparacao_modelos.ipynb) — explains the analytical table and compares the global rate, Poisson, and NB2 on validation.
3. [Zero diagnostics and decision](notebooks/en_us/03_diagnostico_zeros.ipynb) — assesses whether ZIP/ZINB is needed, documents the Poisson choice, and verifies the freeze.
4. **[Final evaluation](notebooks/en_us/04_avaliacao_final.ipynb)** — concentrates the conclusion on the reserved test set. It can be consulted without the ARFF and without refitting the models, using only the published aggregate artifacts.

## 5. Environment

Conda environment (`environment.yml`: Python 3.12, numpy, pandas, scipy, statsmodels, matplotlib, nbclient, pyyaml, ruff, pytest). Notes for this machine: `CONDA_SSL_VERIFY=truststore` to create the environment, OpenBLAS as BLAS (MKL crashed `numpy.linalg`) and `PYTHONNOUSERSITE=1` when running.

```powershell
$env:CONDA_SSL_VERIFY = "truststore"
conda env create -f environment.yml
conda activate freq-sinistros
$env:PYTHONNOUSERSITE = "1"
```

To pin the exact versions used in stages 1 to 5 (Python 3.12.16, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2, OpenBLAS 0.3.34), there are two lock files:

- `environment.lock.yml`: an export of the environment (`conda env export --no-builds`) with exact versions and no builds, useful as a reference and for other platforms. Note: it records the channels `rapidsai`, `defaults` and `conda-forge`, inherited from the local machine configuration, while `environment.yml` uses `conda-forge` only.
- `conda-win-64.lock.txt`: an explicit package list for win-64, to recreate the environment exactly: `conda create -n freq-sinistros --file conda-win-64.lock.txt`, followed by `pip install -e . --no-deps`.

Use `environment.yml` for a fresh, flexible environment; use the win-64 lock when byte-for-byte reproduction needs the same versions. The project requires Python 3.12 or later and is distributed under the MIT license (`LICENSE`).

## 6. Consult the official results (no local data)

From the project root, the [final notebook](notebooks/en_us/04_avaliacao_final.ipynb) can be run with the environment from section 5 and the artifacts already saved in `outputs/`. Preserve the approved files in `outputs/etapa_04` and `outputs/etapa_05`, including `official_run.lock`.

```powershell
python -I -m freq_sinistros --root . verify-stage5     # offline check of the official aggregates
python scripts/run_notebooks.py --final               # runs only the two final notebooks, with the stored aggregates
```

- `verify-stage5` performs the full check, the same one as the `04_avaliacao_final` notebook: the record (`status = completed`), the checksums of all files listed in `SHA256SUMS_etapa05.txt` (and that there are no files outside the list), the reconciliation, the frozen model and the script against the hashes in the record, the integrity gate, the number of test records (checked against stage 2) and the presence of the lock. If the record, the list or a file is missing, it ends with exit code 1 and a message, with no traceback.
- CLI exit codes: 0 success; 1 verification failure; 2 audit mismatch; 3 acquisition refused; 4 `stage5` refused; 5 `check-stage5` or `verify-freeze` refused after the lock.
- Local inputs of the other notebooks: `01_eda_treino` reads the ARFF and the assignments (`data/raw`, `data/splits`); `02_comparacao_modelos` and `03_diagnostico_zeros` check the SHA-256 of the ARFF. Only the `04` pair runs without them.

## 7. Reproduce stages 1 to 4 (isolated copy)

Use the conda environment from section 5 and a copy outside the workspace. The download requires network access (≈ 36 MB). In the recorded environment, the download took about 5 min and the workflow took about 10 min. Stages 1 through 3 are reproduced in this copy, and stage 4 is checked by restoring the approved model on validation. For the final notebooks, follow section 6 with the official aggregates.

```powershell
$copy = Join-Path $env:TEMP ("freq-sinistros-" + [guid]::NewGuid())
robocopy . $copy /E /XD data .venv .pytest_tmp __pycache__ etapa_05
if ($LASTEXITCODE -ge 8) { throw "Robocopy failed." }
Set-Location $copy
python -I -m freq_sinistros --root . acquire           # downloads the ARFF (network), checks the OpenML MD5
python -I -m freq_sinistros --root . audit             # stage 1: audit
python -I -m freq_sinistros --root . partition         # stage 2: 70/15/15 partition by IDpol
python -I -m freq_sinistros --root . eda               # stage 3: EDA on training only
python -I -m freq_sinistros --root . compare           # stage 3: global rate, Poisson and NB2 on validation
python -I -m freq_sinistros --root . verify-freeze     # stage 4: restores the approved frozen model and reproduces validation
python -I -m freq_sinistros --root . check-stage5      # integrity gate (does not make the test set available to the evaluator)
python -m pytest ; ruff check .
```

On Linux or macOS, the bash equivalent (same Python commands):

```bash
copy="$(mktemp -d)/freq-sinistros"
mkdir -p "$copy"
rsync -a --exclude data --exclude .pytest_tmp --exclude __pycache__ --exclude outputs/etapa_05 ./ "$copy"/
cd "$copy"
python -I -m freq_sinistros --root . acquire
python -I -m freq_sinistros --root . audit
python -I -m freq_sinistros --root . partition
python -I -m freq_sinistros --root . eda
python -I -m freq_sinistros --root . compare
python -I -m freq_sinistros --root . verify-freeze
python -m pytest && ruff check .
```

Byte-for-byte identity with the approved artifacts was verified on Windows: the JSON and CSV files in `outputs/` were written with CRLF line endings (the writers use the platform separator). On Linux or macOS the regenerated files will have LF; compare them ignoring `\r` (for example, `diff <(tr -d '\r' < a) b`). That is why `check-stage5` is not in the bash block: the gate would compare the hashes of the six regenerated stage 3 files and refuse them over a line-ending difference, not a content difference.

- `check-stage5` and `verify-freeze` only run where there is no lock: the copy excludes `etapa_05`, so they run normally there. In the workspace, with `outputs/etapa_05/official_run.lock`, both refuse with exit code 5 and write nothing (they would rewrite sealed artifacts).
- The copy carries the approved `outputs/etapa_04` (`frozen_model.json`, `stage5_script.json`). Do **not** run `stage4` over those files: it regenerates the freeze, and the bytes of the JSON may change because they include platform information; the stage 5 gate compares the hashes with the approved values and would refuse the regenerated copy.
- The integrity check compares hashes byte for byte with the approved artifacts. If reproduction yields different hashes, check the environment versions against `conda-win-64.lock.txt` and `environment.lock.yml` (section 5).
- Never run `stage5` in the workspace: the lock `outputs/etapa_05/official_run.lock` refuses a second evaluation. The copy deliberately excludes `etapa_05`.
- The reader parses the whole ARFF in memory; the test rows are made available to the stage 5 evaluator only after the integrity gate and the lock.

## 8. Stages

1. **Acquisition and audit**: ARFF verified by MD5 and SHA-256; 0 missing values; unique `IDpol`.
2. **Evaluation design**: deterministic partition by `IDpol` hash (474,581 / 101,573 / 101,859); protocol of metrics, segments, bootstrap and decision rules.
3. **EDA and comparison**: global rate, Poisson and NB2 on validation; sensitivities.
4. **Zeros, decision and freezing**: fixed-offset Poisson chosen by the approved rule; no ZIP/ZINB; model frozen and verified.
5. **Final evaluation**: frozen script applied once to the test set (`stage5`).

## 9. Limitations

- The fixed offset **underestimates very short exposures** and overestimates long ones; no recalibration was applied.
- Results are internal to the benchmark: no temporal evidence, no `BonusMalus` (unknown measurement time) and no zero-inflated models.
- The 95% intervals are conditional on the fitted models.

## 10. Verification and CI

- **Local checks** (Ruff, pytest with coverage, parity of the four notebook pairs and of the READMEs): passed.
- **Run with real data**: completed and recorded in `outputs/` and `reports/`.
- **Checksum lists**: the `SHA256SUMS*` lists of all stages match and are tested by `tests/test_outputs_integrity.py`. The notebook execution records live in `outputs/notebook_execution_etapa03.json`, `outputs/notebook_execution_etapa04.json` and `outputs/notebook_execution_etapa05.json`, outside the sealed folders. The integrity of the official evaluation is checked by `verify-stage5`.
- **Stage 5 line endings**: `SHA256SUMS_etapa05.txt` has CRLF line endings (written on Windows; it is a sealed file and was not rewritten). For GNU `sha256sum -c`, convert first, for example `tr -d '\r' < SHA256SUMS_etapa05.txt | sha256sum -c`. The CLI and the test accept both formats.
- **CI**: the project is published as the `Claim_Frequency_per_Policy` folder of the portfolio repository [enioRubens_dataScienceProjects](https://github.com/eniorubens/enioRubens_dataScienceProjects). The root `Tests` workflow runs this folder's `pytest` as its own matrix suite (Python 3.12, `pip install -e ".[dev]"`), registered in `.github/ci/test-projects.json`, only when this folder changes (or in the full battery), without mixing dependencies with the other projects. `tests/test_ci_config.py` checks this registration. CI uses `pip`, like the other portfolio suites; the local environment remains conda (section 5). Ruff is checked locally.
- **License**: MIT (`LICENSE`), for the code; the OpenML data are CC0 and stay outside the repository.

## 11. References

Dutang and Charpentier (2018), *CASdatasets: Insurance datasets* (`freMTPL2freq` dictionary); Cameron and Trivedi (2013), *Regression Analysis of Count Data*; Wilson (2015), *Economics Letters*, 127, 51–53. Full references at the end of each notebook.
