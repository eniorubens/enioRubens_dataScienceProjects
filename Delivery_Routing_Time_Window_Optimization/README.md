# Delivery Routing & Time-Window Optimization (VRPTW)

[Português](README.pt-BR.md) · [Executive report](site/en.html)

This project studies how to assign customers to vehicles and sequence deliveries under vehicle capacity and customer time windows, minimising the fleet first and total distance second. It is a prescriptive analysis whose output is an independently verifiable route plan comparable against published reference solutions. Reusable code resides in the `delivery_routing_vrptw` package (`src/`), while the analysis is reported in bilingual Portuguese and English notebooks that share the same executable code.

The data come from the synthetic Solomon benchmark (1987). Within the portfolio, the project provides thematic continuity to the Olist marketplace study in the delivery domain, relying exclusively on benchmark data because Olist contains no fleet, capacity, time-window or route records. Because the method is validated only on synthetic instances (`production_eligible = false`), real operations would require road distances, an actual fleet, measured service times and intraday re-optimisation.

## Question

How should customers be allocated to vehicles and deliveries ordered so that fleet size and then distance are minimised, without violating capacity, time windows or the return to the depot?

## Data

The study uses the 56 Solomon instances with 100 customers (families C1, C2, R1, R2, RC1 and RC2), obtained from SINTEF TOP (`data/raw/sintef/solomon-100.zip`, downloaded manually because the site blocks automated clients) and from CVRPLIB (`data/raw/cvrplib_solomon_100/Solomon.7z`). Both sources are numerically identical, with provenance and SHA-256 hashes recorded in each folder's `MANIFEST.md`. Reference solutions come from SINTEF (SINTEF, 2008; `data/reference/sintef_bks_100.csv`, transcribed from the SINTEF page) and CVRPLIB (CVRPLIB, s.d.; `data/reference/cvrplib_bks_100.csv`). No explicit license was found in the source files; citations should credit Solomon (1987), *Operations Research* 35(2), SINTEF TOP and CVRPLIB. Distances are planar Euclidean as defined in the data contract, without Google Maps, traffic feeds or paid APIs.

### Obtaining the data (third-party files are not redistributed)

The public version of this repository omits third-party files: the SINTEF archive (`data/raw/sintef/solomon-100.zip`), the CVRPLIB archives (`data/raw/cvrplib_solomon_100/*.7z`), the saved SINTEF page (`docs/Solomon-html descriptions.pdf`) and the CVRPLIB solution routes (`data/reference/cvrplib_bks_100.csv`, regenerated from the CVRPLIB archives). The `MANIFEST.md` files, the transcribed SINTEF table of vehicles and distances (`data/reference/sintef_bks_100.csv`) and the development/evaluation split remain in the repository.

~~~
python scripts/fetch_data.py              # verifies SHA-256, downloads the CVRPLIB archives, rebuilds the CVRPLIB csv
python scripts/fetch_data.py --check-only # only verifies the files already present (no network)
~~~

Because the SINTEF site blocks automated clients via Cloudflare, `solomon-100.zip` must be downloaded manually from https://www.sintef.no/projectweb/top/vrptw/100-customers/ and saved to `data/raw/sintef/solomon-100.zip` (the script prints the expected SHA-256 digest). Tests that do not depend on these files run without them.

Sources to cite are Solomon, M. M. (1987), Algorithms for the vehicle routing and scheduling problems with time window constraints (*Operations Research, 35*(2), 254-265); SINTEF (2008), *VRPTW - 100 customers: Solomon benchmark, best known solutions* (SINTEF TOP, https://www.sintef.no/projectweb/top/vrptw/100-customers/); and CVRPLIB (s.d.), *VRPTW instances - Solomon set* (PUC-Rio, https://galgos.inf.puc-rio.br/cvrplib/).

### Objective conventions (central comparability risk)

| | SINTEF (used here) | CVRPLIB |
|---|---|---|
| Objective | hierarchical: vehicles, then distance | distance only |
| Distance | double precision, total rounded to 2 decimals | truncated to 1 decimal |

An independent verifier showed that 8 of the 56 CVRPLIB solutions become infeasible under exact distance and that SINTEF uses fewer vehicles on 37 instances. Results are therefore compared exclusively with SINTEF under its hierarchical convention, and the distance gap is reported only when the fleet equals that of the reference.

## Reproduction

The conda environment relies strictly on `conda-forge`, while OR-Tools is installed from PyPI:

~~~
set CONDA_CHANNELS=conda-forge
set CONDA_CHANNEL_PRIORITY=strict
conda env create -f environment.yml
conda activate delivery-routing-vrptw
~~~

The `environment.yml` specification installs the package in editable mode and sets `MKL_THREADING_LAYER=SEQUENTIAL` to prevent a numpy/MKL crash on Windows; pinned package versions are recorded in `environment.lock.yml`. Unit tests use synthetic fixtures, whereas contract tests read the notebooks and `data/reference/split.csv`, requiring neither network access nor a graphical interface:

~~~
python -m ruff check src tests scripts
python -m pytest -q --cov=delivery_routing_vrptw
~~~

Experiment parameters (solver time limit, budgets, scenario factors and replicates) reside in `config/experiment.toml` and can be overridden via command-line arguments. The table summarises the executed experiments in chronological order:

| Step | Script | Instances |
|---|---|---|
| Development/evaluation split (seed 20261117) | `scripts/make_split.py` | all |
| Gate (baselines + OR-Tools) | `scripts/run_gate.py` | 1 dev per family |
| Pre-registered tournament (18 configurations) | `scripts/run_dev_grid.py`, `scripts/select_config.py` | 14 dev |
| Time-budget sensitivity | `scripts/run_time_sensitivity.py` | 14 dev |
| Wider time windows (business scenario) | `scripts/run_window_scenarios.py` | 14 dev |
| Early delivery allowed (business scenario) | `scripts/run_early_delivery.py` | 14 dev |
| Pre-registered final evaluation | `scripts/run_evaluation.py --confirm-eval` | 42 eval |
| Audit phase A: time metrics and layer-B summaries derived from frozen results (no solver) | `scripts/derive_eval_time_metrics.py`, `scripts/derive_layerb_summaries.py` | 42 eval / 14 dev |
| Audit phase B: v2 diagnostics (B1 scale, B2 fixed cost, B3 fleet strategy, B4 machine load) | `scripts/run_v2_solver_grid.py`, `scripts/select_v2.py` | 14 dev |
| Audit phase B: warm start between layer-B levels (B5) | `scripts/run_v2_layerb_warm.py` | 14 dev |
| Phase C1: robustness of the frozen evaluation plans under delays (no solver) | `scripts/run_robustness.py` | 42 eval plans (descriptive) |
| Phase C2: reduced capacity, limited fleet with priority, soft windows | `scripts/run_phase_c_solver.py` | 14 dev |
| Phase C3: reoptimisation after removing or inserting 10% of the customers | `scripts/run_reoptimisation.py` | 14 dev |
| Phase D: post-hoc v2 re-evaluation and v1-versus-v2 comparison | `scripts/run_evaluation.py --version v2 --confirm-eval --post-hoc`, `scripts/compare_eval_versions.py` | 42 eval (post hoc) |
| Notebook execution | `scripts/execute_notebooks.py` | — |

Runners that resume a folder are protected by a configuration fingerprint (`delivery_routing_vrptw.runs`; legacy folders require `--allow-legacy-resume`). That fingerprint covers all result-determining settings, including the number of worker processes (since wall-clock limits make runs sensitive to CPU contention; block B4 compares 4 versus 8 processes), while ignoring timestamps and file paths. For C3, the protocol aggregate is `aggregate_representative.csv` (representative replicate), whereas `aggregate.csv` pools all replicates for traceability; for the frozen C3 folders, the protocol aggregate is derived with `python scripts/derive_phase_c_corrections.py` without invoking the solver.

Evaluation instances are guarded so every script rejects them without explicit confirmation. Timed runs should not share the machine with other workloads because wall-clock stopping makes results load-dependent. Results consumed by the notebooks are frozen under `outputs/frozen/` (documented in the folder's internal `README.md`).

The run metadata recorded the absolute path of the project folder on the local machine. Before publication those paths were replaced with the neutral placeholder PROJECT_ROOT, and the check can be repeated at any time (the exit code is 1 when a local path remains):

~~~
python scripts/anonymize_local_paths.py --check
~~~

## Notebook map

| Stage | PT-BR | EN-US | What it covers |
| :--- | :---: | :---: | :--- |
| 01 — Presentation and pre-analysis | [PT-BR](notebooks/pt_br/01_apresentacao_e_pre_analise.ipynb) | [EN-US](notebooks/en_us/01_presentation_and_pre_analysis.ipynb) | problem, provenance, data contract, objective conventions |
| 02 — Exploratory analysis of instances | [PT-BR](notebooks/pt_br/02_eda_instancias.ipynb) | [EN-US](notebooks/en_us/02_instance_eda.ipynb) | geography, time windows, capacity vs. time-window-bound fleets |
| 03 — Verifier and baselines | [PT-BR](notebooks/pt_br/03_verificador_e_baselines.ipynb) | [EN-US](notebooks/en_us/03_verifier_and_baselines.ipynb) | independent verifier, constructive baselines |
| 04 — Development tournament | [PT-BR](notebooks/pt_br/04_torneio_desenvolvimento.ipynb) | [EN-US](notebooks/en_us/04_development_tournament.ipynb) | pre-registered tournament, frozen configuration, time budget, v2 diagnostics (phase B) |
| 05 — Final evaluation | [PT-BR](notebooks/pt_br/05_avaliacao_final.ipynb) | [EN-US](notebooks/en_us/05_final_evaluation.ipynb) | final evaluation on 42 unseen instances; post-hoc v2 re-evaluation beside it |
| 06 — Business scenarios | [PT-BR](notebooks/pt_br/06_cenarios_de_negocio.ipynb) | [EN-US](notebooks/en_us/06_business_scenarios.ipynb) | wider windows, early delivery, reallocation, warm start between levels (B5) |
| 07 — Robustness and operational scenarios | [PT-BR](notebooks/pt_br/07_robustez_e_cenarios_operacionais.ipynb) | [EN-US](notebooks/en_us/07_robustness_and_operational_scenarios.ipynb) | layer B: robustness of fixed plans under delays, reduced capacity, limited fleet with priority, soft windows, reoptimisation |

## Method

Every plan is checked in double precision by an independent verifier covering vehicle capacity, single customer visits, time windows, depot return and fleet limits; reported metrics always come from the verifier rather than the solver objective. The constructive baselines comprise a deadline-ordered greedy heuristic (earliest-deadline rule of Jackson, 1955, with first-fit allocation of Johnson, 1974) and Solomon's time-oriented nearest-neighbour heuristic (Solomon, 1987).

Main optimisation uses OR-Tools routing (Perron & Furnon, 2026) with a fixed cost per vehicle to enforce the hierarchical fleet-then-distance objective (Bräysy & Gendreau, 2005a), using integer costs and travel times rounded up so that every solver-feasible schedule remains feasible in double precision. To prevent data leakage, a stratified 14/42 split was drawn before any solver execution, the solver configuration was selected exclusively on the 14 development instances under a pre-registered rule, and the evaluation protocol (time budget, replicates and main-result definition) was fixed before opening the 42 evaluation instances.

## Results

The frozen configuration pairs fixed vehicle cost with `SAVINGS` (Clarke & Wright, 1964) and `GUIDED_LOCAL_SEARCH` (Voudouris & Tsang, 1999). In the final evaluation on the 42 unseen instances (120 s × 5 replicates, taking the median replicate as the main result), all 294 generated plans proved feasible.

| Scope | Mean extra vehicles vs. SINTEF | Fleet equal to SINTEF | Median distance gap (equal fleet) |
|---|---|---|---|
| **Overall** | **0.50** | **24 / 42** | 3.40% |
| C1 | 0.00 | 7 / 7 | 0.00% |
| C2 | 0.00 | 6 / 6 | 0.00% |
| R1 | 0.89 | 1 / 9 | 0.52% |
| R2 | 0.25 | 6 / 8 | 5.02% |
| RC1 | 1.50 | 0 / 6 | — |
| RC2 | 0.33 | 4 / 6 | 4.63% |

The OR-Tools fleet never fell below the reference, requiring +1 vehicle on 15 instances and +2 vehicles on 3 instances (all in RC1), with identical fleet counts across all 5 replicates in every instance. On the C families, the reference fleet was matched in all 13 instances and 10 reproduced the reference distance to two decimal places; R1 and RC1, where time windows are binding, defined the limit of the approach. Both baselines trailed behind: nearest neighbour added +1.36 vehicles on average (12/42 equal fleet) and deadline-first greedy added +3.98 (2/42).

In the business scenarios (run on the 14 development instances and compared solely with each instance's baseline plan, never with SINTEF), symmetrical window widening by a factor of 2 saved 0.79 vehicles on average (reducing the fleet in 8 of 14 instances). Split by fleet class, distance changed by −2.0% where the fleet remained unchanged (6 instances) and by −12.3% where it shrank (8 instances), with zero savings in the C families where capacity dictates fleet size. Allowing unrestricted early delivery saved 0.50 vehicles on average. Evaluated with optimal depot departure (forward time slack; Savelsbergh, 1992), route duration changed by −20.8% and distance by −18.1% where the fleet stayed constant (7 instances), and by −11.7% and −11.0% where the fleet dropped (6 instances); the naive depart-at-zero measure would overstate duration savings by 1.0 to 7.9 percentage points. Duration fell by roughly 37% in R2 and 35% in RC2 among equal-fleet instances, accompanied by vehicle reassignments for roughly 40% of customers, showing that the gain stems from replanning at a tangible operational cost.

## Audit and v2

Following the final evaluation, an external audit (`reports/plano-sanidade-auditoria.md`) reviewed the project and saw most of its ten findings confirmed through inspection of code and frozen outputs: integer scale 100 with ceiling rounding pruned valid routes (R106 and R110), the earlier diagnosis of the `reduce_fleet` failure was incorrect, layer-B levels solved without warm starts produced 20 non-monotone cells (6 in widened windows and 14 in early delivery), the evaluation omitted route duration and waiting times, and several layer-B averages pooled different fleet sizes; the claim that a very high fixed vehicle cost distorted local search remained a testable hypothesis. Subsequent corrections followed four phases under pre-registered protocols (`reports/protocolo-v2.md`, `reports/protocolo-fase-c.md` and `reports/protocolo-fase-d.md`), keeping the original v1 outputs frozen.

In Phase A (without running the solver), route duration and waiting times were derived from frozen evaluation routes (notebook 05), lexicographic representative replicates and fleet-class breakdowns were adopted in layer B (notebook 06), the colour bar in notebook 02 was normalised, and documentation was corrected. In Phase B (solver diagnostics on the 14 development instances), block B1 compared integer scale 10 000 against 100 and selected 10 000 (mean rank 1.32 against 1.68, lowering R110 from 12 to 11 vehicles). Block B2 evaluated relative fixed cost k ∈ {1, 10, 100} and retained k = 10 by mean rank, with identical fleets across all three levels. Block B3 tested the corrected `reduce_fleet` (warm start and penalised dropping), matching SINTEF on R202 (4 to 3 vehicles) while losing on mean rank (1.68 against 1.32), thus preserving `fixed_cost`. Block B4 compared 4 against 8 worker processes and found identical fleets in all 42 cells. Block B5 introduced warm starts between layer-B levels and eliminated all non-monotone cells (6 to 0 and 14 to 0). Consequently, the v2 configuration consists of the v1 settings with integer scale 10 000 and relative fixed cost k = 10, whose magnitude corresponds to roughly 50% to 66% of the v1 absolute fixed cost.

In Phase D, a post-hoc v2 re-evaluation was executed on the 42 evaluation instances (already seen, labelled "v2, post hoc" beside v1 without replacing it), achieving 210/210 feasible plans. Mean extra vehicles moved from 0.50 to 0.48, raising equal-fleet matches against SINTEF from 24 to 25 due to a single fleet change in RC207 (4 to 3 vehicles). With equal fleets, distance moved in both directions (−4.3% to +2.1%; 18 of 41 instances within ±0.5%, 11 shorter and 12 longer), showing that the median-gap shift from 3.40% to 2.23% reflects two-way fluctuations and the inclusion of RC207 in the equal-fleet group rather than a consistent distance gain. The pre-registered expectation of gains on tight-window instances such as R106 did not hold (13 vehicles in both versions). As corrected on 2026-11-19, v2 differs from v1 in both integer scale and fixed-cost magnitude (49.9% to 66.2% of the v1 value in distance units, median 57.7%), meaning the RC207 change is attributable to the full v2 configuration rather than scale alone. While v2 resolves a verified rounding defect, its aggregate impact is marginal and the v1 conclusions stand.

### Phase C: robustness and operational scenarios (layer B, no SINTEF comparison)

Documented in notebook 07 (`reports/fase-c-robustez.md`), Phase C compares each scenario against the reference plan of the same instance, running 30 s × 3 replicates on the 14 development instances for C2 and C3. In C1 (stress testing frozen evaluation plans under delays across 1 000 Monte Carlo draws per plan and scenario), +0–10% travel shocks caused lateness for 6.3% of OR-Tools customers against 9.3% (nearest neighbour) and 21.1% (greedy), and 13.4% late depot returns against 32.9% and 43.3%; per route, OR-Tools accumulated more routes with at least one late visit under service shocks (84.2% against 71.1% and 72.5%). Fragility correlated inversely with nominal slack (Spearman correlation between −0.37 and −0.78).

In the solver-based operational scenarios, reducing capacity to 90% and 80% of Q (C2.1) required 0.21 and 0.57 additional vehicles per instance on average (3 and 6 of 14 instances), increasing distance by roughly 14% where fleets expanded. Under a limited fleet (C2.2), uniform penalties at K − 1 left 111 customers unserved (23 priority), whereas priority penalties left 122 customers unserved with 0 priority customers dropped; at K − 2, counts were 344 (69 priority) versus 363 (3 priority), with minor replicate variability (such as 9, 7 and 7 dropped customers in RC108 under uniform K − 2) because penalties encourage service without guaranteeing a strict minimum drop. The soft-windows block (C2.3) remained exploratory and yielded little information, solving only 3 of 14 instances (4 infeasible by capacity and 7 failing during initial construction), precluding general conclusions without this caveat. In C3 (reoptimisation after removing or inserting 10% of customers, evaluated on the representative replicate per the 2026-11-19 correction), warm starting reassigned 26.7% (removal) and 30.0% (insertion) of common customers against 40.6% and 36.7% cold; although the noise-floor median was 0%, removal showed non-zero churn in 4 of 42 cold-versus-cold comparisons (up to 23.3% in R111 and RC203), confirming substantial operational cost in replanning.

## Limitations

The project relies on synthetic benchmarks with planar Euclidean distances, unit speed and homogeneous fleets, and SINTEF reference values represent best known heuristic solutions rather than proven optima. Wall-clock time limits induce machine-load sensitivity: identical 30 s runs differed by ±1 vehicle on R and RC instances, whereas 120 s runs on an idle machine kept fleet counts stable.

During the initial development tournament, `reduce_fleet` failed almost immediately when attempting a one-vehicle reduction under `PARALLEL_CHEAPEST_INSERTION` and `SAVINGS` (about 10 s total runtime out of the 30 s budget, with only `PATH_CHEAPEST_ARC` exhausting its full 5 s step), discarding the remaining budget and biasing the comparison against `reduce_fleet`. The earlier claim that the 5 s step was too short was retracted on 2026-11-18, and experiment B3 subsequently re-tested fleet reduction with warm starts and penalised dropping at 120 s, matching SINTEF on R202 yet losing on mean rank to `fixed_cost`. In the v1 layer-B scenarios, independent cold starts produced 20 non-monotone cells (6 in widened windows and 14 in early delivery); warm starts in B5 removed these cells, though the main tables in notebook 06 retain the original v1 runs. The post-hoc v2 re-evaluation reuses previously inspected evaluation instances and serves solely as a descriptive comparison alongside v1.

In Phase C, C2.3 (soft windows) failed in 7 of 10 capacity-feasible instances due to initial construction rather than time limits and cannot support broad claims; C1 applies independent arc and service shocks without real traffic correlation; and C2 and C3 cover only the 14 development instances. Business scenarios also treat route-duration savings as a secondary effect of a solver optimising fleet and distance. A documented deviation from the pre-analysis rule restricting evaluation SINTEF reads to the final tournament is also recorded: notebooks 01 and 02 read all 56 published SINTEF entries descriptively to examine conventions and fleet bounds without using them to tune configurations, budgets or parameters.

## Documentation

The `reports/` folder contains the feasibility pre-analysis, the proposal assessment, the pre-registered protocols (grid, evaluation, v2, phase C and phase D) with results, the audit sanity plan, the robustness report, the time and window sensitivity study and the early-delivery scenario. The file `docs/Solomon-html descriptions.pdf` preserves the SINTEF reference web page.

## References

Works cited in this README and the notebooks follow author-year format and are compiled in [`docs/referencias.md`](docs/referencias.md), alongside the closing References section in each notebook.
