# Delivery Routing & Time-Window Optimization (VRPTW)

[Português](README.pt-BR.md) · [Executive report](site/en.html)

This project studies how to assign customers to vehicles and sequence deliveries under vehicle
capacity and customer time windows, minimising **first the fleet and then the total distance**. It is a
**prescriptive** analysis: the output is a route plan that can be checked independently and compared
with published reference solutions, not a prediction.

Reusable code lives in the `delivery_routing_vrptw` package (`src/`); the analysis is told through
bilingual Portuguese and English notebooks that share the same executable code.

The data are **synthetic academic benchmarks** (Solomon, 1987), not real logistics telemetry. The link
with the Olist project is thematic only: Olist contains no fleet, capacity, time windows or routes.
`production_eligible = false`: the method is validated on the benchmark; real operation would require
road distances, the real fleet, measured service times and intraday re-optimisation.

## Question

How should customers be allocated to vehicles and deliveries be ordered so that the fleet and then the
distance are minimised, without violating capacity, time windows or the return to the depot?

## Data

- **Instances:** the 56 Solomon instances with 100 customers (families C1, C2, R1, R2, RC1, RC2), from
  SINTEF TOP (`data/raw/sintef/solomon-100.zip`, downloaded manually because the site blocks automated
  clients) and from CVRPLIB (`data/raw/cvrplib_solomon_100/Solomon.7z`). Both sources are numerically
  identical. Provenance and SHA-256 hashes are in each folder's `MANIFEST.md`.
- **References:** SINTEF best known solutions (SINTEF, 2008; `data/reference/sintef_bks_100.csv`,
  transcribed from the SINTEF page) and CVRPLIB solutions (CVRPLIB, s.d.;
  `data/reference/cvrplib_bks_100.csv`).
- **License:** no explicit license was found; cite Solomon (1987), *Operations Research* 35(2), SINTEF
  TOP and CVRPLIB.
- No Google Maps, traffic data or paid APIs are used. Distances are Euclidean and part of the data
  contract.

### Obtaining the data (third-party files are not redistributed)

The public version of this repository does **not** include third-party files: the SINTEF archive
(`data/raw/sintef/solomon-100.zip`), the CVRPLIB archives (`data/raw/cvrplib_solomon_100/*.7z`), the saved
SINTEF page (`docs/Solomon-html descriptions.pdf`) and the CVRPLIB solution routes
(`data/reference/cvrplib_bks_100.csv`, regenerated from the CVRPLIB archives). The `MANIFEST.md` files, the
transcribed SINTEF table of vehicles and distances (`data/reference/sintef_bks_100.csv`) and the
development/evaluation split are kept.

~~~
python scripts/fetch_data.py              # verifies SHA-256, downloads the CVRPLIB archives, rebuilds the CVRPLIB csv
python scripts/fetch_data.py --check-only # only verifies the files already present (no network)
~~~

The SINTEF site blocks automated clients (Cloudflare): download `solomon-100.zip` manually from
https://www.sintef.no/projectweb/top/vrptw/100-customers/ and save it as `data/raw/sintef/solomon-100.zip`
(the script prints the expected SHA-256). Tests that do not depend on these files run without them.

Sources to cite: Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time
window constraints. *Operations Research, 35*(2), 254-265. SINTEF. (2008). *VRPTW - 100 customers: Solomon
benchmark, best known solutions*. SINTEF TOP. https://www.sintef.no/projectweb/top/vrptw/100-customers/ .
CVRPLIB. (s.d.). *VRPTW instances - Solomon set*. PUC-Rio. https://galgos.inf.puc-rio.br/cvrplib/

### Objective conventions (central comparability risk)

| | SINTEF (used here) | CVRPLIB |
|---|---|---|
| Objective | hierarchical: vehicles, then distance | distance only |
| Distance | double precision, total rounded to 2 decimals | truncated to 1 decimal |

An independent verifier showed that 8 of the 56 CVRPLIB solutions become **infeasible** under exact
distance, and that SINTEF uses fewer vehicles in 37 instances. Results are compared only with SINTEF,
under its convention; the distance gap is reported only when the fleet equals the reference.

## Reproduction

The conda environment uses only conda-forge; OR-Tools comes from PyPI (it is not on conda-forge):

~~~
set CONDA_CHANNELS=conda-forge
set CONDA_CHANNEL_PRIORITY=strict
conda env create -f environment.yml
conda activate delivery-routing-vrptw
~~~

`environment.yml` installs the package in editable mode and sets `MKL_THREADING_LAYER=SEQUENTIAL`
(it avoids a numpy/MKL crash on Windows). Exact versions are in `environment.lock.yml`.

Tests and lint (unit tests use synthetic fixtures; contract tests read the notebooks and
`data/reference/split.csv`; no network or GUI):

~~~
python -m ruff check src tests scripts
python -m pytest -q --cov=delivery_routing_vrptw
~~~

Experiment parameters (solver time limit, budgets, scenario factors, replicates) live in
`config/experiment.toml` and can be overridden on the command line. Experiments, in the order they
were run:

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

Evaluation instances are guarded: every script refuses them unless explicitly confirmed. Timed runs
should not share the machine with other workloads (wall-clock limits make results load-dependent).
Results used by the notebooks are frozen in `outputs/frozen/` (see its `README.md`).

Runners that resume a folder are protected by a configuration fingerprint (`delivery_routing_vrptw.runs`; legacy
folders need `--allow-legacy-resume`). It covers the result-determining configuration, **including the number of
worker processes** (wall-clock limits make results depend on CPU contention; block B4 compares 4 vs 8), but not
timestamps or paths. For C3, the protocol aggregate is `aggregate_representative.csv` (representative replicate);
`aggregate.csv` pools all replicates and is kept for traceability. For the frozen C3 folders the protocol aggregate
is derived with `python scripts/derive_phase_c_corrections.py` (no solver).

## Notebook map

| No. | Portuguese notebook | English notebook | Content |
|---:|---|---|---|
| 01 | `01_apresentacao_e_pre_analise.ipynb` | `01_presentation_and_pre_analysis.ipynb` | problem, provenance, data contract, objective conventions |
| 02 | `02_eda_instancias.ipynb` | `02_instance_eda.ipynb` | geography, time windows, capacity vs. time-window-bound fleets |
| 03 | `03_verificador_e_baselines.ipynb` | `03_verifier_and_baselines.ipynb` | independent verifier, constructive baselines |
| 04 | `04_torneio_desenvolvimento.ipynb` | `04_development_tournament.ipynb` | pre-registered tournament, frozen configuration, time budget, v2 diagnostics (phase B) |
| 05 | `05_avaliacao_final.ipynb` | `05_final_evaluation.ipynb` | final evaluation on 42 unseen instances; post-hoc v2 re-evaluation beside it |
| 06 | `06_cenarios_de_negocio.ipynb` | `06_business_scenarios.ipynb` | wider windows, early delivery, reallocation, warm start between levels (B5) |
| 07 | `07_robustez_e_cenarios_operacionais.ipynb` | `07_robustness_and_operational_scenarios.ipynb` | layer B: robustness of fixed plans under delays, reduced capacity, limited fleet with priority, soft windows, reoptimisation |

## Method

- **Verifier:** every plan is checked in double precision (capacity, single visit, time windows, return
  to the depot, fleet limit); reported metrics come from the verifier, never from the solver objective.
- **Baselines:** deadline-ordered greedy (earliest-deadline rule, Jackson, 1955; first-fit, Johnson, 1974) and
  Solomon's time-oriented nearest neighbour (Solomon, 1987).
- **Solver:** OR-Tools routing (Perron & Furnon, 2026) with a fixed cost per vehicle (fleet first; hierarchical
  objective as in Bräysy & Gendreau, 2005a), integer costs and travel
  times rounded up (a solver-feasible plan is feasible in double precision).
- **Anti-leakage:** stratified 14/42 split drawn before any solver run; configuration chosen on the 14
  development instances under a rule registered beforehand; evaluation protocol (budget, replicates,
  main result) registered before opening the 42 evaluation instances.

## Results

Frozen configuration: fixed vehicle cost + `SAVINGS` (savings heuristic; Clarke & Wright, 1964) +
`GUIDED_LOCAL_SEARCH` (guided local search; Voudouris & Tsang, 1999). Final evaluation: 42
evaluation instances, 120 s × 5 replicates, main result = median replicate, 294/294 feasible plans.

| Scope | Mean extra vehicles vs. SINTEF | Fleet equal to SINTEF | Median distance gap (equal fleet) |
|---|---|---|---|
| **Overall** | **0.50** | **24 / 42** | 3.40% |
| C1 | 0.00 | 7 / 7 | 0.00% |
| C2 | 0.00 | 6 / 6 | 0.00% |
| R1 | 0.89 | 1 / 9 | 0.52% |
| R2 | 0.25 | 6 / 8 | 5.02% |
| RC1 | 1.50 | 0 / 6 | — |
| RC2 | 0.33 | 4 / 6 | 4.63% |

- The fleet is never below the reference: +1 vehicle in 15 instances, +2 in 3 (all RC1).
- In the C families the reference fleet is reached in all 13 instances, and 10 reproduce the reference
  distance to two decimals; R1 and RC1, where time windows bind, are the limit of the approach.
- The fleet was identical across the 5 replicates in all 42 instances.
- Baselines: nearest neighbour +1.36 vehicles (12/42 equal fleet), greedy +3.98 (2/42).

**Business scenarios** (development instances, compared with the plan's own baseline, never with
SINTEF):

- **Wider time windows (×2):** 0.79 vehicles saved on average (8 of 14 instances reduce the fleet).
  Distance is never averaged across different fleets: −2.0% where the fleet is unchanged (6 instances),
  −12.3% where it is reduced (8). **Nothing** is saved in the C families, whose fleet is set by capacity.
- **Early delivery allowed** (the benchmark forces waiting until the window opens; real operations
  often accept early delivery): free early delivery saves 0.50 vehicles on average; where the fleet is
  unchanged (7 instances) route duration is −20.8% and distance −18.1%, where it is reduced (6
  instances) −11.7% and −11.0%, both measured with optimal depot departure (forward time slack; Savelsbergh, 1992) (the naive "depart at 0"
  measure would overstate the duration gain by 1.0–7.9 points). In R2/RC2 the duration falls by about
  37% and 35% among the equal-fleet instances. About 40% of customers change vehicle: the gain comes
  from re-planning, which has an operational cost.

## Audit and v2

After the final evaluation, an external audit (`reports/plano-sanidade-auditoria.md`) reviewed the
project. Its ten findings were triaged by reading the code and the frozen results: most were confirmed,
among them that the integer scale 100 with rounding up prunes valid routes (R106 and R110), that our
diagnosis of the `reduce_fleet` failure was wrong, that layer-B levels solved without a warm start are
not monotone (6 + 14 cells), that the evaluation did not record route duration and waiting, and that
some layer-B averages mixed different fleets; the claim that a very high fixed cost per vehicle distorts
the search was only a hypothesis. The work followed four phases, each with its protocol registered before
running (`reports/protocolo-v2.md`, `reports/protocolo-fase-c.md`, `reports/protocolo-fase-d.md`), and the
v1 results stay frozen as the original result:

- **A, corrections without running the solver:** duration and waiting derived from the frozen routes
  (notebook 05), lexicographic representative replicates and fleet-class separation in layer B (notebook
  06), a normalised colour bar in notebook 02 and corrected documentation.
- **B, solver diagnostics on the 14 development instances (v2):** B1 scale 10 000 against 100 (mean
  rank 1.32 against 1.68; R110 goes from 12 to 11 vehicles), B2 fixed cost k = 1, 10, 100 (same fleet at the
  three levels; k = 10 kept), B3 corrected `reduce_fleet` (matches SINTEF in R202, 4 → 3, but mean rank 1.68
  against 1.32, so `fixed_cost` is kept), B4 4 against 8 processes (same fleet in 42/42 cells), B5 warm
  start between layer-B levels (non-monotone cells 6 → 0 and 14 → 0). **v2 = the v1 configuration with
  integer scale 10 000 and relative fixed cost k = 10, whose magnitude is about 50–66% of the v1 absolute
  fixed cost** (corrected on 2026-11-19; only the scale was isolated, in B1).
- **C, layer B (see below).**
- **D, post-hoc v2 re-evaluation of the 42 evaluation instances** (already seen, so labelled "v2, post hoc"
  and shown beside the v1, never in its place): 210/210 feasible; mean extra vehicles 0.50 → 0.48; fleet
  equal to SINTEF in **25 / 42** instances (24 in the v1), the only fleet change being **RC207, 4 → 3**.
  With equal fleets the distance moves in both directions (from −4.3% to +2.1%; 18 of 41 instances within
  ±0.5%, 11 shorter, 12 longer), so the median-gap drop from 3.40% to 2.23% reflects those oscillations and
  a change in the equal-fleet set, not a consistent gain. The registered expectation of a gain in
  tight-window instances such as R106 was not confirmed (13 vehicles in both versions). The v2 fixes a
  demonstrated defect, but its aggregate impact is marginal; the v1 conclusion stands. **Correction
  (2026-11-19):** the v2 differs from the v1 in the scale **and** in the magnitude of the fixed cost (49.9% to
  66.2% of the v1 value in distance units, median 57.7%), so the comparison is between two configurations and
  the change in RC207 is attributable to the v2 configuration, not to the scale alone.

### Phase C: robustness and operational scenarios (layer B, no SINTEF comparison)

Notebook 07 (`reports/fase-c-robustez.md`; each scenario is compared with the reference plan of the same
instance; C2 and C3 use the 14 development instances, 30 s, 3 replicates):

- **C1, fixed plans under delays** (frozen evaluation plans, used descriptively, 1 000 draws per plan and
  scenario): with travel +0–10%, 6.3% of the OR-Tools customers are served late against 9.3% (nearest
  neighbour) and 21.1% (greedy), and 13.4% of the returns to the depot are late against 32.9% and 43.3%;
  per route, OR-Tools has more routes with some lateness under service shocks (84.2% against 71.1% and
  72.5%). Nominal slack explains the fragility (Spearman correlation with the late rate from −0.37 to
  −0.78).
- **C2.1, reduced capacity:** 90% and 80% of Q need 0.21 and 0.57 extra vehicles per instance (3 and 6 of 14
  instances), with about +14% distance where the fleet grows.
- **C2.2, limited fleet with priority:** with K − 1 the uniform penalty leaves 111 customers unserved (23 of
  them priority) and priority leaves 122 (0 priority); with K − 2, 344 (69) against 363 (3). The number dropped varies a little across replicates of the
  same cell (RC108, K − 2, uniform penalty: 9, 7 and 7), because the penalty encourages service without
  guaranteeing a minimal drop.
- **C2.3, soft windows: exploratory, little information** — only 3 of 14 instances were solved (4 are
  capacity-infeasible and 7 failed in the initial construction), so no general conclusion is drawn and the
  result must not be quoted without this caveat.
- **C3, reoptimisation after removing or inserting 10% of the customers:** a warm start changes the vehicle of
  26.7% (removal) and 30.0% (insertion) of the common customers, against 40.6% and 36.7% cold (representative replicate
  of each instance; corrected on 2026-11-19); the noise floor has a median of 0% but is not zero in removal
  (4 of 42 cold-versus-cold comparisons reassigned customers, up to 23.3%, in R111 and RC203; 0 of 42 in
  insertion). Replanning has a relevant operational cost either way.

## Limitations

- Synthetic benchmark (planar Euclidean distance, speed 1, homogeneous fleet).
- SINTEF values are best known solutions, not proven optima; no optimality claim is made.
- Wall-clock time limits make results load-dependent: at 30 s, identical runs differed by ±1 vehicle on
  R/RC instances; at 120 s on an idle machine the fleet was stable.
- The fleet-reduction strategy (`reduce_fleet`) was not effectively tested in the tournament. The
  attempt with one vehicle fewer failed almost immediately (total runtime about 10 s of the 30 s
  budget with `PARALLEL_CHEAPEST_INSERTION` and `SAVINGS`; only `PATH_CHEAPEST_ARC` used its whole 5 s
  step), and the remaining budget was discarded, so the comparison against the fixed-cost strategy is
  inconclusive and biased against `reduce_fleet`. An earlier statement that the 5 s step was too
  short was wrong (corrected on 2026-11-18). Experiment B3 later repeated the comparison with a corrected
  `reduce_fleet` (warm start, penalised dropping, 120 s): it matched SINTEF in R202 but lost on mean rank,
  so `fixed_cost` stays (see Audit and v2).
- In the v1 layer-B runs the scenario levels were solved without a warm start between them, so results are
  not monotone: 20 cells (6 widened-window and 14 early-delivery) are lexicographically worse (vehicles,
  then distance) than the best plan of a less permissive level of the same instance. With a warm start
  (experiment B5) these cells disappear, but the main tables of notebook 06 remain the v1 ones.
- The post-hoc v2 re-evaluation reuses instances that were already seen; its marginal difference from the v1
  is descriptive and does not replace the v1 result.
- Phase C: C2.3 (soft windows) is exploratory (3 of 14 instances solved): it failed in 7 of the 10
  feasible-by-capacity instances because of the initial construction (not time) and is not conclusive; C1 uses independent shocks without the spatial and temporal
  correlation of real traffic; C2 and C3 use 14 development instances.
- Deviation from the pre-analysis rule that the evaluation SINTEF values would be read only in the final
  tournament: notebooks 01 and 02 read all 56 published values descriptively (conventions and fleet
  bounds); they were not used to select a configuration, a budget or a parameter.
- Business scenarios use the 14 development instances only and treat time gains as a by-product (the
  solver optimises fleet and distance, not duration).

## Documentation

`reports/` holds the feasibility pre-analysis, the proposal assessment, the pre-registered protocols
(grid, evaluation, v2, phase C, phase D) with their results, the audit sanity plan, the robustness report,
the time/time-window sensitivity study and the early-delivery scenario.
`docs/Solomon-html descriptions.pdf` is the saved SINTEF page with the reference table.

## References

The works cited in the README and in the notebooks, in author-year format, are listed in
[`docs/referencias.md`](docs/referencias.md) (curated list; each notebook ends with a References section).
