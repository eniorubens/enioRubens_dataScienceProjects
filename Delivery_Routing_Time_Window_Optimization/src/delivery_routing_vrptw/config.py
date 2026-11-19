"""Central experiment configuration (``config/experiment.toml``) and shared CLI plumbing."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .io import Instance
from .paths import project_root
from .solution import Solution
from .solver import SolverConfig, minimize_fleet, solve

FLEET_STRATEGIES = ("fixed_cost", "reduce_fleet")
WIDENING_MODES = ("symmetric", "later_due")
INSTANCE_SETS = ("dev", "eval", "all")


def _check_instance_set(value: str) -> None:
    if value not in INSTANCE_SETS:
        raise ValueError(f"instance_set must be one of {INSTANCE_SETS}, got {value!r}")


@dataclass(frozen=True)
class SolverSection:
    """Frozen OR-Tools configuration shared by every experiment.

    Attributes
    ----------
    time_limit_s : float
        Search budget per instance, in seconds.
    first_solution, metaheuristic : str
        OR-Tools strategy names (validated by building a :class:`SolverConfig`).
    fleet_strategy : str
        ``"fixed_cost"`` (single solve) or ``"reduce_fleet"`` (fleet-reduction loop).
    workers : int
        Parallel worker processes.
    """

    time_limit_s: float = 30.0
    first_solution: str = "SAVINGS"
    metaheuristic: str = "GUIDED_LOCAL_SEARCH"
    fleet_strategy: str = "fixed_cost"
    workers: int = 8

    def __post_init__(self) -> None:
        """Validate values."""
        if self.fleet_strategy not in FLEET_STRATEGIES:
            raise ValueError(f"fleet_strategy must be one of {FLEET_STRATEGIES}, got {self.fleet_strategy!r}")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        SolverConfig(
            first_solution=self.first_solution, metaheuristic=self.metaheuristic, time_limit_s=self.time_limit_s
        )


@dataclass(frozen=True)
class TimeSensitivitySection:
    """Layer A: search budgets of the time-sensitivity experiment."""

    budgets_s: tuple[float, ...] = (10, 30, 60, 120, 300)
    instance_set: str = "dev"

    def __post_init__(self) -> None:
        """Validate values."""
        if not self.budgets_s or any(b <= 0 for b in self.budgets_s):
            raise ValueError("budgets_s must be a non-empty list of positive numbers")
        _check_instance_set(self.instance_set)


@dataclass(frozen=True)
class WindowWideningSection:
    """Layer B: customer time-window widening scenario."""

    factors: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
    mode: str = "symmetric"
    instance_set: str = "dev"
    time_limit_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate values."""
        if not self.factors or any(f < 1.0 for f in self.factors):
            raise ValueError("factors must be a non-empty list of values >= 1.0")
        if 1.0 not in self.factors:
            raise ValueError("factors must include 1.0 (scenario baseline)")
        if self.mode not in WIDENING_MODES:
            raise ValueError(f"mode must be one of {WIDENING_MODES}, got {self.mode!r}")
        _check_instance_set(self.instance_set)
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")


@dataclass(frozen=True)
class EarlyDeliverySection:
    """Layer B: early-delivery scenario (deliveries before the window opens are accepted).

    Attributes
    ----------
    tolerances : tuple of float
        Fractions of the window width by which ``ready`` may be anticipated; must include 0.0.
    include_full : bool
        Adds the ``"full"`` level (``ready = 0``).
    replicates : int
        Repeated runs per cell (wall-clock variability).
    instance_set : str
        ``"dev"``, ``"eval"`` or ``"all"``.
    time_limit_s : float
        Search budget per run, in seconds.
    """

    tolerances: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    include_full: bool = True
    replicates: int = 3
    instance_set: str = "dev"
    time_limit_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate values."""
        if not self.tolerances or any(not math.isfinite(t) or t < 0 for t in self.tolerances):
            raise ValueError("tolerances must be a non-empty list of finite values >= 0")
        if 0.0 not in self.tolerances:
            raise ValueError("tolerances must include 0.0 (scenario baseline)")
        if len(set(self.tolerances)) != len(self.tolerances):
            raise ValueError("tolerances must not repeat")
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        _check_instance_set(self.instance_set)
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")

    def levels(self) -> list[float]:
        """Return the tolerance levels in increasing order, with ``math.inf`` last when ``include_full``."""
        levels = sorted(float(t) for t in self.tolerances)
        return [*levels, math.inf] if self.include_full else levels


@dataclass(frozen=True)
class EvaluationSection:
    """Final evaluation, pre-registered in ``reports/protocolo-avaliacao.md``.

    Attributes
    ----------
    time_limit_s : float
        Search budget per OR-Tools run, in seconds.
    replicates : int
        OR-Tools replicates per instance (wall-clock variability).
    instance_set : str
        ``"dev"``, ``"eval"`` or ``"all"``.
    include_baselines : bool
        Also run the deterministic baselines once per instance.
    """

    time_limit_s: float = 120.0
    replicates: int = 5
    instance_set: str = "eval"
    include_baselines: bool = True

    def __post_init__(self) -> None:
        """Validate values."""
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        _check_instance_set(self.instance_set)


@dataclass(frozen=True)
class V2Section:
    """Solver diagnostics of ``reports/protocolo-v2.md`` (blocks B1-B5, development instances only).

    Attributes
    ----------
    scales : tuple of int
        Integer scales compared in B1.
    k_levels : tuple of float
        Fixed-cost multipliers compared in B2 (``vehicle_fixed_cost = k * scale * D_ub``); must include
        ``k_default``.
    k_default : float
        Multiplier of B1 (and the v1-equivalent level), reused rather than re-run in B2.
    time_limit_s : float
        Search budget per run in B1-B4, in seconds.
    replicates : int
        Replicates per cell in B1-B4.
    workers : int
        Worker processes of B1-B3 and B5.
    b4_workers : int
        Worker processes of B4 (the *throttling* check).
    first_fraction : float
        B3: fraction of the budget used by the first (cold) solve of ``reduce_fleet`` v2.
    step_time_s : float
        B3: nominal time limit of one reduction attempt, in seconds.
    drop_penalty_factor : int
        B3: dropping penalty as a multiple of the fixed cost per vehicle.
    b5_time_limit_s : float
        B5: search budget per level, in seconds (the same as v1).
    protocol : str
        Path of the pre-registered protocol, recorded in the run metadata.
    """

    scales: tuple[int, ...] = (100, 10000)
    k_levels: tuple[float, ...] = (1.0, 10.0, 100.0)
    k_default: float = 10.0
    time_limit_s: float = 120.0
    replicates: int = 3
    workers: int = 8
    b4_workers: int = 4
    first_fraction: float = 1 / 3
    step_time_s: float = 20.0
    drop_penalty_factor: int = 100
    b5_time_limit_s: float = 30.0
    protocol: str = "reports/protocolo-v2.md"

    def __post_init__(self) -> None:
        """Validate values."""
        if not self.scales or any(int(x) != x or x < 1 for x in self.scales):
            raise ValueError("scales must be a non-empty list of integers >= 1")
        if not self.k_levels or any(k <= 0 for k in self.k_levels):
            raise ValueError("k_levels must be a non-empty list of positive numbers")
        if self.k_default not in self.k_levels:
            raise ValueError("k_levels must include k_default")
        if self.time_limit_s <= 0 or self.step_time_s <= 0 or self.b5_time_limit_s <= 0:
            raise ValueError("time limits must be positive")
        if self.replicates < 1 or self.workers < 1 or self.b4_workers < 1:
            raise ValueError("replicates, workers and b4_workers must be >= 1")
        if not 0 < self.first_fraction < 1:
            raise ValueError("first_fraction must be in (0, 1)")
        if self.drop_penalty_factor < 1:
            raise ValueError("drop_penalty_factor must be >= 1")


@dataclass(frozen=True)
class PhaseCSection:
    """Robustness and operational scenarios of ``reports/protocolo-fase-c.md`` (phase C, layer B).

    Attributes
    ----------
    seed : int
        Seed of every random draw of the phase (priorities, samples, Monte Carlo).
    scale, fixed_cost_k : int, float
        Solver v2 configuration: integer scale and fixed-cost multiplier (``vehicle_fixed_cost = k * scale *
        D_ub``). The first solution strategy and metaheuristic come from ``[solver]``.
    time_limit_s : float
        Search budget per solver run (C2 and C3), in seconds.
    replicates : int
        Replicates per cell (representative replicate = position ``ceil(R / 2)``, lexicographic).
    workers : int
        Parallel worker processes of the solver blocks (and of C1).
    capacity_factors : tuple of float
        C2.1 capacity multipliers; must include 1.0 (the reference).
    fleet_reductions : tuple of int
        C2.2 fleet reductions: ``K = reference vehicles - r``.
    priority_share : float
        C2.2 share of customers marked as priority (per instance).
    priority_penalty_multiplier : int
        C2.2 priority penalty as a multiple of the common penalty.
    soft_cost_multipliers : tuple of int
        C2.3 cost per unit of lateness as a multiple of the cost per unit of distance (low, high).
    soft_fleet_reduction : int
        C2.3 fleet reduction: ``K = reference vehicles - soft_fleet_reduction``.
    reopt_share : float
        C3 share of customers removed or inserted.
    c1_draws : int
        C1 Monte Carlo draws per plan and scenario.
    c1_travel_a : tuple of float
        C1 travel-time shock sizes ``a`` (``eps ~ U(0, a)``).
    c1_service_b : float
        C1 service-time shock size (``eta ~ U(0, b)``).
    c1_combined_travel_a : float
        C1 travel shock of the combined (travel + service) scenario.
    protocol : str
        Path of the pre-registered protocol, recorded in the run metadata.
    """

    seed: int = 20261118
    scale: int = 10000
    fixed_cost_k: float = 10.0
    time_limit_s: float = 30.0
    replicates: int = 3
    workers: int = 8
    capacity_factors: tuple[float, ...] = (1.0, 0.9, 0.8)
    fleet_reductions: tuple[int, ...] = (1, 2)
    priority_share: float = 0.2
    priority_penalty_multiplier: int = 10
    soft_cost_multipliers: tuple[int, ...] = (1, 10)
    soft_fleet_reduction: int = 1
    reopt_share: float = 0.1
    c1_draws: int = 1000
    c1_travel_a: tuple[float, ...] = (0.05, 0.10, 0.20)
    c1_service_b: float = 0.20
    c1_combined_travel_a: float = 0.10
    protocol: str = "reports/protocolo-fase-c.md"

    def __post_init__(self) -> None:
        """Validate values."""
        if self.scale < 1 or self.fixed_cost_k <= 0:
            raise ValueError("scale must be >= 1 and fixed_cost_k positive")
        if self.time_limit_s <= 0 or self.replicates < 1 or self.workers < 1 or self.c1_draws < 1:
            raise ValueError("time_limit_s must be positive; replicates, workers and c1_draws >= 1")
        if not self.capacity_factors or any(not 0 < f <= 1 for f in self.capacity_factors):
            raise ValueError("capacity_factors must be non-empty values in (0, 1]")
        if 1.0 not in self.capacity_factors:
            raise ValueError("capacity_factors must include 1.0 (the reference)")
        if not self.fleet_reductions or any(int(r) != r or r < 1 for r in self.fleet_reductions):
            raise ValueError("fleet_reductions must be non-empty integers >= 1")
        if not self.soft_cost_multipliers or any(int(m) != m or m < 1 for m in self.soft_cost_multipliers):
            raise ValueError("soft_cost_multipliers must be non-empty integers >= 1")
        if int(self.soft_fleet_reduction) != self.soft_fleet_reduction or self.soft_fleet_reduction < 1:
            raise ValueError("soft_fleet_reduction must be an integer >= 1")
        if not 0 < self.priority_share < 1 or not 0 < self.reopt_share < 1:
            raise ValueError("priority_share and reopt_share must be in (0, 1)")
        if self.priority_penalty_multiplier < 1:
            raise ValueError("priority_penalty_multiplier must be >= 1")
        if not self.c1_travel_a or any(a <= 0 for a in self.c1_travel_a):
            raise ValueError("c1_travel_a must be non-empty positive values")
        if self.c1_service_b <= 0 or self.c1_combined_travel_a <= 0:
            raise ValueError("c1_service_b and c1_combined_travel_a must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Whole configuration file."""

    solver: SolverSection = field(default_factory=SolverSection)
    time_sensitivity: TimeSensitivitySection = field(default_factory=TimeSensitivitySection)
    window_widening: WindowWideningSection = field(default_factory=WindowWideningSection)
    early_delivery: EarlyDeliverySection = field(default_factory=EarlyDeliverySection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    v2: V2Section = field(default_factory=V2Section)
    phase_c: PhaseCSection = field(default_factory=PhaseCSection)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of the effective configuration."""
        return json.loads(json.dumps(asdict(self)))


def default_config_path() -> Path:
    """Return ``<project root>/config/experiment.toml``."""
    return project_root() / "config" / "experiment.toml"


def _section(cls, data: dict, name: str):
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown key(s) in [{name}]: {sorted(unknown)}")
    data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return cls(**data)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Read and validate the experiment configuration.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        TOML file; defaults to ``config/experiment.toml`` at the project root.

    Returns
    -------
    ExperimentConfig
        Frozen, validated configuration. Missing sections or keys take the dataclass defaults.
    """
    path = default_config_path() if path is None else Path(path)
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - {"solver", "time_sensitivity", "window_widening", "early_delivery", "evaluation", "v2", "phase_c"}
    if unknown:
        raise ValueError(f"unknown section(s): {sorted(unknown)}")
    return ExperimentConfig(
        solver=_section(SolverSection, raw.get("solver", {}), "solver"),
        time_sensitivity=_section(TimeSensitivitySection, raw.get("time_sensitivity", {}), "time_sensitivity"),
        window_widening=_section(WindowWideningSection, raw.get("window_widening", {}), "window_widening"),
        early_delivery=_section(EarlyDeliverySection, raw.get("early_delivery", {}), "early_delivery"),
        evaluation=_section(EvaluationSection, raw.get("evaluation", {}), "evaluation"),
        v2=_section(V2Section, raw.get("v2", {}), "v2"),
        phase_c=_section(PhaseCSection, raw.get("phase_c", {}), "phase_c"),
    )


def solver_config(cfg: ExperimentConfig, time_limit_s: float | None = None) -> SolverConfig:
    """Build the OR-Tools :class:`SolverConfig` from the ``[solver]`` section.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    time_limit_s : float, optional
        Overrides ``solver.time_limit_s`` (for example one time-sensitivity budget).

    Returns
    -------
    SolverConfig
        Solver configuration.
    """
    s = cfg.solver
    return SolverConfig(
        first_solution=s.first_solution,
        metaheuristic=s.metaheuristic,
        time_limit_s=s.time_limit_s if time_limit_s is None else time_limit_s,
    )


def solve_configured(instance: Instance, cfg: ExperimentConfig, time_limit_s: float | None = None) -> Solution:
    """Solve with the configured strategy (``fixed_cost`` or ``reduce_fleet``) and budget.

    ``reduce_fleet`` stops at the first failed reduction, which happens almost immediately (see the notes of
    :func:`delivery_routing_vrptw.solver.minimize_fleet`), so it uses far less than the budget.
    """
    sc = solver_config(cfg, time_limit_s)
    if cfg.solver.fleet_strategy == "reduce_fleet":
        budget = sc.time_limit_s
        return minimize_fleet(instance, sc, step_time_s=5, total_time_s=budget, first_step_time_s=budget / 3)
    return solve(instance, sc)


def select_instances(
    split: dict[str, str], instance_set: str, names: list[str] | None = None, allow_eval: bool = False
) -> list[str]:
    """Return the instance names of a split set, refusing the evaluation set unless allowed.

    Parameters
    ----------
    split : dict
        Instance name -> ``"dev"`` or ``"eval"``.
    instance_set : str
        ``"dev"``, ``"eval"`` or ``"all"``.
    names : list of str, optional
        Restrict to these instances; each must belong to the set.
    allow_eval : bool
        Must be true to touch ``"eval"`` or ``"all"``.

    Returns
    -------
    list of str
        Selected instance names.

    Raises
    ------
    PermissionError
        If the evaluation set is requested without ``allow_eval``.
    ValueError
        If ``names`` contains instances outside the set, or nothing is selected.
    """
    _check_instance_set(instance_set)
    if instance_set != "dev" and not allow_eval:
        raise PermissionError(
            f"refused: instance_set={instance_set!r} includes the evaluation set. Evaluation must only be "
            "opened after the protocol says so; pass --allow-eval to override."
        )
    wanted = {"dev": {"dev"}, "eval": {"eval"}, "all": {"dev", "eval"}}[instance_set]
    pool = sorted(n for n, s in split.items() if s in wanted)
    if names:
        names = [n.strip().upper() for n in names if n.strip()]
        bad = [n for n in names if n not in pool]
        if bad:
            raise ValueError(f"instances not in set {instance_set!r}: {bad}")
        pool = names
    if not pool:
        raise ValueError("no instances selected")
    return pool


def add_config_args(parser: argparse.ArgumentParser, *, budgets: bool = False, factors: bool = False, early: bool = False
) -> None:
    """Add ``--config`` and the common override flags to a script parser."""
    parser.add_argument("--config", default=None, help="TOML file (default: config/experiment.toml)")
    parser.add_argument("--time-limit", type=float, default=None, help="override the search budget (s)")
    parser.add_argument("--workers", type=int, default=None, help="override solver.workers")
    parser.add_argument(
        "--set", dest="instance_set", choices=INSTANCE_SETS, default=None, help="override instance_set"
    )
    if budgets:
        parser.add_argument("--budgets", default=None, help="comma list of budgets in s, e.g. 10,30,60")
    if factors:
        parser.add_argument("--factors", default=None, help="comma list of widening factors, e.g. 1.0,1.5")
        parser.add_argument("--mode", choices=WIDENING_MODES, default=None, help="override window mode")
    if early:
        parser.add_argument(
            "--tolerances", default=None, help="comma list, e.g. 0,0.5,full ('full' adds that level)"
        )
        parser.add_argument("--replicates", type=int, default=None, help="override early_delivery.replicates")


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(",") if v.strip())


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load the TOML file and apply the CLI overrides found in ``args`` (CLI wins)."""
    cfg = load_config(args.config)
    solver, ts, ww, ed = cfg.solver, cfg.time_sensitivity, cfg.window_widening, cfg.early_delivery
    ev = cfg.evaluation
    if args.time_limit is not None:
        ev = replace(ev, time_limit_s=args.time_limit)
        solver = replace(solver, time_limit_s=args.time_limit)
        ww = replace(ww, time_limit_s=args.time_limit)
        ed = replace(ed, time_limit_s=args.time_limit)
    if args.workers is not None:
        solver = replace(solver, workers=args.workers)
    if args.instance_set is not None:
        ev = replace(ev, instance_set=args.instance_set)
        ts = replace(ts, instance_set=args.instance_set)
        ww = replace(ww, instance_set=args.instance_set)
        ed = replace(ed, instance_set=args.instance_set)
    if getattr(args, "budgets", None):
        ts = replace(ts, budgets_s=_floats(args.budgets))
    if getattr(args, "factors", None):
        ww = replace(ww, factors=_floats(args.factors))
    if getattr(args, "mode", None):
        ww = replace(ww, mode=args.mode)
    if getattr(args, "tolerances", None):
        tokens = [t.strip() for t in args.tolerances.split(",") if t.strip()]
        ed = replace(
            ed,
            tolerances=tuple(float(t) for t in tokens if t.lower() != "full"),
            include_full=any(t.lower() == "full" for t in tokens),
        )
    if getattr(args, "replicates", None) is not None:
        ed = replace(ed, replicates=args.replicates)
        ev = replace(ev, replicates=args.replicates)
    return ExperimentConfig(
        solver=solver, time_sensitivity=ts, window_widening=ww, early_delivery=ed, evaluation=ev, v2=cfg.v2,
        phase_c=cfg.phase_c,
    )
