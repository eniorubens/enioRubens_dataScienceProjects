import argparse
import json

import pytest

from delivery_routing_vrptw import SolverConfig, parse_instance, verify
from delivery_routing_vrptw.config import (
    ExperimentConfig,
    SolverSection,
    TimeSensitivitySection,
    WindowWideningSection,
    add_config_args,
    default_config_path,
    load_config,
    resolve_config,
    select_instances,
    solve_configured,
    solver_config,
)

TOML = """
[solver]
time_limit_s = 7
first_solution = "SAVINGS"
metaheuristic = "GUIDED_LOCAL_SEARCH"
fleet_strategy = "fixed_cost"
workers = 2

[time_sensitivity]
budgets_s = [1, 2.5]
instance_set = "dev"

[window_widening]
factors = [1.0, 1.5]
mode = "later_due"
instance_set = "dev"
time_limit_s = 3
"""


def write(tmp_path, text=TOML):
    path = tmp_path / "exp.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.solver.time_limit_s == 7 and cfg.solver.workers == 2
    assert cfg.time_sensitivity.budgets_s == (1, 2.5)
    assert cfg.window_widening.factors == (1.0, 1.5) and cfg.window_widening.mode == "later_due"
    assert json.dumps(cfg.to_dict())


def test_missing_sections_use_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "[solver]\nworkers = 3\n"))
    assert cfg.solver.workers == 3 and cfg.window_widening.factors[0] == 1.0


def test_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        load_config(write(tmp_path, "[solver]\nbogus = 1\n"))
    with pytest.raises(ValueError, match="unknown section"):
        load_config(write(tmp_path, "[nope]\na = 1\n"))


@pytest.mark.parametrize("bad", [
    lambda: SolverSection(first_solution="NOPE"),
    lambda: SolverSection(metaheuristic="NOPE"),
    lambda: SolverSection(time_limit_s=0),
    lambda: SolverSection(fleet_strategy="x"),
    lambda: SolverSection(workers=0),
    lambda: TimeSensitivitySection(budgets_s=(10, 0)),
    lambda: TimeSensitivitySection(budgets_s=()),
    lambda: TimeSensitivitySection(instance_set="test"),
    lambda: WindowWideningSection(factors=(1.5, 2.0)),
    lambda: WindowWideningSection(factors=(0.9, 1.0)),
    lambda: WindowWideningSection(mode="x"),
    lambda: WindowWideningSection(instance_set="x"),
    lambda: WindowWideningSection(time_limit_s=0),
])
def test_validation(bad):
    with pytest.raises(ValueError):
        bad()


def test_solver_config():
    cfg = ExperimentConfig()
    sc = solver_config(cfg)
    assert isinstance(sc, SolverConfig) and sc.time_limit_s == 30.0 and sc.first_solution == "SAVINGS"
    assert solver_config(cfg, 4).time_limit_s == 4


def test_project_file_matches_frozen_config():
    cfg = load_config()
    frozen_path = default_config_path().parent.parent / "outputs" / "frozen" / "selected_config.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["config"]
    assert cfg.solver.first_solution == frozen["first_solution"]
    assert cfg.solver.metaheuristic == frozen["metaheuristic"]
    assert cfg.solver.fleet_strategy == frozen["fleet_strategy"]
    assert cfg.solver.time_limit_s == 30


def test_cli_overrides(tmp_path):
    parser = argparse.ArgumentParser()
    add_config_args(parser, budgets=True, factors=True)
    args = parser.parse_args([
        "--config", str(write(tmp_path)), "--time-limit", "2", "--workers", "5", "--budgets", "3,4",
        "--factors", "1.0,2.0", "--mode", "symmetric", "--set", "all",
    ])
    cfg = resolve_config(args)
    assert cfg.solver.time_limit_s == 2 and cfg.window_widening.time_limit_s == 2
    assert cfg.solver.workers == 5 and cfg.time_sensitivity.budgets_s == (3.0, 4.0)
    assert cfg.window_widening.factors == (1.0, 2.0) and cfg.window_widening.mode == "symmetric"
    assert cfg.time_sensitivity.instance_set == "all"
    plain = resolve_config(parser.parse_args(["--config", str(write(tmp_path))]))
    assert plain.solver.time_limit_s == 7 and plain.window_widening.time_limit_s == 3


def test_select_instances_guard():
    split = {"A": "dev", "B": "eval", "C": "dev"}
    assert select_instances(split, "dev") == ["A", "C"]
    assert select_instances(split, "dev", ["c"]) == ["C"]
    with pytest.raises(PermissionError):
        select_instances(split, "eval")
    with pytest.raises(PermissionError):
        select_instances(split, "all")
    assert select_instances(split, "all", allow_eval=True) == ["A", "B", "C"]
    with pytest.raises(ValueError):
        select_instances(split, "dev", ["B"])
    with pytest.raises(ValueError):
        select_instances({"B": "eval"}, "dev")


@pytest.mark.parametrize("fleet", ["fixed_cost", "reduce_fleet"])
def test_solve_configured(text, fleet):
    inst = parse_instance(text)
    cfg = ExperimentConfig(solver=SolverSection(time_limit_s=3, fleet_strategy=fleet, workers=1))
    sol = solve_configured(inst, cfg, time_limit_s=3)
    assert verify(inst, sol.routes).feasible


def test_early_delivery_section(tmp_path):
    from delivery_routing_vrptw.config import EarlyDeliverySection

    sec = EarlyDeliverySection()
    assert sec.levels() == [0.0, 0.25, 0.5, 1.0, float("inf")]
    assert EarlyDeliverySection(tolerances=(0.0, 1.0), include_full=False).levels() == [0.0, 1.0]
    for bad in (
        lambda: EarlyDeliverySection(tolerances=(0.5,)),
        lambda: EarlyDeliverySection(tolerances=(0.0, -1.0)),
        lambda: EarlyDeliverySection(tolerances=(0.0, float("inf"))),
        lambda: EarlyDeliverySection(tolerances=(0.0, 0.0)),
        lambda: EarlyDeliverySection(tolerances=()),
        lambda: EarlyDeliverySection(replicates=0),
        lambda: EarlyDeliverySection(instance_set="x"),
        lambda: EarlyDeliverySection(time_limit_s=0),
    ):
        with pytest.raises(ValueError):
            bad()
    text = TOML + "\n[early_delivery]\ntolerances = [0.0, 0.3]\ninclude_full = false\nreplicates = 2\n"
    cfg = load_config(write(tmp_path, text))
    assert cfg.early_delivery.tolerances == (0.0, 0.3) and not cfg.early_delivery.include_full
    assert json.dumps(cfg.to_dict())
    assert load_config().early_delivery.include_full


def test_early_cli_overrides(tmp_path):
    parser = argparse.ArgumentParser()
    add_config_args(parser, early=True)
    args = parser.parse_args(
        ["--config", str(write(tmp_path)), "--tolerances", "0,0.5,full", "--replicates", "2",
         "--time-limit", "4", "--set", "all"]
    )
    ed = resolve_config(args).early_delivery
    assert ed.tolerances == (0.0, 0.5) and ed.include_full and ed.replicates == 2
    assert ed.time_limit_s == 4 and ed.instance_set == "all"
    args2 = parser.parse_args(["--config", str(write(tmp_path)), "--tolerances", "0,1"])
    assert not resolve_config(args2).early_delivery.include_full


def test_evaluation_section_preregistered():
    cfg = load_config()
    ev = cfg.evaluation
    assert (ev.time_limit_s, ev.replicates, ev.instance_set, ev.include_baselines) == (120, 5, "eval", True)
    frozen_path = default_config_path().parent.parent / "outputs" / "frozen" / "selected_config.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["config"]
    assert {
        "first_solution": cfg.solver.first_solution,
        "metaheuristic": cfg.solver.metaheuristic,
        "fleet_strategy": cfg.solver.fleet_strategy,
    } == frozen
    assert cfg.solver.time_limit_s == 30


def test_evaluation_validation_and_overrides(tmp_path):
    from delivery_routing_vrptw.config import EvaluationSection

    for bad in (
        lambda: EvaluationSection(time_limit_s=0),
        lambda: EvaluationSection(replicates=0),
        lambda: EvaluationSection(instance_set="x"),
    ):
        with pytest.raises(ValueError):
            bad()
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args(["--config", str(write(tmp_path)), "--time-limit", "3", "--replicates", "2", "--set", "dev"])
    ev = resolve_config(args).evaluation
    assert (ev.time_limit_s, ev.replicates, ev.instance_set) == (3, 2, "dev")
