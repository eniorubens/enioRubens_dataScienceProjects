"""Self-tests for select_tests.py (stdlib unittest; run by the detect job).

    python3 -m unittest discover -s .github/ci -p "test_*.py"
"""
from __future__ import annotations

import json
import unittest

import select_tests as st

CFG = st.load_config()
ALL = st.all_suites(CFG)
BIKE = "Bike-Sharing-Demand"
VRPTW = "Delivery_Routing_Time_Window_Optimization"


def ns(*entries: tuple[str, ...]) -> str:
    """Build a ``git diff --name-status -z`` dump."""
    return "".join("\0".join(e) + "\0" for e in entries)


def run(diff: str | None = None, full: str | None = None) -> dict:
    out = st.decide(CFG, full_reason=full, diff_text=diff)
    out["suites"] = set(json.loads(out["suites"]))
    out["matrix_projects"] = {p["project"] for p in json.loads(out["matrix"])["include"]}
    return out


class SelectionTests(unittest.TestCase):
    def assertSuites(self, out: dict, expected: set[str], mode: str = "selective") -> None:
        self.assertEqual(out["mode"], mode, out["reason"])
        self.assertEqual(out["suites"], expected)
        self.assertEqual(out["matrix_projects"], expected - {BIKE})
        self.assertEqual(out["bike"], "true" if BIKE in expected else "false")
        self.assertEqual(out["has_matrix"], "true" if expected - {BIKE} else "false")

    def test_vrptw_only(self):
        self.assertSuites(run(ns(("M", f"{VRPTW}/src/vrptw/solver.py"))), {VRPTW})

    def test_vrptw_notebook_is_in_scope(self):
        self.assertSuites(run(ns(("M", f"{VRPTW}/notebooks/en-US/01.ipynb"))), {VRPTW})

    def test_bike_only(self):
        self.assertSuites(run(ns(("M", f"{BIKE}/src/features.py"))), {BIKE})

    def test_two_projects(self):
        out = run(ns(("M", f"{VRPTW}/README.md"), ("A", "ModelRisk_ResponsibleML/tests/test_x.py")))
        self.assertSuites(out, {VRPTW, "ModelRisk_ResponsibleML"})

    def test_project_with_spaces(self):
        self.assertSuites(run(ns(("D", "Customer Lifetime Value/src/clv.py"))), {"Customer Lifetime Value"})

    def test_ds_toolkit_multilang(self):
        out = run(ns(("M", "ds_toolkit/multilang/multilang/core.py")))
        self.assertSuites(out, {"Marketing_Promotion_Campaign_Uplift_Modelling", BIKE, "Customer Lifetime Value"})

    def test_ds_toolkit_opt_binary(self):
        out = run(ns(("M", "ds_toolkit/opt_binary_clf_pipe/opt_binary_clf_pipe/pipe.py")))
        self.assertSuites(out, {"ds_toolkit/opt_binary_clf_pipe", "customer-churn-prediction"})

    def test_ds_toolkit_unmapped_is_full(self):
        self.assertSuites(run(ns(("A", "ds_toolkit/new_pkg/x.py"))), ALL, "full")
        self.assertSuites(run(ns(("M", "ds_toolkit/README.md"))), ALL, "full")

    def test_root_readme_only(self):
        self.assertSuites(run(ns(("M", "README.md"), ("M", "README.pt-BR.md"))), set())

    def test_showcase_and_license(self):
        self.assertSuites(run(ns(("M", "pages-publicacao/index.html"), ("M", "LICENSE"))), set())

    def test_project_without_suite(self):
        self.assertSuites(run(ns(("M", "CMAPSS_Predictive_Maintenance/site/index.html"))), set())

    def test_workflow_change_is_full(self):
        self.assertSuites(run(ns(("M", ".github/workflows/tests.yml"))), ALL, "full")
        self.assertSuites(run(ns(("M", ".github/ci/test-projects.json"))), ALL, "full")

    def test_global_config_is_full(self):
        self.assertSuites(run(ns(("M", ".gitignore"))), ALL, "full")
        self.assertSuites(run(ns(("A", "conftest.py"))), ALL, "full")

    def test_editorial_plus_code(self):
        self.assertSuites(run(ns(("M", "README.md"), ("M", f"{BIKE}/src/x.py"))), {BIKE})

    def test_rename_between_projects(self):
        out = run(ns(("R087", f"{VRPTW}/src/util.py", "E-commerce_Delivery_Network_SLA_Risk/src/util.py")))
        self.assertSuites(out, {VRPTW, "E-commerce_Delivery_Network_SLA_Risk"})

    def test_rename_out_of_editorial(self):
        out = run(ns(("R100", "pages-publicacao/data.csv", f"{BIKE}/data/data.csv")))
        self.assertSuites(out, {BIKE})

    def test_copy_counts_both_paths(self):
        out = run(ns(("C075", "Cross_Sell_Association_Rules/a.py", f"{VRPTW}/a.py")))
        self.assertSuites(out, {"Cross_Sell_Association_Rules", VRPTW})

    def test_prefix_is_not_a_match(self):
        # "Bike-Sharing-Demand2" must not be read as Bike-Sharing-Demand.
        self.assertSuites(run(ns(("A", f"{BIKE}2/x.py"))), ALL, "full")

    def test_diff_unavailable(self):
        self.assertSuites(run(None), ALL, "full")

    def test_malformed_diff(self):
        self.assertSuites(run("garbage\0path\0"), ALL, "full")
        self.assertSuites(run("R100\0only-one-path\0"), ALL, "full")

    def test_manual_dispatch(self):
        out = run(full="manual dispatch")
        self.assertSuites(out, ALL, "full")
        self.assertEqual(out["reason"], "manual dispatch")

    def test_empty_diff(self):
        self.assertSuites(run(""), set())

    def test_full_matrix_matches_config(self):
        out = run(full="x")
        include = json.loads(out["matrix"])["include"]
        settings = json.loads(out["settings"])
        self.assertEqual(include, [{"project": p["project"]} for p in CFG["matrix_projects"]])
        for p in CFG["matrix_projects"]:
            self.assertEqual(
                settings[p["project"]],
                {"python": p["python"], "install": p["install"], "prepare": p.get("prepare", "")},
            )

    def test_settings_follow_selection(self):
        out = run(ns(("M", f"{VRPTW}/README.md")))
        self.assertEqual(set(json.loads(out["settings"])), {VRPTW})
        self.assertEqual(json.loads(run(ns(("M", "README.md")))["settings"]), {})

if __name__ == "__main__":
    unittest.main()
