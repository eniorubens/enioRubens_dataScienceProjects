"""Static checks for S0 public docstrings and source line width."""

import inspect
import pathlib
import unittest

import consumer_complaint_intelligence as package
from consumer_complaint_intelligence import audit, config, contracts, data
from consumer_complaint_intelligence import service, temporal_split, tracking
from consumer_complaint_intelligence import deduplication, s1, taxonomy
from consumer_complaint_intelligence import s3
from consumer_complaint_intelligence import s3_reporting
from consumer_complaint_intelligence import s4
from consumer_complaint_intelligence import s4_reporting
from consumer_complaint_intelligence import s5
from consumer_complaint_intelligence import s5_reporting
from consumer_complaint_intelligence import s6
from consumer_complaint_intelligence import s6_reporting
from consumer_complaint_intelligence import s7
from consumer_complaint_intelligence import s7_reporting
from consumer_complaint_intelligence import s8
from consumer_complaint_intelligence import s8_reporting


MODULES = (
    package,
    audit,
    config,
    contracts,
    data,
    deduplication,
    s1,
    service,
    taxonomy,
    temporal_split,
    tracking,
    s3,
    s3_reporting,
    s4,
    s4_reporting,
    s5,
    s5_reporting,
    s6,
    s6_reporting,
    s7,
    s7_reporting,
    s8,
    s8_reporting,
)


class DocumentationTests(unittest.TestCase):
    """Keep public S0 APIs documented and readable without extra tooling."""

    def test_public_definitions_have_docstrings(self) -> None:
        """Require PEP 257-style non-empty docstrings on public definitions."""

        missing: list[str] = []
        for module in MODULES:
            for name, member in inspect.getmembers(module):
                if name.startswith("_"):
                    continue
                if inspect.isfunction(member) or inspect.isclass(member):
                    if getattr(member, "__module__", None) == module.__name__:
                        if not inspect.getdoc(member):
                            missing.append(f"{module.__name__}.{name}")
                        if inspect.isclass(member):
                            for method_name, method in inspect.getmembers(
                                member, inspect.isfunction
                            ):
                                if not method_name.startswith("_"):
                                    if not inspect.getdoc(method):
                                        missing.append(
                                            f"{module.__name__}."
                                            f"{name}.{method_name}"
                                        )
        self.assertEqual(missing, [])

    def test_source_lines_are_controlled(self) -> None:
        """Keep source modules within an 88-character line budget."""

        source_root = pathlib.Path(__file__).parents[1] / "src"
        violations = []
        for path in source_root.rglob("*.py"):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if len(line) > 88:
                    violations.append(f"{path}:{number}:{len(line)}")
        self.assertEqual(violations, [])
