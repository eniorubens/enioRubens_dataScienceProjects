import zipfile
from dataclasses import replace

import pytest
from conftest import ROWS, make_text

from delivery_routing_vrptw import (
    family_of,
    parse_instance,
    read_archive,
    read_instance,
    read_references,
    validate_instance,
)


def test_parse(inst):
    assert inst.name == "R101"
    assert inst.n_customers == 5
    assert inst.vehicle_limit == 3 and inst.capacity == 10
    assert inst.horizon == 100
    assert inst.family == "R1"
    assert inst.demand.tolist() == [0, 5, 5, 6, 5, 4]
    assert inst.due[3] == 70 and inst.service[5] == 0


def test_read_instance(tmp_path, text):
    path = tmp_path / "r101.txt"
    path.write_text(text, encoding="utf-8")
    assert read_instance(path).n_customers == 5


def test_bad_columns():
    rows = [*ROWS[:-1], (5, 0, -5, 4, 30, 90)]
    with pytest.raises(ValueError, match="7 columns"):
        parse_instance(make_text(rows=rows))


def test_bad_header():
    bad = make_text().replace("  3    10", "  three")
    with pytest.raises(ValueError, match="VEHICLE"):
        parse_instance(bad)


def test_demand_above_capacity():
    rows = [*ROWS[:-1], (5, 0, -5, 11, 30, 90, 0)]
    with pytest.raises(ValueError, match="exceeds"):
        parse_instance(make_text(rows=rows))


def test_ready_after_due():
    rows = [*ROWS[:-1], (5, 0, -5, 4, 95, 90, 0)]
    with pytest.raises(ValueError, match="ready"):
        parse_instance(make_text(rows=rows))


def test_non_contiguous_ids():
    rows = [*ROWS[:-1], (7, 0, -5, 4, 30, 90, 0)]
    with pytest.raises(ValueError, match="contiguous"):
        parse_instance(make_text(rows=rows))


def test_window_after_horizon():
    rows = [*ROWS[:-1], (5, 0, -5, 4, 30, 101, 0)]
    with pytest.raises(ValueError, match="horizon"):
        parse_instance(make_text(rows=rows))


def test_depot_and_negative_contract(inst):
    bad = inst.demand.copy()
    bad[0] = 1
    with pytest.raises(ValueError, match="depot"):
        validate_instance(replace(inst, demand=bad))
    bad = inst.service.copy()
    bad[2] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_instance(replace(inst, service=bad))
    with pytest.raises(ValueError, match="positive"):
        validate_instance(replace(inst, capacity=0))


@pytest.mark.parametrize(
    ("name", "fam"),
    [("C101", "C1"), ("c208", "C2"), ("R105", "R1"), ("r211", "R2"), ("RC104", "RC1"), ("rc208", "RC2"),
     ("C101.25", "C1")],
)
def test_family_of(name, fam):
    assert family_of(name) == fam


def test_family_of_bad():
    with pytest.raises(ValueError):
        family_of("X999")


def test_subset(inst):
    sub = inst.subset(3)
    assert sub.n_customers == 3 and sub.name == "R101.3"
    assert sub.x.tolist() == inst.x[:4].tolist()
    with pytest.raises(ValueError):
        inst.subset(0)
    with pytest.raises(ValueError):
        inst.subset(6)


def test_read_references(tmp_path):
    path = tmp_path / "bks.csv"
    path.write_text("instance,vehicles,distance,reference,note\nr101,19,1650.80,RT,\nC101,10,828.94,RT,x\n")
    refs = read_references(path)
    assert refs["R101"].vehicles == 19
    assert refs["R101"].distance == pytest.approx(1650.80)
    assert refs["C101"].note == "x" and refs["C101"].source == "RT"


def test_read_archive(tmp_path):
    path = tmp_path / "solomon.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("In/r101.txt", make_text("R101"))
        zf.writestr("In/c101.txt", make_text("C101"))
        zf.writestr("readme.md", "ignored")
    instances = read_archive(path)
    assert list(instances) == ["C101", "R101"]
    assert instances["C101"].n_customers == 5

