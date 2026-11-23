"""Testes com fixture sintetica pequena: sem dados locais nem rede."""

import json

import numpy as np
import pandas as pd
import pytest

from freq_sinistros import audit
from freq_sinistros.arff import read_arff
from freq_sinistros.labels import describe_column

ARFF = """@relation toy
@attribute IDpol numeric
@attribute ClaimNb numeric
@attribute Exposure numeric
@attribute Area {'A','B'}
@attribute VehGas string
@data
1,0,0.5,'A','Diesel'
2,1,1.0,'B','Regular'
1e+05,5,1.5,'A','Diesel'
4,0,?,'B','Regular'
5,x,0.2,'A','Diesel'
"""


@pytest.fixture
def toy(tmp_path):
    path = tmp_path / "toy.arff"
    path.write_text(ARFF, encoding="utf-8")
    return read_arff(path, chunk_rows=2)


def test_reader_keeps_every_row_and_counts_conversions(toy):
    diag = toy.diagnostics
    assert diag["raw_data_lines"] == diag["parsed_rows"] == 5
    assert diag["rows_rejected"] == 0
    assert diag["numeric"]["Exposure"]["missing"] == 1
    assert diag["numeric"]["ClaimNb"]["non_numeric"] == 1
    assert diag["chunks"] == 3


def test_scientific_id_is_exact_but_flagged(toy):
    assert 100000 in set(toy.frame["IDpol"])
    assert toy.diagnostics["id_noncanonical_tokens"] == ["1e+05"]
    assert toy.diagnostics["id_non_integer_or_unsafe"] == 0


def test_validity_masks():
    claim = pd.Series([0, 1, 2.5, -1, np.nan, np.inf])
    assert claim_list(audit.claim_valid_mask(claim)) == [
        True, True, False, False, False, False]
    expo = pd.Series([0.0, 0.5, -1, np.nan, np.inf, 2.0])
    assert claim_list(audit.exposure_valid_mask(expo)) == [
        False, True, False, False, False, True]


def claim_list(mask):
    return [bool(x) for x in mask]


def test_bands_cover_all_rows_without_overlap():
    expo = pd.Series([0.01, 0.02, 0.05, 0.2, 0.5, 0.75, 0.99, 1.0, 1.2,
                      0.0, np.nan])
    bands = audit.assign_band(expo).tolist()
    assert bands == ["(0, 0.02]", "(0, 0.02]", "(0.02, 0.08]",
                     "(0.08, 0.25]", "(0.25, 0.50]", "(0.50, 0.75]",
                     "(0.75, 1)", "= 1", "> 1", audit.INVALID_BAND,
                     audit.INVALID_BAND]


def synthetic_frame():
    return pd.DataFrame({
        "IDpol": [1, 2, 3, 4, 5],
        "ClaimNb": [0.0, 1.0, 6.0, 0.0, 2.0],
        "Exposure": [0.5, 1.0, 1.5, np.nan, 0.01],
        "Area": pd.Categorical(["A", "B", "A", "B", "A"]),
    })


def test_band_table_reconciles_with_raw_totals():
    frame = synthetic_frame()
    bands = audit.band_table(frame)
    total = bands[bands["band"] == "total"].iloc[0]
    assert total["rows"] == 5
    invalid = bands[bands["band"] == audit.INVALID_BAND].iloc[0]
    assert invalid["rows"] == 1 and invalid["rows_rate_subset"] == 0
    assert total["rows_rate_subset"] == 4
    assert total["claims_rate_subset"] == 9.0
    assert total["exposure_rate_subset"] == pytest.approx(3.01)
    assert total["zeros_rate_subset"] == 1  # zero valido, nao descartado
    rec = audit.reconciliation(frame, bands)
    assert rec["ok"].all()


def test_clip_sensitivity_is_aggregate_on_excess():
    table = audit.clip_sensitivity(synthetic_frame())
    both = table[table["scenario"] == "ClaimNb<=4 e Exposure<=1"].iloc[0]
    assert both["claims_excess_removed"] == 2.0   # 6 -> 4
    assert both["exposure_excess_removed"] == pytest.approx(0.5)
    assert both["rows_affected"] == 1
    raw = table[table["scenario"] == "bruto"].iloc[0]
    assert raw["total_claims"] == 9.0


def test_id_audit_flags_repeated_and_conflicting_ids():
    frame = pd.DataFrame({
        "IDpol": [1, 1, 2, 2, 3],
        "ClaimNb": [0.0, 0.0, 1.0, 2.0, 0.0],
        "Exposure": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    summary, rows = audit.id_audit(frame)
    assert summary["ids_distinct"] == 3
    assert summary["ids_repeated"] == 2
    assert summary["rows_in_repeated_ids"] == 4
    assert summary["fully_duplicated_rows_incl_id"] == 2
    assert summary["repeated_id_groups_with_conflicting_values"] == 1
    assert len(rows) == 4


def test_claimnb_summary_moments_and_tail():
    frame = synthetic_frame()
    summary = audit.claimnb_summary(
        frame, {"missing": 0, "non_numeric": 0})
    assert summary["valid_rows"] == 5
    assert summary["zeros"] == 2
    assert summary["total_claims"] == 9.0
    assert summary["mean"] == pytest.approx(1.8)
    assert summary["variance_ddof1"] == pytest.approx(
        np.var([0, 1, 6, 0, 2], ddof=1))
    assert summary["rows_gt_4"] == 1
    assert summary["claims_in_rows_gt_4"] == 6.0


def test_audit_frame_end_to_end_on_toy(toy):
    summary, tables = audit.audit_frame(toy)
    assert summary["production_eligible"] is False
    assert summary["dimensions"]["rows"] == 5
    assert tables["reconciliation"]["ok"].all()
    assert summary["claimnb"]["validity"]["non_numeric"] == 1


def test_write_outputs_is_deterministic(toy, tmp_path):
    summary, tables = audit.audit_frame(toy)
    audit.write_outputs(tmp_path / "a", summary, tables)
    audit.write_outputs(tmp_path / "b", summary, tables)
    for path in (tmp_path / "a").iterdir():
        assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes()


def test_labels_bilingual():
    assert describe_column("Exposure", "en_us")["unit"] == "years of coverage"
    assert "INDETERMINADA" in describe_column("BonusMalus")["availability"]
    with pytest.raises(ValueError):
        describe_column("Exposure", "fr_fr")


def _read_toy(tmp_path, rows):
    header = ARFF.split("@data")[0] + "@data\n"
    path = tmp_path / "t.arff"
    path.write_text(header + rows, encoding="utf-8")
    return read_arff(path)


def test_clip_sensitivity_all_zero_claims_keeps_zero_rate(tmp_path):
    frame = pd.DataFrame({"IDpol": [1, 2], "ClaimNb": [0.0, 0.0],
                          "Exposure": [0.5, 1.0]})
    table = audit.clip_sensitivity(frame)
    assert (table["rate_per_100_policy_years"] == 0.0).all()
    assert table["claims_change_pct"].isna().all()
    assert table["rate_change_pct"].isna().all()
    assert table["exposure_change_pct"].notna().all()


def test_clip_sensitivity_without_valid_rows_is_missing_not_error():
    frame = pd.DataFrame({"IDpol": [1, 2], "ClaimNb": [1.0, 2.0],
                          "Exposure": [np.nan, -1.0]})
    table = audit.clip_sensitivity(frame)
    assert len(table) == 4
    assert table["rate_per_100_policy_years"].isna().all()
    assert table["rate_change_pct"].isna().all()
    assert (table["rows_in_scope"] == 0).all()


def test_safe_ratio():
    assert audit.safe_ratio(0.0, 2.0) == 0.0
    assert np.isnan(audit.safe_ratio(1.0, 0.0))
    assert np.isnan(audit.safe_ratio(1.0, float("nan")))


def test_audit_terminates_when_all_claims_are_zero(tmp_path):
    rows = "1,0,0.5,'A','Diesel'\n2,0,1.0,'B','Regular'\n"
    result = _read_toy(tmp_path, rows)
    summary, tables = audit.audit_frame(result)
    assert summary["claimnb"]["total_claims"] == 0.0
    assert summary["claimnb"]["zeros"] == 2
    assert summary["rate_subset"]["rate_per_100_policy_years"] == 0.0
    assert tables["reconciliation"]["ok"].all()
    audit.write_outputs(tmp_path / "o", summary, tables)


def test_audit_terminates_without_valid_rate_rows(tmp_path):
    result = _read_toy(tmp_path, "1,1,?,'A','Diesel'\n2,x,0.5,'B','Regular'\n")
    summary, tables = audit.audit_frame(result)
    assert summary["rate_subset"]["rows"] == 0
    assert np.isnan(summary["rate_subset"]["rate_per_100_policy_years"])
    assert summary["rate_subset"]["rows_outside_subset"] == 2
    assert tables["reconciliation"]["ok"].all()
    audit.write_outputs(tmp_path / "o", summary, tables)
    written = json.loads((tmp_path / "o" / "audit_summary.json").read_text())
    assert written["rate_subset"]["rate_per_100_policy_years"] is None


def test_covariate_availability_is_a_hypothesis_when_date_is_omitted():
    for name in ("Area", "VehAge", "DrivAge", "Density", "Region"):
        for lang in ("pt_br", "en_us"):
            text = describe_column(name, lang)["availability"]
            assert text.startswith(("hipotese", "hypothesis"))
