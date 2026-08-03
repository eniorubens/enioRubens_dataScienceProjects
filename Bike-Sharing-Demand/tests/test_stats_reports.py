"""Tests for the nb02 compact report-subset builders in src/stats_tests.py."""

from __future__ import annotations

import pandas as pd

from src import stats_tests as st


def test_seasonal_report_table_selects_columns():
    results = pd.DataFrame(
        {
            "Test": ["A"],
            "N": ["1/2"],
            "Group medians": ["0.1 / 0.2"],
            "Effect measure": ["r"],
            "Effect size": [0.3],
            "Reference": ["Normal"],  # extra column dropped by the report
            "p-value": [0.01],
            "Holm p-value": [0.02],
            "Holm Decision": ["Reject H0"],
        }
    )
    out = st.seasonal_report_table(results)
    assert "Reference" not in out.columns
    assert list(out.columns) == st._SEASONAL_REPORT_COLS


def test_chi_square_report_sorted_by_cramers_v():
    scores = pd.DataFrame(
        {
            "Cramer's V": [0.1, 0.5, 0.3],
            "Chi Squared Score": [1, 2, 3],
            "df": [1, 1, 1],
            "Min expected count": [10, 20, 30],
            "Expected cells < 5 (%)": [0, 0, 0],
            "Holm p-value": [0.1, 0.01, 0.05],
            "Holm Decision": ["x", "y", "z"],
        },
        index=["a", "b", "c"],
    )
    out = st.chi_square_report(scores)
    assert list(out["Cramer's V"]) == [0.5, 0.3, 0.1]


def test_anova_and_f_reports_columns():
    anova = pd.DataFrame(
        {
            "Eta squared": [0.2, 0.4],
            "ANOVA Score": [1, 2],
            "df1": [4, 4],
            "df2": [10, 10],
            "Holm p-value": [0.1, 0.01],
            "Holm Decision": ["a", "b"],
        }
    )
    a_out = st.anova_report(anova)
    assert list(a_out["Eta squared"]) == [0.4, 0.2]

    freg = pd.DataFrame(
        {
            "Univariate R2": [0.3, 0.1],
            "F Score": [5, 2],
            "Relative F share (%)": [70, 30],
            "Holm p-value": [0.01, 0.2],
            "Holm Decision": ["a", "b"],
        }
    )
    f_out = st.f_regression_report(freg)
    assert list(f_out.columns) == st._F_REGRESSION_REPORT_COLS


def test_localize_report_renames_headers_and_maps_decisions_without_mutation():
    internal = pd.DataFrame(
        {
            "Test": ["Mann-Whitney: A vs B"],
            "Effect size": [0.42],
            "p-value": [0.001],
            "Holm Decision": ["Reject H0"],
        }
    )
    before = internal.copy()

    display_df = st.localize_report(internal)

    assert list(display_df.columns) == ["Teste", "Tamanho do efeito", "valor-p", "Decisão de Holm"]
    assert display_df["Tamanho do efeito"].iloc[0] == 0.42  # numbers unchanged
    assert display_df["Decisão de Holm"].iloc[0] == "Rejeita H₀"
    pd.testing.assert_frame_equal(internal, before)  # internal schema untouched


def test_localize_report_handles_fail_to_reject_and_not_evaluated():
    internal = pd.DataFrame({"Holm Decision": ["Fail to reject H0", "Not evaluated"]})
    display_df = st.localize_report(internal)
    assert list(display_df["Decisão de Holm"]) == ["Não rejeita H₀", "Não avaliado"]


def test_localize_report_translates_test_names_and_reference_without_mutation():
    internal = pd.DataFrame(
        {
            "Test": [
                "Mann-Whitney: Spring vs Winter",
                "Mann-Whitney: Summer vs Autumn",
                "Mann-Whitney: Summer vs Winter",
                "Kruskal-Wallis: All seasons",
                "Spearman: Demand vs Temperature",
                "Mann-Whitney: Holiday vs Non-Holiday",
            ],
            "Reference": [
                "Normal approximation",
                "Normal approximation",
                "Normal approximation",
                "Chi-square(df=3)",
                "t(df=1000)",
                "Normal approximation",
            ],
        }
    )
    before = internal.copy()

    display_df = st.localize_report(internal)

    assert list(display_df["Teste"]) == [
        "Mann-Whitney: Primavera vs Inverno",
        "Mann-Whitney: Verão vs Outono",
        "Mann-Whitney: Verão vs Inverno",
        "Kruskal-Wallis: Todas as estações",
        "Spearman: Demanda vs Temperatura",
        "Mann-Whitney: Feriado vs Não feriado",
    ]
    assert list(display_df["Referência"]) == [
        "Aproximação normal",
        "Aproximação normal",
        "Aproximação normal",
        "Chi-square(df=3)",  # parameterized reference, left as-is (universal notation)
        "t(df=1000)",
        "Aproximação normal",
    ]
    # internal schema (English test names / reference labels) is untouched
    pd.testing.assert_frame_equal(internal, before)
    assert "Test" in internal.columns and "Teste" not in internal.columns


def test_univariate_selection_summary_is_portuguese_and_preserves_numbers():
    chi_report = pd.DataFrame(
        {
            "Cramer's V": [0.363],
            "df": [4],
            "Min expected count": [64.4],
            "Holm p-value": [1.2e-10],
            "Holm Decision": ["Reject H0"],
        },
        index=["Time_Period"],
    )
    anova_report = pd.DataFrame(
        {
            "Eta squared": [0.280],
            "ANOVA Score": [1234.5],
            "Holm p-value": [1e-9],
            "Holm Decision": ["Reject H0"],
        },
        index=["Ground Temp(C)"],
    )
    f_report = pd.DataFrame(
        {
            "Univariate R2": [0.223, 0.000021],
            "F Score": [500.1, 0.02],
            "Relative F share (%)": [12.3, 0.01],
            "Holm p-value": [1e-8, 0.2127],
            "Holm Decision": ["Reject H0", "Fail to reject H0"],
        },
        index=["Temperature(C)", "Weekday_Monday"],
    )

    text = st.univariate_selection_summary(chi_report, anova_report, f_report, top_n=15)

    assert "CHI-SQUARE" in text
    assert "ANOVA" in text
    assert "F-REGRESSION" in text
    assert "não retidas por Holm" in text
    assert "Rejeita H₀" in text
    assert "Não rejeita H₀" in text
    assert "0.363" in text  # Cramer's V preserved
    assert "0.223" in text  # Univariate R2 preserved
    assert "Weekday_Monday" in text  # excluded-by-Holm row surfaced
