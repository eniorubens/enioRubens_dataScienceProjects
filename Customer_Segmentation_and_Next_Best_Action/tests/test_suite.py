"""
tests/test_suite.py
-------------------
Unit and integration tests for customer_segmentation_nba_v2.

Run with:
    pytest tests/ -v --tb=short
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Raw-transaction dataframe with enough customers for clustering/ML tests.

    Contains deliberate dirty rows (cancellation, missing ID, bad quantity/price,
    null description) alongside 20 valid transactions across 8 customers.
    """
    rng = np.random.default_rng(42)
    n_valid = 20
    customer_ids = rng.choice([11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888], n_valid)
    quantities   = rng.integers(1, 20, n_valid).tolist()
    prices       = rng.uniform(0.5, 50.0, n_valid).round(2).tolist()
    dates        = pd.date_range("2010-12-01", periods=n_valid, freq="D").astype(str).tolist()
    invoices     = [f"{i:04d}" for i in range(1, n_valid + 1)]
    stock_codes  = [f"SC{i:02d}" for i in range(n_valid)]
    descriptions = [f"Product {i}" for i in range(n_valid)]
    countries    = rng.choice(["UK", "Germany", "France"], n_valid).tolist()

    valid_rows = pd.DataFrame({
        "Invoice":     invoices,
        "StockCode":   stock_codes,
        "Description": descriptions,
        "Quantity":    quantities,
        "InvoiceDate": dates,
        "Price":       prices,
        "Customer ID": customer_ids.astype(float),
        "Country":     countries,
    })

    dirty_rows = pd.DataFrame({
        "Invoice":     ["C900", "901",  "902",  "903"],
        "StockCode":   ["X1",   "X2",   "X3",   "X4"],
        "Description": ["Ret",  None,   "Adj",  "Bad"],
        "Quantity":    [-1,     2,      0,      3],
        "InvoiceDate": ["2010-12-25"] * 4,
        "Price":       [5.0,    3.0,    2.0,    -1.0],
        "Customer ID": [99999.0, np.nan, 99998.0, 99997.0],
        "Country":     ["UK"] * 4,
    })

    return pd.concat([valid_rows, dirty_rows], ignore_index=True)


@pytest.fixture
def clean_df(sample_raw_df) -> pd.DataFrame:
    from src.data import clean
    return clean(sample_raw_df, verbose=False)


@pytest.fixture
def rfm_df(clean_df) -> pd.DataFrame:
    from src.data import build_rfm
    return build_rfm(clean_df)


@pytest.fixture
def rfm_with_proba(rfm_df) -> pd.DataFrame:
    """RFM table with synthetic prediction columns for prescriptive tests."""
    segment_pool = [
        "Champions", "Loyal Customers", "High Value at Risk",
        "Occasional Buyers", "Low Value", "Inactive Customers",
    ]
    df = rfm_df.copy()
    n = len(df)
    rng = np.random.default_rng(42)
    df["prob_repurchase_90d"] = rng.uniform(0.1, 0.9, n)
    # Cycle through segment pool to fill all rows
    df["PredictiveSegment"] = [segment_pool[i % len(segment_pool)] for i in range(n)]
    return df


# ══════════════════════════════════════════════════════════════════════════
# multilang
# ══════════════════════════════════════════════════════════════════════════

class TestMultilang:
    def test_default_language_is_en(self):
        from src.multilang import get_language, set_language
        set_language("en")
        assert get_language() == "en"

    def test_t_returns_original_in_en(self):
        from src.multilang import set_language, t
        set_language("en")
        assert t("Customers") == "Customers"

    def test_t_translates_to_pt(self):
        from src.multilang import set_language, t
        set_language("pt")
        assert t("Customers") == "Clientes"
        set_language("en")  # restore

    def test_t_falls_back_for_missing_key(self):
        from src.multilang import set_language, t
        set_language("pt")
        assert t("__nonexistent_key__") == "__nonexistent_key__"
        set_language("en")

    def test_unsupported_language_raises(self):
        from src.multilang import set_language
        with pytest.raises(ValueError, match="not supported"):
            set_language("fr")

    def test_segment_names_translate(self):
        from src.multilang import set_language, t
        set_language("pt")
        assert t("Champions") == "Campeões"
        assert t("Inactive Customers") == "Clientes Inativos"
        set_language("en")


# ══════════════════════════════════════════════════════════════════════════
# data — clean()
# ══════════════════════════════════════════════════════════════════════════

class TestClean:
    def test_removes_missing_customer_id(self, sample_raw_df):
        from src.data import clean
        result = clean(sample_raw_df, verbose=False)
        assert result["Customer ID"].notna().all()

    def test_removes_cancellations(self, sample_raw_df):
        from src.data import clean
        result = clean(sample_raw_df, verbose=False)
        assert not result["Invoice"].astype(str).str.startswith("C").any()

    def test_removes_non_positive_quantity(self, sample_raw_df):
        from src.data import clean
        result = clean(sample_raw_df, verbose=False)
        assert (result["Quantity"] > 0).all()

    def test_removes_non_positive_price(self, sample_raw_df):
        from src.data import clean
        result = clean(sample_raw_df, verbose=False)
        assert (result["Price"] > 0).all()

    def test_removes_null_description(self, sample_raw_df):
        from src.data import clean
        result = clean(sample_raw_df, verbose=False)
        assert result["Description"].notna().all()

    def test_revenue_column_created(self, clean_df):
        assert "Revenue" in clean_df.columns
        assert (clean_df["Revenue"] == clean_df["Quantity"] * clean_df["Price"]).all()

    def test_customer_id_is_string(self, clean_df):
        # pandas 2.x may return StringDtype instead of object — both are string-like
        assert pd.api.types.is_string_dtype(clean_df["Customer ID"])

    def test_result_is_non_empty(self, clean_df):
        assert len(clean_df) > 0


# ══════════════════════════════════════════════════════════════════════════
# data — build_rfm()
# ══════════════════════════════════════════════════════════════════════════

class TestBuildRFM:
    def test_one_row_per_customer(self, clean_df, rfm_df):
        assert rfm_df["Customer ID"].nunique() == len(rfm_df)

    def test_required_columns_present(self, rfm_df):
        required = {"Customer ID", "Recency", "Frequency", "Monetary",
                    "AverageTicket", "QuantityTotal", "UniqueProducts", "CountryMode"}
        assert required.issubset(rfm_df.columns)

    def test_recency_non_negative(self, rfm_df):
        assert (rfm_df["Recency"] >= 0).all()

    def test_frequency_positive(self, rfm_df):
        assert (rfm_df["Frequency"] > 0).all()

    def test_monetary_positive(self, rfm_df):
        assert (rfm_df["Monetary"] > 0).all()

    def test_average_ticket_no_division_by_zero(self, rfm_df):
        assert rfm_df["AverageTicket"].notna().all()
        assert np.isfinite(rfm_df["AverageTicket"]).all()


# ══════════════════════════════════════════════════════════════════════════
# segmentation
# ══════════════════════════════════════════════════════════════════════════

class TestSegmentation:
    @pytest.fixture
    def scaled_array(self, rfm_df) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        features = rfm_df[["Recency", "Frequency", "Monetary"]].values
        return StandardScaler().fit_transform(features)

    def test_fit_kmeans_returns_correct_shape(self, scaled_array, rfm_df):
        from src.segmentation import fit_kmeans
        labels, inertia = fit_kmeans(scaled_array, k=2)
        assert len(labels) == len(rfm_df)
        assert inertia > 0

    def test_silhouette_between_minus1_and_1(self, scaled_array):
        from src.segmentation import fit_kmeans, silhouette
        labels, _ = fit_kmeans(scaled_array, k=2)
        score = silhouette(scaled_array, labels)
        assert -1.0 <= score <= 1.0

    def test_search_k_returns_dataframe(self, scaled_array):
        from src.segmentation import search_k
        result = search_k(scaled_array, k_range=range(2, 4))
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"k", "inertia", "silhouette_score"}
        assert len(result) == 2

    def test_segment_names_returns_dict(self):
        from src.segmentation import segment_names
        profiles = pd.DataFrame({
            "Cluster":   [0, 1, 2],
            "Recency":   [10, 90, 300],
            "Frequency": [20,  5,   1],
            "Monetary":  [5000, 800, 100],
        })
        names = segment_names(profiles)
        assert isinstance(names, dict)
        assert set(names.keys()) == {0, 1, 2}
        assert "Champions" in names.values()

    def test_validate_segment_names_warns_on_low_monetary(self):
        from src.segmentation import validate_segment_names
        # Two clusters: one high-value labelled Champions with LOW monetary → should warn
        profiles = pd.DataFrame({
            "Cluster":  [0, 1],
            "Monetary": [1000.0, 1.0],  # median = 500.5; cluster 1 << median
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_segment_names(profiles, {1: "Champions"})
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1


# ══════════════════════════════════════════════════════════════════════════
# segmentation — KMeansClusterAdder
# ══════════════════════════════════════════════════════════════════════════

class TestKMeansClusterAdder:
    @pytest.fixture
    def adder(self):
        from src.segmentation import KMeansClusterAdder
        return KMeansClusterAdder(n_clusters=2, n_init=5)

    @pytest.fixture
    def X(self, rfm_df) -> pd.DataFrame:
        return rfm_df[["Recency", "Frequency", "Monetary"]].copy()

    def test_fit_returns_self(self, adder, X):
        result = adder.fit(X)
        assert result is adder

    def test_transform_adds_column(self, adder, X):
        adder.fit(X)
        out = adder.transform(X)
        assert "PredictiveCluster" in out.columns

    def test_transform_before_fit_raises(self, X):
        from sklearn.exceptions import NotFittedError
        from src.segmentation import KMeansClusterAdder
        adder = KMeansClusterAdder(n_clusters=2)
        with pytest.raises(NotFittedError):
            adder.transform(X)

    def test_cluster_labels_are_strings(self, adder, X):
        adder.fit(X)
        out = adder.transform(X)
        # pandas 2.x may use StringDtype; both are string-like
        assert pd.api.types.is_string_dtype(out["PredictiveCluster"])

    def test_no_leakage_between_train_test(self, adder, rfm_df):
        """Transformer fitted on train must still work on unseen test rows."""
        X = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
        # Ensure at least 1 row in each split
        if len(X) < 4:
            pytest.skip("Not enough customers in fixture for train/test split")
        split = max(1, len(X) // 2)
        train, test = X.iloc[:split], X.iloc[split:]
        adder.fit(train)
        out = adder.transform(test)
        assert "PredictiveCluster" in out.columns
        assert len(out) == len(test)


# ══════════════════════════════════════════════════════════════════════════
# prediction
# ══════════════════════════════════════════════════════════════════════════

class TestPrediction:
    @pytest.fixture
    def pipeline(self, rfm_df):
        from src.prediction import make_repurchase_pipeline
        numeric    = ["Recency", "Frequency", "Monetary", "AverageTicket"]
        cluster    = ["Recency", "Frequency", "Monetary"]
        log_feats  = ["Monetary", "AverageTicket"]
        estimator  = LogisticRegression(max_iter=200, random_state=42)
        return make_repurchase_pipeline(estimator, numeric, cluster, log_feats, selected_k=2)

    @pytest.fixture
    def X_y(self, rfm_df):
        X = rfm_df[["Recency", "Frequency", "Monetary", "AverageTicket", "CountryMode"]].copy()
        rng = np.random.default_rng(0)
        n = len(X)
        # Guarantee both classes present by forcing at least one of each
        labels = rng.integers(0, 2, n)
        labels[0] = 0
        labels[1] = 1
        y = pd.Series(labels, name="target")
        return X, y

    def test_pipeline_fits_without_error(self, pipeline, X_y):
        X, y = X_y
        pipeline.fit(X, y)

    def test_predict_proba_shape(self, pipeline, X_y):
        X, y = X_y
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, pipeline, X_y):
        X, y = X_y
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_classifier_keys(self, pipeline, X_y):
        from src.prediction import evaluate_classifier
        X, y = X_y
        pipeline.fit(X, y)
        metrics = evaluate_classifier("LR", "Test", pipeline, X, y)
        assert {"Model", "Sample", "ROC_AUC", "Recall", "Precision",
                "Accuracy", "BrierScore"}.issubset(metrics.keys())


# ══════════════════════════════════════════════════════════════════════════
# prescriptive
# ══════════════════════════════════════════════════════════════════════════

class TestPrescriptive:
    def test_simulate_actions_row_count(self, rfm_with_proba):
        from src.prescriptive import simulate_actions
        from src.config import INCENTIVE_CATALOGUE
        result = simulate_actions(rfm_with_proba)
        assert len(result) == len(rfm_with_proba) * len(INCENTIVE_CATALOGUE)

    def test_select_best_action_one_per_customer(self, rfm_with_proba):
        from src.prescriptive import select_best_action, simulate_actions
        sim    = simulate_actions(rfm_with_proba)
        best   = select_best_action(sim)
        assert len(best) == len(rfm_with_proba)

    def test_expected_profit_no_action_baseline(self, rfm_with_proba):
        from src.prescriptive import simulate_actions
        from src.config import MARGIN_RATE
        sim = simulate_actions(rfm_with_proba)
        no_action = sim.loc[sim["Action"] == "No Action"]
        expected  = no_action["AdjustedProb"] * rfm_with_proba.set_index("Customer ID").loc[
            no_action["Customer ID"], "Monetary"
        ].values * MARGIN_RATE
        np.testing.assert_allclose(
            no_action["ExpectedProfit"].values,
            expected,
            rtol=1e-5,
        )

    def test_build_recommendation_returns_string(self, rfm_with_proba):
        from src.prescriptive import build_recommendation, select_best_action, simulate_actions
        sim  = simulate_actions(rfm_with_proba)
        best = select_best_action(sim)
        best = best.merge(
            rfm_with_proba[["Customer ID", "PredictiveSegment"]],
            on="Customer ID", how="left",
        )
        rec = build_recommendation(best.iloc[0])
        assert isinstance(rec, str)
        assert len(rec) > 0

    def test_allocate_budget_respects_limit(self, rfm_with_proba):
        from src.prescriptive import allocate_budget, select_best_action, simulate_actions
        sim  = simulate_actions(rfm_with_proba)
        best = select_best_action(sim)
        result = allocate_budget(best, budget=5.0)
        allocated_cost = result.loc[result["BudgetAllocated"], "incentive_cost"].sum()
        assert allocated_cost <= 5.0 + 1e-6

    def test_budget_curve_length(self, rfm_with_proba):
        from src.prescriptive import build_budget_curve, select_best_action, simulate_actions
        sim   = simulate_actions(rfm_with_proba)
        best  = select_best_action(sim)
        curve, breakpoints = build_budget_curve(best, n_points=10)
        assert len(curve) == 10
        assert len(breakpoints) == 4

    def test_sensitivity_analysis_noise_levels(self, rfm_with_proba):
        from src.prescriptive import select_best_action, sensitivity_analysis, simulate_actions
        sim    = simulate_actions(rfm_with_proba)
        best   = select_best_action(sim)
        best   = best.merge(
            rfm_with_proba[["Customer ID", "prob_repurchase_90d", "Monetary"]],
            on="Customer ID", how="left",
        )
        result = sensitivity_analysis(best, noise_levels=[-0.10, 0.0, 0.10])
        assert len(result) == 3
        assert "NoiseDelta" in result.columns


# ══════════════════════════════════════════════════════════════════════════
# Integration: data → rfm → segmentation
# ══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_clean_to_rfm_pipeline(self, sample_raw_df):
        from src.data import build_rfm, clean
        cleaned = clean(sample_raw_df, verbose=False)
        rfm     = build_rfm(cleaned)
        assert len(rfm) > 0
        assert "Recency" in rfm.columns

    def test_rfm_to_cluster_adder(self, rfm_df):
        from src.segmentation import KMeansClusterAdder
        X = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
        adder = KMeansClusterAdder(n_clusters=2, n_init=5)
        adder.fit(X)
        out = adder.transform(X)
        assert "PredictiveCluster" in out.columns
        assert out["PredictiveCluster"].nunique() <= 2
