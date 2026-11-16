import numpy as np
import pandas as pd

from ecom_delivery_sla.drift import feature_psi_table, population_stability_index


def test_population_stability_index_is_zero_for_identical_samples():
    values = np.arange(100, dtype=float)

    assert population_stability_index(values, values) == 0.0


def test_feature_psi_table_ranks_shifted_feature():
    expected = pd.DataFrame({"stable": np.arange(100), "shifted": np.arange(100)})
    actual = pd.DataFrame({"stable": np.arange(100), "shifted": np.arange(100) + 50})

    result = feature_psi_table(expected, actual)

    assert result.iloc[0]["feature"] == "shifted"
    assert result.iloc[0]["psi"] > 0
