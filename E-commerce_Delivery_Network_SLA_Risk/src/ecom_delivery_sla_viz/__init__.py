"""Presentation-only Plotly charts for the delivery SLA notebooks.

This package never trains models, computes model metrics or reads data files.
It does not import ``ecom_delivery_sla``: keeping it separate leaves the source
fingerprint of the run cache untouched when a chart changes.
"""

from .charts import (
    plot_ablation,
    plot_expanding_folds,
    plot_gain_over_baseline,
    plot_label_availability,
    plot_metric_by_fold,
    plot_metric_intervals,
    plot_promise_regime,
    plot_psi,
)
from .labels import get_labels
from .routes import (
    node_table,
    plot_route_matrix,
    plot_route_network,
    plot_top_city_routes,
    route_pairs,
    route_table,
    wilson_interval,
)

__all__ = [
    "get_labels",
    "node_table",
    "plot_ablation",
    "plot_expanding_folds",
    "plot_gain_over_baseline",
    "plot_label_availability",
    "plot_metric_by_fold",
    "plot_metric_intervals",
    "plot_promise_regime",
    "plot_psi",
    "plot_route_matrix",
    "plot_route_network",
    "plot_top_city_routes",
    "route_pairs",
    "route_table",
    "wilson_interval",
]
