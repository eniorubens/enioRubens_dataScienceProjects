import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecom_delivery_sla.geo import haversine_km  # noqa: E402
from ecom_delivery_sla_viz import (  # noqa: E402
    get_labels,
    node_table,
    plot_route_matrix,
    plot_route_network,
    plot_top_city_routes,
    route_pairs,
    route_table,
    wilson_interval,
)

LANGUAGES = ["pt", "en"]


@pytest.fixture()
def tables():
    """Synthetic tables: SP sellers s1/s2, RJ seller s3; customers in SP, RJ, MG."""
    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1, 1, 2, 3, 4, 4, 5],
            "geolocation_lat": [-23.5, -23.7, -22.9, -19.9, -23.6, -23.4, -22.0],
            "geolocation_lng": [-46.6, -46.8, -43.2, -43.9, -46.7, -46.5, -43.0],
            "geolocation_city": ["sao paulo", "sao paulo", "rio", "bh", "sao paulo", "sao paulo", "rio"],
            "geolocation_state": ["SP", "SP", "RJ", "MG", "SP", "SP", "RJ"],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["s1", "s2", "s3"],
            "seller_zip_code_prefix": [1, 4, 2],
            "seller_city": ["sao paulo", "sao paulo", "rio"],
            "seller_state": ["SP", "SP", "RJ"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(1, 7)],
            "customer_unique_id": ["u1", "u2", "u3", "u4", "u5", "u1"],
            "customer_zip_code_prefix": [2, 2, 3, 1, 1, 2],
            "customer_city": ["rio", "rio", "bh", "sao paulo", "sao paulo", "rio"],
            "customer_state": ["RJ", "RJ", "MG", "SP", "SP", "RJ"],
        }
    )
    # o1: s1 twice (counts once) -> RJ; o2: s1 and s2 (two pairs) -> RJ;
    # o3: s1 -> MG; o4: s1 -> SP (self-loop); o5: s2 -> SP; o6: s3 -> RJ.
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o2", "o3", "o4", "o5", "o6"],
            "seller_id": ["s1", "s1", "s1", "s2", "s1", "s1", "s2", "s3"],
        }
    )
    supervised = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
            "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "sla_breach": [True, False, True, False, True, False],
            "promised_lead_days": [10.0, 12.0, 20.0, 5.0, 6.0, 8.0],
            "actual_lead_days": [12.0, 9.0, 25.0, 4.0, 7.0, 7.0],
        }
    )
    return supervised, items, sellers, customers, geolocation


@pytest.fixture()
def pairs(tables):
    return route_pairs(*tables)


def test_pairs_count_each_order_seller_once(pairs):
    assert len(pairs) == 7
    assert not pairs.duplicated(["order_id", "seller_id"]).any()
    o2 = pairs[pairs["order_id"] == "o2"]
    assert sorted(o2["seller_id"]) == ["s1", "s2"]
    assert (o2["destination_state"] == "RJ").all()
    assert set(pairs["sla_breach"].unique()) <= {0, 1}
    # Multi-seller order counted once per route.
    table = route_table(pairs, "state", min_orders=1)
    sp_rj = table[(table["origin"] == "SP") & (table["destination"] == "RJ")].iloc[0]
    assert sp_rj["orders"] == 3  # o1 (s1), o2 (s1), o2 (s2)
    assert sp_rj["breaches"] == 1


def test_pairs_distance_matches_haversine(pairs):
    expected = haversine_km(
        pairs["origin_lat"], pairs["origin_lng"], pairs["destination_lat"], pairs["destination_lng"]
    )
    assert np.max(np.abs(pairs["distance_km"].to_numpy() - expected)) < 1e-9


def test_pairs_coordinates_are_median_per_zip_prefix(pairs):
    s1 = pairs[pairs["seller_id"] == "s1"].iloc[0]
    assert s1["origin_lat"] == pytest.approx(-23.6)
    assert s1["origin_lng"] == pytest.approx(-46.7)


def test_pairs_take_outcome_columns_from_supervised(tables, pairs):
    supervised = tables[0]
    row = pairs[pairs["order_id"] == "o3"].iloc[0]
    assert row["promised_lead_days"] == 20.0
    assert row["actual_lead_days"] == 25.0
    assert row["sla_breach"] == 1
    assert supervised["order_id"].nunique() == pairs["order_id"].nunique()


def test_wilson_interval_matches_hand_computation():
    low, high = wilson_interval(np.array([5, 0, 10]), np.array([10, 10, 10]))
    # n=10, k=5: centre 0.5, half = z*sqrt(.25/10 + z^2/400)/(1+z^2/10)
    z = 1.959963984540054
    half = z * np.sqrt(0.25 / 10 + z**2 / 400) / (1 + z**2 / 10)
    assert low[0] == pytest.approx(0.5 - half)
    assert high[0] == pytest.approx(0.5 + half)
    assert low[0] == pytest.approx(0.2365931, abs=1e-6)
    assert high[0] == pytest.approx(0.7634069, abs=1e-6)
    # k=0, n=10: lower bound 0, upper z^2/(n+z^2)
    assert low[1] == 0.0
    assert high[1] == pytest.approx(z**2 / (10 + z**2))
    assert high[1] == pytest.approx(0.2775, abs=1e-4)
    assert 0 <= low[2] < high[2] and high[2] == pytest.approx(1.0)


def test_wilson_interval_empty_sample_is_nan():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_route_table_values_and_min_orders(pairs):
    table = route_table(pairs, "state", min_orders=1)
    assert table["orders"].sum() == len(pairs)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table["orders"].is_monotonic_decreasing
    row = table[(table["origin"] == "SP") & (table["destination"] == "MG")].iloc[0]
    assert row["breach_rate"] == 1.0 and row["breaches"] == 1
    assert row["ci_low"] <= row["breach_rate"] <= row["ci_high"]
    filtered = route_table(pairs, "state", min_orders=2)
    assert (filtered["orders"] >= 2).all()
    assert len(filtered) < len(table)
    # Share keeps the unfiltered denominator.
    assert filtered["share"].sum() < 1.0
    assert route_table(pairs, "state", min_orders=100).empty


def test_route_table_city_level_and_invalid_level(pairs):
    cities = route_table(pairs, "city", min_orders=1)
    assert "Sao Paulo (SP)" in set(cities["origin"])
    assert cities["orders"].sum() == len(pairs)
    with pytest.raises(ValueError):
        route_table(pairs, "region")


def test_self_loops_are_flagged_and_separated(pairs):
    table = route_table(pairs, "state", min_orders=1)
    loops = table[table["is_self_loop"]]
    assert set(loops["origin"]) == {"SP", "RJ"}
    figure = plot_route_network(pairs, geolocation=_geo(), thresholds=(1, 2))
    names = {trace.name for trace in figure.data if trace.meta in {"edge-line", "edge-marker"}}
    assert "SP→SP" not in names and "RJ→RJ" not in names
    assert "SP→RJ" in names


def _geo():
    return pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1, 1, 2, 3, 4, 4, 5],
            "geolocation_lat": [-23.5, -23.7, -22.9, -19.9, -23.6, -23.4, -22.0],
            "geolocation_lng": [-46.6, -46.8, -43.2, -43.9, -46.7, -46.5, -43.0],
            "geolocation_city": ["sao paulo", "sao paulo", "rio", "bh", "sao paulo", "sao paulo", "rio"],
            "geolocation_state": ["SP", "SP", "RJ", "MG", "SP", "SP", "RJ"],
        }
    )


def test_node_table_degrees_strength_and_totals(pairs):
    edges = route_table(pairs, "state", min_orders=1)
    nodes = node_table(pairs, edges, _geo()).set_index("state")
    assert set(nodes.index) == {"SP", "RJ", "MG"}
    sp = nodes.loc["SP"]
    assert sp["sent"] == 6 and sp["received"] == 2
    assert sp["loop_orders"] == 2 and sp["loop_breach_rate"] == pytest.approx(0.5)
    assert sp["total_orders"] == 6 + 2 - 2
    assert sp["out_degree"] == 2 and sp["in_degree"] == 0  # SP->RJ, SP->MG; no inbound non-loop route
    assert sp["out_strength"] == 4  # 3 to RJ + 1 to MG
    assert sp["sellers"] == 2
    assert sp["customers"] == 2  # c4 -> u4, c5 -> u5 received in SP
    rj = nodes.loc["RJ"]
    assert rj["in_degree"] == 1 and rj["in_strength"] == 3
    assert rj["out_degree"] == 0 and rj["out_strength"] == 0
    assert rj["customers"] == 2  # u1 (c1, c6) and u2 -> distinct keys, not customer ids
    assert rj["in_breach_rate"] == pytest.approx(1 / 4)
    assert nodes.loc["MG", "loop_orders"] == 0
    # Centroid: median of the geolocation rows of that state.
    assert sp["lat"] == pytest.approx(np.median([-23.5, -23.7, -23.6, -23.4]))
    assert sp["lng"] == pytest.approx(np.median([-46.6, -46.8, -46.7, -46.5]))


def test_node_degree_respects_edge_filter(pairs):
    edges = route_table(pairs, "state", min_orders=2)
    nodes = node_table(pairs, edges, _geo()).set_index("state")
    assert nodes.loc["SP", "out_degree"] == 1  # only SP->RJ (3 pairs) survives
    assert nodes.loc["SP", "out_strength"] == 3


@pytest.mark.parametrize("language", LANGUAGES)
def test_network_figure_traces_and_hover(pairs, language):
    text = get_labels(language)["routes"]
    figure = plot_route_network(pairs, _geo(), language=language, thresholds=(1, 2, 3))
    assert isinstance(figure, go.Figure) and figure.to_json()
    markers = [t for t in figure.data if t.meta == "edge-marker"]
    nodes = [t for t in figure.data if t.meta == "node"]
    lines = [t for t in figure.data if t.meta == "edge-line"]
    assert len(markers) == len(lines) == 2  # SP->RJ and SP->MG; loops are not drawn
    assert len(nodes) == 3
    for key in ("orders", "breaches", "rate", "ci", "share", "distance", "promised", "actual"):
        assert text[key] in markers[0].hovertemplate
    for key in (
        "sent",
        "received",
        "rate_out",
        "rate_in",
        "loop",
        "sellers",
        "customers",
        "out_degree",
        "in_degree",
        "out_strength",
        "in_strength",
    ):
        assert text[key] in nodes[0].hovertemplate
    assert markers[0].marker.angleref == "previous" and markers[0].marker.symbol == "arrow"
    assert len(figure.layout.updatemenus[0].buttons) == 3
    visible = figure.layout.updatemenus[0].buttons[2].args[0]["visible"]
    assert len(visible) == len(figure.data)
    assert sum(visible[-3:]) == 1 and visible[-1]
    assert figure.layout.geo.scope == "south america"
    assert "%" in str(markers[0].customdata[0][4])


def test_network_edge_width_grows_with_volume(pairs):
    figure = plot_route_network(pairs, _geo(), thresholds=(1,))
    lines = [t for t in figure.data if t.meta == "edge-line"]
    markers = [t for t in figure.data if t.meta == "edge-marker"]
    by_name = {t.name: t.line.width for t in lines}
    orders = {m.name: m.customdata[0][2] for m in markers}
    big = max(orders, key=lambda k: int(orders[k]))
    small = min(orders, key=lambda k: int(orders[k]))
    assert by_name[big] > by_name[small]


@pytest.mark.parametrize("language", LANGUAGES)
def test_matrix_figure_blank_cells_and_hover(pairs, language):
    text = get_labels(language)["routes"]
    figure = plot_route_matrix(pairs, language=language, top_origins=2, min_cell=2)
    heat = figure.data[0]
    assert isinstance(heat, go.Heatmap)
    assert list(heat.y) == ["SP", "RJ"]
    assert set(heat.x) == {"SP", "RJ", "MG"}
    z = np.array(heat.z, dtype=float)
    x = list(heat.x)
    assert np.isnan(z[0, x.index("MG")])  # 1 pair < min_cell
    assert not np.isnan(z[0, x.index("RJ")])
    assert heat.text[0][x.index("RJ")] == "3"
    assert heat.text[0][x.index("MG")] == ""
    for key in ("orders", "breaches", "rate", "ci", "share", "distance", "promised", "actual"):
        assert text[key] in heat.hovertemplate
    assert heat.hoverongaps is False
    assert figure.to_json()


@pytest.mark.parametrize("language", LANGUAGES)
def test_city_routes_figure(pairs, language):
    text = get_labels(language)["routes"]
    figure = plot_top_city_routes(pairs, language=language, top=3)
    bar = figure.data[0]
    assert isinstance(bar, go.Bar) and bar.orientation == "h"
    assert len(bar.y) == 3
    assert bar.error_x.symmetric is False
    assert all(a >= 0 for a in bar.error_x.array) and all(a >= 0 for a in bar.error_x.arrayminus)
    for key in ("orders", "breaches", "rate", "ci", "distance"):
        assert text[key] in bar.hovertemplate
    assert "Sao Paulo (SP)" in bar.y[0]
    assert figure.to_json()


def test_labels_have_same_routes_keys():
    assert get_labels("pt")["routes"].keys() == get_labels("en")["routes"].keys()


def test_functions_do_not_mutate_inputs(tables):
    snapshots = [frame.copy(deep=True) for frame in tables]
    pairs = route_pairs(*tables)
    for frame, before in zip(tables, snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, before)
    pairs_before = pairs.copy(deep=True)
    edges = route_table(pairs, "state", min_orders=1)
    edges_before = edges.copy(deep=True)
    geo = tables[4]
    geo_before = geo.copy(deep=True)
    node_table(pairs, edges, geo)
    plot_route_network(pairs, geo, thresholds=(1, 2))
    plot_route_matrix(pairs, top_origins=2, min_cell=1)
    plot_top_city_routes(pairs, top=2)
    pd.testing.assert_frame_equal(pairs, pairs_before)
    pd.testing.assert_frame_equal(edges, edges_before)
    pd.testing.assert_frame_equal(geo, geo_before)
