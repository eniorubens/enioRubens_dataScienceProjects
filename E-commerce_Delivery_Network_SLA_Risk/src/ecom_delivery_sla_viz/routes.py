"""Origin-destination route tables and Plotly route graphs (presentation only).

The functions here take the raw Olist tables already loaded by the notebooks
and describe, per origin-destination route, how many order-seller pairs flow
and how often the SLA is breached. Nothing is trained or scored, and no file is
read. The package deliberately does not import ``ecom_delivery_sla``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from .labels import get_labels

EARTH_RADIUS_KM = 6_371.0088
Z_95 = 1.959963984540054
COLORSCALE = "YlOrRd"
NODE_OUTLINE = "#111827"
_ROUTE_KEYS = {
    "state": (["origin_state"], ["destination_state"]),
    "city": (["origin_state", "origin_city"], ["destination_state", "destination_city"]),
}


def _haversine_km(lat_1, lng_1, lat_2, lng_2) -> np.ndarray:
    """Great-circle distance in kilometres, with R = 6371.0088 km."""
    p1, l1 = np.radians(np.asarray(lat_1, dtype=float)), np.radians(np.asarray(lng_1, dtype=float))
    p2, l2 = np.radians(np.asarray(lat_2, dtype=float)), np.radians(np.asarray(lng_2, dtype=float))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _zip_coordinates(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Median latitude and longitude per zip-code prefix."""
    return geolocation.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median")
    )


def route_pairs(
    supervised: pd.DataFrame,
    items: pd.DataFrame,
    sellers: pd.DataFrame,
    customers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per (order, seller) pair with route, distance and outcome.

    An order served by two sellers yields two pairs; the order outcome is
    attributed to each pair. Pairs are de-duplicated on ``(order_id, seller_id)``
    so several items from the same seller count once.

    Parameters
    ----------
    supervised : pandas.DataFrame
        Supervised orders with ``order_id``, ``customer_id``, ``sla_breach``,
        ``promised_lead_days`` and ``actual_lead_days``.
    items : pandas.DataFrame
        Order items with ``order_id`` and ``seller_id``.
    sellers : pandas.DataFrame
        Sellers with ``seller_id``, ``seller_zip_code_prefix``, ``seller_city``
        and ``seller_state``.
    customers : pandas.DataFrame
        Customers with ``customer_id``, ``customer_zip_code_prefix``,
        ``customer_city``, ``customer_state`` and, optionally,
        ``customer_unique_id``.
    geolocation : pandas.DataFrame
        Olist geolocation table; coordinates are the median per zip prefix.

    Returns
    -------
    pandas.DataFrame
        Columns ``order_id``, ``seller_id``, ``customer_key``, ``origin_state``,
        ``origin_city``, ``destination_state``, ``destination_city``,
        ``origin_lat``, ``origin_lng``, ``destination_lat``, ``destination_lng``,
        ``distance_km`` (NaN without coordinates), ``promised_lead_days``,
        ``actual_lead_days`` and ``sla_breach`` (0/1 integer). Inputs are not
        modified.
    """
    coordinates = _zip_coordinates(geolocation)
    pairs = items[["order_id", "seller_id"]].drop_duplicates()
    orders = supervised[
        ["order_id", "customer_id", "sla_breach", "promised_lead_days", "actual_lead_days"]
    ].drop_duplicates("order_id")
    pairs = pairs.merge(orders, on="order_id", how="inner")

    seller_cols = sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]]
    seller_cols = seller_cols.drop_duplicates("seller_id").merge(
        coordinates, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left"
    )
    seller_cols = seller_cols.rename(
        columns={
            "seller_city": "origin_city",
            "seller_state": "origin_state",
            "lat": "origin_lat",
            "lng": "origin_lng",
        }
    )[["seller_id", "origin_state", "origin_city", "origin_lat", "origin_lng"]]

    key = "customer_unique_id" if "customer_unique_id" in customers.columns else "customer_id"
    customer_cols = customers.drop_duplicates("customer_id").merge(
        coordinates, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left"
    )
    customer_cols = customer_cols.assign(customer_key=customer_cols[key]).rename(
        columns={
            "customer_city": "destination_city",
            "customer_state": "destination_state",
            "lat": "destination_lat",
            "lng": "destination_lng",
        }
    )[
        [
            "customer_id",
            "customer_key",
            "destination_state",
            "destination_city",
            "destination_lat",
            "destination_lng",
        ]
    ]

    pairs = pairs.merge(seller_cols, on="seller_id", how="inner").merge(
        customer_cols, on="customer_id", how="inner"
    )
    pairs["distance_km"] = _haversine_km(
        pairs["origin_lat"], pairs["origin_lng"], pairs["destination_lat"], pairs["destination_lng"]
    )
    pairs["sla_breach"] = pairs["sla_breach"].astype(int)
    columns = [
        "order_id",
        "seller_id",
        "customer_key",
        "origin_state",
        "origin_city",
        "destination_state",
        "destination_city",
        "origin_lat",
        "origin_lng",
        "destination_lat",
        "destination_lng",
        "distance_km",
        "promised_lead_days",
        "actual_lead_days",
        "sla_breach",
    ]
    return pairs.sort_values(["order_id", "seller_id"]).reset_index(drop=True)[columns]


def wilson_interval(breaches, n, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion (Wilson, 1927).

    Parameters
    ----------
    breaches : array-like
        Number of successes (SLA breaches).
    n : array-like
        Number of trials; where ``n == 0`` the bounds are NaN.
    z : float, default 1.959963984540054
        Normal quantile; the default gives a 95% interval.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper bounds, same shape as the inputs.
    """
    k = np.asarray(breaches, dtype=float)
    total = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / total
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    low = np.clip(centre - half, 0.0, 1.0)
    high = np.clip(centre + half, 0.0, 1.0)
    empty = total <= 0
    return np.where(empty, np.nan, low), np.where(empty, np.nan, high)


def _label(frame: pd.DataFrame, side: str, level: str) -> pd.Series:
    if level == "state":
        return frame[f"{side}_state"]
    return frame[f"{side}_city"].astype(str).str.title() + " (" + frame[f"{side}_state"].astype(str) + ")"


def route_table(pairs: pd.DataFrame, level: str = "state", min_orders: int = 100) -> pd.DataFrame:
    """Aggregate order-seller pairs into one row per origin-destination route.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`route_pairs`.
    level : {"state", "city"}, default "state"
        Route granularity.
    min_orders : int, default 100
        Keep routes with at least this many pairs. The share of volume is
        computed over all pairs before the filter.

    Returns
    -------
    pandas.DataFrame
        One row per route, sorted by descending ``orders``, with the route keys,
        ``origin`` and ``destination`` labels, ``orders``, ``breaches``,
        ``breach_rate``, Wilson ``ci_low``/``ci_high``, ``share`` of all pairs,
        ``distance_median_km``, ``promised_median_days``,
        ``actual_median_days`` and ``is_self_loop`` (same origin and
        destination state, or same city for ``level="city"``).
    """
    if level not in _ROUTE_KEYS:
        raise ValueError("level must be 'state' or 'city'")
    origin_keys, destination_keys = _ROUTE_KEYS[level]
    keys = origin_keys + destination_keys
    grouped = pairs.groupby(keys, as_index=False, dropna=False).agg(
        orders=("sla_breach", "size"),
        breaches=("sla_breach", "sum"),
        distance_median_km=("distance_km", "median"),
        promised_median_days=("promised_lead_days", "median"),
        actual_median_days=("actual_lead_days", "median"),
    )
    grouped["breach_rate"] = grouped["breaches"] / grouped["orders"]
    grouped["ci_low"], grouped["ci_high"] = wilson_interval(grouped["breaches"], grouped["orders"])
    grouped["share"] = grouped["orders"] / max(len(pairs), 1)
    grouped["origin"] = _label(grouped, "origin", level)
    grouped["destination"] = _label(grouped, "destination", level)
    grouped["is_self_loop"] = grouped["origin"] == grouped["destination"]
    grouped = grouped[grouped["orders"] >= min_orders]
    ordered = ["origin", "destination"] + keys + [
        "orders",
        "breaches",
        "breach_rate",
        "ci_low",
        "ci_high",
        "share",
        "distance_median_km",
        "promised_median_days",
        "actual_median_days",
        "is_self_loop",
    ]
    grouped = grouped.sort_values(["orders"] + keys, ascending=[False] + [True] * len(keys))
    return grouped.reset_index(drop=True)[ordered]


def node_table(pairs: pd.DataFrame, edges: pd.DataFrame, geolocation: pd.DataFrame) -> pd.DataFrame:
    """Summarise each state as a network node.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`route_pairs`.
    edges : pandas.DataFrame
        State-level ``route_table`` already filtered by minimum volume. Rows
        with ``is_self_loop`` are ignored for degree and strength.
    geolocation : pandas.DataFrame
        Olist geolocation table; the node position is the median latitude and
        longitude of the rows of that state.

    Returns
    -------
    pandas.DataFrame
        One row per state with ``state``, ``lat``, ``lng``, ``sent``,
        ``received``, ``total_orders`` (pairs that touch the state, a
        self-loop counted once), ``out_breach_rate``, ``in_breach_rate``,
        ``loop_orders``, ``loop_breach_rate``, ``sellers`` (distinct sellers
        located there), ``customers`` (distinct customers located there),
        ``out_degree``, ``in_degree``, ``out_strength`` and ``in_strength``
        (degree counts routes, strength sums their pairs).
    """
    centroids = geolocation.groupby("geolocation_state").agg(
        lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median")
    )
    states = sorted(set(pairs["origin_state"]) | set(pairs["destination_state"]))
    out = pairs.groupby("origin_state").agg(
        sent=("sla_breach", "size"), out_breaches=("sla_breach", "sum"), sellers=("seller_id", "nunique")
    )
    inn = pairs.groupby("destination_state").agg(
        received=("sla_breach", "size"), in_breaches=("sla_breach", "sum"), customers=("customer_key", "nunique")
    )
    loops = (
        pairs[pairs["origin_state"] == pairs["destination_state"]]
        .groupby("origin_state")
        .agg(loop_orders=("sla_breach", "size"), loop_breaches=("sla_breach", "sum"))
    )
    result = pd.DataFrame({"state": states}).set_index("state")
    result = result.join(centroids).join(out).join(inn).join(loops)
    for column in (
        "sent",
        "out_breaches",
        "sellers",
        "received",
        "in_breaches",
        "customers",
        "loop_orders",
        "loop_breaches",
    ):
        result[column] = result[column].fillna(0).astype(int)
    result["total_orders"] = result["sent"] + result["received"] - result["loop_orders"]
    with np.errstate(divide="ignore", invalid="ignore"):
        result["out_breach_rate"] = (result["out_breaches"] / result["sent"]).replace([np.inf], np.nan)
        result["in_breach_rate"] = (result["in_breaches"] / result["received"]).replace([np.inf], np.nan)
        result["loop_breach_rate"] = (result["loop_breaches"] / result["loop_orders"]).replace([np.inf], np.nan)
    real = edges[~edges["is_self_loop"]]
    result["out_degree"] = real.groupby("origin_state").size().reindex(result.index).fillna(0).astype(int)
    result["in_degree"] = real.groupby("destination_state").size().reindex(result.index).fillna(0).astype(int)
    result["out_strength"] = real.groupby("origin_state")["orders"].sum().reindex(result.index).fillna(0).astype(int)
    result["in_strength"] = real.groupby("destination_state")["orders"].sum().reindex(result.index).fillna(0).astype(int)
    columns = [
        "lat",
        "lng",
        "sent",
        "received",
        "total_orders",
        "out_breach_rate",
        "in_breach_rate",
        "loop_orders",
        "loop_breach_rate",
        "sellers",
        "customers",
        "out_degree",
        "in_degree",
        "out_strength",
        "in_strength",
    ]
    return result[columns].reset_index()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _fmt(value: float, language: str, digits: int = 1, pct: bool = False) -> str:
    """Format a number (or percentage) with the decimal mark of the language."""
    if pd.isna(value):
        return "–"
    text = f"{value * 100:,.{digits}f}%" if pct else f"{value:,.{digits}f}"
    if language == "pt":
        text = text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return text


def _int(value: float, language: str) -> str:
    return _fmt(value, language, digits=0)


def _edge_custom(row: pd.Series, language: str) -> list[str]:
    """Pre-formatted edge fields, in the order used by ``_edge_template``."""
    return [
        str(row["origin"]),
        str(row["destination"]),
        _int(row["orders"], language),
        _int(row["breaches"], language),
        _fmt(row["breach_rate"], language, pct=True),
        _fmt(row["ci_low"], language, pct=True),
        _fmt(row["ci_high"], language, pct=True),
        _fmt(row["share"], language, digits=2, pct=True),
        _int(row["distance_median_km"], language),
        _fmt(row["promised_median_days"], language),
        _fmt(row["actual_median_days"], language),
    ]


def _edge_template(text: dict[str, str]) -> str:
    return (
        "<b>%{customdata[0]} → %{customdata[1]}</b><br>"
        f"{text['orders']}: %{{customdata[2]}}<br>"
        f"{text['breaches']}: %{{customdata[3]}}<br>"
        f"{text['rate']}: %{{customdata[4]}} ({text['ci']}: %{{customdata[5]}} – %{{customdata[6]}})<br>"
        f"{text['share']}: %{{customdata[7]}}<br>"
        f"{text['distance']}: %{{customdata[8]}} km<br>"
        f"{text['promised']}: %{{customdata[9]}} {text['days']}<br>"
        f"{text['actual']}: %{{customdata[10]}} {text['days']}<extra></extra>"
    )


def _node_custom(row: pd.Series, language: str) -> list[str]:
    return [
        str(row["state"]),
        _int(row["sent"], language),
        _int(row["received"], language),
        _fmt(row["out_breach_rate"], language, pct=True),
        _fmt(row["in_breach_rate"], language, pct=True),
        _int(row["loop_orders"], language),
        _fmt(row["loop_breach_rate"], language, pct=True),
        _int(row["sellers"], language),
        _int(row["customers"], language),
        _int(row["out_degree"], language),
        _int(row["in_degree"], language),
        _int(row["out_strength"], language),
        _int(row["in_strength"], language),
    ]


def _node_template(text: dict[str, str]) -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        f"{text['sent']}: %{{customdata[1]}}<br>"
        f"{text['received']}: %{{customdata[2]}}<br>"
        f"{text['rate_out']}: %{{customdata[3]}}<br>"
        f"{text['rate_in']}: %{{customdata[4]}}<br>"
        f"{text['loop']}: %{{customdata[5]}} ({text['rate']}: %{{customdata[6]}})<br>"
        f"{text['sellers']}: %{{customdata[7]}}<br>"
        f"{text['customers']}: %{{customdata[8]}}<br>"
        f"{text['out_degree']}: %{{customdata[9]}}<br>"
        f"{text['in_degree']}: %{{customdata[10]}}<br>"
        f"{text['out_strength']}: %{{customdata[11]}}<br>"
        f"{text['in_strength']}: %{{customdata[12]}}<extra></extra>"
    )


def _curve(lat_1: float, lng_1: float, lat_2: float, lng_2: float, bend: float = 0.15, points: int = 25):
    """Quadratic Bezier arc between two points, bent to the right of the direction."""
    p0 = np.array([lng_1, lat_1])
    p1 = np.array([lng_2, lat_2])
    chord = p1 - p0
    normal = np.array([chord[1], -chord[0]])
    control = (p0 + p1) / 2 + bend * normal
    t = np.linspace(0.0, 1.0, points)[:, None]
    path = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p1
    return path[:, 1], path[:, 0]


def _color_at(value: float, low: float, high: float) -> str:
    position = 0.0 if high <= low else float(np.clip((value - low) / (high - low), 0, 1))
    return sample_colorscale(COLORSCALE, [position])[0]


def _separators(language: str) -> str:
    return ",." if language == "pt" else ".,"


def plot_route_network(
    pairs: pd.DataFrame,
    geolocation: pd.DataFrame,
    language: str = "pt",
    thresholds: Sequence[int] = (100, 300, 1000),
) -> go.Figure:
    """Plot the state-to-state route network on a map of Brazil.

    Edges are curved and directed (an arrow at the midpoint, where the hover
    lives), with width growing with the square root of volume and colour given
    by the breach rate. Nodes are sized by total volume and coloured by the
    inbound breach rate. A menu switches the minimum route volume among
    ``thresholds`` without recomputing. Self-loops (same origin and destination
    state) appear only in node hovers.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`route_pairs`.
    geolocation : pandas.DataFrame
        Olist geolocation table, used for the state centroids.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    thresholds : sequence of int, default (100, 300, 1000)
        Minimum number of pairs per drawn route; the smallest is shown first.

    Returns
    -------
    plotly.graph_objects.Figure
        Scattergeo figure with one line and one marker trace per route, plus one
        node trace per threshold.
    """
    text = get_labels(language)["routes"]
    thresholds = tuple(sorted(thresholds))
    all_edges = route_table(pairs, "state", min_orders=1)
    base = all_edges[(all_edges["orders"] >= thresholds[0]) & ~all_edges["is_self_loop"]].reset_index(drop=True)
    nodes_by_threshold = {t: node_table(pairs, all_edges[all_edges["orders"] >= t], geolocation) for t in thresholds}
    nodes = nodes_by_threshold[thresholds[0]].set_index("state")
    low = float(min(base["breach_rate"].min(), nodes["in_breach_rate"].min()))
    high = float(max(base["breach_rate"].max(), nodes["in_breach_rate"].max()))
    max_orders = float(base["orders"].max()) if len(base) else 1.0
    max_total = float(nodes["total_orders"].max())
    edge_template = _edge_template(text)
    node_template = _node_template(text)
    figure = go.Figure()
    edge_volumes: list[int] = []

    for _, row in base.iterrows():
        origin, destination = nodes.loc[row["origin_state"]], nodes.loc[row["destination_state"]]
        lat, lng = _curve(origin["lat"], origin["lng"], destination["lat"], destination["lng"])
        middle = int(0.6 * (len(lat) - 1))  # off-centre, so arrows rarely sit under a node
        width = 1 + 9 * np.sqrt(row["orders"] / max_orders)
        hover = _edge_custom(row, language)
        name = f"{row['origin']}→{row['destination']}"
        figure.add_trace(
            go.Scattergeo(
                lat=lat,
                lon=lng,
                mode="lines",
                line={"width": width, "color": _color_at(row["breach_rate"], low, high)},
                opacity=0.75,
                hoverinfo="skip",
                showlegend=False,
                name=name,
                meta="edge-line",
            )
        )
        figure.add_trace(
            go.Scattergeo(
                lat=[lat[middle - 1], lat[middle]],
                lon=[lng[middle - 1], lng[middle]],
                mode="markers",
                marker={
                    "symbol": "arrow",
                    "angleref": "previous",
                    "size": [0, 6 + 0.6 * width],
                    "color": [row["breach_rate"]] * 2,
                    "coloraxis": "coloraxis",
                    "line": {"width": 0.5, "color": NODE_OUTLINE},
                },
                customdata=[hover, hover],
                hovertemplate=edge_template,
                showlegend=False,
                name=name,
                meta="edge-marker",
            )
        )
        edge_volumes.append(int(row["orders"]))

    for index, threshold in enumerate(thresholds):
        frame = nodes_by_threshold[threshold]
        figure.add_trace(
            go.Scattergeo(
                lat=frame["lat"],
                lon=frame["lng"],
                mode="markers+text",
                text=frame["state"],
                textposition="middle center",
                textfont={"size": 9, "color": NODE_OUTLINE},
                marker={
                    "size": 10 + 30 * np.sqrt(frame["total_orders"] / max_total),
                    "color": frame["in_breach_rate"],
                    "coloraxis": "coloraxis",
                    "line": {"width": 1, "color": NODE_OUTLINE},
                    "opacity": 0.9,
                },
                customdata=[_node_custom(row, language) for _, row in frame.iterrows()],
                hovertemplate=node_template,
                showlegend=False,
                name=text["node"],
                meta="node",
                visible=index == 0,
            )
        )

    def visibility(threshold: int, node_position: int) -> list[bool]:
        flags: list[bool] = []
        for volume in edge_volumes:
            flags.extend([volume >= threshold, volume >= threshold])
        return flags + [i == node_position for i in range(len(thresholds))]

    figure.update_layout(
        title={"text": text["title_network"], "y": 0.985, "yanchor": "top"},
        template="plotly_white",
        separators=_separators(language),
        coloraxis={
            "colorscale": COLORSCALE,
            "cmin": low,
            "cmax": high,
            "colorbar": {
                "title": {"text": text["colorbar"]},
                "tickformat": ".0%",
                "len": 0.7,
                "x": 0.92,
                "xanchor": "left",
            },
        },
        geo={
            "scope": "south america",
            "projection": {"type": "mercator"},
            "lataxis": {"range": [-34.5, 5.8]},
            "lonaxis": {"range": [-74.5, -34]},
            "domain": {"x": [0.0, 0.9], "y": [0.0, 1.0]},
            "showcountries": True,
            "showland": True,
            "landcolor": "#F3F4F6",
            "countrycolor": "#9CA3AF",
            "showocean": True,
            "oceancolor": "#EFF6FF",
        },
        # The title sits at the top of the figure and the volume menu right below it,
        # above the map, so the two never overlap at narrow widths.
        margin={"l": 5, "r": 5, "t": 100, "b": 5},
        height=790,
        dragmode="pan",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.0,
                "yanchor": "bottom",
                "pad": {"b": 6},
                "xanchor": "left",
                "buttons": [
                    {
                        "label": f"{text['min_volume']} {_int(t, language)}",
                        "method": "update",
                        "args": [{"visible": visibility(t, i)}],
                    }
                    for i, t in enumerate(thresholds)
                ],
            }
        ],
    )
    return figure


def plot_route_matrix(
    pairs: pd.DataFrame,
    language: str = "pt",
    top_origins: int = 8,
    min_cell: int = 30,
) -> go.Figure:
    """Plot a heatmap of breach rate for the top origin states by destination.

    Rows are the ``top_origins`` states with most pairs sent; columns are all
    destination states ordered by volume received. Cells with fewer than
    ``min_cell`` pairs are blank; other cells show the pair count as text.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`route_pairs`.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    top_origins : int, default 8
        Number of origin states (rows).
    min_cell : int, default 30
        Minimum pairs for a cell to be drawn.

    Returns
    -------
    plotly.graph_objects.Figure
        Single Heatmap figure with the full edge hover in every drawn cell.
    """
    text = get_labels(language)["routes"]
    table = route_table(pairs, "state", min_orders=1)
    origins = pairs["origin_state"].value_counts().head(top_origins).index.tolist()
    destinations = pairs["destination_state"].value_counts().index.tolist()
    lookup = {(r.origin_state, r.destination_state): r for r in table.itertuples(index=False)}
    z, cell_text, custom = [], [], []
    for origin in origins:
        z_row, text_row, custom_row = [], [], []
        for destination in destinations:
            route = lookup.get((origin, destination))
            if route is None or route.orders < min_cell:
                z_row.append(np.nan)
                text_row.append("")
                custom_row.append([""] * 11)
                continue
            z_row.append(float(route.breach_rate))
            text_row.append(_int(route.orders, language))
            custom_row.append(_edge_custom(pd.Series(route._asdict()), language))
        z.append(z_row)
        cell_text.append(text_row)
        custom.append(custom_row)
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=destinations,
            y=origins,
            text=cell_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            customdata=custom,
            hovertemplate=_edge_template(text),
            hoverongaps=False,
            colorscale=COLORSCALE,
            colorbar={"title": {"text": text["colorbar"]}, "tickformat": ".0%"},
        )
    )
    figure.update_layout(
        title=text["title_matrix"].format(min_cell=min_cell),
        template="plotly_white",
        separators=_separators(language),
        xaxis={"title": {"text": text["destination"]}, "type": "category"},
        yaxis={"title": {"text": text["origin"]}, "autorange": "reversed", "type": "category"},
        height=70 * len(origins) + 150,
        margin={"l": 60, "r": 20, "t": 70, "b": 70},
    )
    return figure


def plot_top_city_routes(pairs: pd.DataFrame, language: str = "pt", top: int = 15) -> go.Figure:
    """Plot the largest city-to-city routes with Wilson error bars.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`route_pairs`.
    language : str, default "pt"
        ``"pt"`` or ``"en"``.
    top : int, default 15
        Number of routes, by descending number of pairs.

    Returns
    -------
    plotly.graph_objects.Figure
        Horizontal bar chart of the breach rate with 95% Wilson error bars.
    """
    text = get_labels(language)["routes"]
    table = route_table(pairs, "city", min_orders=1).head(top).reset_index(drop=True)
    labels = table["origin"] + " → " + table["destination"]
    figure = go.Figure(
        go.Bar(
            x=table["breach_rate"],
            y=labels,
            orientation="h",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": table["ci_high"] - table["breach_rate"],
                "arrayminus": table["breach_rate"] - table["ci_low"],
            },
            marker={"color": "#2563EB"},
            customdata=[_edge_custom(row, language) for _, row in table.iterrows()],
            hovertemplate=_edge_template(text),
        )
    )
    figure.update_layout(
        title=text["title_cities"].format(top=top),
        template="plotly_white",
        separators=_separators(language),
        xaxis={"title": {"text": text["rate"]}, "tickformat": ".0%", "rangemode": "tozero"},
        yaxis={"autorange": "reversed", "title": {"text": text["route"]}},
        height=520,
        margin={"l": 280},
    )
    return figure
