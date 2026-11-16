"""Reproducible Folium maps for the Haversine demonstration."""

from __future__ import annotations


def build_haversine_map(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    haversine_distance_km: float,
) -> object:
    """Build a Folium map with origin, destination, and Haversine line."""
    import folium

    center = [
        (origin_latitude + destination_latitude) / 2,
        (origin_longitude + destination_longitude) / 2,
    ]
    route_map = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")
    origin = [origin_latitude, origin_longitude]
    destination = [destination_latitude, destination_longitude]
    folium.Marker(origin, tooltip="Seller origin", icon=folium.Icon(color="blue")).add_to(route_map)
    folium.Marker(destination, tooltip="Customer destination", icon=folium.Icon(color="green")).add_to(route_map)
    folium.PolyLine(
        [origin, destination],
        color="#1f77b4",
        weight=4,
        dash_array="8 8",
        tooltip=f"Haversine: {haversine_distance_km:.1f} km",
    ).add_to(route_map)
    route_map.fit_bounds([origin, destination])
    return route_map
