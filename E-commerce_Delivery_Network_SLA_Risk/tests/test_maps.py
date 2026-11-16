from ecom_delivery_sla.maps import build_haversine_map


def test_build_haversine_map_returns_folium_map():
    result = build_haversine_map(0.0, 0.0, 1.0, 1.0, 157.2)

    assert result.__class__.__name__ == "Map"
