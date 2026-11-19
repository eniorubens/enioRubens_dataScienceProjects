import numpy as np
import pytest

from delivery_routing_vrptw import exact, route_length, scaled_ceil, scaled_round, truncated


def test_exact(inst):
    d = exact(inst)
    assert d.shape == (6, 6)
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)
    assert d[0, 1] == pytest.approx(5.0)
    assert d[3, 5] == pytest.approx(np.sqrt(90))


def test_truncated(inst):
    t = truncated(inst)
    assert t[3, 5] == pytest.approx(9.4)
    assert np.all(t <= exact(inst) + 1e-12)
    assert truncated(inst, 0)[3, 5] == 9.0


def test_scaled(inst):
    d = exact(inst)
    c = scaled_ceil(inst, 100)
    r = scaled_round(inst, 100)
    assert c.dtype == np.int64
    assert np.all(c >= d * 100 - 1e-9)
    assert c[0, 1] == 500  # an integer distance is not bumped by float noise
    assert c[3, 5] == 949 and r[3, 5] == 949
    assert np.all(c >= r)


def test_route_length(inst):
    d = exact(inst)
    assert route_length([], d) == 0.0
    assert route_length([1, 2], d) == pytest.approx(20.0)
    assert route_length([3, 5], d) == pytest.approx(10 + np.sqrt(90))
