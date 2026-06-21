"""Grid geometry, date helpers, file-size inference, and resolution dims.

These lock the coordinate conventions and the simple-but-load-bearing helpers
that the NetCDF export and the combine step depend on. The generate_netcdf
pieces need netCDF4 (the module imports it), so they skip cleanly where it is
absent; get_dims needs only numpy.
"""
import os
import tempfile

import pytest

np = pytest.importorskip("numpy")


def _gn():
    pytest.importorskip("netCDF4")
    import generate_netcdf as gn
    return gn


def test_grid25_coords():
    gn = _gn()
    lats, lons = gn.make_coords(gn.GRID_25)
    assert (len(lats), len(lons)) == (72, 144)
    assert np.isclose(lats[0], -88.75) and np.isclose(lats[-1], 88.75)
    assert np.isclose(lons[0], 1.25) and np.isclose(lons[-1], 358.75)


def test_grid10_coords():
    gn = _gn()
    lats, lons = gn.make_coords(gn.GRID_10)
    assert (len(lats), len(lons)) == (180, 360)
    assert np.isclose(lats[0], -89.5) and np.isclose(lats[-1], 89.5)
    assert np.isclose(lons[0], 0.5) and np.isclose(lons[-1], 359.5)


def test_days_since_ref():
    gn = _gn()
    assert gn.days_since_ref(1987, 1) == 0
    assert gn.days_since_ref(1987, 2) == 31
    assert gn.days_since_ref(1988, 1) == 365  # 1987 is not a leap year


def test_iso8601_format():
    gn = _gn()
    assert gn.iso8601(2012, 7) == '2012-07-01T00:00:00Z'


def test_get_n_months_from_file_size():
    gn = _gn()
    grid = gn.GRID_25
    month_bytes = grid['nlat'] * grid['nlon'] * 4
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, 'three_months.bin')
        np.zeros(3 * grid['nlat'] * grid['nlon'], dtype=np.float32).tofile(fn)
        assert gn.get_n_months(fn, grid) == 3
        assert os.path.getsize(fn) == 3 * month_bytes
    assert gn.get_n_months('/no/such/file', grid) == 0


def test_combine_get_dims():
    import combine
    assert combine.get_dims(2.5) == (72, 144)
    assert combine.get_dims(1.0) == (180, 360)
    with pytest.raises(ValueError):
        combine.get_dims(0.25)
