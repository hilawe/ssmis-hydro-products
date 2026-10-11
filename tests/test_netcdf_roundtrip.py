"""NetCDF export round-trip.

Write one month of a product to NetCDF with write_netcdf, reopen it, and verify
the data survives unchanged and the key CF/CDR attributes are present. This also
pins the link to the centralized version token: the file's product_version must
equal product_version.PRODUCT_VERSION_ATTR.
"""
import os
import tempfile

import pytest

np = pytest.importorskip("numpy")
netCDF4 = pytest.importorskip("netCDF4")
import generate_netcdf as gn
import product_version as pv

FILL = np.float32(-999.99)


def test_write_netcdf_roundtrip_and_version_attr():
    grid = gn.GRID_25
    lats, lons = gn.make_coords(grid)
    rng = np.random.default_rng(7)
    data = rng.uniform(0.0, 1.0, size=(grid['nlat'], grid['nlon'])).astype(np.float32)
    data[0, 0] = FILL  # include a fill cell

    prod_key = 'CFR'
    var_name = gn.CF_STANDARD[prod_key][0]

    with tempfile.TemporaryDirectory() as d:
        ncfile = os.path.join(d, 'roundtrip.nc')
        gn.write_netcdf(ncfile, data, lats, lons, 2012, 7, prod_key,
                        title='test', initial_year=1987, constellation='test-const',
                        history='test-history', summary='test-summary',
                        dataset_name='test-dataset')

        with netCDF4.Dataset(ncfile) as ds:
            assert ds.product_version == pv.PRODUCT_VERSION_ATTR
            assert 'CF-1.5' in ds.Metadata_Conventions
            assert len(ds.dimensions['lat']) == grid['nlat']
            assert len(ds.dimensions['lon']) == grid['nlon']

            raw = ds.variables[var_name][:]          # (time, lat, lon), masked
            got = np.ma.filled(raw[0], FILL).astype(np.float32)
            assert got.shape == data.shape
            assert np.allclose(got, data, atol=1e-5)


@pytest.mark.parametrize("grid_name,expected", [("GRID_25", "2.5 Degree"), ("GRID_10", "1.0 Degree")])
def test_long_name_names_the_written_resolution(grid_name, expected):
    """The variable long_name must name the grid it is written on. Until 2026-10-09 every file took the
    2.5-degree title table, so 1.0-degree files said "2.5 Degree" (found by the 2024 release validation)."""
    grid = getattr(gn, grid_name)
    lats, lons = gn.make_coords(grid)
    data = np.zeros((grid['nlat'], grid['nlon']), np.float32)
    with tempfile.TemporaryDirectory() as d:
        ncfile = os.path.join(d, 'title.nc')
        gn.write_netcdf(ncfile, data, lats, lons, 2024, 1, 'LWP',
                        title='test', initial_year=1987, constellation='test-const',
                        history='test-history', summary='test-summary',
                        dataset_name='test-dataset')
        with netCDF4.Dataset(ncfile) as ds:
            assert expected in ds.variables[gn.CF_STANDARD['LWP'][0]].long_name


def test_units_are_physical_units_not_encoding_phrases():
    """Every units attribute must be a UDUNITS-parsable physical unit. LWP is g m-2 (1000 times the mean liquid
    water path in mm); the phrase '1000*mm' used until 2026-10-09 is an encoding description, not a unit."""
    for key, (_, units, _) in gn.CF_STANDARD.items():
        assert '*' not in units, (key, units)
    assert gn.CF_STANDARD['LWP'][1] == 'g m-2'
    for titles in (gn.PRODUCT_TITLES_25, gn.PRODUCT_TITLES_10):
        assert '1000*mm' not in titles['LWP']
