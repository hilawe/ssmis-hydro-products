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
