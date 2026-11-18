"""GPCP-input netCDF files: the label, the input file, the chain and the date agree.

Between 14 July and 18 November 2026 the two writers labeled the F-08 to F-17
chain `_late_` and the F-10 to F-16 chain `_early_`, the reverse of the legacy
writers and of every archived file. Nothing caught it, because the validator's
expected names had been written from the same belief as the writer.

These tests do not take the expected values from the module under test. They
are written out below from what the legacy writers do. One test reads the
legacy IDL sources, where the checkout holds them, and confirms the values
written here for the two single-chain families. The dual family's array origin
of 1987 is the Python writer's own contract and differs on purpose from the
legacy dual array, which starts in 1992.

The end-to-end test writes every family from small real binaries in which each
month of each file holds its own number, with three cells set apart so that a
grid written upside down or back to front shows. For every netCDF file it then
checks the label in the name, the `source` attribute, the satellites in the
`constellation` attribute, the date in the name, the time coordinate, the
first latitude and longitude, and that the grid is the month of the right file.
A label, an origin year or an input file exchanged between two families fails
it.

Not covered here: the validator's comparison with the legacy writer's files
(`validate_netcdf.py`), which needs the operational tree and is run on the
processing host, and the 2.5-degree files, which this module never opens.
"""
import datetime
import os
import re

import pytest

np = pytest.importorskip("numpy")
netCDF4 = pytest.importorskip("netCDF4")
import generate_netcdf as gn

from conftest import REPO_ROOT

NLAT, NLON = 72, 144
FILL = np.float32(-999.99)

MORNING = {'F-08', 'F-11', 'F-13', 'F-17'}
LATE_MORNING = {'F-10', 'F-14', 'F-15', 'F-16'}

# What each label stands for. From gpcp_early_netcdf.pro, gpcp_late_netcdf.pro and
# gpcp_dual_netcdf.pro, and from the `constellation` attribute of the files those
# writers made, read on 17 November 2026.
# The dual's array origin is the one deliberate difference: the legacy dual array starts
# in 1992, the Python one in 1987 with fill before 1992.
EXPECTED = {
    'early': dict(sats=MORNING, origin=1987,
                  files={'PR1': 'gpcp_nesdis_pr1.dat',
                         'PR2': 'gpcp_nesdis_pr2.dat',
                         'SSA': 'gpcp_nesdis_ssa.dat'}),
    'late': dict(sats=LATE_MORNING, origin=1992,
                 files={'PR1': 'gpcp_nesdis_f10_pr1.dat',
                        'PR2': 'gpcp_nesdis_f10_pr2.dat',
                        'SSA': 'gpcp_nesdis_f10_ssa.dat'}),
    'dual': dict(sats=MORNING | LATE_MORNING, origin=1987,
                 files={'PR1': 'gpcp_nesdis_dual_pr1.dat',
                        'PR2': 'gpcp_nesdis_dual_pr2.dat',
                        'SSA': 'gpcp_nesdis_dual_ssa.dat'}),
}
VARIABLE = {'PR1': 'rainfall_alg1', 'PR2': 'rainfall_alg2', 'SSA': 'sampling_fraction'}


def _sats(text):
    return set(re.findall(r'F-\d\d', text))


def test_each_label_names_its_legacy_files_origin_and_chain():
    assert set(gn.GPCP_FAMILIES) == set(EXPECTED)
    for tag, want in EXPECTED.items():
        fam = gn.GPCP_FAMILIES[tag]
        assert fam['files'] == want['files'], tag
        assert fam['initial_year'] == want['origin'], tag
        assert _sats(gn.gpcp_constellation(tag)) == want['sats'], tag


def test_the_expected_values_here_are_what_the_legacy_sources_say():
    """Reads the IDL writers. A checkout without them (the public export carries
    no IDL) skips this one test and keeps the others."""
    for tag in ('early', 'late'):
        path = os.path.join(REPO_ROOT, 'monthly', 'netcdf', f'gpcp_{tag}_netcdf.pro')
        if not os.path.exists(path):
            pytest.skip('the legacy IDL writers are not in this checkout')
        with open(path) as fh:
            text = fh.read()
        want = EXPECTED[tag]
        assert int(re.search(r'^initial_year=(\d{4})', text, re.M).group(1)) == want['origin']
        inputs = re.findall(r"^fname\(\d\)=filepin\+'([^']+)'", text, re.M)
        assert inputs == [want['files'][k] for k in ('PR1', 'PR2', 'SSA')]
        labels = set(re.findall(r"^outname\(\d\)=filepout\+'mw-hydro_v01_gpcp-input_[a-z0-9]+_([a-z]+)_'",
                                text, re.M))
        assert labels == {tag}
        chain = re.search(r"^constellation=(.*?)^title=", text, re.M | re.S).group(1)
        assert _sats(chain) == want['sats']


def test_the_two_single_chain_labels_take_the_chain_history_the_25_degree_writers_take():
    """The GPCP history of each single-chain label is the history string that the
    2.5-degree writer of the same label reads from satellite_config. This compares
    the strings. It does not write or open a 2.5-degree file."""
    from satellite_config import (get_constellation_history,
                                  LATE_CONSTELLATION_SAT, EARLY_CONSTELLATION_SAT)
    # run_early_25deg takes its chain from LATE_CONSTELLATION_SAT and run_late_25deg from
    # EARLY_CONSTELLATION_SAT. The identifiers hold the opposite words on purpose.
    assert gn.gpcp_constellation('early') == get_constellation_history(LATE_CONSTELLATION_SAT)
    assert gn.gpcp_constellation('late') == get_constellation_history(EARLY_CONSTELLATION_SAT)
    assert _sats(get_constellation_history(LATE_CONSTELLATION_SAT)) == MORNING
    assert _sats(get_constellation_history(EARLY_CONSTELLATION_SAT)) == LATE_MORNING


# ----------------------------------------------------------------------- end to end

CODE = {('early', 'PR1'): 1000.0, ('early', 'PR2'): 2000.0, ('early', 'SSA'): 3000.0,
        ('late', 'PR1'): 4000.0, ('late', 'PR2'): 5000.0, ('late', 'SSA'): 6000.0,
        ('dual', 'PR1'): 7000.0, ('dual', 'PR2'): 8000.0, ('dual', 'SSA'): 9000.0}

# The months each family holds data for, as (year, month). August 1987 exercises the
# 1987 origin, and January and February 1992 the 1992 origin and the first dual months.
MONTHS = {'early': [(1987, 8), (1992, 1), (1992, 2)],
          'late': [(1992, 1), (1992, 2)],
          'dual': [(1992, 1), (1992, 2)]}


def _absolute_month(year, month):
    """Months since December 1986, so that January 1987 is 1."""
    return (year - 1987) * 12 + month


def _make_binaries(gpcp_dir):
    """One binary per family and product. A month's grid is the file's code plus the
    month's absolute number, except three cells near the first corner of the file,
    which hold that value plus 1, 2 and 3. The file's first cell is the southernmost
    latitude and the first longitude, so the three cells fix which way round the
    grid is. Every other month is fill."""
    gpcp_dir.mkdir()
    for (tag, prod), code in CODE.items():
        origin = EXPECTED[tag]['origin']
        last = max(MONTHS[tag])
        n_months = (last[0] - origin) * 12 + last[1]
        arr = np.full((n_months, NLAT, NLON), FILL, dtype=np.float32)
        for (y, m) in MONTHS[tag]:
            month = arr[(y - origin) * 12 + (m - 1)]
            month[:] = code + _absolute_month(y, m)
            month[0, 0] += 1.0          # first latitude, first longitude
            month[0, 1] += 2.0          # first latitude, second longitude
            month[1, 0] += 3.0          # second latitude, first longitude
        arr.tofile(gpcp_dir / EXPECTED[tag]['files'][prod])


def _check_written(out_dir, tags):
    names = sorted(os.listdir(out_dir))
    expected_names = sorted(
        f'mw-hydro_{gn.PRODUCT_VERSION}_gpcp-input_{prod.lower()}_{tag}_{y:04d}{m:02d}.nc'
        for tag in tags for prod in ('PR1', 'PR2', 'SSA') for (y, m) in MONTHS[tag])
    assert names == expected_names

    for name in names:
        prod, tag, yyyymm = re.fullmatch(
            rf'mw-hydro_{gn.PRODUCT_VERSION}_gpcp-input_([a-z0-9]+)_([a-z]+)_(\d{{6}})\.nc', name).groups()
        prod = prod.upper()
        year, month = int(yyyymm[:4]), int(yyyymm[4:])
        with netCDF4.Dataset(os.path.join(out_dir, name)) as ds:
            # label and input file
            assert ds.source == EXPECTED[tag]['files'][prod], name
            # label and chain
            assert _sats(ds.constellation) == EXPECTED[tag]['sats'], name
            # date in the name and time coordinate
            days = (datetime.date(year, month, 1) - datetime.date(1987, 1, 1)).days
            assert ds.variables['time'].units == 'days since 1987-01-01'
            assert float(ds.variables['time'][0]) == days, name
            iso = b''.join(ds.variables['time_iso8601'][:, 0].tolist()).decode().strip()
            assert iso == f'{year:04d}-{month:02d}-01T00:00:00Z', name
            # the grid is that month of that family's file
            grid = np.ma.filled(ds.variables[VARIABLE[prod]][0], FILL)
            want = np.float32(CODE[(tag, prod)] + _absolute_month(year, month))
            assert grid.shape == (NLAT, NLON)
            # which way round: the south-west corner and its two neighbors
            lat, lon = ds.variables['lat'][:], ds.variables['lon'][:]
            assert float(lat[0]) == -88.75 and float(lat[-1]) == 88.75
            assert float(lon[0]) == 1.25 and float(lon[-1]) == 358.75
            assert (grid[0, 0], grid[0, 1], grid[1, 0]) == (want + 1, want + 2, want + 3), name
            rest = grid.copy()
            rest[0, 0] = rest[0, 1] = rest[1, 0] = want
            assert np.all(rest == want), (name, float(grid[5, 5]), float(want))
            assert ds.id == name[:-3]
            if tag == 'dual':
                assert '1992' in ds.constellation and '1992' in ds.summary


def test_written_files_tie_label_source_chain_and_date_together(tmp_path, monkeypatch):
    _make_binaries(tmp_path / 'gpcp')
    work = tmp_path / 'netcdf'
    work.mkdir()
    monkeypatch.chdir(work)

    gn.run_gpcp_early_netcdf()
    gn.run_gpcp_late_netcdf()
    gn.run_gpcp_dual_netcdf()

    _check_written(work / 'gpcp-input', ('early', 'late', 'dual'))


@pytest.mark.parametrize("option,tag", [('gpcpearly', 'early'), ('gpcplate', 'late'), ('dual', 'dual')])
def test_each_command_line_option_writes_the_family_of_its_name(tmp_path, monkeypatch, option, tag):
    _make_binaries(tmp_path / 'gpcp')
    work = tmp_path / 'netcdf'
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr('sys.argv', ['generate_netcdf.py', '--dataset', option])

    gn.main()

    _check_written(work / 'gpcp-input', (tag,))
