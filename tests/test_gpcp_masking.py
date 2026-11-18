"""The snow, ice and coastal masking of the GPCP input rain (gpcp_late, gpcp_early).

The two functions turn a chain's monthly rain into the masked field GPCP is
given, and that field is also what the archive distributes as the rain product
of each chain. The rule, from gpcp_f08.pro and gpcp_f10.pro, in the order it is
applied:

    a negative rain value                         becomes fill (-999.99)
    snow fraction of the month at least 0.20      rain becomes -100 times that fraction
    sea ice of the month at least 25 percent      rain becomes minus that percent
    second rain product only, two coastal masks   rain becomes fill

For the morning chain alone, in the eighteen months July 1990 to December 1991
(months 43 to 60 from January 1987), the month's own snow field is not used. A
snow climatology of the calendar month is used in its place, and a cell at or
above 0.20 there becomes fill, not a flag.

Each case sets a handful of cells to known values in small real files and reads
the masked files back. The expected numbers are worked from the rule above.
The coastal cells are chosen here from the legacy rule itself, by row and land
fraction, and not from the module's own masks.

These are selected cases: the snow, ice and climatology thresholds at, below and
above their values, the order of the snow and ice steps, the entry into the
climatology window at month 43 and the return to the month's own snow at month
61, and one cell of each coastal mask in each chain. The coastal rain limit of
1000 is tried at 1500 only. They do not test every cell of the masks or any
retrieval.
"""
import os
import shutil

import pytest

np = pytest.importorskip("numpy")
import gpcp_processing as gp

from conftest import MONTHLY_DIR

# The masking reads the 2.5-degree land/sea file from the working folder. A checkout
# without it cannot run these cases, and they are skipped there and say so.
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MONTHLY_DIR, 'lndsea.25d')),
    reason='monthly/lndsea.25d, the 2.5-degree land/sea file the masking reads, is not in this checkout')

FILL = np.float32(-999.99)
G = gp.GRID_SIZE

# Cells by their position in the file. Rain 50 mm everywhere unless a case says otherwise.
PLAIN, SNOW_LOW, SNOW, ICE_LOW, ICE, BOTH, NEGATIVE, ZERO = 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007
SNOW_AT, ICE_AT = 5008, 5009


def is_fill(x):
    return np.isclose(x, FILL, atol=1e-3)


def _fields(n_months):
    rain = np.full((n_months, G), 50.0, np.float32)
    snow = np.zeros((n_months, G), np.float32)
    ice = np.zeros((n_months, G), np.float32)
    snow[:, SNOW_LOW] = 0.19
    snow[:, SNOW] = 0.45
    snow[:, SNOW_AT] = 0.20
    ice[:, ICE_LOW] = 24.0
    ice[:, ICE] = 80.0
    ice[:, ICE_AT] = 25.0
    snow[:, BOTH] = 0.45
    ice[:, BOTH] = 80.0
    rain[:, NEGATIVE] = -3.0
    rain[:, ZERO] = 0.0
    return rain, snow, ice


CLIM_CELL = 6000          # the climatology cell of calendar month m is CLIM_CELL + m


def _workdir(tmp_path, monkeypatch, sat, clock, climatology=None):
    """A working folder with the land/sea tag, the chain's three inputs and a snow
    climatology. If `climatology` is given, the file of calendar month m holds that
    value at cell CLIM_CELL + m and zero elsewhere, so each calendar month marks a
    cell of its own."""
    monkeypatch.setattr(gp, '_current_year_month', lambda: clock)
    shutil.copy(os.path.join(MONTHLY_DIR, 'lndsea.25d'), tmp_path / 'lndsea.25d')
    (tmp_path / 'snow').mkdir()
    for m in range(1, 13):
        clim = np.zeros(G, np.float32)
        if climatology is not None:
            clim[CLIM_CELL + m] = climatology
        clim.tofile(tmp_path / 'snow' / f'SN{m:02d}')
    monkeypatch.chdir(tmp_path)
    d = tmp_path / f'{sat}-2.5'
    d.mkdir()
    return d


def _write_inputs(d, sat, rain, snow, ice, rain2=None):
    rain.tofile(d / f'PR1-{sat}-2.5')
    (rain if rain2 is None else rain2).tofile(d / f'PR2-{sat}-2.5')
    snow.tofile(d / f'SNW-{sat}-2.5')
    ice.tofile(d / f'ICE-{sat}-2.5')


def _read(tmp_path, name, n_months):
    return np.fromfile(tmp_path / 'gpcp' / name, dtype=np.float32).reshape(n_months, G)


def _check_flags(out, first_month=0):
    for row in out[first_month:]:
        assert row[PLAIN] == 50.0
        assert row[SNOW_LOW] == 50.0                               # 0.19 is below the threshold
        assert row[SNOW] == pytest.approx(-45.0, rel=1e-6)         # -100 times 0.45
        assert row[SNOW_AT] == pytest.approx(-20.0, rel=1e-6)      # the threshold itself is masked
        assert row[ICE_LOW] == 50.0                                # 24 percent is below the threshold
        assert row[ICE] == -80.0
        assert row[ICE_AT] == -25.0
        assert row[BOTH] == -80.0                                  # ice is applied after snow
        assert is_fill(row[NEGATIVE])
        assert row[ZERO] == 0.0                                    # zero rain is an amount


@pytest.mark.parametrize("func,sat,origin,prefix", [
    ('gpcp_late', 'f17', 1987, 'gpcp_nesdis_pr'),
    ('gpcp_early', 'f16', 1992, 'gpcp_nesdis_f10_pr'),
])
def test_snow_and_ice_flags_of_each_chain(tmp_path, monkeypatch, func, sat, origin, prefix):
    monkeypatch.setattr(gp, '_LATE_SAT', 'f17')
    monkeypatch.setattr(gp, '_EARLY_SAT', 'f16')
    d = _workdir(tmp_path, monkeypatch, sat, (origin, 4))          # twelve months on the axis
    rain, snow, ice = _fields(12)
    _write_inputs(d, sat, rain, snow, ice)

    getattr(gp, func)(path_in=str(d) + os.sep, path_out=str(tmp_path / 'gpcp') + os.sep)

    for product in (1, 2):
        out = _read(tmp_path, f'{prefix}{product}.dat', 12)
        _check_flags(out)
        # Nothing else was touched in the first product.
        if product == 1:
            untouched = np.ones(G, bool)
            untouched[[SNOW, SNOW_AT, ICE, ICE_AT, BOTH, NEGATIVE, ZERO]] = False
            assert np.all(out[:, untouched] == 50.0)


def _coastal_cells():
    """One cell for each line of the legacy coastal rule, chosen from the rule.

    gpcp_f08.pro and gpcp_f10.pro read each field as fltarr(144,72) and loop over
    (i, j), so the value at position k of the file is row j = k // 144. The rule:
        j <= 11 or j >= 60, and land fraction 25 to 75 percent   ->  fill
        j <= 19 or j >= 53, land fraction below 25, rain > 1000  ->  fill
    """
    tag = np.fromfile(os.path.join(MONTHLY_DIR, 'lndsea.25d'), dtype=np.float32)
    row = np.arange(G) // 144
    polar = (row <= 11) | (row >= 60)
    wide = (row <= 19) | (row >= 53)
    mixed = np.where(polar & (tag >= 25.0) & (tag <= 75.0))[0]
    ocean = np.where(wide & (tag < 25.0))[0]
    mixed_outside = np.where(~wide & (tag >= 25.0) & (tag <= 75.0))[0]
    ocean_outside = np.where(~wide & (tag < 25.0))[0]
    # a mixed cell in the wider band but outside the polar one: not masked by the first line
    mixed_between = np.where(wide & ~polar & (tag >= 25.0) & (tag <= 75.0))[0]
    for group in (mixed, ocean, mixed_outside, ocean_outside, mixed_between):
        assert len(group), 'the land/sea file holds no cell for one line of the rule'
    return dict(mixed=int(mixed[0]), ocean=int(ocean[0]), mixed_outside=int(mixed_outside[0]),
                ocean_outside=int(ocean_outside[0]), mixed_between=int(mixed_between[0]))


@pytest.mark.parametrize("func,sat,origin,prefix", [
    ('gpcp_late', 'f17', 1987, 'gpcp_nesdis_pr'),
    ('gpcp_early', 'f16', 1992, 'gpcp_nesdis_f10_pr'),
])
def test_second_product_coastal_masks(tmp_path, monkeypatch, func, sat, origin, prefix):
    """A mixed coastal cell near the poles is fill in the second product only, and so
    is a value above 1000 over ocean at extended high latitudes. The same land
    fraction, and the same high value, away from those rows are left alone."""
    monkeypatch.setattr(gp, '_LATE_SAT', 'f17')
    monkeypatch.setattr(gp, '_EARLY_SAT', 'f16')
    d = _workdir(tmp_path, monkeypatch, sat, (origin, 4))
    c = _coastal_cells()
    rain = np.full((12, G), 50.0, np.float32)
    rain2 = rain.copy()
    rain2[:, c['ocean']] = 1500.0                                  # above 1000 over ocean, inside the band
    rain2[:, c['ocean_outside']] = 1500.0                          # the same value outside the band
    zero = np.zeros((12, G), np.float32)
    _write_inputs(d, sat, rain, zero, zero, rain2=rain2)

    getattr(gp, func)(path_in=str(d) + os.sep, path_out=str(tmp_path / 'gpcp') + os.sep)

    pr1, pr2 = _read(tmp_path, f'{prefix}1.dat', 12), _read(tmp_path, f'{prefix}2.dat', 12)
    assert np.all(pr1 == 50.0)                                     # the first product has no coastal mask
    assert np.all(is_fill(pr2[:, c['mixed']]))
    assert np.all(is_fill(pr2[:, c['ocean']]))
    assert np.all(pr2[:, c['mixed_outside']] == 50.0)
    assert np.all(pr2[:, c['mixed_between']] == 50.0)
    assert np.all(pr2[:, c['ocean_outside']] == 1500.0)


def test_morning_chain_uses_the_snow_climatology_from_july_1990_to_december_1991(tmp_path, monkeypatch):
    """Months 43 to 60. The month's own snow field, 0.45 at one cell, is ignored
    there, and the climatology of the calendar month gives fill and not a flag. The
    files run on to February 1992, so the return to the month's own field at month
    61 is checked as well. Each calendar month's climatology marks a different
    cell, so a climatology read for the wrong calendar month shows."""
    monkeypatch.setattr(gp, '_LATE_SAT', 'f17')
    d = _workdir(tmp_path, monkeypatch, 'f17', (1992, 3), climatology=0.30)
    n = 72                                                         # the axis runs to December 1992
    data = 62                                                      # data from January 1987 to February 1992
    rain, snow, ice = _fields(data)
    _write_inputs(d, 'f17', rain, snow, ice)

    gp.gpcp_late(path_in=str(d) + os.sep, path_out=str(tmp_path / 'gpcp') + os.sep)

    out = _read(tmp_path, 'gpcp_nesdis_pr1.dat', n)
    every_clim_cell = [CLIM_CELL + m for m in range(1, 13)]
    _check_flags(out[:42])                                         # through June 1990: the month's own field
    assert np.all(out[:42][:, every_clim_cell] == 50.0)            # and no climatology
    for jmon in range(43, 61):                                     # July 1990 to December 1991
        row = out[jmon - 1]
        calendar_month = (jmon - 1) % 12 + 1
        assert row[SNOW] == 50.0 and row[SNOW_AT] == 50.0          # the month's own 0.45 and 0.20 are not used
        for m in range(1, 13):                                     # only this calendar month's climatology
            if m == calendar_month:
                assert is_fill(row[CLIM_CELL + m]), (jmon, m)
            else:
                assert row[CLIM_CELL + m] == 50.0, (jmon, m)
        assert row[ICE] == -80.0 and row[BOTH] == -80.0            # ice is applied as in any month
        assert is_fill(row[NEGATIVE])
    _check_flags(out[60:data])                                     # January and February 1992: back to the month's own field
    assert np.all(out[60:data][:, every_clim_cell] == 50.0)
    assert np.all(is_fill(out[data:]))                             # no input, fill


@pytest.mark.parametrize('level', [0.19, 0.20, 0.21])
def test_climatology_threshold(tmp_path, monkeypatch, level):
    """The climatology masks at 0.20 and above, like the month's own snow field. It
    is a separate line of the rule and needs its own threshold case. Added 18
    November 2026 because the earlier cases did not fail with the threshold set
    above 0.20."""
    monkeypatch.setattr(gp, '_LATE_SAT', 'f17')
    d = _workdir(tmp_path, monkeypatch, 'f17', (1992, 3), climatology=level)
    rain, snow, ice = _fields(62)
    _write_inputs(d, 'f17', rain, snow, ice)
    gp.gpcp_late(path_in=str(d) + os.sep, path_out=str(tmp_path / 'gpcp') + os.sep)
    for product in (1, 2):
        out = _read(tmp_path, f'gpcp_nesdis_pr{product}.dat', 72)
        value = out[42, CLIM_CELL + 7]                             # July 1990, the July cell
        if level >= 0.20:
            assert is_fill(value)
        else:
            assert value == 50.0


def test_late_morning_chain_never_uses_the_climatology(tmp_path, monkeypatch):
    """gpcp_f10.pro has no climatology window. Its record begins in 1992."""
    monkeypatch.setattr(gp, '_EARLY_SAT', 'f16')
    d = _workdir(tmp_path, monkeypatch, 'f16', (1997, 1), climatology=0.30)
    n = 60                                                         # January 1992 to December 1996
    rain, snow, ice = _fields(n)
    _write_inputs(d, 'f16', rain, snow, ice)

    gp.gpcp_early(path_in=str(d) + os.sep, path_out=str(tmp_path / 'gpcp') + os.sep)

    out = _read(tmp_path, 'gpcp_nesdis_f10_pr1.dat', n)
    _check_flags(out)
    assert np.all(out[:, [CLIM_CELL + m for m in range(1, 13)]] == 50.0)
