"""The pentad processor against the monthly processor, on the same input.

The legacy pentad and monthly programs share their per-pixel code, so on the
same daily files the two Python processors must agree. For the first five days
of a year the pentad processor's first pentad and the monthly processor's
January are made from exactly the same samples. Eight of the ten products do
not depend on the length of the averaging period and must be EQUAL, value for
value. The two rainfall totals scale with the number of days (5 against 31).

This comparison is the regression case the pentad processor lacked. Until 18
November 2026 it carried its own copy of the per-file code, and that copy had
missed two repairs the monthly processor received on 9 October: the six-channel
test for a missing pixel, and the row at which equatorial sea ice is masked. It
also never selected a satellite's own scattering-index table. Each of the three
shows here as a difference between the two processors.

The input is synthetic and built in the test from a fixed seed: full-size
passes of antenna temperatures between 140 and 289 K, with a tenth of the
pixels missing in every channel and a few percent holding the 85 GHz vertical
channel alone. No operational daily grid is needed. The tests do need NumPy,
SciPy, the repository's land tag and lookup tables, and, for the table case, a
satellite in satellite_config that carries its own table.

Agreement is shown for the synthetic days executed, which are days 1 and 3 of
2026. Those days take one branch of the seasonal snow check and one of the two
sea-ice channel choices, so a separate test records every argument the pentad
wrapper hands to the shared function, for a day and a year on the other
branches. The comparison says the two processors agree with each other. Whether
either agrees with the legacy Fortran is the business of the golden month and
the conversion record, which need the operational grids.
"""
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
import climalg_ssmis as cl
import pentad_ssmis as pn
import satellite_config as sc

from conftest import MONTHLY_DIR

FILL = np.float32(-999.99)
YEAR = 2026
PASSES = (('as', 1), ('ds', 1), ('as', 3))          # the files that exist in days 1 to 5
SAME = ('PF1', 'PF2', 'LWP', 'CFR', 'WVP', 'SNW', 'ICE', 'SSA')
TOTALS = ('PR1', 'PR2')
# Rows whose equatorial sea-ice mask differed between the two processors: with rows
# counted from 1, as the Fortran does, the band is rows 150 to 390, which are the
# zero-based rows 149 to 389. The copy in the pentad processor used 150 to 390.
BOUNDARY_ROWS = (149, 390)


def _synthetic_pass(seed):
    """One pass file's bytes, shape (NROW, NCOL, 7), and the mask of 85V-only pixels."""
    rng = np.random.default_rng(seed)
    byte = (rng.integers(140, 290, size=(cl.NROW, cl.NCOL, 7)) - 70).astype(np.uint8)
    pick = rng.uniform(size=(cl.NROW, cl.NCOL))
    byte[pick < 0.10] = 32                                   # no channel observed
    only85v = (pick >= 0.10) & (pick < 0.14)
    keep = byte[only85v, 5].copy()
    byte[only85v] = 32
    byte[only85v, 5] = keep                                  # 85 GHz vertical alone
    return byte, only85v


@pytest.fixture(scope='module')
def grids(tmp_path_factory):
    """The synthetic passes on disk, under two satellite names."""
    d = tmp_path_factory.mktemp('grids')
    own = [name for name in sc.SATELLITE_REGISTRY if sc.get_lut_add_si(name) is not None]
    sats = ['f17'] + own[:1]
    info = {'dir': str(d) + os.sep, 'own_table_sat': own[0] if own else None, 'only85v': 0, 'ice_rows': {}}
    tag = cl.load_land_tag(os.path.join(MONTHLY_DIR, 'NLNDSEA.TAG'))
    for k, (prefix, jday) in enumerate(PASSES):
        byte, only85v = _synthetic_pass(100 + k)
        info['only85v'] += int(only85v.sum())
        ice = cl.seaice(byte.astype(np.float32) + 70.0, tag == 0, 8)
        for r in BOUNDARY_ROWS:
            info['ice_rows'][r] = info['ice_rows'].get(r, 0) + int((ice[r] == 100.0).sum())
        for sat in sats:
            byte.tofile(d / f'{prefix}{YEAR % 100:02d}{jday:03d}.{sat}')
    return info


def _run_both(sat, grids, out, coeff_dir, monkeypatch):
    """Pentad 1 and January from the same five days. Returns two dicts of (72, 144) fields."""
    monkeypatch.chdir(MONTHLY_DIR)                 # the tables are read from the working folder
    pn.run(sat, YEAR, 1, 5, outdir=str(out), inpath=grids['dir'], coeff_dir=coeff_dir)
    cl.run(sat, YEAR, 1, 5, res=2.5, outdir=str(out), inpath=grids['dir'], coeff_dir=coeff_dir)
    d = os.path.join(str(out), f'{sat}-2.5')
    pentad, month = {}, {}
    for p in SAME + TOTALS:
        pen = np.fromfile(os.path.join(d, f'{p}{YEAR % 100:02d}-{sat}-2.5.pen'), dtype=np.float32)
        assert pen.size == pn.N_PENTADS * 72 * 144
        pen = pen.reshape(pn.N_PENTADS, 72, 144)
        assert np.all(pen[1:] == FILL), 'a pentad after the first holds data'
        pentad[p] = pen[0]
        month[p] = np.fromfile(os.path.join(d, f'{p}{YEAR % 100:02d}-01-{sat}-2.5'),
                               dtype=np.float32).reshape(72, 144)
    return pentad, month


def _assert_agree(pentad, month):
    for p in SAME:
        assert np.array_equal(pentad[p], month[p]), (
            f'{p}: {int((pentad[p] != month[p]).sum())} cells differ between the pentad and the month')
    for p in TOTALS:
        filled = month[p] == FILL
        assert np.array_equal(pentad[p] == FILL, filled), p
        # mean rate times 24 times the number of days: 5 for a pentad, 31 for January
        np.testing.assert_allclose(pentad[p][~filled], month[p][~filled] * (5.0 / 31.0),
                                   rtol=1e-5, atol=1e-6, err_msg=p)


def test_the_synthetic_input_exercises_the_cases_that_were_wrong(grids):
    """Without these the agreement below would say nothing about the repairs."""
    assert grids['only85v'] > 10000, 'no pixel holds the 85 GHz vertical channel alone'
    for r in BOUNDARY_ROWS:
        assert grids['ice_rows'][r] > 0, f'no sea ice detected in boundary row {r}'


def test_first_pentad_agrees_with_january_from_the_same_five_days(grids, tmp_path, monkeypatch):
    pentad, month = _run_both('f17', grids, tmp_path, MONTHLY_DIR, monkeypatch)
    _assert_agree(pentad, month)
    # The fields are not trivially equal: each holds data and some cells are positive.
    for p in SAME + TOTALS:
        assert (pentad[p] != FILL).sum() > 1000, p
        assert (pentad[p] > 0).any(), p


def test_pentad_reads_the_satellites_own_scattering_index_table(grids, tmp_path, monkeypatch):
    """With no coefficient folder named, a satellite that carries its own table in
    satellite_config reads it, in both processors."""
    sat = grids['own_table_sat']
    if sat is None:
        pytest.skip('no satellite in satellite_config carries its own scattering-index table')
    own = sc.get_lut_add_si(sat)
    standard = os.path.join(MONTHLY_DIR, 'lut_add_si')
    assert open(own, 'rb').read() != open(standard, 'rb').read()

    pentad, month = _run_both(sat, grids, tmp_path / 'own', None, monkeypatch)
    _assert_agree(pentad, month)

    # The table matters on this input: the same satellite run with the standard table,
    # named through the coefficient folder, gives a different rainfall field.
    with_standard, _ = _run_both(sat, grids, tmp_path / 'standard', MONTHLY_DIR, monkeypatch)
    assert not np.array_equal(pentad['PR1'], with_standard['PR1'])


@pytest.mark.parametrize("year,feb_end,last", [(2026, 60, 365), (2024, 61, 366), (2100, 60, 365)])
def test_pentad_boundaries(year, feb_end, last):
    """Seventy-three pentads of five days, the twelfth taking the leap day."""
    ipen = pn.build_pentad_boundaries(year)
    assert len(ipen) == 73
    assert list(ipen[:11]) == [5 * (i + 1) for i in range(11)]
    assert ipen[11] == feb_end
    assert ipen[-1] == last
    assert set(np.diff(ipen[12:])) == {5}


def test_the_wrapper_hands_every_argument_on_unchanged(monkeypatch):
    """The pentad wrapper is one call. Each argument is given a value that no other
    argument has, and the shared function must receive them in its own order, with
    the 2.5-degree grid size and half the row count added. A year or a day replaced
    by a constant, or two arguments exchanged, fails here."""
    seen = {}

    def record(*args):
        seen['args'] = args
        return 'returned'

    monkeypatch.setattr(pn, 'process_file', record)
    given = ('the path', 'the tag', 'ocean table', 'land table', 'ice table',
             'the accumulator', 'row map', 'column map')
    assert pn.process_file_pentad(*given, 1990, 181) == 'returned'
    assert seen['args'] == given + (72, 144, 1990, 181, cl.NROW // 2)


def test_the_run_hands_each_days_year_and_day_to_the_wrapper(tmp_path, monkeypatch):
    """A run over days 1 to 185 of 1990 with two files present, one of them on day
    181, where the sea-ice channel choice changes. The year and the day reach the
    shared function as they are, and the run asks for the standard table when a
    coefficient folder is named."""
    calls = []

    def record(fpath, tag, add_oce, add_lnd, add_si, rrmon, ii, jj, m, n, yyyy, jday, half):
        calls.append((os.path.basename(fpath), m, n, yyyy, jday, half))
        return True

    monkeypatch.setattr(pn, 'process_file', record)
    for name in ('as90091.f17', 'ds90181.f17'):
        (tmp_path / name).write_bytes(b'')
    monkeypatch.chdir(MONTHLY_DIR)

    pn.run('f17', 1990, 1, 185, outdir=str(tmp_path / 'out'), inpath=str(tmp_path) + os.sep,
           coeff_dir=MONTHLY_DIR)

    assert calls == [('as90091.f17', 72, 144, 1990, 91, cl.NROW // 2),
                     ('ds90181.f17', 72, 144, 1990, 181, cl.NROW // 2)]
    pen = np.fromfile(tmp_path / 'out' / 'f17-2.5' / 'PR190-f17-2.5.pen', dtype=np.float32)
    assert pen.size == pn.N_PENTADS * 72 * 144


def test_process_file_pentad_is_the_monthly_function(grids):
    """One file through each entry point gives the same accumulator, cell for cell."""
    tag = cl.load_land_tag(os.path.join(MONTHLY_DIR, 'NLNDSEA.TAG'))
    add_oce, add_lnd, add_si = cl.load_luts(MONTHLY_DIR)
    res = 360.0 / cl.NCOL
    ii = np.minimum(((np.arange(cl.NROW) + 1) * res / 2.5).astype(int), 71)
    jj = np.minimum(((np.arange(cl.NCOL) + 1) * res / 2.5).astype(int), 143)
    f = os.path.join(grids['dir'], f'as{YEAR % 100:02d}001.f17')
    a = np.zeros((72, 144, cl.NT), np.float32)
    b = np.zeros((72, 144, cl.NT), np.float32)
    assert pn.process_file_pentad(f, tag, add_oce, add_lnd, add_si, a, ii, jj, YEAR, 1)
    assert cl.process_file(f, tag, add_oce, add_lnd, add_si, b, ii, jj, 72, 144, YEAR, 1, cl.NROW // 2)
    assert np.array_equal(a, b)
    assert a[:, :, cl.NT - 3].sum() == cl.NROW * cl.NCOL
