"""Reference-parity tests: the WSF-M adapters must reproduce the OPERATIONAL
IDL chain, not merely be self-consistent.

Why this file exists. Seven WSF-M defects were found by independent checking,
and every one was an input-correctness failure that the project's own
verification could not see:

    1. missing Ta encoded as byte 0, decoding to a valid-looking 70 K
    2. monthly accumulations reported as mm/day
    3. a scattering proxy the retrieval does not compute
    4. floor gridding against the decoder's nearest-node gridding
    5. np.rint half-to-even against IDL ROUND half-away-from-zero
    6. a raw scattering index against the retrieval's corrected one
    7. finite out-of-range Ta clipped into range instead of set to fill

Self-consistency checking is structurally blind to all of these, because each one
is applied identically to both sides of whatever comparison is being made. A
half-cell offset, a wrong unit, and a wrong fill sentinel all cancel in a
difference and leave the difference looking healthy.

So these tests encode the operational RULES, transcribed from the IDL with the
source lines cited, and assert the Python reproduces them. The reference side is
written in plain Python and deliberately avoids the numpy function the
implementation uses, because a reference that calls the same primitive is a
restatement, not a check. That mistake was live in
test_wsfm_grid_registration.py until 2026-07-29: it claimed an independent
reimplementation while calling np.rint on both sides, and so could not have
caught defect 5.

Transcribed rules:

  ssmis_tdr_decoder.pro:229-233   grid placement
      ii = ROUND(SCALE*(180+lon)),  jj = ROUND(SCALE*(90+lat))
      if (ii ge LONLEN or ii lt 0) then ii = 0      ; dateline fold
      jj clipped to [0, NROW-1]                     ; pole clip
      IDL ROUND is half AWAY FROM ZERO.

  ssmis_tdr_format.pro:120-137    storage order and fill folding
      i_index = 539 - iii                           ; south-first -> north-first
      index = where(out_chan le 32) -> out_chan = 32 ; EVERY low byte is fill

  ssmis_tdr_decoder.pro:351-395   valid interval
      where(Ta gt 70 and Ta le 325) -> byte = Ta-70, complement -> byte 0
      combined with the format stage: valid data requires Ta > 102 K.
"""
import importlib.util
import math
import os

import pytest

np = pytest.importorskip("numpy")

from conftest import MONTHLY_DIR

NROW, NCOL, SCALE = 540, 1080, 3.0
FILL_BYTE = 32
# The IMG+ENV adapter reads an evaluation delivery and is not part of every tree. Where it is absent its cases are
# reported as skipped with this reason, never silently dropped.
IMGENV = "wsfm_adapter_imgenv.py"
SKIP_IMGENV = pytest.mark.skipif(not os.path.isfile(os.path.join(MONTHLY_DIR, IMGENV)),
                                 reason="the IMG+ENV adapter is not part of this tree")
ADAPTERS = ["wsfm_adapter.py", pytest.param(IMGENV, marks=SKIP_IMGENV)]


def _load(name):
    path = os.path.join(MONTHLY_DIR, name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Reference side, plain Python, no numpy primitives shared with the adapters
# --------------------------------------------------------------------------

def ref_round(x):
    """IDL ROUND: nearest integer, halves away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def ref_encode_one(ta):
    """Operational byte for a single Ta, both IDL stages combined.

    Stage 1 (decoder): valid iff 70 < Ta <= 325, else byte 0.
    Stage 2 (format):  any byte <= 32 becomes 32.
    Net effect: Ta must exceed 102 K to survive as data.
    """
    if ta != ta or ta in (float("inf"), float("-inf")):      # NaN or inf
        return FILL_BYTE
    if not (70.0 < ta <= 325.0):
        return FILL_BYTE                                     # byte 0 -> 32
    b = int(ta - 70.0)                                       # IDL byte() truncates
    return FILL_BYTE if b <= 32 else b


# --------------------------------------------------------------------------
# Encoding parity
# --------------------------------------------------------------------------

# Spans every boundary of the combined rule, including the band that the old
# clipping implementation silently turned into valid cold data.
TA_CASES = [
    float("nan"), float("inf"), float("-inf"),
    -40.0, 0.0, 69.9, 70.0, 70.1,          # at or below the decoder's floor
    71.0, 85.0, 100.0, 101.9, 102.0,       # the band that used to become valid
    102.1, 102.9,                          # truncate to byte 32 -> fill
    103.0, 150.0, 249.5, 300.0,            # genuinely valid (byte >= 33)
    325.0, 325.1, 400.0, 1e6,              # at and above the ceiling
]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_encoding_matches_operational_rule(adapter):
    m = _load(adapter)
    got = m._encode_ta(np.array([TA_CASES], dtype=np.float32))
    want = [ref_encode_one(v) for v in TA_CASES]
    assert list(np.ravel(got)) == want, (
        f"{adapter} encoding differs from the operational chain\n"
        f"  Ta   : {TA_CASES}\n  got  : {list(np.ravel(got))}\n  want : {want}"
    )


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_cold_band_is_fill_not_valid_data(adapter):
    """The regression that motivated this file.

    Ta in (70, 102] used to encode to bytes 1..32, decoding to a valid-looking
    71-102 K. climalg's skip test fires only at exactly 102 K, so those entered
    the retrievals as real data, and a spuriously cold 91V inflates the
    scattering index and fabricates rain. Audited at 81,446 PointCloud and
    13,114 ENV values over 24 delivered orbits.
    """
    m = _load(adapter)
    cold = np.array([[70.5, 75.0, 85.0, 95.0, 101.0, 102.0]], dtype=np.float32)
    assert set(np.ravel(m._encode_ta(cold))) == {FILL_BYTE}


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_no_finite_input_decodes_to_impossible_temperature(adapter):
    """No input may produce a byte that decodes below the fill temperature.

    A byte under 32 decodes to under 102 K, which is not physically reachable for
    an Earth-viewing microwave antenna temperature. This is the invariant that
    would have caught defects 1 and 7 directly.
    """
    m = _load(adapter)
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        rng.uniform(-100.0, 500.0, 20000),
        np.array(TA_CASES, dtype=np.float64),
    ]).astype(np.float32)
    out = m._encode_ta(vals.reshape(1, -1))
    assert int(np.ravel(out).min()) >= FILL_BYTE


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_encoding_truncates_like_idl_byte(adapter):
    """IDL byte() truncates toward zero; it does not round."""
    m = _load(adapter)
    got = np.ravel(m._encode_ta(np.array([[150.9, 151.0, 151.9]], dtype=np.float32)))
    assert list(got) == [80, 81, 81]


# --------------------------------------------------------------------------
# Grid placement parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("adapter", ADAPTERS)
def test_grid_placement_matches_decoder_on_a_sweep(adapter):
    """Dense sweep, not a hand-picked list, so an off-by-one cannot hide."""
    m = _load(adapter)
    lons = [-180.0 + i * 0.05 for i in range(7201)]
    lats = [-90.0 + i * 0.05 for i in range(3601)]

    got_c = list(m._lon_to_col(np.array(lons, dtype=np.float64)))
    want_c = []
    for v in lons:
        ii = ref_round(SCALE * (180.0 + v))
        want_c.append(0 if (ii >= NCOL or ii < 0) else ii)
    assert got_c == want_c

    got_r = list(m._lat_to_row(np.array(lats, dtype=np.float64)))
    want_r = []
    for v in lats:
        jj = min(max(ref_round(SCALE * (90.0 + v)), 0), NROW - 1)
        want_r.append((NROW - 1) - jj)
    assert got_r == want_r


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_storage_is_north_first(adapter):
    """ssmis_tdr_format.pro:126 reverses latitude: i_index = 539 - iii."""
    m = _load(adapter)
    assert int(m._lat_to_row(np.array([90.0]))[0]) == 0
    assert int(m._lat_to_row(np.array([-90.0]))[0]) == NROW - 1


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_out_of_domain_geolocation_cannot_land_on_a_real_cell(adapter):
    """Longitude outside [-180, 180] folds to column 0, as the IDL does.

    Guards the overflow path too: a large NetCDF sentinel must not wrap into an
    arbitrary column via the int32 cast.
    """
    m = _load(adapter)
    for v in (-200.0, 180.5, 360.0, 1e6, -1e6):
        c = int(m._lon_to_col(np.array([v], dtype=np.float64))[0])
        assert c == 0, f"longitude {v} produced column {c}, expected the fold to 0"
    for v in (-95.0, 91.0, 1e6, -1e6):
        r = int(m._lat_to_row(np.array([v], dtype=np.float64))[0])
        assert 0 <= r < NROW, f"latitude {v} produced row {r}, out of range"


# --------------------------------------------------------------------------
# Multi-orbit merge policy
# --------------------------------------------------------------------------

def test_merge_is_last_write_wins_including_fill():
    """A later footprint must overwrite an earlier one even when it is fill.

    ssmis_tdr_decoder.pro:235-241 assigns every gridded footprint
    unconditionally (`ssmis_91v_as(ii,jj) = a91v`) and only applies the
    valid-range test afterward, to the whole accumulated grid. So when two
    orbits cover one cell and the second carries an out-of-range Ta, the cell
    ends as FILL, not as the first orbit's value.

    wsfm_adapter.py merged on `!= FILL_BYTE` until 2026-07-29, which made a fill
    fail to overwrite and left a stale value from an earlier orbit. That was
    concealed while out-of-range Ta was still being clipped to a non-fill byte:
    the two defects hid each other, which is why this test exists rather than
    only the encoder test above.
    """
    m = _load("wsfm_adapter.py")
    NCH = m.N_CHAN

    def swath(lat, lon, ta_value):
        n = 4          # >1 scan so the ascending logic has something to compare
        return {
            "lat": np.full((n, 1), lat, dtype=np.float64),
            "lon": np.full((n, 1), lon, dtype=np.float64),
            "ta": np.full((n, 1, NCH), ta_value, dtype=np.float32),
            "ascending": np.ones(n, dtype=bool),
            "scan_day": np.full(n, 2026060, dtype=np.int64),
        }

    valid_grid, valid_touched = m.grid_swath_to_binary(swath(10.0, 20.0, 250.0), True)
    cold_grid, cold_touched = m.grid_swath_to_binary(swath(10.0, 20.0, 85.0), True)

    r = int(m._lat_to_row(np.array([10.0]))[0])
    c = int(m._lon_to_col(np.array([20.0]))[0])

    # The cold orbit encodes to fill, yet still reports the cell as OBSERVED.
    assert cold_grid[r, c, 0] == FILL_BYTE
    assert cold_touched[r, c], "an observed cell must be reported touched even when it encodes to fill"
    assert valid_touched[r, c]

    # Exercise the REAL merge in process_day, not a replay of it. A replay would
    # restate the implementation, which is the defect this file exists to avoid.
    import tempfile

    orbits = [swath(10.0, 20.0, 250.0), swath(10.0, 20.0, 85.0)]

    def fake_read(path):
        return orbits[int(os.path.basename(path))]

    real_read = m.read_wsfm_orbit
    m.read_wsfm_orbit = fake_read
    try:
        with tempfile.TemporaryDirectory() as d:
            m.process_day(["0", "1"], d, 2026, 60)
            out = np.fromfile(os.path.join(d, "as26060.wsfm"), dtype=np.uint8)
            out = out.reshape(540, 1080, NCH)
            assert out[r, c, 0] == FILL_BYTE, (
                "the later out-of-range footprint must overwrite the earlier valid "
                "one, leaving fill; got a stale value, which is the "
                "last-valid-wins defect"
            )
            assert out[0, 0, 0] == FILL_BYTE, "an unobserved cell must stay fill"

        # Reverse order keeps the valid value, so this is genuinely ordering and
        # not simply everything collapsing to fill.
        orbits.reverse()
        with tempfile.TemporaryDirectory() as d:
            m.process_day(["0", "1"], d, 2026, 60)
            out = np.fromfile(os.path.join(d, "as26060.wsfm"), dtype=np.uint8)
            out = out.reshape(540, 1080, NCH)
            assert out[r, c, 0] == int(250.0 - 70.0)
    finally:
        m.read_wsfm_orbit = real_read


@pytest.mark.parametrize("name, fn", [("wsfm_adapter.py", "_is_ascending"),
                                      pytest.param(IMGENV, "_is_ascending_per_scan", marks=SKIP_IMGENV)])
def test_single_scan_granule_fails_closed(name, fn):
    """Granules with fewer than 2 scans must be REJECTED, not guessed.

    Pass direction comes from the latitude trend between scans, so a 0- or
    1-scan granule is unclassifiable. An earlier version defaulted these to
    ascending, inventing a pass class that could contaminate the ascending
    composite. Now the classifier raises and the day loops
    skip the granule visibly. Audited on the full delivery (2 x 1,264 orbit
    files): zero 1-scan granules and 28 empty 0-scan granules exist, so the
    change is byte-identical on all assessed data.
    """
    m = _load(name)
    f = getattr(m, fn)
    for n in (0, 1):
        with pytest.raises(ValueError):
            f(np.linspace(0.0, 1.0, max(n, 1) * 1).reshape(n, 1) if n else
              np.empty((0, 1)))
    out = f(np.array([[0.0], [1.0], [2.0]]))
    assert len(np.atleast_1d(out)) == 3


# --------------------------------------------------------------------------
# Longitude convention at the input boundary (fixed 2026-09-08)
# --------------------------------------------------------------------------

def test_unsigned_longitude_lands_on_the_same_column_as_signed():
    """270 E and -90 are the same meridian and must hit the same column.

    The delivered PointCloud `lon` is 0 to 360; the node mapping assumes
    -180 to 180 and folds anything else onto column 0. Before the fix every
    western-hemisphere pixel landed on column 0.
    """
    m = _load("wsfm_adapter.py")
    unsigned = np.array([270.0, 200.5, 359.0, 360.0, 180.0, 90.0, 0.0], dtype=np.float32)
    signed = np.array([-90.0, -159.5, -1.0, 0.0, 180.0, 90.0, 0.0], dtype=np.float32)
    got = m.normalize_longitude(unsigned.copy(), "synthetic")
    assert np.array_equal(m._lon_to_col(got), m._lon_to_col(signed))
    # the western column is a real interior column, not the fold column
    assert int(m._lon_to_col(got)[0]) == ref_round(SCALE * 90.0)
    assert int(m._lon_to_col(got)[0]) != 0


def test_signed_longitude_passes_through_unchanged():
    m = _load("wsfm_adapter.py")
    signed = np.array([-179.9, -90.0, 0.0, 90.0, 180.0, np.nan], dtype=np.float32)
    got = m.normalize_longitude(signed.copy(), "synthetic")
    assert np.array_equal(got[:5], signed[:5])
    assert np.isnan(got[5])


def test_longitude_outside_both_conventions_is_refused():
    m = _load("wsfm_adapter.py")
    for bad in ([-181.0, 0.0], [0.0, 360.5], [1e6, 0.0]):
        with pytest.raises(ValueError, match="neither signed"):
            m.normalize_longitude(np.array(bad, dtype=np.float32), "synthetic")


def test_unsigned_swath_populates_the_western_hemisphere():
    """An unsigned swath spanning 190 to 350 E must fill columns 30 to 510,
    the western interior, and leave column 0 empty."""
    m = _load("wsfm_adapter.py")
    nscan, npix = 40, 50
    lon = np.tile(np.linspace(190.0, 350.0, npix, dtype=np.float32), (nscan, 1))
    lat = np.tile(np.linspace(-40.0, 40.0, nscan, dtype=np.float32)[:, None], (1, npix))
    tas = np.full((nscan, npix, 7), 250.0, dtype=np.float32)
    lon = m.normalize_longitude(lon, "synthetic")
    swath = {"lat": lat, "lon": lon, "ta": tas,
             "ascending": np.ones(nscan, dtype=bool)}
    grid, _touched = m.grid_swath_to_binary(swath, ascending_only=True)
    valid = grid[:, :, 0] != m.FILL_BYTE
    west = int(valid[:, :NCOL // 2].sum())
    east = int(valid[:, NCOL // 2:].sum())
    assert west > 0 and east == 0, f"west {west} east {east}"
    assert int(valid[:, 0].sum()) == 0, "nothing may fold onto the dateline column"


# --------------------------------------------------------------------------
# Hemisphere balance gate and fatal unreadable input
# --------------------------------------------------------------------------

def test_hemisphere_gate_refuses_a_collapsed_hemisphere():
    m = _load("wsfm_adapter.py")
    grid = np.full((NROW, NCOL, 7), m.FILL_BYTE, dtype=np.uint8)
    grid[100:300, 600:900, :] = 180          # east only
    grid[100:101, 0:1, :] = 180              # the 119-cell shape, in miniature
    with pytest.raises(ValueError, match="hemisphere balance failed"):
        m.hemisphere_balance(grid, "as26075.wsfm", 14)


def test_hemisphere_gate_does_not_judge_a_sparse_day():
    """One orbit's ascending half is on one side of the globe and its
    descending half on the other; 28 January 2026 has a single granule."""
    m = _load("wsfm_adapter.py")
    grid = np.full((NROW, NCOL, 7), m.FILL_BYTE, dtype=np.uint8)
    grid[100:300, 600:900, :] = 180
    m.hemisphere_balance(grid, "as26028.wsfm", 1)
    m.hemisphere_balance(grid, "as26028.wsfm", m.FULL_DAY_MIN_ORBITS - 1)
    with pytest.raises(ValueError, match="hemisphere balance failed"):
        m.hemisphere_balance(grid, "as26028.wsfm", m.FULL_DAY_MIN_ORBITS)


def test_hemisphere_gate_accepts_a_real_looking_pass():
    m = _load("wsfm_adapter.py")
    grid = np.full((NROW, NCOL, 7), m.FILL_BYTE, dtype=np.uint8)
    grid[50:450, 100:400, :] = 180
    grid[50:450, 640:1000, :] = 180
    m.hemisphere_balance(grid, "as26075.wsfm", 14)
    empty = np.full((NROW, NCOL, 7), m.FILL_BYTE, dtype=np.uint8)
    m.hemisphere_balance(empty, "empty is a data gap, not a convention defect", 14)


def test_unreadable_granule_makes_the_day_fatal(tmp_path):
    """A first version skipped unreadable granules and wrote all-fill grids
    with exit 0. A read failure must refuse the day and write nothing."""
    m = _load("wsfm_adapter.py")
    bad = tmp_path / "1C.WSFM1.MWI.XCAL2026-V.20260316-S000000-E013000.000001.PARV817.nc"
    bad.write_bytes(b"not a netcdf file")
    out = tmp_path / "grids"
    out.mkdir()
    with pytest.raises(RuntimeError, match="unreadable granule"):
        m.process_day([str(bad)], str(out), 2026, 75)
    assert list(out.iterdir()) == [], "no grid may be written for a refused day"


# --------------------------------------------------------------------------
# UTC-day scan assignment (fixed 2026-09-08)
# --------------------------------------------------------------------------

_DAY_FUNCS = {
    "wsfm_adapter.py": ("read_scan_days", "crossing_granules", "hemisphere_balance"),
    "wsfm_adapter_imgenv.py": ("_read_scan_days", "_crossing_granules", "_hemisphere_balance"),
}

_STAMPS = ["2026-02-28T23:04:23.045Z", "2026-02-28T23:59:59.999Z",
           "2026-03-01T00:00:00.000Z", "2026-03-01T00:45:45.563Z"]


def _write_stamp_file(path, stamps):
    import netCDF4
    with netCDF4.Dataset(str(path), "w") as f:
        f.createDimension("nscans", len(stamps))
        v = f.createVariable("scan_datetime", str, ("nscans",))
        for i, s in enumerate(stamps):
            v[i] = s


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_scan_days_follow_the_scan_time_not_the_filename(adapter, tmp_path):
    """Granule 009775 is named for 28 Feb and ends 1 Mar 00:45. Each scan
    must carry its own UTC day, as the F-17 decoder assigns it."""
    import netCDF4
    m = _load(adapter)
    fn = getattr(m, _DAY_FUNCS[adapter][0])
    p = tmp_path / "g.nc"
    _write_stamp_file(p, _STAMPS)
    with netCDF4.Dataset(str(p)) as f:
        days = fn(f, 4, str(p))
    assert list(days) == [2026059, 2026059, 2026060, 2026060]
    with netCDF4.Dataset(str(p)) as f:
        assert list(fn(f, 0, str(p))) == []          # empty granule, no stamps needed
        with pytest.raises(ValueError, match="scan_datetime has 4 entries"):
            fn(f, 5, str(p))


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_missing_scan_datetime_is_refused_for_a_populated_granule(adapter, tmp_path):
    import netCDF4
    m = _load(adapter)
    fn = getattr(m, _DAY_FUNCS[adapter][0])
    p = tmp_path / "nostamp.nc"
    with netCDF4.Dataset(str(p), "w") as f:
        f.createDimension("nscans", 3)
    with netCDF4.Dataset(str(p)) as f:
        with pytest.raises(ValueError, match="scan_datetime"):
            fn(f, 3, str(p))


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_only_midnight_crossing_granules_come_from_the_preceding_day(adapter, tmp_path):
    m = _load(adapter)
    fn = getattr(m, _DAY_FUNCS[adapter][1])
    names = [
        "1C.WSFM1.MWI.XCAL2026-V.20260228-S000001-E013000.009761.PARV817.nc",
        "1C.WSFM1.MWI.XCAL2026-V.20260228-S213000-E230422.009774.PARV817.nc",
        "1C.WSFM1.MWI.XCAL2026-V.20260228-S230423-E004546.009775.PARV817.nc",
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    got = [os.path.basename(x) for x in fn(str(tmp_path), "*.nc")]
    assert got == [names[2]]


def test_select_scans_on_day_keeps_only_the_target_day():
    m = _load("wsfm_adapter.py")
    nscan = 6
    swath = {
        "lat": np.arange(nscan, dtype=np.float32)[:, None].repeat(3, 1),
        "lon": np.zeros((nscan, 3), dtype=np.float32),
        "ta": np.zeros((nscan, 3, 7), dtype=np.float32),
        "ascending": np.array([True, True, False, False, True, False]),
        "scan_day": np.array([2026059] * 4 + [2026060] * 2),
    }
    out = m.select_scans_on_day(swath, 2026, 60)
    assert out["n_scans_total"] == 6 and out["n_scans_kept"] == 2
    assert list(out["lat"][:, 0]) == [4.0, 5.0]
    assert list(out["ascending"]) == [True, False]
    none = m.select_scans_on_day(swath, 2026, 61)
    assert none["n_scans_kept"] == 0 and none["ta"].shape == (0, 3, 7)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_hemisphere_gate_in_both_adapters(adapter):
    m = _load(adapter)
    gate = getattr(m, _DAY_FUNCS[adapter][2])
    grid = np.full((NROW, NCOL, 7), m.FILL_BYTE, dtype=np.uint8)
    grid[100:300, 600:900, :] = 180
    with pytest.raises(ValueError, match="hemisphere balance failed"):
        gate(grid, "ds26060", 14)
    gate(grid, "ds26060", 1)                 # a one-orbit day is not judged
    grid[100:300, 100:400, :] = 180
    gate(grid, "ds26060", 14)


def test_zero_scan_gap_granule_is_skipped_not_fatal(tmp_path):
    """The delivery carries 28 zero-scan granules as data gaps; 13 March 2026
    holds three. They must be skipped visibly while a corrupt file still
    refuses the day (the first fatal-input version conflated the two)."""
    import netCDF4
    m = _load("wsfm_adapter.py")
    gap = tmp_path / "1C.WSFM1.MWI.XCAL2026-V.20260313-S184147-E202311.009957.PARV817.nc"
    with netCDF4.Dataset(str(gap), "w") as f:
        f.createDimension("nscans", 0)
        f.createDimension("npixel", 3)
        f.createDimension("nchan", 11)
        f.createVariable("lat", "f4", ("nscans", "npixel"))
        f.createVariable("lon", "f4", ("nscans", "npixel"))
        f.createVariable("Predicted_SSMIS_Tas", "f4", ("nscans", "npixel", "nchan"))
    out = tmp_path / "grids"
    out.mkdir()
    m.process_day([str(gap)], str(out), 2026, 72)
    written = sorted(p.name for p in out.iterdir())
    assert written == ["as26072.wsfm", "ds26072.wsfm"], written
    corrupt = tmp_path / "1C.WSFM1.MWI.XCAL2026-V.20260313-S202312-E220436.009958.PARV817.nc"
    corrupt.write_bytes(b"not a netcdf file")
    out2 = tmp_path / "grids2"
    out2.mkdir()
    with pytest.raises(RuntimeError, match="1 unreadable granule"):
        m.process_day([str(gap), str(corrupt)], str(out2), 2026, 72)
    assert list(out2.iterdir()) == []
