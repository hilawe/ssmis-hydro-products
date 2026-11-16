"""Checks of route_a.py on synthetic granules built from a literal schema written here, independent of the reader's own
constants."""
import datetime

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")      # the reader's one third-party package beyond numpy: without it these tests skip

import route_a as R  # noqa: E402

A = R.A
NAME = "CSU_AMSR2_ICDR_V02R00_GCOMW1_D20220405_S0107_E0246_R050000.nc"
EPOCH = datetime.datetime(1987, 1, 1)
# variable: (quality variable, polarization index in its last axis, value written at the footprints the conversion uses)
LOW = {"fcdr_tb18v": ("quality_tb18", 0, 181.0), "fcdr_tb18h": ("quality_tb18", 1, 182.0),
       "fcdr_tb23v": ("quality_tb23", 0, 231.0), "fcdr_tb36v": ("quality_tb36", 0, 301.0),
       "fcdr_tb36h": ("quality_tb36", 1, 302.0)}
HIGH = {"fcdr_tb89v_ascan": ("quality_tb89_ascan", 0, 291.0), "fcdr_tb89h_ascan": ("quality_tb89_ascan", 1, 292.0)}
EXPECTED = [181.0, 182.0, 231.0, 301.0, 302.0, 291.0, 292.0]      # 19V 19H 22V 37V 37H 91V 91H
UNUSED_HIGH = {"fcdr_tb89v_ascan": 191.0, "fcdr_tb89h_ascan": 192.0}  # the even A-scan footprints, not used


def make_granule(path, nscan=6, nlo=4, values=None, quality=None, start=None, lats=None):
    """A granule in CSU's layout: values maps a variable to its (nscan, n) array, quality maps a quality variable to its
    array, lats gives each scan's latitude (default: marching north one degree a scan)."""
    start = start or datetime.datetime(2022, 4, 5, 1, 7, 0)
    values, quality = values or {}, quality or {}
    lats = lats if lats is not None else [-10.0 + s for s in range(nscan)]
    with h5py.File(path, "w") as f:
        for var, (_, _, v) in LOW.items():
            f[var] = values.get(var, np.full((nscan, nlo), v, np.float32))
        for var, (_, _, v) in HIGH.items():
            hi = np.empty((nscan, 2 * nlo), np.float32)
            hi[:, 1::2], hi[:, 0::2] = v, UNUSED_HIGH[var]
            f[var] = values.get(var, hi)
        for q, n in (("quality_tb18", nlo), ("quality_tb23", nlo), ("quality_tb36", nlo), ("quality_tb89_ascan", 2 * nlo)):
            f[q] = quality.get(q, np.zeros((nscan, n, 2), np.int8))
        f["lat_tb36"] = np.array([[lat] * nlo for lat in lats], np.float32)
        f["lon_tb36"] = np.array([[100.0 + k for k in range(nlo)] for _ in range(nscan)], np.float32)
        f["scan_time"] = np.array([(start - EPOCH).total_seconds() + 1.5 * s for s in range(nscan)])
    return path


def test_each_channel_comes_from_its_declared_variable_and_footprint(tmp_path):
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME))
    assert sw["ta"].shape == (6, 4, 7)
    for c, v in enumerate(EXPECTED):
        assert np.all(sw["ta"][..., c] == v), c


@pytest.mark.parametrize("qvar,pol,width,expect_bad", [
    ("quality_tb18", 0, 4, True), ("quality_tb18", 1, 4, True), ("quality_tb23", 0, 4, True),
    ("quality_tb23", 1, 4, False),                               # 23.8 H is not used
    ("quality_tb36", 0, 4, True), ("quality_tb36", 1, 4, True),
    ("quality_tb89_ascan", 0, 8, True), ("quality_tb89_ascan", 1, 8, True)])
def test_each_used_quality_flag_and_no_unused_one_invalidates_the_footprint(tmp_path, qvar, pol, width, expect_bad):
    q = np.zeros((6, width, 2), np.int8)
    q[2, 3 if width == 8 else 1, pol] = 1                      # A-scan footprint 3 = 2*1+1 pairs with footprint 1
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME, quality={qvar: q}))
    bad = sorted(zip(*np.nonzero(~np.isfinite(sw["ta"]).all(axis=-1))))
    assert bad == ([(2, 1)] if expect_bad else [])
    even = np.zeros((6, 8, 2), np.int8)
    even[2, 2, pol] = 1                                        # an even A-scan footprint: not used
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME, quality={"quality_tb89_ascan": even}))
    assert np.isfinite(sw["ta"]).all()


@pytest.mark.parametrize("var", list(LOW) + list(HIGH))
@pytest.mark.parametrize("value,valid", [(50.0, True), (350.0, True), (49.99, False), (350.01, False),
                                         (np.nan, False), (-9999.9, False)])
def test_the_valid_range_is_inclusive_and_fill_and_nonfinite_invalidate(tmp_path, var, value, valid):
    if var in LOW:
        a = np.full((6, 4), LOW[var][2], np.float32)
        a[3, 2] = value
    else:
        a = np.empty((6, 8), np.float32)
        a[:, 1::2], a[:, 0::2] = HIGH[var][2], UNUSED_HIGH[var]
        a[3, 5] = value                                        # odd footprint 5 pairs with footprint 2
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME, values={var: a}))
    assert bool(np.isfinite(sw["ta"][3, 2]).all()) is valid
    assert np.isfinite(np.delete(sw["ta"].reshape(-1, 7), 3 * 4 + 2, axis=0)).all()


def test_the_pass_follows_the_latitude_trend_both_ways(tmp_path):
    lats = [-10.0, -9.0, -8.0, -9.0, -10.0, -11.0]            # north for two steps, then south
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME, lats=lats))
    assert sw["ascending"].tolist() == [True, True, True, False, False, False]


def test_scans_are_routed_to_their_utc_day(tmp_path):
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME, start=datetime.datetime(2022, 4, 5, 23, 59, 56)))
    assert sw["scan_day"].tolist() == [2022095] * 3 + [2022096] * 3


def test_the_day_keeps_only_its_scans_and_merges_granules_by_last_write(tmp_path):
    # a preceding-day granule crossing midnight (its off-day scans at 40 to 42N, its on-day scans at 43 to 45N), then two
    # granules of the day: the second writes 250 K over the first, and its one invalid footprint blanks the cell it
    # touches; a cell no granule touches stays empty
    low = lambda v: {k: np.full((6, 4), v, np.float32) for k in LOW}
    prev = make_granule(tmp_path / NAME.replace("D20220405_S0107_E0246", "D20220404_S2359_E0130"), values=low(111.0),
                        start=datetime.datetime(2022, 4, 4, 23, 59, 56), lats=[40.0, 41, 42, 43, 44, 45])
    first = make_granule(tmp_path / NAME, start=datetime.datetime(2022, 4, 5, 1, 7, 0))
    values = low(250.0)
    values["fcdr_tb18v"][1, 1] = -9999.9
    second = make_granule(tmp_path / NAME.replace("S0107_E0246", "S0246_E0425"), values=values,
                          start=datetime.datetime(2022, 4, 5, 2, 46, 0), lats=[-10.0, -9, -8, -7, 20, 21])
    asc, dsc, info = R.composite_day([str(prev), str(first), str(second)], 2022, 95)
    assert [f["scans_on_day"] for f in info["files"]] == [3, 6, 6] and info["observed_minutes"] == 3

    def cell(lat, lon):
        return asc[int(A._lat_to_row(np.array([lat]))[0]), int(A._lon_to_col(np.array([lon]))[0])]
    assert np.isnan(cell(41.0, 100.0)).all()                    # the preceding day's own scans are not the day's
    assert cell(44.0, 100.0)[0] == 111.0                         # its scans after midnight are
    assert np.isnan(cell(-9.0, 101.0)).all()                     # blanked by the later invalid footprint
    assert cell(-8.0, 101.0).tolist() == [250.0] * 5 + [291.0, 292.0]   # overwritten by the later granule
    assert cell(-6.0, 100.0).tolist() == EXPECTED                # the first granule only
    assert np.isnan(cell(0.0, 100.0)).all() and np.isnan(dsc).all()


def test_the_hemisphere_guard_counts_observed_time_not_files(tmp_path, monkeypatch):
    # twelve granules holding the same minute, a preceding-day granule with no scan on the day, and a one-scan granule
    # that is skipped: one minute observed, so with an orbit length of one minute balance is not judged. Counting the
    # fourteen files would judge and refuse it. Ten granules of ten distinct minutes are judged and refused.
    monkeypatch.setattr(A, "MIN_CELLS_FOR_BALANCE", 1)
    monkeypatch.setattr(R, "ORBIT_MINUTES", 1.0)
    files = [make_granule(tmp_path / NAME.replace("R050000", f"R0501{k:02d}")) for k in range(12)]
    files.append(make_granule(tmp_path / NAME.replace("D20220405", "D20220404"),
                              start=datetime.datetime(2022, 4, 4, 10, 0, 0)))
    files.append(make_granule(tmp_path / NAME.replace("R050000", "R050200"), nscan=1))
    asc, dsc, info = R.composite_day([str(f) for f in files], 2022, 95)
    assert info["observed_minutes"] == 1 and len(info["gaps"]) == 1
    ten = [make_granule(tmp_path / NAME.replace("R050000", f"R0503{k:02d}"),
                        start=datetime.datetime(2022, 4, 5, 3, k, 0)) for k in range(10)]
    with pytest.raises(ValueError, match="hemisphere balance"):
        R.composite_day([str(f) for f in ten], 2022, 95)


def test_the_float_composite_encodes_to_exactly_the_adapter_byte_grid(tmp_path):
    sw = R.read_amsr2_granule(make_granule(tmp_path / NAME, lats=[-10.0, -9, -8, -9, -10, -11]))
    for flag in (True, False):
        byte_grid, touched_b = A.grid_swath_to_binary(sw, ascending_only=flag)
        float_grid, touched_f = R.grid_float(sw, flag)
        assert touched_f.any() and np.array_equal(touched_b, touched_f)
        assert np.array_equal(A._encode_ta(float_grid), byte_grid)


def test_names_and_midnight_crossing():
    assert R.parse_name(NAME) == {"date": "20220405", "start": "0107", "end": "0246", "rev": 50000}
    assert R.crosses_midnight(NAME.replace("S0107_E0246", "S2330_E0109"))
    with pytest.raises(R.Refused):
        R.parse_name("CSU_SSMIS_ICDR_V02R00_F16_D20220405_S0107_E0246_R050000.nc")
