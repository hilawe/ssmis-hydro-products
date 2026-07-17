"""1.0-degree anomaly imagery contracts (SNW/ICE/WVP).

The monthly anomaly panels compute a climatology from the 1.0-degree per-year
binaries (ncdc-bin for the morning primary, {sat}-1.0 otherwise) and difference
the current month against it. The things in that path that are easy to break
silently are pinned here:

  1. the byte layout of a 12-month 1.0-degree yearly file, read one month at a
     time (offset, orientation, fill sentinel, all-fill-month rejection),
  2. the per-product anomaly scale (SNW is a 0-1 fraction scaled x100 to %
     points; ICE is already percent; WVP is mm) - applying the snow factor to
     ICE or WVP would inflate the anomaly 100-fold,
  3. the SYMMETRIC integrity gating: the MIN_VALID_FRAC floor admits or rejects
     a month by the same rule whether it is a baseline year or the current
     field, MIN_BASELINE_YEARS declines a degenerate climatology, and
     BAD_1DEG_YEARS rejects a documented-bad year on both sides (baseline and
     current), reporting the exclusion only when it removed real data.

The module lives under monthly/grads and is imported as a script elsewhere, so
it is loaded here by explicit file path to avoid picking up any same-named file
on sys.path. cartopy/scipy are optional in the module, so a headless host still
imports it.
"""
import importlib.util
import os
import struct

import pytest

np = pytest.importorskip("numpy")
# run_mon_image hard-requires matplotlib at import (it is the renderer), so a
# host without it must skip this module at collection rather than error, per
# the suite convention of guarding heavy dependencies (see conftest docstring).
pytest.importorskip("matplotlib")

from conftest import MONTHLY_DIR

_RMI_PATH = os.path.join(MONTHLY_DIR, "grads", "run_mon_image.py")


def _load_rmi():
    # monthly/ is already on sys.path (conftest) so the module's own
    # `from product_version import PRODUCT_VERSION` resolves. Load by path so the
    # test targets THIS file, not any run_mon_image.py that shadows it on a host.
    spec = importlib.util.spec_from_file_location("_rmi_under_test", _RMI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rmi = _load_rmi()

FILL = np.float32(-999.99)


def _grid(value=0.0):
    return np.full((rmi.N_LAT_1, rmi.N_LON_1), np.float32(value), np.float32)


def _write_year(path, months):
    """Write a 12-month 1.0-degree yearly binary. `months` maps 0-based month
    index to a (180, 360) float32 array; absent months are written all-fill."""
    with open(path, "wb") as f:
        for m in range(12):
            g = months.get(m)
            if g is None:
                g = _grid(FILL)
            f.write(np.asarray(g, dtype=np.float32).tobytes())


def _seed_baseline(ncdc_dir, prod, years_values, mon0):
    """Write one yearly file per (year -> constant value) for month mon0."""
    for yr, val in years_values.items():
        _write_year(os.path.join(ncdc_dir, f"{prod}.{yr % 100:02d}"),
                    {mon0: _grid(val)})


def test_read_1deg_month_offset_and_fill(tmp_path):
    # Distinct constant per month, month 3 all fill. Confirms the reader seeks to
    # the right month, applies the shape, maps <= -999 to NaN, and returns None
    # for an all-fill month (not an all-NaN grid).
    p = str(tmp_path / "SNW.99")
    months = {m: _grid(m + 1) for m in range(12) if m != 3}
    _write_year(p, months)
    for m in range(12):
        g = rmi._read_1deg_month(p, m)
        if m == 3:
            assert g is None
        else:
            assert g.shape == (rmi.N_LAT_1, rmi.N_LON_1)
            assert np.allclose(g, m + 1)


def test_read_1deg_month_orientation(tmp_path):
    # A single marked cell at the south-west origin: row 0 = 89.5S, col 0 = 0.5E,
    # lon fastest. A transposed, north-first, or lon/lat-swapped reader would put
    # the marker elsewhere. The second marker (row 2, col 5) pins lon-fastest
    # ordering: its flat offset must be 2*360 + 5.
    flat = np.full(rmi.N_LAT_1 * rmi.N_LON_1, np.float32(1.0), np.float32)
    flat[0] = 77.0                      # (lat 89.5S, lon 0.5E)
    flat[2 * rmi.N_LON_1 + 5] = 88.0    # (lat 87.5S, lon 5.5E)
    p = str(tmp_path / "SNW.98")
    with open(p, "wb") as f:
        f.write(flat.tobytes() * 12)
    g = rmi._read_1deg_month(p, 7)
    assert g[0, 0] == np.float32(77.0)
    assert g[2, 5] == np.float32(88.0)
    assert g[5, 2] == np.float32(1.0)   # the transpose would land here


def test_read_1deg_month_missing_and_short(tmp_path):
    assert rmi._read_1deg_month(str(tmp_path / "nope.00"), 0) is None
    short = str(tmp_path / "SNW.00")
    with open(short, "wb") as f:
        f.write(struct.pack("<f", 1.0) * 10)  # far shorter than one month
    assert rmi._read_1deg_month(short, 5) is None


def test_month_is_valid_floor():
    # Exactly at the MIN_VALID_FRAC boundary on each side, plus None.
    cells = rmi.N_LAT_1 * rmi.N_LON_1
    need = int(np.ceil(cells * rmi.MIN_VALID_FRAC))
    g = _grid(FILL)
    g_flat = g.ravel()
    g_flat[:need - 1] = 5.0
    g2 = g_flat.reshape(g.shape).copy()
    g2[g2 <= -999.0] = np.nan
    assert not rmi._month_is_valid(g2)          # one below the floor
    g_flat[need - 1] = 5.0
    g3 = g_flat.reshape(g.shape).copy()
    g3[g3 <= -999.0] = np.nan
    assert rmi._month_is_valid(g3)              # at the floor
    assert not rmi._month_is_valid(None)


def test_climatology_averages_and_skips_sparse(tmp_path):
    # MIN_BASELINE_YEARS good years (values 10..) plus one all-fill year and one
    # too-sparse year for May; the sparse/fill years must be skipped, and the
    # mean must come out over the good years only.
    root = str(tmp_path)
    ncdc = os.path.join(root, "ncdc-bin")
    os.makedirs(ncdc)
    may = 4
    n = rmi.MIN_BASELINE_YEARS
    vals = {2000 + i: 10.0 + i for i in range(n)}
    _seed_baseline(ncdc, "ICE", vals, may)
    _write_year(os.path.join(ncdc, f"ICE.{(2000 + n) % 100:02d}"), {})  # all fill
    sparse = _grid(FILL)
    sparse[0, 0] = 50.0                                       # 1 cell, below floor
    _write_year(os.path.join(ncdc, f"ICE.{(2001 + n) % 100:02d}"), {may: sparse})
    clim, ny = rmi._compute_climatology_1deg(
        root, rmi.NCDC_BIN_SAT, "ICE", may,
        baseline_start=2000, baseline_end=2001 + n)
    assert ny == n
    assert np.allclose(clim, np.mean(list(vals.values())))


def test_climatology_declines_below_min_years(tmp_path):
    # One fewer valid year than MIN_BASELINE_YEARS -> (None, 0), not a
    # degenerate "normal" labeled with a tiny year count.
    root = str(tmp_path)
    ncdc = os.path.join(root, "ncdc-bin")
    os.makedirs(ncdc)
    may = 4
    _seed_baseline(ncdc, "ICE",
                   {2000 + i: 20.0 for i in range(rmi.MIN_BASELINE_YEARS - 1)},
                   may)
    clim, ny = rmi._compute_climatology_1deg(
        root, rmi.NCDC_BIN_SAT, "ICE", may,
        baseline_start=2000, baseline_end=2010)
    assert clim is None and ny == 0


def test_bad_year_excluded_only_when_present(tmp_path, capsys):
    # WVP 1998 is in BAD_1DEG_YEARS. When its file holds real data it must be
    # dropped from the mean AND reported; when the record never spans it, the
    # log must not claim an exclusion.
    assert 1998 in rmi.BAD_1DEG_YEARS.get("WVP", ())
    root = str(tmp_path)
    ncdc = os.path.join(root, "ncdc-bin")
    os.makedirs(ncdc)
    may = 4
    n = rmi.MIN_BASELINE_YEARS
    good_years = {1992 + i: 25.0 for i in range(n)}           # 1992..1997
    _seed_baseline(ncdc, "WVP", good_years, may)
    _write_year(os.path.join(ncdc, "WVP.98"), {may: _grid(9.0)})   # corrupt
    clim, ny = rmi._compute_climatology_1deg(
        root, rmi.NCDC_BIN_SAT, "WVP", may,
        baseline_start=1992, baseline_end=1998)
    out = capsys.readouterr().out
    assert ny == n
    assert np.allclose(clim, 25.0)          # not dragged toward 9.0
    assert "excluding [1998]" in out

    # Same window but 1998 absent on disk: silent about the exclusion.
    os.remove(os.path.join(ncdc, "WVP.98"))
    clim2, ny2 = rmi._compute_climatology_1deg(
        root, rmi.NCDC_BIN_SAT, "WVP", may,
        baseline_start=1992, baseline_end=1998)
    out2 = capsys.readouterr().out
    assert ny2 == n
    assert "excluding" not in out2


def test_current_field_rejects_bad_year_and_sparse(tmp_path, capsys):
    # read_current_1deg must apply the same two gates to the current field:
    # a documented-bad product-year is refused outright, and a too-sparse month
    # (below MIN_VALID_FRAC) is refused, while a healthy month passes.
    root = str(tmp_path)
    sat = rmi.NCDC_BIN_SAT
    d = os.path.join(root, f"{sat}-1.0")
    os.makedirs(d)
    may0 = 4

    # documented-bad year: refused before any file I/O matters
    assert rmi.read_current_1deg(root, sat, "WVP", "98", "05") is None
    assert "documented-bad" in capsys.readouterr().out

    # too-sparse month: written but under the floor
    sparse = _grid(FILL)
    sparse[10, 10] = 30.0
    _write_year(os.path.join(d, f"WVP.26-{sat}-1.0"), {may0: sparse})
    assert rmi.read_current_1deg(root, sat, "WVP", "26", "05") is None

    # healthy month passes
    _write_year(os.path.join(d, f"WVP.26-{sat}-1.0"), {may0: _grid(30.0)})
    g = rmi.read_current_1deg(root, sat, "WVP", "26", "05")
    assert g is not None and np.allclose(g, 30.0)


def test_anom_scale_distinguishes_products():
    # The scale table is the one guard against differencing SNW (fraction) and
    # ICE/WVP (already in display units) with the same factor.
    assert rmi.ANOM_SCALE["SNW"] == 100.0
    assert rmi.ANOM_SCALE["ICE"] == 1.0
    assert rmi.ANOM_SCALE["WVP"] == 1.0


def test_ice_uses_ssmis_baseline_others_wmo():
    # ICE must not share the WMO window (sensor step); SNW and WVP must keep it.
    assert rmi.ANOM_BASELINE["ICE"] == (2009, 2020)
    assert rmi.ANOM_BASELINE["SNW"] == (1991, 2020)
    assert rmi.ANOM_BASELINE["WVP"] == (1991, 2020)


def test_baseline_desc_reports_passed_window():
    # The label must describe the window actually passed, not re-read a default,
    # and must tag 1991-2020 as WMO and anything else as SSMIS era.
    assert "1991-2020" in rmi._baseline_desc((1991, 2020), 29)
    assert "WMO" in rmi._baseline_desc((1991, 2020), 29)
    assert "29 yr" in rmi._baseline_desc((1991, 2020), 29)
    assert "2009-2020" in rmi._baseline_desc((2009, 2020), 12)
    assert "SSMIS era" in rmi._baseline_desc((2009, 2020), 12)


def test_snow_anomaly_gates_sparse_current(tmp_path, capsys):
    # The snow 1.0-degree anomaly branch must apply the same MIN_VALID_FRAC
    # floor as the baseline (multi-model review find): a sparse current month
    # must NOT be differenced against the gated WMO baseline. We call the
    # branch condition directly: _month_is_valid on a sparse grid is False, so
    # the 1.0-degree anomaly branch is skipped.
    sparse = _grid(FILL)
    sparse[0, :10] = 0.5
    sparse[sparse <= -999.0] = np.nan
    assert not rmi._month_is_valid(sparse)
    healthy = _grid(0.5)
    assert rmi._month_is_valid(healthy)


def test_snow_mixed_resolution_fallback_policy(tmp_path, monkeypatch):
    # PINS A DELIBERATE POLICY (do not "fix" without an owner decision): when
    # the 1.0-degree current snow field renders but its 1.0-degree baseline
    # declines (fewer than MIN_BASELINE_YEARS, e.g. a future satellite's first
    # years), gen_snw falls back to the 2.5-degree anomaly beside the
    # 1.0-degree field panels - a mixed-resolution figure with an honest
    # non-WMO label - rather than the placeholder ICE uses. plot_polar_4panel
    # takes separate field/anomaly coordinate arrays for exactly this case.
    root = str(tmp_path)
    sat = rmi.NCDC_BIN_SAT
    os.makedirs(os.path.join(root, f"{sat}-1.0"))
    os.makedirs(os.path.join(root, "ncdc-bin"))
    os.makedirs(os.path.join(root, f"{sat}-2.5"))
    may0 = 4
    # healthy 1.0-degree current month, but too few baseline years (2 < 6)
    _write_year(os.path.join(root, f"{sat}-1.0", f"SNW.26-{sat}-1.0"),
                {may0: _grid(0.5)})
    _seed_baseline(os.path.join(root, "ncdc-bin"), "SNW",
                   {2010: 0.4, 2011: 0.6}, may0)
    captured = {}
    def fake_plot(data, anom, header_title, cmap, levels, anom_cmap,
                  anom_levels, cbar_label, anom_label, outpath, **kw):
        captured.update(kw, anom=anom, data=data, anom_label=anom_label)
        return None
    monkeypatch.setattr(rmi, "plot_polar_4panel", fake_plot)
    rmi.gen_snw(sat, "26", "05", str(tmp_path), root)
    # field stayed 1.0-degree
    assert captured["snw_lons"] is rmi.LONS_1
    assert captured["data"].shape == (rmi.N_LAT_1, rmi.N_LON_1)
    # the 1.0-degree anomaly declined (baseline below MIN_BASELINE_YEARS) and
    # no 2.5-degree data exists in this sandbox, so the anomaly is None with
    # the neutral label - crucially NOT a WMO-labeled 1.0-degree anomaly
    assert captured["anom"] is None
    assert "WMO" not in captured["anom_label"]
