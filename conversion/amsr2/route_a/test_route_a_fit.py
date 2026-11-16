"""Checks of route_a_fit.py on synthetic composites whose F-16 grids are the AMSR2 values shifted by a known amount per
surface class, encoded as the adapter encodes, with each side's missing cells drawn on their own."""
import datetime
import json

import numpy as np
import pytest

pytest.importorskip("h5py")             # the reader's one third-party package beyond numpy: without it these tests skip

import route_a_calendar as Q  # noqa: E402
import route_a_fit as F  # noqa: E402

A = F.A
DAYS = ["2022-04-05", "2022-04-15"]
SHIFT = {0: -3.0, 1: 2.0}                                   # ocean, land
LAT = (269 - np.arange(A.NROW)) / 3.0


def write_case(tmp_path, seed=0, days=DAYS, edit=None):
    """Composites and F-16 grids for the days. AMSR2 misses 20 percent of cells and F-16 a different 15 percent.
    edit(key, x, f16, tag) may change a pass's AMSR2 floats or F-16 bytes in place before they are written."""
    rng = np.random.default_rng(seed)
    tag = (rng.random((A.NROW, A.NCOL)) < 0.4).astype(np.int8)
    tag.tofile(tmp_path / "NLNDSEA.TAG")
    comp, grids = tmp_path / "comp", tmp_path / "grids"
    comp.mkdir()
    grids.mkdir()
    for s in days:
        d = datetime.date.fromisoformat(s)
        arrays = {}
        for key, prefix in F.PASSES.items():
            full = rng.uniform(150.0, 280.0, (A.NROW, A.NCOL, 7)).astype(np.float32)
            f16 = A._encode_ta((full + np.where(tag == 0, SHIFT[0], SHIFT[1])[..., None]).astype(np.float32))
            f16[rng.random((A.NROW, A.NCOL)) < 0.15] = A.FILL_BYTE
            x = full.copy()
            x[rng.random((A.NROW, A.NCOL)) < 0.2] = np.nan
            if edit is not None:
                edit(key, x, f16, tag)
            f16.tofile(grids / f"{prefix}{d.year % 100:02d}{d.timetuple().tm_yday:03d}.f16")
            arrays[key] = x
        np.savez_compressed(comp / f"{s}.npz", **arrays)
        json.dump({"day": s, "observed_minutes": 1440, "npz_sha256": F.R.sha256_file(comp / f"{s}.npz")},
                  open(comp / f"{s}.json", "w"))
    return comp, grids, tmp_path / "NLNDSEA.TAG"


def small_calendar(monkeypatch, days):
    monkeypatch.setattr(Q, "FIT_DAYS", list(days))


def test_the_maps_recover_the_shift_and_the_mapped_grid_reencodes_to_f16(tmp_path, monkeypatch):
    small_calendar(monkeypatch, DAYS)
    comp, grids, tag = write_case(tmp_path)
    rec = F.fit(str(comp), DAYS, str(grids), str(tag), str(tmp_path / "maps.json"))
    assert len(rec["maps"]) == 2 * 2 * 6 and all(k.startswith("04|") for k in rec["maps"])
    for name, mp in rec["maps"].items():
        shift = SHIFT[F.CLASSES[name.split("|")[2]]]
        diff = np.asarray(mp["f16"])[:, 1:-1] - np.asarray(mp["amsr2"])[:, 1:-1]
        # the two sides are independent samples of one distribution, so interior knots carry sampling noise as well
        # as the half-kelvin byte steps
        assert abs(np.median(diff) - shift) < 0.1 and np.abs(diff - shift).max() < 1.2
    F.apply(str(tmp_path / "maps.json"), str(comp / f"{DAYS[0]}.npz"), DAYS[0], str(tag), str(tmp_path / "out"))
    d = datetime.date.fromisoformat(DAYS[0])
    for prefix in ("as", "ds"):
        mine = np.fromfile(tmp_path / "out" / f"{prefix}{d.year % 100:02d}{d.timetuple().tm_yday:03d}.amsr2", np.uint8)
        ref = np.fromfile(grids / f"{prefix}{d.year % 100:02d}{d.timetuple().tm_yday:03d}.f16", np.uint8)
        ok = (mine != A.FILL_BYTE) & (ref != A.FILL_BYTE)
        diff = mine[ok].astype(int) - ref[ok].astype(int)
        assert abs(diff.mean()) < 0.05 and np.mean(np.abs(diff) <= 1) > 0.999
    assert (tmp_path / "maps.json.started").is_dir()
    npz, rec_path = comp / f"{DAYS[0]}.npz", comp / f"{DAYS[0]}.json"
    ident = F.R.verify_composite(str(npz), str(rec_path), DAYS[0])
    with pytest.raises(F.Refused, match="changed since the preflight"):
        F.apply(str(tmp_path / "maps.json"), str(npz), DAYS[0], str(tag), str(tmp_path / "o2"),
                expect_npz_sha256="0" * 64, expect_json_sha256=ident["json_sha256"])
    # the record alone edited after planning, the archive untouched: refused before anything is converted
    r = json.load(open(rec_path))
    r["observed_minutes"] = 1439
    json.dump(r, open(rec_path, "w"))
    with pytest.raises(F.Refused, match="changed since the preflight"):
        F.apply(str(tmp_path / "maps.json"), str(npz), DAYS[0], str(tag), str(tmp_path / "o3"),
                expect_npz_sha256=ident["npz_sha256"], expect_json_sha256=ident["json_sha256"])
    # a planned application keeps the minimum of observed minutes even with matching digests
    r["observed_minutes"] = 900
    json.dump(r, open(rec_path, "w"))
    short = F.R.verify_composite(str(npz), str(rec_path), DAYS[0])
    with pytest.raises(F.Refused, match="fewer than 990"):
        F.apply(str(tmp_path / "maps.json"), str(npz), DAYS[0], str(tag), str(tmp_path / "o4"),
                expect_npz_sha256=short["npz_sha256"], expect_json_sha256=short["json_sha256"])
    assert not any((tmp_path / o).exists() for o in ("o2", "o3", "o4"))


def _percentile(v, method):
    try:
        return np.percentile(v, F.KNOTS, axis=0, method=method).T
    except TypeError:                                           # numpy before 1.22
        return np.percentile(v, F.KNOTS, axis=0, interpolation=method).T


def test_each_side_is_counted_on_its_own_cells_and_the_knots_are_linear_percentiles(tmp_path):
    assert F.KNOTS == [0.1] + list(range(1, 100)) + [99.9] and len(F.KNOTS) == 101
    comp, grids, tag_path = write_case(tmp_path, seed=4, days=DAYS[:1])
    rec = F.fit_kernel(str(comp), DAYS[:1], str(grids), str(tag_path), str(tmp_path / "maps.json"))
    tag = np.fromfile(tag_path, np.int8).reshape(A.NROW, A.NCOL)
    z = np.load(comp / f"{DAYS[0]}.npz")
    f16 = np.fromfile(grids / "as22095.f16", np.uint8).reshape(A.NROW, A.NCOL, 7)
    cell = (tag == 0) & (np.broadcast_to(F.zones()[:, None], tag.shape) == 3)
    a = z["asc"][cell & np.isfinite(z["asc"]).all(axis=-1)].astype(np.float64)
    f = f16[cell & (f16 > A.FILL_BYTE).all(axis=-1)].astype(np.float64) + 70.5
    mp = rec["maps"]["04|asc|ocean|3"]
    assert (mp["n_amsr2"], mp["n_f16"]) == (len(a), len(f)) and len(a) != len(f)
    assert np.allclose(mp["amsr2"], _percentile(a, "linear"), atol=1e-9)
    assert np.allclose(mp["f16"], _percentile(f, "linear"), atol=1e-9)
    assert not np.allclose(mp["amsr2"], _percentile(a, "nearest"), atol=1e-6)


def _north(key, x, f16, tag, side, keep):
    """Leave only `keep` valid cells of each class north of 60N on one side of the ascending pass."""
    if key != "asc":
        return
    north = np.zeros(tag.shape, bool)
    north[LAT >= 60] = True
    for cv in (0, 1):
        cells = np.argwhere(north & (tag == cv))
        drop = cells[keep:]
        if side == "amsr2":
            x[drop[:, 0], drop[:, 1]] = np.nan
            x[cells[:keep, 0], cells[:keep, 1]] = np.linspace(160.0, 260.0, keep)[:, None]
        else:
            f16[drop[:, 0], drop[:, 1]] = A.FILL_BYTE
            f16[cells[:keep, 0], cells[:keep, 1]] = 150


@pytest.mark.parametrize("side,keep,refusal", [("amsr2", 100, "fewer than"), ("f16", 100, "fewer than"),
                                               ("amsr2", 101, None)])
def test_a_map_needs_as_many_samples_as_knots_on_each_side(tmp_path, side, keep, refusal):
    comp, grids, tag = write_case(tmp_path, seed=1, days=DAYS[:1],
                                  edit=lambda key, x, f16, t: _north(key, x, f16, t, side, keep))
    if refusal:
        with pytest.raises(F.Refused, match=refusal):
            F.fit_kernel(str(comp), DAYS[:1], str(grids), str(tag), str(tmp_path / "maps.json"))
    else:
        rec = F.fit_kernel(str(comp), DAYS[:1], str(grids), str(tag), str(tmp_path / "maps.json"))
        assert rec["maps"]["04|asc|ocean|5"]["n_amsr2"] == 101


def test_tied_amsr2_knots_refuse(tmp_path):
    def flat(key, x, f16, tag):
        if key == "asc":
            x[LAT >= 60] = np.where(np.isfinite(x[LAT >= 60]), 200.0, np.nan)
    comp, grids, tag = write_case(tmp_path, seed=2, days=DAYS[:1], edit=flat)
    with pytest.raises(F.Refused, match="strictly increase"):
        F.fit_kernel(str(comp), DAYS[:1], str(grids), str(tag), str(tmp_path / "maps.json"))


def test_the_entry_point_refuses_any_other_schedule_or_an_unverified_composite_before_reading(tmp_path,
                                                                                           monkeypatch):
    def no_input(path):
        raise AssertionError("the fit read its inputs")
    monkeypatch.setattr(F, "load_tag", no_input)
    cal = Q.FIT_DAYS
    assert len(cal) == 72
    for days in (cal[:-1] + ["2025-02-05"], cal[:-1] + [cal[0]], cal[:-1]):
        with pytest.raises(F.Refused, match="declared calendar"):
            F.fit(str(tmp_path), days, "g", "t", str(tmp_path / "maps.json"))
    comp = tmp_path / "comp"
    comp.mkdir()
    np.savez_compressed(comp / f"{cal[0]}.npz", asc=np.zeros((A.NROW, A.NCOL, 7), np.float32),
                        dsc=np.zeros((A.NROW, A.NCOL, 7), np.float32))
    json.dump({"day": cal[1], "observed_minutes": 1440, "npz_sha256": F.R.sha256_file(comp / f"{cal[0]}.npz")},
              open(comp / f"{cal[0]}.json", "w"))
    with pytest.raises(F.Refused, match="fitting composite"):
        F.fit(str(comp), cal, "g", "t", str(tmp_path / "maps.json"))
    assert not (tmp_path / "maps.json.started").exists()


def test_a_second_fit_refuses_before_reading_any_input(tmp_path, monkeypatch):
    def no_input(*a, **k):
        raise AssertionError("a second fit read its inputs")
    monkeypatch.setattr(F, "load_tag", no_input)
    monkeypatch.setattr(F.R, "verify_composite", no_input)
    (tmp_path / "maps.json").write_text("{}")
    with pytest.raises(FileExistsError):
        F.fit("c", Q.FIT_DAYS, "g", "t", str(tmp_path / "maps.json"))
    (tmp_path / "other.json.started").mkdir()                 # a marker left by an earlier attempt
    with pytest.raises(FileExistsError):
        F.fit("c", Q.FIT_DAYS, "g", "t", str(tmp_path / "other.json"))


def test_beyond_the_end_knots_the_end_offset_applies():
    a, f = np.array([100.0, 200.0, 300.0]), np.array([110.0, 205.0, 290.0])
    y = F.map_values(np.array([50.0, 150.0, 250.0, 320.0]), a, f)
    assert y.tolist() == [60.0, 157.5, 247.5, 310.0]


def test_zones_are_six_bands_by_center_latitude():
    z = F.zones()
    assert z[LAT >= 60].min() == 5 and z[(LAT >= 0) & (LAT < 30)].tolist() == [3] * 90 and z[-1] == 0
    assert np.bincount(z).tolist() == [90, 90, 90, 90, 90, 90]


def test_an_output_created_while_the_fit_computes_is_not_overwritten(tmp_path, monkeypatch):
    small_calendar(monkeypatch, DAYS)
    comp, grids, tag = write_case(tmp_path, seed=3)
    out = tmp_path / "maps.json"
    real_mkdir = F.os.mkdir

    def racing_mkdir(path, *args, **kwargs):            # another writer creates the output just after the marker
        real_mkdir(path, *args, **kwargs)
        if str(path).endswith(".started"):
            out.write_text("{}")
    monkeypatch.setattr(F.os, "mkdir", racing_mkdir)
    with pytest.raises(FileExistsError):
        F.fit(str(comp), DAYS, str(grids), str(tag), str(out))
    assert out.read_text() == "{}"
