"""One synthetic day through the whole conversion: the reader, the fixed maps and the processor.

The test needs no data file. It writes two small granules in CSU's layout from a literal schema kept here, apart from
the reader's own constants, one marching north and one marching south. It then runs the three commands a user would
run: the compositor, the map application with the committed maps and the land/sea tag, and the monthly processor with
its lookup tables. It requires the day's two grids in the SSMIS daily format, valid exactly where the granules put
footprints, and from them the ten monthly product files at 2.5 degrees.

The grids hold the mapped temperatures of made-up brightness temperatures, so the products' values mean nothing. The
test shows that the pieces fit together and that the tag, the maps and the tables are the ones the commands find.
"""
import datetime
import os
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")      # the reader's one third-party package beyond numpy: without it these tests skip

HERE = os.path.dirname(os.path.abspath(__file__))
MONTHLY = os.path.normpath(os.path.join(HERE, "..", "..", "..", "monthly"))
MAPS = next(p for p in (os.path.join(HERE, "route_a_maps.json"), os.path.join(HERE, "fit_record", "maps.json"))
            if os.path.isfile(p))
EPOCH = datetime.datetime(1987, 1, 1)
DAY = datetime.date(2022, 4, 5)
JDAY = DAY.timetuple().tm_yday
NROW, NCOL, NCHAN, FILL_BYTE = 540, 1080, 7, 32
# variable: (quality variable, value written at the footprints the conversion uses)
LOW = {"fcdr_tb18v": ("quality_tb18", 181.0), "fcdr_tb18h": ("quality_tb18", 122.0), "fcdr_tb23v": ("quality_tb23", 201.0),
       "fcdr_tb36v": ("quality_tb36", 211.0), "fcdr_tb36h": ("quality_tb36", 152.0)}
HIGH = {"fcdr_tb89v_ascan": 251.0, "fcdr_tb89h_ascan": 232.0}
PRODUCTS = ("PR1", "PF1", "PR2", "PF2", "LWP", "CFR", "WVP", "SNW", "ICE", "SSA")
NSCAN, NLO = 40, 12


def granule(path, start, lats, lon0):
    """A granule whose scans lie at the given latitudes, with NLO low-resolution footprints a degree apart."""
    with h5py.File(path, "w") as f:
        for var, (_, value) in LOW.items():
            f[var] = np.full((NSCAN, NLO), value, np.float32)
        for var, value in HIGH.items():
            f[var] = np.full((NSCAN, 2 * NLO), value, np.float32)
        for q, n in (("quality_tb18", NLO), ("quality_tb23", NLO), ("quality_tb36", NLO), ("quality_tb89_ascan", 2 * NLO)):
            f[q] = np.zeros((NSCAN, n, 2), np.int8)
        f["lat_tb36"] = np.array([[lat] * NLO for lat in lats], np.float32)
        f["lon_tb36"] = np.array([[lon0 + k for k in range(NLO)] for _ in range(NSCAN)], np.float32)
        f["scan_time"] = np.array([(start - EPOCH).total_seconds() + 1.5 * s for s in range(NSCAN)])
    return str(path)


def run(argv, cwd):
    p = subprocess.run([sys.executable] + argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert p.returncode == 0, p.stdout
    return p.stdout


def test_a_synthetic_day_becomes_two_daily_grids_and_ten_monthly_products(tmp_path):
    north = granule(tmp_path / "CSU_AMSR2_ICDR_V02R00_GCOMW1_D20220405_S0107_E0108_R050000.nc",
                    datetime.datetime(2022, 4, 5, 1, 7, 0), [-20.0 + 0.5 * s for s in range(NSCAN)], 100.0)
    south = granule(tmp_path / "CSU_AMSR2_ICDR_V02R00_GCOMW1_D20220405_S1307_E1308_R050000.nc",
                    datetime.datetime(2022, 4, 5, 13, 7, 0), [10.0 - 0.5 * s for s in range(NSCAN)], -110.0)
    composite, grids, out = str(tmp_path / "day.npz"), str(tmp_path / "grids"), str(tmp_path / "out")

    said = run(["route_a.py", "composite", "--day", DAY.isoformat(), "--files", north, south, "--out", composite], HERE)
    assert "2 files, %d scans" % (2 * NSCAN) in said
    run(["route_a_fit.py", "apply", "--maps", MAPS, "--composite", composite, "--day", DAY.isoformat(),
         "--tag", os.path.join(MONTHLY, "NLNDSEA.TAG"), "--out-dir", grids], HERE)

    names = sorted(os.listdir(grids))
    assert names == ["as22%03d.amsr2" % JDAY, "ds22%03d.amsr2" % JDAY]
    valid = {}
    for name in names:
        data = np.fromfile(os.path.join(grids, name), dtype=np.uint8)
        assert data.size == NROW * NCOL * NCHAN
        data = data.reshape(NROW, NCOL, NCHAN)
        some = data > FILL_BYTE
        assert (some.all(axis=2) == some.any(axis=2)).all()       # a cell is valid in all seven channels or in none
        assert (data[~some] == FILL_BYTE).all()
        valid[name[:2]] = some.all(axis=2)
    # The northward granule is the ascending pass and the southward one the descending pass, each in its own box.
    # Rows run from 89.667 N southward and columns from 180 W eastward, a third of a degree apart.
    lat = 89.667 - np.arange(NROW) / 3.0
    lon = -180.0 + np.arange(NCOL) / 3.0
    for key, lats, lons in (("as", (-20.0, -0.5), (100.0, 111.0)), ("ds", (-9.5, 10.0), (-110.0, -99.0))):
        rows, cols = np.nonzero(valid[key])
        assert rows.size > NSCAN * NLO / 4
        assert lats[0] - 0.5 <= lat[rows].min() and lat[rows].max() <= lats[1] + 0.5
        assert lons[0] - 0.5 <= lon[cols].min() and lon[cols].max() <= lons[1] + 0.5

    run(["climalg_ssmis.py", "amsr2", str(DAY.year), str(JDAY), str(JDAY), "--res", "2.5",
         "--inpath", grids + "/", "--outdir", out], MONTHLY)
    made = sorted(os.listdir(os.path.join(out, "amsr2-2.5")))
    assert made == sorted("%s22-04-amsr2-2.5" % p for p in PRODUCTS)
    assert all(os.path.getsize(os.path.join(out, "amsr2-2.5", n)) == 72 * 144 * 4 for n in made)
    ssa = np.fromfile(os.path.join(out, "amsr2-2.5", "SSA22-04-amsr2-2.5"), dtype=np.float32)
    assert np.isfinite(ssa).all() and (ssa > 0).any() and (ssa > 0).sum() < ssa.size / 20     # only the two boxes
