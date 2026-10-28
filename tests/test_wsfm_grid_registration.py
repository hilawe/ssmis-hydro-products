"""WSF-M adapters must place samples on the SAME 1/3-degree grid as the
operational F-17 decoder.

Until 2026-07-29 both adapters used floor((90-lat)/RES) and floor((lon+180)/RES)
while the operational decoder
(third_degree/src/ssmis_tdr_decoder.pro:229-233) uses nearest-node,
ROUND(3*(90+lat)) and ROUND(3*(180+lon)), with the formatter reversing latitude
into north-first storage. That is a half-cell offset: the longitude formulas
disagreed on 50.2 percent of positions and on a real delivered orbit row AND
column agreed for only 25.0 percent of 1.53 million pixels, so every
WSF-M-versus-F-17 comparison was pairing cells that were not the same cells.

These tests reimplement the IDL formulas independently and pin the adapters to
them, including the half-cell boundaries, the dateline fold, and both poles.
"""
import importlib.util
import math
import os

import pytest

np = pytest.importorskip("numpy")

from conftest import MONTHLY_DIR

NROW, NCOL, SCALE = 540, 1080, 3.0


def _load(name):
    path = os.path.join(MONTHLY_DIR, name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def idl_round(x):
    """IDL ROUND, in pure Python: nearest integer, halves AWAY FROM ZERO.

    Deliberately does NOT call np.rint. np.rint rounds halves to EVEN, so using
    it here would make the reference a restatement of the implementation rather
    than an independent check, and the two genuinely differ on exact halves
    (np.rint(268.5) = 268, IDL ROUND(268.5) = 269).
    """
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def idl_col(lon):
    """ii = ROUND(SCALE*(180+lon)); out of range folds to 0 (IDL wrap)."""
    ii = idl_round(SCALE * (180.0 + lon))
    return 0 if (ii >= NCOL or ii < 0) else ii


def idl_row_north(lat):
    """jj = ROUND(SCALE*(90+lat)) clipped, then reversed to north-first."""
    jj = idl_round(SCALE * (90.0 + lat))
    jj = min(max(jj, 0), NROW - 1)
    return (NROW - 1) - jj


# The IMG+ENV adapter reads an evaluation delivery and is not part of every tree. Where it is absent its cases are
# reported as skipped with this reason, never silently dropped.
IMGENV = "wsfm_adapter_imgenv.py"
SKIP_IMGENV = pytest.mark.skipif(not os.path.isfile(os.path.join(MONTHLY_DIR, IMGENV)),
                                 reason="the IMG+ENV adapter is not part of this tree")
ADAPTERS = ["wsfm_adapter.py", pytest.param(IMGENV, marks=SKIP_IMGENV)]

# Values chosen to sit on BOTH sides of half-cell boundaries, where floor and
# nearest-node disagree, plus the wrap and pole edges.
LONS = [-180.0, -179.9, -179.834, -179.833, -179.7, -90.0, -0.2, 0.0, 0.2,
        0.166, 0.167, 90.0, 179.7, 179.833, 179.9, 180.0,
        # Exact half nodes, where IDL ROUND and numpy rint disagree.
        -178.5, -179.5, -0.5, 0.5, 45.5, -45.5, 178.5]
LATS = [90.0, 89.9, 89.834, 89.833, 89.667, 45.0, 0.166, 0.0, -0.166,
        -45.0, -89.667, -89.833, -89.9, -90.0,
        # Exact half nodes, as above.
        -0.5, 0.5, 45.5, -45.5, -89.5, 89.5, 30.5]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_longitude_matches_operational_decoder(adapter):
    m = _load(adapter)
    got = m._lon_to_col(np.array(LONS, dtype=np.float64))
    want = [idl_col(v) for v in LONS]
    assert list(got) == want, f"{adapter} column mapping differs from the IDL decoder"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_latitude_matches_operational_decoder(adapter):
    m = _load(adapter)
    got = m._lat_to_row(np.array(LATS, dtype=np.float64))
    want = [idl_row_north(v) for v in LATS]
    assert list(got) == want, f"{adapter} row mapping differs from the IDL decoder"


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_indices_in_range(adapter):
    m = _load(adapter)
    lons = np.linspace(-180.0, 180.0, 4001)
    lats = np.linspace(-90.0, 90.0, 4001)
    c = m._lon_to_col(lons)
    r = m._lat_to_row(lats)
    assert c.min() >= 0 and c.max() < NCOL
    assert r.min() >= 0 and r.max() < NROW


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_poles_and_orientation(adapter):
    """Row 0 is the NORTH pole row, matching the north-first storage order."""
    m = _load(adapter)
    assert int(m._lat_to_row(np.array([90.0]))[0]) == 0
    assert int(m._lat_to_row(np.array([-90.0]))[0]) == NROW - 1
    assert int(m._lat_to_row(np.array([89.9]))[0]) < int(m._lat_to_row(np.array([-89.9]))[0])


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_exact_half_nodes_round_away_from_zero(adapter):
    """Halves must round AWAY FROM ZERO (IDL ROUND), not to even (numpy rint).

    Latitude -0.5 gives 3*(90-0.5) = 268.5 and longitude -178.5 gives
    3*(180-178.5) = 4.5. Under rint those land on 268 and 4; under IDL they land
    on 269 and 5, one cell away. Fixed 2026-07-29 and measured at
    about 2 rows and 6 columns per million real footprints, so it moved no
    published number, but it is the same class as the floor defect.
    """
    m = _load(adapter)
    assert int(m._lat_to_row(np.array([-0.5]))[0]) == idl_row_north(-0.5)
    assert int(m._lat_to_row(np.array([-0.5]))[0]) == (NROW - 1) - 269
    assert int(m._lon_to_col(np.array([-178.5]))[0]) == idl_col(-178.5)
    assert int(m._lon_to_col(np.array([-178.5]))[0]) == 5


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_floor_regression_is_gone(adapter):
    """The specific case that exposed the defect: longitude 0.20 degrees mapped
    to column 540 under floor but 541 under the operational formula."""
    m = _load(adapter)
    assert int(m._lon_to_col(np.array([0.20]))[0]) == 541
