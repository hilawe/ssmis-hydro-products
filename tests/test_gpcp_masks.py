"""gpcp_processing PR2 coastal masks: the IDL index-convention geometry.

These masks encode the subtle Fortran-column-major to C-row-major latitude index
mapping that the module docstring warns corrupts roughly half the grid if done
wrong. They are fixed geometry (independent of any month data), which makes them
a clean regression target. The tests lock the convention with independent
hand-computed anchors, a structural subset invariant, and an exact match to the
documented formula.
"""
import pytest

np = pytest.importorskip("numpy")
import gpcp_processing as gp


def test_mask_shapes():
    assert gp._ICOAST.shape == (gp.N_LON, gp.N_LAT)
    assert gp._JCOAST.shape == (gp.N_LON, gp.N_LAT)


def test_convention_anchor_cells():
    ic, jc = gp._build_coastal_masks()
    # Python[a, b] maps to idl_lat_j = (a * N_LAT + b) // N_LON.
    assert ic[0, 0] and jc[0, 0]                       # idl_lat_j 0 (polar) -> True
    assert ic[gp.N_LON - 1, gp.N_LAT - 1]              # idl_lat_j 71 (>=60) -> True
    # Python[60, 0] -> (60*72 + 0)//144 = 30: mid-latitude, both False.
    assert (60 * gp.N_LAT + 0) // gp.N_LON == 30
    assert not ic[60, 0] and not jc[60, 0]


def test_icoast_is_subset_of_jcoast():
    ic, jc = gp._build_coastal_masks()
    assert np.all(jc[ic])  # jcoast band is wider, so it covers all of icoast


def test_masks_match_documented_formula():
    ic, jc = gp._build_coastal_masks()
    k = np.arange(gp.GRID_SIZE, dtype=np.int32).reshape(gp.N_LON, gp.N_LAT)
    idl_lat_j = k // gp.N_LON
    assert np.array_equal(ic, (idl_lat_j <= 11) | (idl_lat_j >= 60))
    assert np.array_equal(jc, (idl_lat_j <= 19) | (idl_lat_j >= 53))
