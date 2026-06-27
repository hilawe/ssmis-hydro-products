"""Radiometric antenna-to-brightness-temperature conversion.

ta2tb is a pure, exact function with no data dependency, so it is an ideal
regression anchor. Two complementary checks: the calibration constants are
locked so they cannot drift silently, and the vectorized implementation is
verified against the explicit per-channel formula so an indexing or
cross-polarization mix-up is caught.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")  # climalg_ssmis imports scipy.ndimage at module load
import climalg_ssmis as ca


def test_ap_bp_constants_locked():
    assert np.allclose(ca.AP, [0.969, 0.969, 0.974, 0.986, 0.986, 0.988, 0.988])
    assert np.allclose(ca.BP, [0.00473, 0.00415, 0.0107, 0.0217, 0.02612, 0.01383, 0.01947])


def test_shape_preserved():
    ta = np.full((5, 3, 7), 200.0, dtype=np.float32)
    assert ca.ta2tb(ta).shape == ta.shape


def test_matches_explicit_per_channel_formula():
    ta = np.array([[180., 170., 200., 210., 205., 250., 245.]], dtype=np.float32)
    C, D = ca.C_TA2TB, ca.D_TA2TB
    a = ta[0]
    exp = np.empty(7, dtype=np.float64)
    exp[0] = C[0] * a[0] - D[0] * a[1]
    exp[1] = C[1] * a[1] - D[1] * a[0]
    exp[2] = C[2] * a[2] - D[2] * (0.653 * a[1] + 96.6)
    exp[3] = C[3] * a[3] - D[3] * a[4]
    exp[4] = C[4] * a[4] - D[4] * a[3]
    exp[5] = C[5] * a[5] - D[5] * a[6]
    exp[6] = C[6] * a[6] - D[6] * a[5]
    tb = ca.ta2tb(ta)
    assert np.allclose(tb[0], exp, rtol=1e-5, atol=1e-3)


def test_finite_on_physical_input():
    rng = np.random.default_rng(0)
    ta = rng.uniform(150.0, 295.0, size=(200, 7)).astype(np.float32)
    tb = ca.ta2tb(ta)
    assert np.all(np.isfinite(tb))
