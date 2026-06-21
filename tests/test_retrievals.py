"""Structural invariants for the retrieval algorithms.

These deliberately do not assert exact geophysical magnitudes (that is what the
golden-month regression and the Fortran cross-validation cover). They assert the
contracts that must always hold regardless of the science: output shape, the
missing-channel sentinel, finiteness, and physical sign where a value is valid.
A change that breaks one of these is a bug, not a science update.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
import climalg_ssmis as ca

FILL = np.float32(-999.99)


def _physical_ta(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(150.0, 295.0, size=(n, 7)).astype(np.float32)


def test_precip8_shape_and_fill_on_missing():
    ta = np.full((6, 7), 102.0, dtype=np.float32)  # all channels missing
    tb = ca.ta2tb(ta)
    add_si = np.zeros((149, 2), dtype=np.float32)
    out = ca.precip8(ta, tb, np.zeros(6, bool), np.zeros(6, bool), add_si)
    assert out.shape == (6,)
    assert np.all(out == FILL)


def test_precip8_finite_and_nonnegative_where_valid():
    ta = _physical_ta(300, 3)
    tb = ca.ta2tb(ta)
    add_si = np.zeros((149, 2), dtype=np.float32)
    out = ca.precip8(ta, tb, np.zeros(300, bool), np.zeros(300, bool), add_si)
    assert np.all(np.isfinite(out))
    valid = out != FILL
    assert np.all(out[valid] >= 0.0)


def test_precip3_shape_and_finite():
    ta = _physical_ta(120, 4)
    out = ca.precip3(ta, np.zeros(120, bool))
    assert out.shape == (120,)
    assert np.all(np.isfinite(out))


def test_snowc_returns_flag_array():
    ta = _physical_ta(80, 5)
    out = ca.snowc(ta, np.ones(80, bool))
    assert out.shape == (80,)
    assert np.all(np.isfinite(out))
