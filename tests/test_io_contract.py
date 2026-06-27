"""Binary I/O format contract.

The monthly products are raw float32 with a specific layout (south-first rows,
longitude varying fastest) and specific fill sentinels. That convention is the
kind of thing that is otherwise documented only in prose and easy to break
silently. These tests pin it down: the GrADS layout transform round-trips, the
on-disk byte order is what downstream readers assume, and the fill value is
exactly float32(-999.99).
"""
import os
import tempfile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
import climalg_ssmis as ca


def test_to_grads_is_invertible():
    # to_grads(z, M, N): (M,N) north-first/180W-first -> (N,M) south-first/1.25E.
    M, N = 72, 144
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 50, size=(M, N)).astype(np.float32)
    xp = ca.to_grads(z, M, N)
    assert xp.shape == (N, M)
    # Apply the documented inverse: transpose back, roll +N/2, flip lat.
    z_rolled = xp.T
    z_flip = np.roll(z_rolled, N // 2, axis=1)
    z_back = z_flip[::-1, :]
    assert np.array_equal(z_back, z)


def test_write_grads_on_disk_layout_is_lon_fastest_south_first():
    # write_grads writes xp.T (=(M,N)=(lat,lon)) in C order: each south-to-north
    # row holds all lon values in order. A plain fromfile+reshape(M,N) must
    # recover xp.T bit-for-bit.
    M, N = 72, 144
    rng = np.random.default_rng(2)
    xp = rng.uniform(0, 50, size=(N, M)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, 'grid.bin')
        ca.write_grads(fn, xp)
        assert os.path.getsize(fn) == M * N * 4  # float32
        back = np.fromfile(fn, dtype=np.float32).reshape(M, N)
        assert np.array_equal(back, xp.T)


def test_fill_value_contract():
    fill = np.float32(-999.99)
    # precip8 emits exactly this sentinel for invalid cells.
    ta = np.full((4, 7), 102.0, dtype=np.float32)  # 102.0 == missing channel
    tb = ca.ta2tb(ta)
    add_si = np.zeros((149, 2), dtype=np.float32)
    out = ca.precip8(ta, tb, np.zeros(4, bool), np.zeros(4, bool), add_si)
    assert np.all(out == fill)
