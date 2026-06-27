"""combine.py multi-year assembly: seek offsets and record ordering.

combine writes a time-stacked binary where each month occupies one fixed-size
record. The load-bearing correctness property is that a given (year, month)
lands in exactly the right slot and that missing or future months are filled
with the sentinel, not left as garbage or misaligned. These tests exercise that
with synthetic per-month inputs in a temporary basedir (the --basedir hook the
module exposes for exactly this).
"""
import os
import tempfile

import pytest

np = pytest.importorskip("numpy")
import combine

FILL = np.float32(-999.99)


def _write_grid(path, n_elem, value):
    np.full(n_elem, value, dtype=np.float32).tofile(path)


def test_combine_25deg_offsets_and_fill():
    M, N = combine.get_dims(2.5)  # 72, 144
    sat, prod = 'ftest', 'CFR'
    with tempfile.TemporaryDirectory() as base:
        d = os.path.join(base, f'{sat}-2.5')
        os.makedirs(d)
        # Provide input only for 2000-02 and 2001-01, with distinct values.
        _write_grid(os.path.join(d, f'{prod}00-02-{sat}-2.5'), M * N, 5.0)
        _write_grid(os.path.join(d, f'{prod}01-01-{sat}-2.5'), M * N, 7.0)

        combine.combine_25deg_efficient(sat, 2000, 2001, 3, basedir=base)

        out = np.fromfile(os.path.join(d, f'{prod}-{sat}-2.5'), dtype=np.float32)
        assert out.size == 24 * M * N            # 2 years x 12 months
        out = out.reshape(24, M * N)             # month-major records

        # Slot index = (year - start) * 12 + (month - 1)
        assert np.all(out[1] == 5.0)             # 2000-02 -> slot 1
        assert np.all(out[12] == 7.0)            # 2001-01 -> slot 12
        assert np.all(out[0] == FILL)            # 2000-01 missing -> fill
        assert np.all(out[15] == FILL)           # 2001-04 (> current_month 3) -> fill


def test_combine_10deg_record_order():
    M, N = combine.get_dims(1.0)  # 180, 360
    sat, prod, yy = 'ftest', 'PR1', '01'
    with tempfile.TemporaryDirectory() as base:
        d = os.path.join(base, f'{sat}-1.0')
        os.makedirs(d)
        _write_grid(os.path.join(d, f'{prod}01-05-{sat}-1.0'), M * N, 3.0)  # May

        combine.combine_10deg(sat, 2001, 2001, yy, basedir=base)

        out = np.fromfile(os.path.join(d, f'{prod}.{yy}-{sat}-1.0'), dtype=np.float32)
        assert out.size == 12 * M * N
        out = out.reshape(12, M * N)
        assert np.all(out[4] == 3.0)             # May -> slot 4
        assert np.all(out[0] == FILL)            # Jan missing -> fill
