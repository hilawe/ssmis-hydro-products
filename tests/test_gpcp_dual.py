"""The dual-satellite merge of the GPCP input (gpcp_processing.gpcp_dual).

The rule for one month is gpcp_processing.merge_dual_month. These tests state
it from the outside, in three layers.

1. Named cases, one per way a cell can stand: both chains valid, a snow flag,
   an ice flag or the fill in either chain, zero rain, zero sampling, missing
   sampling, sampling out of range, and non-finite values. Rain validity and
   sampling validity are tested separately, because until 18 November 2026 the
   sampling field depended on a rain value and a negative flag of the second
   chain entered the rain mean.
2. An oracle written here cell by cell, with plain `if` statements and no
   reference to the module's masks, compared with the module on random fields.
3. The file-level function on small real files: the time origin, the months
   before the second chain's record, the cutoff at the last complete month, the
   year boundary, a second chain shorter than the first, the second chain's
   rain and sampling ending at different months, both rain products, and each
   of the six inputs missing at the start.

These are selected cases. They do not cover an input that changes or vanishes
during a run, an interrupted run, or the real record, which is measured
separately.

The primary chain is the morning chain (F-17 today), the secondary chain is the
late-morning chain (F-16 today).
"""
import math
import os

import pytest

np = pytest.importorskip("numpy")
import gpcp_processing as gp

FILL = np.float32(-999.99)
SNOW = np.float32(-45.0)      # the snow mask writes -100 times a fraction of 0.45
ICE = np.float32(-80.0)       # the ice mask writes minus an ice cover of 80 percent
FLAGS = (SNOW, ICE, FILL)


def merge(rp, sp, rs, ss):
    """One cell through the module's rule. Returns (rain, sampling) as floats."""
    rain, samp = gp.merge_dual_month(np.array([rp], np.float32), np.array([sp], np.float32),
                                     np.array([rs], np.float32), np.array([ss], np.float32))
    assert rain.dtype == np.float32 and samp.dtype == np.float32
    return float(rain[0]), float(samp[0])


def is_fill(x):
    return x == pytest.approx(float(FILL), abs=1e-3)


# ------------------------------------------------------------------ the two validity rules

def test_a_rain_amount_is_finite_and_zero_or_more():
    values = np.array([np.inf, -np.inf, np.nan, float(FILL), float(SNOW), float(ICE), -0.0001, 0.0, 0.3, 8693.0],
                      np.float32)
    assert gp.valid_rain(values).tolist() == [False, False, False, False, False, False, False, True, True, True]


def test_a_sampling_fraction_is_finite_and_from_zero_to_one():
    values = np.array([np.inf, -np.inf, np.nan, float(FILL), -0.1, 0.0, 0.00025, 0.5, 1.0, 1.0001, 1.5], np.float32)
    assert gp.valid_sampling(values).tolist() == [False, False, False, False, False, True, True, True, True,
                                                  False, False]


def test_the_merge_mask_is_the_stated_rule():
    """Merged only where both rain values are amounts and both sampling fractions
    are valid and positive. The mask is also what the run counts and reports."""
    inf = float("inf")
    cases = [  # primary rain, primary sampling, secondary rain, secondary sampling, merged
        (10.0, 0.5, 20.0, 0.5, True), (0.0, 0.5, 0.0, 0.5, True), (10.0, 1.0, 20.0, 1.0, True),
        (inf, 0.5, 20.0, 0.5, False), (10.0, 0.5, inf, 0.5, False),
        (float(SNOW), 0.5, 20.0, 0.5, False), (10.0, 0.5, float(ICE), 0.5, False),
        (10.0, 0.0, 20.0, 0.5, False), (10.0, 0.5, 20.0, 0.0, False),
        (10.0, float(FILL), 20.0, 0.5, False), (10.0, 0.5, 20.0, 1.5, False), (10.0, inf, 20.0, 0.5, False),
    ]
    a = np.array(cases, np.float32)
    got = gp.dual_merge_mask(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    assert got.tolist() == [bool(c[4]) for c in cases]


# ------------------------------------------------------------------ rain

def test_both_chains_valid_gives_the_sampling_weighted_mean():
    rain, samp = merge(10.0, 0.25, 20.0, 0.75)
    assert rain == pytest.approx(17.5)
    assert samp == pytest.approx(0.5)


def test_equal_weights_give_the_plain_mean():
    rain, _ = merge(10.0, 0.5, 20.0, 0.5)
    assert rain == pytest.approx(15.0)


@pytest.mark.parametrize("flag", FLAGS)
def test_a_negative_secondary_value_never_enters_the_mean(flag):
    """The defect of the legacy and of the Python until 18 November 2026: the
    primary amount 10 with a secondary snow flag of -100 at equal sampling came
    out as -45."""
    rain, _ = merge(10.0, 0.5, float(flag), 0.5)
    assert rain == 10.0


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("secondary", [20.0, 0.0, float(SNOW), float(FILL)])
def test_a_primary_flag_passes_through_whatever_the_secondary_holds(flag, secondary):
    """F-17 is primary. Its snow flag, ice flag or fill is the dual value, and the
    secondary chain is never used on its own."""
    rain, _ = merge(float(flag), 0.5, secondary, 0.5)
    assert rain == pytest.approx(float(flag))


def test_zero_rain_is_an_amount_in_either_chain():
    assert merge(0.0, 0.5, 20.0, 0.5)[0] == pytest.approx(10.0)
    assert merge(10.0, 0.5, 0.0, 0.5)[0] == pytest.approx(5.0)
    assert merge(0.0, 0.5, 0.0, 0.5)[0] == 0.0


def test_zero_primary_sampling_keeps_the_primary_value():
    """Before 18 November 2026 this cell took the secondary value alone (20),
    since the primary weight was zero. The primary value now passes through."""
    rain, samp = merge(10.0, 0.0, 20.0, 0.5)
    assert rain == 10.0
    assert samp == pytest.approx(0.25)


def test_zero_secondary_sampling_keeps_the_primary_value_exactly():
    """A weighted mean with a zero secondary weight is the primary value times
    its weight divided by its weight, which in single precision is not always
    the primary value. For 10.3 at a sampling of 0.1 it is 10.300001. The
    primary value passes through bit for bit instead."""
    a, w = np.float32(10.3), np.float32(0.1)
    assert (a * w) / w != a                      # the pair shows the difference
    rain, samp = merge(10.3, 0.1, 20.0, 0.0)
    assert rain == float(a)
    assert samp == pytest.approx(0.05)


def test_zero_sampling_in_both_chains():
    rain, samp = merge(10.0, 0.0, 20.0, 0.0)
    assert rain == 10.0
    assert samp == 0.0


@pytest.mark.parametrize("sp,ss", [(float(FILL), 0.5), (0.5, float(FILL)),
                                   (float(FILL), float(FILL))])
def test_missing_sampling_keeps_the_primary_value_and_fills_the_sampling(sp, ss):
    rain, samp = merge(10.0, sp, 20.0, ss)
    assert rain == 10.0
    assert is_fill(samp)


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["primary", "secondary"])
def test_a_sampling_value_that_is_not_a_fraction_is_not_used(bad, side):
    sp, ss = (bad, 0.5) if side == "primary" else (0.5, bad)
    rain, samp = merge(10.0, sp, 20.0, ss)
    assert rain == 10.0
    assert is_fill(samp)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_secondary_rain_keeps_the_primary_value(bad):
    rain, samp = merge(10.0, 0.5, bad, 0.5)
    assert rain == 10.0
    assert samp == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_primary_rain_is_written_as_fill(bad):
    rain, _ = merge(bad, 0.5, 20.0, 0.5)
    assert is_fill(rain)


# ------------------------------------------------------------------ sampling

@pytest.mark.parametrize("rp", [10.0, 0.0, float(SNOW), float(ICE), float(FILL)])
@pytest.mark.parametrize("rs", [20.0, 0.0, float(SNOW), float(ICE), float(FILL)])
def test_the_sampling_field_does_not_depend_on_either_rain_value(rp, rs):
    """The Python until 18 November 2026 wrote the sampling as fill wherever the
    primary rain of the second rain product was negative. The legacy writes the
    plain mean at every cell."""
    assert merge(rp, 0.2, rs, 0.6)[1] == pytest.approx(0.4)


def test_sampling_is_the_same_for_two_rain_products_of_one_month():
    rng = np.random.default_rng(11)
    sp = rng.uniform(0, 1, 500).astype(np.float32)
    ss = rng.uniform(0, 1, 500).astype(np.float32)
    rain_a = rng.uniform(0, 300, 500).astype(np.float32)
    rain_b = np.where(rng.uniform(size=500) < 0.4, SNOW, rain_a).astype(np.float32)
    rs = rng.uniform(0, 300, 500).astype(np.float32)
    samp_a = gp.merge_dual_month(rain_a, sp, rs, ss)[1]
    samp_b = gp.merge_dual_month(rain_b, sp, rs, ss)[1]
    np.testing.assert_array_equal(samp_a, samp_b)


# ------------------------------------------------------------------ oracle

def _oracle(rp, sp, rs, ss):
    """The rule for one cell, written without the module's helpers."""
    def amount(x):
        return math.isfinite(x) and x >= 0.0

    def fraction(x):
        return math.isfinite(x) and 0.0 <= x <= 1.0

    if amount(rp) and amount(rs) and fraction(sp) and fraction(ss) and sp > 0.0 and ss > 0.0:
        rain = (rp * sp + rs * ss) / (sp + ss)
    elif math.isfinite(rp):
        rain = rp
    else:
        rain = float(FILL)
    samp = (sp + ss) / 2.0 if fraction(sp) and fraction(ss) else float(FILL)
    return rain, samp


def _random_fields(seed, n=4000):
    rng = np.random.default_rng(seed)

    def rain():
        x = rng.uniform(0.0, 400.0, n)
        pick = rng.uniform(size=n)
        x[pick < 0.10] = 0.0
        x[(pick >= 0.10) & (pick < 0.20)] = -100.0 * rng.uniform(0.2, 1.0, n)[(pick >= 0.10) & (pick < 0.20)]
        x[(pick >= 0.20) & (pick < 0.30)] = -rng.uniform(25.0, 100.0, n)[(pick >= 0.20) & (pick < 0.30)]
        x[(pick >= 0.30) & (pick < 0.38)] = FILL
        x[(pick >= 0.38) & (pick < 0.39)] = np.nan
        return x.astype(np.float32)

    def samp():
        x = rng.uniform(0.0, 1.0, n)
        pick = rng.uniform(size=n)
        x[pick < 0.08] = 0.0
        x[(pick >= 0.08) & (pick < 0.16)] = FILL
        x[(pick >= 0.16) & (pick < 0.18)] = 1.0
        x[(pick >= 0.18) & (pick < 0.19)] = 1.7
        return x.astype(np.float32)

    return rain(), samp(), rain(), samp()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_module_agrees_with_a_cell_by_cell_oracle(seed):
    rp, sp, rs, ss = _random_fields(seed)
    rain, samp = gp.merge_dual_month(rp, sp, rs, ss)
    want = [_oracle(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(rp, sp, rs, ss)]
    want_rain = np.array([w[0] for w in want])
    want_samp = np.array([w[1] for w in want])
    np.testing.assert_allclose(rain, want_rain, rtol=2e-6, atol=1e-4)
    np.testing.assert_allclose(samp, want_samp, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("seed", [4, 5])
def test_the_dual_never_has_more_or_less_coverage_than_the_primary(seed):
    rp, sp, rs, ss = _random_fields(seed)
    rain, _ = gp.merge_dual_month(rp, sp, rs, ss)
    finite = np.isfinite(rp)
    # An amount stays an amount and a flag stays that flag.
    np.testing.assert_array_equal(rain[finite] >= 0.0, rp[finite] >= 0.0)
    np.testing.assert_array_equal(rain[finite & (rp < 0)], rp[finite & (rp < 0)])
    # A merged cell lies between the two amounts it was made from.
    merged = rain != np.where(finite, rp, FILL)
    lo, hi = np.minimum(rp, rs), np.maximum(rp, rs)
    assert merged.sum() > 500
    assert np.all(rain[merged] >= lo[merged] - 1e-3) and np.all(rain[merged] <= hi[merged] + 1e-3)
    assert np.all(rs[merged] >= 0.0)


# ------------------------------------------------------------------ extreme values

def test_the_mean_is_formed_in_single_precision_like_the_legacy():
    """The weighted mean is computed in single precision, as the legacy computes it,
    so that a cell the repair does not change stays bit for bit what it was. The
    result for ordinary values is the single-precision formula exactly."""
    rng = np.random.default_rng(21)
    rp = rng.uniform(0, 2000, 2000).astype(np.float32)
    rs = rng.uniform(0, 2000, 2000).astype(np.float32)
    sp = rng.uniform(0.01, 1, 2000).astype(np.float32)
    ss = rng.uniform(0.01, 1, 2000).astype(np.float32)
    rain, _ = gp.merge_dual_month(rp, sp, rs, ss)
    want = (rp * sp + rs * ss) / (ss + sp)
    assert want.dtype == np.float32
    np.testing.assert_array_equal(rain, want)


def test_a_mean_that_overflows_single_precision_passes_the_primary_value_through():
    """Two amounts of 3e38 with full sampling overflow in the sum before the
    division. The largest rain value in either tree's files on 18 November 2026 is
    8,693 mm, so the case is one of arithmetic and not of the record. The mean is not finite and is
    not used."""
    big = float(np.float32(3e38))
    rain, samp = merge(big, 1.0, big, 1.0)
    assert rain == big
    assert samp == pytest.approx(1.0)


def test_single_precision_has_a_known_lower_limit_on_sampling():
    """With both sampling fractions at the smallest positive single-precision number
    the two products underflow and the mean of 0.1 and 0.2 comes out as zero. This is
    recorded, not repaired: a sampling fraction is a ratio of sample counts, and the
    smallest positive value in either tree's files on 18 November 2026 is 0.00025.
    Widening the arithmetic would change the stored value of about a third of the
    merged cells of the version 2 files, each by less than 0.001 mm."""
    tiny = float(np.nextafter(np.float32(0), np.float32(1)))
    rain, _ = merge(0.1, tiny, 0.2, tiny)
    assert rain == 0.0
    # At any sampling fraction a file could hold, the mean is right.
    assert merge(0.1, 1e-6, 0.2, 1e-6)[0] == pytest.approx(0.15, rel=1e-5)


# ------------------------------------------------------------------ files

N = 4            # cells per month in the small files


def _write(path, months):
    np.asarray(months, dtype=np.float32).tofile(path)


def _samp17(jmon, cell):
    """Primary sampling of month jmon (1 = January 1987) at a cell: its own value everywhere."""
    return 0.10 + 0.05 * cell + 0.001 * jmon


def _samp16(k, cell):
    """Secondary sampling of month k (1 = January 1992) at a cell."""
    return 0.40 + 0.07 * cell + 0.002 * k


def _setup(tmp_path, monkeypatch, clock, n17=None, n16=None, n16_rain=None, n16_samp=None):
    """Small real files for gpcp_dual. Returns (paths, inputs).

    Rain amounts and both sampling fields differ in every month and every cell, so
    a month or a cell read from the wrong place shows in the rain and in the
    sampling. The flags and the missing value repeat from month to month by design:
    cell 1 holds a secondary snow flag, cell 2 a primary snow flag in the second rain
    product only, and cell 3 is missing in the secondary sampling.
    """
    monkeypatch.setattr(gp, 'N_LON', N)
    monkeypatch.setattr(gp, 'N_LAT', 1)
    monkeypatch.setattr(gp, 'GRID_SIZE', N)
    monkeypatch.setattr(gp, '_current_year_month', lambda: clock)
    monkeypatch.setattr(gp, '_LATE_INIT_YEAR', 1987)
    monkeypatch.setattr(gp, '_EARLY_INIT_YEAR', 1992)
    monkeypatch.setattr(gp, '_LATE_SAT', 'f17')
    monkeypatch.setattr(gp, '_EARLY_SAT', 'f16')
    cy, cm = clock
    full17 = (cy - 1987 + 1) * 12 - (12 if cm == 1 else 0)
    full16 = (cy - 1992 + 1) * 12 - (12 if cm == 1 else 0)
    n17 = full17 if n17 is None else n17
    n16 = full16 if n16 is None else n16
    n16_rain = n16 if n16_rain is None else n16_rain
    n16_samp = n16 if n16_samp is None else n16_samp

    d17, d16, out = tmp_path / 'f17-2.5', tmp_path / 'f16-2.5', tmp_path / 'gpcp'
    for d in (d17, d16, out):
        d.mkdir()
    cells = np.arange(N, dtype=np.float64)[None, :]
    m17 = np.arange(1, n17 + 1, dtype=np.float64)[:, None]      # month number, 1 = Jan 1987
    r16 = np.arange(1, n16_rain + 1, dtype=np.float64)[:, None]  # month number, 1 = Jan 1992
    s16 = np.arange(1, n16_samp + 1, dtype=np.float64)[:, None]
    inp = {
        'r17_1': (m17 + 0.01 * cells).astype(np.float32),            # primary rain: month and cell
        'r17_2': (m17 + 0.01 * cells + 0.5).astype(np.float32),
        'r16_1': (1000.0 + r16 + 0.01 * cells).astype(np.float32),   # secondary rain: 1000 + month and cell
        'r16_2': (2000.0 + r16 + 0.01 * cells).astype(np.float32),
        's17': _samp17(m17, cells).astype(np.float32),
        's16': _samp16(s16, cells).astype(np.float32),
    }
    inp['r16_1'][:, 1] = SNOW
    inp['r16_2'][:, 1] = SNOW
    inp['r17_2'][:, 2] = SNOW
    inp['s16'][:, 3] = FILL
    _write(out / 'gpcp_nesdis_pr1.dat', inp['r17_1'])
    _write(out / 'gpcp_nesdis_pr2.dat', inp['r17_2'])
    _write(out / 'gpcp_nesdis_f10_pr1.dat', inp['r16_1'])
    _write(out / 'gpcp_nesdis_f10_pr2.dat', inp['r16_2'])
    _write(d17 / 'SSA-f17-2.5', inp['s17'])
    _write(d16 / 'SSA-f16-2.5', inp['s16'])
    return (str(d17) + os.sep, str(d16) + os.sep, str(out) + os.sep), inp


def _read(out, name):
    return np.fromfile(os.path.join(out, name), dtype=np.float32).reshape(-1, N)


def _mean(jmon, cell, product):
    """The weighted mean a merged cell must hold, worked here in double precision."""
    k = jmon - 60
    a = jmon + 0.01 * cell + (0.5 if product == 2 else 0.0)
    b = (1000.0 if product == 1 else 2000.0) + k + 0.01 * cell
    wa, wb = _samp17(jmon, cell), _samp16(k, cell)
    return (a * wa + b * wb) / (wa + wb)


def _primary(jmon, cell, product):
    return float(np.float32(jmon + 0.01 * cell + (0.5 if product == 2 else 0.0)))


def test_file_level_merge_time_axis_and_both_products(tmp_path, monkeypatch):
    """Clock in March 1993: 84 months on the primary timeline, the last complete
    month is February 1993 (month 74)."""
    (d17, d16, out), inp = _setup(tmp_path, monkeypatch, (1993, 3))
    gp.gpcp_dual(d17, d16, out)

    pr1, pr2, ssa = (_read(out, f'gpcp_nesdis_dual_{k}.dat') for k in ('pr1', 'pr2', 'ssa'))
    assert pr1.shape == pr2.shape == ssa.shape == (84, N)

    # Months 1 to 60, January 1987 to December 1991: fill, although the primary
    # chain holds valid data there. The dual product does not begin before 1992.
    for f in (pr1, pr2, ssa):
        assert np.all(np.isclose(f[:60], FILL, atol=1e-3))
    # Months 75 to 84, after the last complete month: fill.
    for f in (pr1, pr2, ssa):
        assert np.all(np.isclose(f[74:], FILL, atol=1e-3))

    for jmon in range(61, 75):
        i = jmon - 1
        # Cell 0, both valid: the weighted mean of THIS month and THIS cell of each chain.
        assert pr1[i, 0] == pytest.approx(_mean(jmon, 0, 1), rel=2e-6)
        assert pr2[i, 0] == pytest.approx(_mean(jmon, 0, 2), rel=2e-6)
        # Cell 1, secondary snow flag: the primary amount.
        assert pr1[i, 1] == _primary(jmon, 1, 1) and pr2[i, 1] == _primary(jmon, 1, 2)
        # Cell 2, primary snow flag in the second product only.
        assert pr1[i, 2] == pytest.approx(_mean(jmon, 2, 1), rel=2e-6)
        assert pr2[i, 2] == SNOW
        # Cell 3, secondary sampling missing: the primary amount.
        assert pr1[i, 3] == _primary(jmon, 3, 1) and pr2[i, 3] == _primary(jmon, 3, 2)
        # Sampling: the mean of this month's two fractions at each cell, the flagged
        # cells 1 and 2 included, and fill where one is missing.
        for cell in range(3):
            want = (_samp17(jmon, cell) + _samp16(jmon - 60, cell)) / 2.0
            assert ssa[i, cell] == pytest.approx(want, rel=2e-6), (jmon, cell)
        assert is_fill(float(ssa[i, 3]))

    # The two sampling copies are the inputs unchanged.
    np.testing.assert_array_equal(_read(out, 'gpcp_nesdis_ssa.dat'), inp['s17'])
    np.testing.assert_array_equal(_read(out, 'gpcp_nesdis_f10_ssa.dat'), inp['s16'])


def test_january_writes_through_december_of_the_year_before(tmp_path, monkeypatch):
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1994, 1))
    gp.gpcp_dual(d17, d16, out)
    pr1 = _read(out, 'gpcp_nesdis_dual_pr1.dat')
    assert pr1.shape == (84, N)                              # January 1987 to December 1993
    assert pr1[83, 0] == pytest.approx(_mean(84, 0, 1), rel=2e-6)             # December 1993
    assert not np.any(np.isclose(pr1[60:, 0], FILL, atol=1e-3))


def test_february_writes_january_and_fills_the_rest_of_the_year(tmp_path, monkeypatch):
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1994, 2))
    gp.gpcp_dual(d17, d16, out)
    pr1 = _read(out, 'gpcp_nesdis_dual_pr1.dat')
    assert pr1.shape == (96, N)
    assert pr1[84, 0] == pytest.approx(_mean(85, 0, 1), rel=2e-6)             # January 1994
    assert np.all(np.isclose(pr1[85:], FILL, atol=1e-3))


def test_a_shorter_secondary_record_falls_back_to_the_primary(tmp_path, monkeypatch):
    """The secondary files end after 5 months (May 1992). Later months hold the
    primary value in rain and fill in sampling."""
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1993, 3), n16=5)
    gp.gpcp_dual(d17, d16, out)
    pr1, ssa = _read(out, 'gpcp_nesdis_dual_pr1.dat'), _read(out, 'gpcp_nesdis_dual_ssa.dat')
    assert pr1[64, 0] == pytest.approx(_mean(65, 0, 1), rel=2e-6)             # May 1992, merged
    for jmon in range(66, 75):
        for cell in range(N):
            assert pr1[jmon - 1, cell] == _primary(jmon, cell, 1)
        assert np.all(np.isclose(ssa[jmon - 1], FILL, atol=1e-3))


def test_secondary_rain_shorter_than_its_sampling(tmp_path, monkeypatch):
    """Secondary rain ends after 5 months and secondary sampling runs on. The later
    months hold the primary rain, and the sampling field still holds the mean of the
    two fractions, because both chains sampled. The two files are read separately
    and neither shifts the other."""
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1993, 3), n16_rain=5)
    gp.gpcp_dual(d17, d16, out)
    pr1, ssa = _read(out, 'gpcp_nesdis_dual_pr1.dat'), _read(out, 'gpcp_nesdis_dual_ssa.dat')
    for jmon in range(66, 75):
        assert pr1[jmon - 1, 0] == _primary(jmon, 0, 1)
        want = (_samp17(jmon, 0) + _samp16(jmon - 60, 0)) / 2.0
        assert ssa[jmon - 1, 0] == pytest.approx(want, rel=2e-6)


def test_secondary_sampling_shorter_than_its_rain(tmp_path, monkeypatch):
    """Secondary sampling ends after 5 months and secondary rain runs on. Without a
    secondary sampling fraction nothing is merged: the later months hold the primary
    rain and fill in the sampling field."""
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1993, 3), n16_samp=5)
    gp.gpcp_dual(d17, d16, out)
    pr1, ssa = _read(out, 'gpcp_nesdis_dual_pr1.dat'), _read(out, 'gpcp_nesdis_dual_ssa.dat')
    assert pr1[64, 0] == pytest.approx(_mean(65, 0, 1), rel=2e-6)
    for jmon in range(66, 75):
        assert pr1[jmon - 1, 0] == _primary(jmon, 0, 1)
        assert np.all(np.isclose(ssa[jmon - 1], FILL, atol=1e-3))


def test_primary_sampling_shorter_than_its_rain(tmp_path, monkeypatch):
    """Primary sampling ends after month 65 and primary rain runs on. Without a
    primary sampling fraction nothing is merged: the later months hold the primary
    rain, in both products, and fill in the sampling field. A missing primary
    sampling value taken for a full one would merge them. Added 18 November 2026
    because the earlier cases did not fail under that error."""
    (d17, d16, out), inp = _setup(tmp_path, monkeypatch, (1993, 3))
    _write(os.path.join(d17, 'SSA-f17-2.5'), inp['s17'][:65])
    gp.gpcp_dual(d17, d16, out)
    for product in (1, 2):
        rain = _read(out, f'gpcp_nesdis_dual_pr{product}.dat')
        np.testing.assert_array_equal(rain[65:74], inp[f'r17_{product}'][65:74])
    sampling = _read(out, 'gpcp_nesdis_dual_ssa.dat')
    np.testing.assert_array_equal(sampling[65:74], np.full((9, N), FILL))


def test_a_shorter_primary_record_gives_fill_not_the_secondary(tmp_path, monkeypatch):
    """The primary files end at December 1992 (month 72). The secondary chain is
    never used on its own, so January and February 1993 are fill."""
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1993, 3), n17=72)
    gp.gpcp_dual(d17, d16, out)
    pr1 = _read(out, 'gpcp_nesdis_dual_pr1.dat')
    assert pr1[71, 0] == pytest.approx(_mean(72, 0, 1), rel=2e-6)
    assert np.all(np.isclose(pr1[72:74], FILL, atol=1e-3))


@pytest.mark.parametrize("missing", ['gpcp/gpcp_nesdis_pr1.dat', 'gpcp/gpcp_nesdis_pr2.dat',
                                     'gpcp/gpcp_nesdis_f10_pr1.dat', 'gpcp/gpcp_nesdis_f10_pr2.dat',
                                     'f16-2.5/SSA-f16-2.5', 'f17-2.5/SSA-f17-2.5'])
def test_a_missing_input_raises_and_leaves_the_previous_dual_files(tmp_path, monkeypatch, missing):
    """Opening an output truncates it. A run without one of its six inputs must stop
    before that, or it replaces the previous dual product with fill. This covers a
    file that is absent at the start. It does not cover a file that vanishes or
    cannot be read once the run is under way, and the outputs are written in place."""
    (d17, d16, out), _ = _setup(tmp_path, monkeypatch, (1993, 3))
    previous = {}
    for k in ('pr1', 'pr2', 'ssa'):
        f = os.path.join(out, f'gpcp_nesdis_dual_{k}.dat')
        with open(f, 'wb') as fh:
            fh.write(b'previous run ' + k.encode())
        previous[f] = open(f, 'rb').read()
    os.remove(tmp_path / missing)

    with pytest.raises(FileNotFoundError) as err:
        gp.gpcp_dual(d17, d16, out)

    assert os.path.basename(missing) in str(err.value)
    for f, content in previous.items():
        assert open(f, 'rb').read() == content
