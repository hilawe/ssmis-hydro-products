"""The ten monthly products, each worked by hand from its definition.

Every case builds the accumulator of ONE grid cell with counts and sums chosen
so the result can be checked on paper, and compares it with what the averaging
functions of climalg_ssmis return. The expected numbers are written out below
from the definitions, not taken from the code:

    rainfall      mean rain rate over the valid samples, times 24, times the days of the month
    rain fraction raining samples over valid samples
    liquid water  1000 times the mean over the cloudy samples (g m-2), ocean cells only
    cloud fraction cloudy samples over ocean samples, ocean cells only
    water vapor   mean over the samples with vapor, ocean cells only
    snow cover    snow detections over land samples, land cells only
    sea ice       100 times ice detections over ocean samples, ocean cells only
    sampling      valid samples over all samples

An ocean cell is one whose valid samples are at most 5 percent land, and a land
cell one with at least 95 percent. Rain is set to zero over an ocean cell with
more than 10 percent ice detections. Cloud and vapor are set to zero there too
where they were observed, and stay fill where they were not. The first rain
product alone, its amount and its fraction, is set to zero over a land cell with
more than 99 percent snow.

These cases check selected numerical consequences of the definitions, with each
threshold tried at its value and on either side of it. They work one cell at a
time, so they do not test that each averaging function lays its grid out
correctly, beyond the one layout case at the end. They do not test the
retrievals that fill the accumulator, which are compared with the legacy program
on real grids.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
import climalg_ssmis as cl

FILL = pytest.approx(-999.99, abs=1e-3)
NT = cl.NT


def cell(**kw):
    """The accumulator of one cell. Unnamed entries are zero."""
    index = dict(rain1_sum=0, rain1_n=1, rain2_sum=2, rain2_n=3, lwp_sum=4, cloudy_n=5,
                 wvp_sum=6, wvp_n=7, snow_n=8, ice_n=9, all_n=10, land_n=11, ocean_n=12)
    r = np.zeros((1, 1, NT), np.float32)
    for name, value in kw.items():
        r[0, 0, index[name]] = value
    return r


def one(field):
    assert field.shape == (1, 1)
    return float(field[0, 0])


OCEAN = dict(all_n=45, ocean_n=40, land_n=0)
LAND = dict(all_n=45, ocean_n=0, land_n=40)


def test_rainfall_is_mean_rate_times_24_times_days():
    """Sum of rates 6.0 mm/h over 40 valid samples is 0.15 mm/h. Over 30 days that is 108 mm."""
    r = cell(rain1_sum=6.0, rain1_n=8, rain2_sum=3.0, rain2_n=5, **OCEAN)
    assert one(cl.mean_precip(r, 1, 1, NT, 30, 1, 8)) == pytest.approx(108.0, rel=1e-6)
    assert one(cl.mean_precip(r, 1, 1, NT, 31, 1, 8)) == pytest.approx(111.6, rel=1e-6)
    assert one(cl.mean_precip(r, 1, 1, NT, 30, 1, 3)) == pytest.approx(54.0, rel=1e-6)


def test_rain_fraction_is_raining_samples_over_valid_samples():
    r = cell(rain1_sum=6.0, rain1_n=8, rain2_sum=3.0, rain2_n=5, **OCEAN)
    assert one(cl.mean_precip(r, 1, 1, NT, 30, 2, 8)) == pytest.approx(0.2)
    assert one(cl.mean_precip(r, 1, 1, NT, 30, 2, 3)) == pytest.approx(0.125)
    # Over all valid samples, land and ocean together: 8 of 25 ocean and 15 land samples is 0.2.
    mixed = cell(rain1_sum=6.0, rain1_n=8, all_n=45, ocean_n=25, land_n=15)
    assert one(cl.mean_precip(mixed, 1, 1, NT, 30, 2, 8)) == pytest.approx(0.2)
    assert one(cl.mean_precip(mixed, 1, 1, NT, 30, 1, 8)) == pytest.approx(108.0, rel=1e-6)


def test_a_cell_with_no_valid_sample_is_fill_in_every_product():
    r = cell(all_n=45)
    for value in (cl.mean_precip(r, 1, 1, NT, 30, 1, 8), cl.mean_precip(r, 1, 1, NT, 30, 2, 3),
                  cl.mean_lwp(r, 1, 1, NT, 1), cl.mean_lwp(r, 1, 1, NT, 2), cl.mean_wvp(r, 1, 1, NT),
                  cl.mean_snw(r, 1, 1, NT), cl.mean_ice(r, 1, 1, NT)):
        assert one(value) == FILL
    assert one(cl.mean_ssa(r, 1, 1, NT)) == 0.0
    assert one(cl.mean_ssa(cell(), 1, 1, NT)) == FILL


def test_rain_is_zero_over_an_ocean_cell_with_more_than_a_tenth_ice():
    icy = cell(rain1_sum=6.0, rain1_n=8, rain2_sum=3.0, rain2_n=5, ice_n=5, **OCEAN)     # 12.5 percent
    clear = cell(rain1_sum=6.0, rain1_n=8, rain2_sum=3.0, rain2_n=5, ice_n=4, **OCEAN)   # 10 percent, not more
    for ichan in (8, 3):
        assert one(cl.mean_precip(icy, 1, 1, NT, 30, 1, ichan)) == 0.0
        assert one(cl.mean_precip(clear, 1, 1, NT, 30, 1, ichan)) > 0.0


def test_only_the_first_rain_product_is_zero_over_a_snow_covered_land_cell():
    snowy = cell(rain1_sum=6.0, rain1_n=8, rain2_sum=3.0, rain2_n=5, snow_n=40, **LAND)   # 100 percent
    partly = cell(rain1_sum=6.0, rain1_n=8, rain2_sum=3.0, rain2_n=5, snow_n=39, **LAND)  # 97.5 percent
    assert one(cl.mean_precip(snowy, 1, 1, NT, 30, 1, 8)) == 0.0
    assert one(cl.mean_precip(snowy, 1, 1, NT, 30, 2, 8)) == 0.0                           # its fraction too
    assert one(cl.mean_precip(snowy, 1, 1, NT, 30, 1, 3)) == pytest.approx(54.0, rel=1e-6)
    assert one(cl.mean_precip(snowy, 1, 1, NT, 30, 2, 3)) == pytest.approx(0.125)
    assert one(cl.mean_precip(partly, 1, 1, NT, 30, 1, 8)) == pytest.approx(108.0, rel=1e-6)


def test_the_snow_screen_needs_more_than_99_percent_not_99():
    """Exactly 99 of 100 land samples with snow is not more than 99 percent, so the
    first rain product is kept: 30 days times 24 times 6.0 over 100 samples is 43.2
    mm. With all 100 it is zero."""
    at = cell(rain1_sum=6.0, rain1_n=8, snow_n=99, all_n=110, ocean_n=0, land_n=100)
    above = cell(rain1_sum=6.0, rain1_n=8, snow_n=100, all_n=110, ocean_n=0, land_n=100)
    assert one(cl.mean_precip(at, 1, 1, NT, 30, 1, 8)) == pytest.approx(43.2, rel=1e-6)
    assert one(cl.mean_precip(at, 1, 1, NT, 30, 2, 8)) == pytest.approx(0.08)
    assert one(cl.mean_precip(above, 1, 1, NT, 30, 1, 8)) == 0.0


def test_liquid_water_is_the_mean_over_cloudy_samples_in_grams():
    """0.9 mm summed over 6 cloudy samples is 0.15 mm, which is 150 g m-2."""
    r = cell(lwp_sum=0.9, cloudy_n=6, **OCEAN)
    assert one(cl.mean_lwp(r, 1, 1, NT, 1)) == pytest.approx(150.0, rel=1e-6)


def test_cloud_fraction_is_cloudy_samples_over_ocean_samples():
    """Over ocean samples, not over all valid samples. With 38 ocean and 2 land
    samples (5 percent land, still an ocean cell) the answer is 6/38, not 6/40."""
    assert one(cl.mean_lwp(cell(lwp_sum=0.9, cloudy_n=6, **OCEAN), 1, 1, NT, 2)) == pytest.approx(0.15)
    mixed = cell(lwp_sum=0.9, cloudy_n=6, all_n=45, ocean_n=38, land_n=2)
    assert one(cl.mean_lwp(mixed, 1, 1, NT, 2)) == pytest.approx(6.0 / 38.0)


def test_water_vapor_is_the_mean_over_samples_with_vapor():
    r = cell(wvp_sum=120.0, wvp_n=4, **OCEAN)
    assert one(cl.mean_wvp(r, 1, 1, NT)) == pytest.approx(30.0)


def test_ocean_products_are_fill_where_land_exceeds_five_percent():
    """3 land samples of 40 is 7.5 percent land."""
    coast = cell(lwp_sum=0.9, cloudy_n=6, wvp_sum=120.0, wvp_n=4, ice_n=2, all_n=45, ocean_n=37, land_n=3)
    assert one(cl.mean_lwp(coast, 1, 1, NT, 1)) == FILL
    assert one(cl.mean_lwp(coast, 1, 1, NT, 2)) == FILL
    assert one(cl.mean_wvp(coast, 1, 1, NT)) == FILL
    assert one(cl.mean_ice(coast, 1, 1, NT)) == FILL


def test_cloud_and_vapor_are_zero_over_ice_and_stay_fill_where_never_observed():
    icy = cell(lwp_sum=0.9, cloudy_n=6, wvp_sum=120.0, wvp_n=4, ice_n=5, **OCEAN)
    assert one(cl.mean_lwp(icy, 1, 1, NT, 1)) == 0.0
    assert one(cl.mean_lwp(icy, 1, 1, NT, 2)) == 0.0
    assert one(cl.mean_wvp(icy, 1, 1, NT)) == 0.0
    # An icy ocean cell with no cloudy sample and no vapor sample holds fill, not zero.
    unseen = cell(ice_n=5, **OCEAN)
    assert one(cl.mean_lwp(unseen, 1, 1, NT, 1)) == FILL
    assert one(cl.mean_lwp(unseen, 1, 1, NT, 2)) == FILL
    assert one(cl.mean_wvp(unseen, 1, 1, NT)) == FILL


def test_snow_cover_is_detections_over_land_samples_on_land_cells():
    assert one(cl.mean_snw(cell(snow_n=30, **LAND), 1, 1, NT)) == pytest.approx(0.75)
    # 38 land and 2 ocean samples is exactly 95 percent land and counts as a land cell.
    edge = cell(snow_n=19, all_n=45, land_n=38, ocean_n=2)
    assert one(cl.mean_snw(edge, 1, 1, NT)) == pytest.approx(0.5)
    assert one(cl.mean_snw(cell(snow_n=10, all_n=45, land_n=30, ocean_n=10), 1, 1, NT)) == FILL
    assert one(cl.mean_snw(cell(**OCEAN), 1, 1, NT)) == FILL


def test_sea_ice_is_percent_of_ocean_samples_on_ocean_cells():
    assert one(cl.mean_ice(cell(ice_n=10, **OCEAN), 1, 1, NT)) == pytest.approx(25.0)
    assert one(cl.mean_ice(cell(ice_n=0, **OCEAN), 1, 1, NT)) == 0.0
    assert one(cl.mean_ice(cell(**LAND), 1, 1, NT)) == FILL


def test_sampling_is_valid_samples_over_all_samples():
    assert one(cl.mean_ssa(cell(**OCEAN), 1, 1, NT)) == pytest.approx(40.0 / 45.0)
    assert one(cl.mean_ssa(cell(all_n=45, ocean_n=25, land_n=20), 1, 1, NT)) == pytest.approx(1.0)


def test_output_layout_puts_the_south_first_and_the_prime_meridian_first():
    """A 2 by 4 accumulator, north row first and 180 W first, comes back as
    (lon, lat) with the south row first and the longitudes rolled by half."""
    z = np.arange(8, dtype=np.float32).reshape(2, 4)       # row 0 north: 0 1 2 3, row 1 south: 4 5 6 7
    out = cl.to_grads(z, 2, 4)
    assert out.shape == (4, 2)
    assert out.T.tolist() == [[6.0, 7.0, 4.0, 5.0], [2.0, 3.0, 0.0, 1.0]]
