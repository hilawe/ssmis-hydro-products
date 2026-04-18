#!/usr/bin/env python3
"""
climalg_ssmis.py

PURPOSE
    Generates climatological monthly products (PR1, PF1, PR2, PF2, LWP, CFR,
    WVP, SNW, ICE, SSA) from 1/3-degree SSMIS gridded TDR files.
    Replaces the Fortran programs climalg-ssmis-2.5deg.f and climalg-ssmis-1.0deg.f

SYNOPSIS
    python climalg_ssmis.py <sat> <yyyy> <jday_start> <jday_end> [--res 2.5|1.0]

EXAMPLE
    python climalg_ssmis.py f16 2012 001 031 --res 2.5
    python climalg_ssmis.py f16 2012 001 031 --res 1.0

NOTES
    - Input grid: 1/3-degree, 1080 cols x 540 rows
      * Row 1 = 89.667°N (north pole), going south
      * Col 1 = 180°W (date line), going east
    - Output grid (2.5-deg): 144 cols x 72 rows (GrADS format, S -> N, 1.25°E -> 358.75°E)
    - Output grid (1.0-deg): 360 cols x 180 rows (GrADS format, S -> N, 0.5°E -> 359.5°E)
    - Uses numpy vectorized operations throughout - no pixel-level Python loops.

AUTHORS
    Ralph Ferraro (NOAA/NESDIS/STAR), Fuzhong Weng, Wanchun Chen,
    Daniel Vila (UMD/EESIC), Hilawe Semunegus (NOAA/NCEI)
    Python conversion: 2024
"""

import sys
import os
import argparse
import calendar
import numpy as np
from scipy.ndimage import maximum_filter1d

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
NCOL = 1080      # input 1/3-degree columns
NROW = 540       # input 1/3-degree rows
NT = 13          # number of accumulator channels

# Default input path for gridded TDR files (1/3-degree binary, ascending/descending half-passes).
# Override at runtime with --inpath command-line argument to support test runs in non-standard
# working directories without modifying source code.
INPATH = '../SSMIS_Grid/'

# Channel names (0-indexed): TV19,TH19,TV22,TV37,TH37,TV85,TH85
# Stored in TA and TB arrays index 0..6

# ---------------------------------------------------------------------------
# TA -> TB conversion constants
# ---------------------------------------------------------------------------
AP = np.array([0.969, 0.969, 0.974, 0.986, 0.986, 0.988, 0.988], dtype=np.float32)
BP = np.array([0.00473, 0.00415, 0.0107, 0.0217, 0.02612, 0.01383, 0.01947], dtype=np.float32)

def _ta2tb_coeffs():
    C = 1.0 / (AP * (1.0 - BP))
    D = C * BP
    return C, D

C_TA2TB, D_TA2TB = _ta2tb_coeffs()


def ta2tb(ta):
    """
    Convert antenna temperature to brightness temperature.
    ta: ndarray (..., 7)
    returns tb: ndarray (..., 7)
    Channels: 0=TV19, 1=TH19, 2=TV22, 3=TV37, 4=TH37, 5=TV85, 6=TH85
    """
    tb = np.empty_like(ta)
    tb[..., 0] = C_TA2TB[0] * ta[..., 0] - D_TA2TB[0] * ta[..., 1]
    tb[..., 1] = C_TA2TB[1] * ta[..., 1] - D_TA2TB[1] * ta[..., 0]
    tb[..., 2] = C_TA2TB[2] * ta[..., 2] - D_TA2TB[2] * (0.653 * ta[..., 1] + 96.6)
    tb[..., 3] = C_TA2TB[3] * ta[..., 3] - D_TA2TB[3] * ta[..., 4]
    tb[..., 4] = C_TA2TB[4] * ta[..., 4] - D_TA2TB[4] * ta[..., 3]
    tb[..., 5] = C_TA2TB[5] * ta[..., 5] - D_TA2TB[5] * ta[..., 6]
    tb[..., 6] = C_TA2TB[6] * ta[..., 6] - D_TA2TB[6] * ta[..., 5]
    return tb


# ---------------------------------------------------------------------------
# Retrieval algorithms (fully vectorized)
# ---------------------------------------------------------------------------

def precip8(ta, tb, lsrain, isnchk, add_si):
    """
    Blended 85 GHz scattering/emission rainfall algorithm.
    ta, tb: (..., 7) antenna and brightness temperatures
    lsrain: (...) bool array - coastal flag (True=near land)
    isnchk: (...) bool array - snow check flag
    add_si: (149, 2) scattering index bias lookup table
    Returns: rain1 (...) in mm/day, -999.99 where invalid
    """
    RT = 285.0
    precip = np.full(ta.shape[:-1], -999.99, dtype=np.float32)

    # Check required channels valid (not missing=102)
    ch_valid = ((ta[..., 0] != 102.0) & (ta[..., 1] != 102.0) &
                (ta[..., 2] != 102.0) & (ta[..., 3] != 102.0) &
                (ta[..., 5] != 102.0))

    ch1, ch2, ch3, ch4, ch5, ch6, ch7 = (ta[..., i] for i in range(7))
    # PRECIP8 uses TA (not TB)
    TTT = 44.0 + 0.85 * ch1
    TT  = 168.0 + 0.49 * ch6

    # ---- OCEAN ----
    sct_oce = -182.7 + 0.75*ch1 + 2.543*ch3 - 0.00543*ch3*ch3 - ch6
    ki_oce = np.clip(np.round(sct_oce).astype(np.int32), 0, 148)  # 0-indexed into 149
    sct_oce_corr = sct_oce - add_si[ki_oce, 0]

    # Scattering component over ocean
    rain_sct_oce = np.where(
        sct_oce_corr >= 10.0,
        0.00188 * np.maximum(sct_oce_corr, 0.0) ** 2.03434,
        0.0
    )
    rain_sct_oce = np.where(ch3 <= TTT, 0.0, rain_sct_oce)
    rain_sct_oce = np.where((ch3 > 257.0) & ((ch3 - ch1) < 2.0), 0.0, rain_sct_oce)

    # Emission component over ocean (used when sct < 10)
    q19_valid = (ch1 < RT) & (ch3 < RT)
    q19 = np.where(q19_valid, -2.70 * (np.log(np.maximum(290.0 - ch1, 1e-6)) - 2.80
                                        - 0.42 * np.log(np.maximum(290.0 - ch3, 1e-6))), 0.0)
    rain_em_q19 = np.where(q19 >= 0.60,
                           0.001707 * np.maximum(q19 * 100.0, 0.0) ** 1.7359, 0.0)

    q37_valid = (ch4 < RT) & (ch3 < RT)
    q37 = np.where(q37_valid, -1.15 * (np.log(np.maximum(290.0 - ch4, 1e-6)) - 2.90
                                        - 0.349 * np.log(np.maximum(290.0 - ch3, 1e-6))), 0.0)
    rain_em_q37 = np.where(q37 >= 0.20,
                           0.001707 * np.maximum(q37 * 100.0, 0.0) ** 1.7359, 0.0)

    # Select emission estimate
    rain_em_oce = np.where(q19 >= 0.60, rain_em_q19, rain_em_q37)
    rain_em_oce = np.where(ch3 <= TTT, 0.0, rain_em_oce)

    # Ocean precip: scattering if SCT>=10 else emission
    rain_oce = np.where(sct_oce_corr >= 10.0, rain_sct_oce, rain_em_oce)
    rain_oce = 1.25 * rain_oce  # beam filling correction

    # ---- LAND ----
    sct_lnd = 438.5 - 0.46*ch1 - 1.735*ch3 + 0.00589*ch3*ch3 - ch6
    ki_lnd = np.clip(np.round(sct_lnd).astype(np.int32), 0, 148)
    sct_lnd_corr = sct_lnd - add_si[ki_lnd, 1]

    rain_lnd = np.where(
        sct_lnd_corr >= 10.0,
        0.00513 * np.maximum(sct_lnd_corr, 0.0) ** 1.9468,
        0.0
    )
    rain_lnd = np.where((ch1 - ch2) > 20.0, 0.0, rain_lnd)
    # Snow check
    snow_check = isnchk & (sct_lnd_corr <= 60.0) & ((ch3 >= 257.0) & (ch3 <= 261.0)) & (ch3 < TT)
    rain_lnd = np.where(snow_check, 0.0, rain_lnd)
    rain_lnd = np.where((ch3 < 257.0) & (ch3 < TT) & (sct_lnd_corr >= 10.0), 0.0, rain_lnd)
    rain_lnd = np.where((ch6 > 250.0) & ((ch1 - ch2) > 7.0), 0.0, rain_lnd)

    # Combine ocean/land
    is_land = lsrain  # coastal or land: use land algorithm
    rain_combined = np.where(is_land, rain_lnd, rain_oce)
    rain_combined = np.clip(rain_combined, -999.99, 35.0)
    rain_combined = np.where(rain_combined > 35.0, 35.0, rain_combined)

    # Apply valid mask
    precip = np.where(ch_valid, rain_combined, -999.99)
    return precip.astype(np.float32)


def precip3(ta, lsrain):
    """
    37 GHz scattering (land) / emission (ocean) rainfall algorithm.
    ta: (..., 7) antenna temperatures
    lsrain: (...) bool coastal flag
    Returns: rain2 (...) in mm/day, -999.99 where invalid
    """
    RT = 285.0
    ch1, ch2, ch3, ch4 = ta[..., 0], ta[..., 1], ta[..., 2], ta[..., 3]

    ch_valid = ((ta[..., 0] != 102.0) & (ta[..., 1] != 102.0) &
                (ta[..., 2] != 102.0) & (ta[..., 3] != 102.0))

    TTT = 44.0 + 0.85 * ch1

    # Ocean: emission only
    q19_valid = (ch1 < RT) & (ch3 < RT)
    q19 = np.where(q19_valid, -2.70*(np.log(np.maximum(290.0-ch1, 1e-6)) - 2.80
                                      - 0.42*np.log(np.maximum(290.0-ch3, 1e-6))), 0.0)
    rain_em_q19 = np.where(q19 >= 0.60,
                           0.001707 * np.maximum(q19 * 100.0, 0.0) ** 1.7359, 0.0)

    q37_valid = (ch4 < RT) & (ch3 < RT)
    q37 = np.where(q37_valid, -1.15*(np.log(np.maximum(290.0-ch4, 1e-6)) - 2.90
                                      - 0.349*np.log(np.maximum(290.0-ch3, 1e-6))), 0.0)
    rain_em_q37 = np.where(q37 >= 0.20,
                           0.001707 * np.maximum(q37 * 100.0, 0.0) ** 1.7359, 0.0)

    rain_oce = np.where(q19 >= 0.60, rain_em_q19, rain_em_q37)
    rain_oce = np.where(ch3 <= TTT, 0.0, rain_oce)
    rain_oce = 1.25 * rain_oce

    # Land: 37 GHz scattering
    sct_lnd = 62.18 + 0.773*ch1 - ch4
    rain_lnd = np.where(sct_lnd >= 5.0, 1.3 + 1.46*sct_lnd, 0.0)
    rain_lnd = np.where(ch3 <= 261.0, 0.0, rain_lnd)
    rain_lnd = np.where((ch1 - ch2) > 20.0, 0.0, rain_lnd)
    rain_lnd = np.where((ch4 > 250.0) & ((ch1 - ch2) > 7.0), 0.0, rain_lnd)

    rain_combined = np.where(lsrain, rain_lnd, rain_oce)
    rain_combined = np.where(rain_combined > 35.0, 35.0, rain_combined)
    return np.where(ch_valid, rain_combined, np.float32(-999.99)).astype(np.float32)


def snowc(ta, is_land):
    """
    Snow cover algorithm. is_land: bool array (...).
    Returns: snow (...) = 100.0 for snow, 0.0 for no snow, -999.99 for invalid/ocean.
    """
    ch_valid = ((ta[..., 0] != 102.0) & (ta[..., 1] != 102.0) &
                (ta[..., 2] != 102.0) & (ta[..., 3] != 102.0) &
                (ta[..., 5] != 102.0))
    ch1, ch2, ch3, ch4, _, ch6, _ = (ta[..., i] for i in range(7))

    SCAT = ch3 - ch6
    SC37 = ch1 - ch4
    PD19 = ch1 - ch2
    SCX  = ch4 - ch6
    SCAT_eff = np.maximum(SCAT, SC37)
    TT = 165.0 + 0.49 * ch6

    snow = np.full(ta.shape[:-1], -999.99, dtype=np.float32)

    scat_pos = SCAT_eff > 0.0
    has_snow = scat_pos.copy()
    has_snow = np.where((ch3 >= 254.0) & (SCAT_eff <= 2.0), False, has_snow)
    has_snow = np.where((ch3 >= 258.0) | (ch3 >= TT), False, has_snow)
    has_snow = np.where((PD19 >= 18.0) & (SC37 <= 10.0) & (SCX <= 10.0), False, has_snow)
    has_snow = np.where((SCAT_eff <= 6.0) & (PD19 >= 8.0), False, has_snow)

    snow_val = np.where(has_snow, 100.0, 0.0)
    # Glacial ice check
    snow_val = np.where((ch3 <= 210.0) | ((ch3 <= 229.0) & (PD19 >= 23.0)), 100.0, snow_val)

    snow = np.where(is_land & ch_valid, snow_val, -999.99)
    return snow.astype(np.float32)


def seaice(ta, is_ocean, ichan):
    """
    Sea ice algorithm. is_ocean: bool array (...).
    ichan: int (3 or 8) flag for 85GHz availability.
    Returns: sice (...) = 100.0 for ice, 0.0 for no ice, -999.99 for invalid/land.
    """
    # Uses TA channels
    ch_valid = ((ta[..., 0] != 102.0) & (ta[..., 1] != 102.0) &
                (ta[..., 2] != 102.0) & (ta[..., 3] != 102.0) &
                (ta[..., 4] != 102.0))
    tv19, th19, tv22, tv37, th37, tv85, th85 = (ta[..., i] for i in range(7))

    if ichan == 8:
        ch85_valid = ta[..., 5] != 102.0
        sice_val = np.where(ch85_valid,
                            91.9 - 2.994*tv22 + 2.846*tv19 - 0.386*tv37 + 0.495*tv85 + 1.005*th19 - 0.904*th37,
                            ((36.4 + tv19 - 0.788*tv22) - 60.0) / 0.35)
    else:
        sice_val = ((36.4 + tv19 - 0.788*tv22) - 60.0) / 0.35

    sice_binary = np.where(sice_val >= 70.0, 100.0, 0.0)
    return np.where(is_ocean & ch_valid, sice_binary, np.float32(-999.99)).astype(np.float32)


def cloud(ta, tb, is_ocean, sice):
    """
    Cloud liquid water path (Weng et al.).
    ta: antenna temps (...,7), tb: brightness temps (...,7)
    is_ocean: bool (...), sice: (...) sea ice array
    Returns: lwp (...) in mm (kg/m²), -999.99 where invalid
    """
    ch_valid = ((ta[..., 0] != 102.0) & (ta[..., 2] != 102.0) &
                (ta[..., 3] != 102.0) & (ta[..., 6] != 102.0))
    # Uses TA for retrieval but TB for some intermediate
    tv19_a, _, tv22_a, tv37_a, _, _, th85_a = (ta[..., i] for i in range(7))
    tv19_b, _, tv22_b, tv37_b, _, _, th85_b = (tb[..., i] for i in range(7))

    no_ice = sice != 100.0
    mask = is_ocean & no_ice & ch_valid

    # WVP intermediate for cloud discriminator (using TB)
    rwvp = 232.89 - 0.1486*tv19_b - 0.3695*tv37_b - (1.8291 - 0.006193*tv22_b)*tv22_b

    RT = 285.0
    alg1 = np.where((tv19_a < RT) & (tv22_a < RT),
                    -3.20*(np.log(np.maximum(290.0-tv19_a, 1e-6)) - 2.80
                           - 0.42*np.log(np.maximum(290.0-tv22_a, 1e-6))),
                    -999.99)
    alg2 = np.where((tv37_a < RT) & (tv22_a < RT),
                    -1.66*(np.log(np.maximum(290.0-tv37_a, 1e-6)) - 2.90
                           - 0.349*np.log(np.maximum(290.0-tv22_a, 1e-6))),
                    -999.99)
    alg3 = np.where((th85_a < RT) & (tv22_a < RT),
                    -0.44*(np.log(np.maximum(290.0-th85_a, 1e-6)) + 1.60
                           - 1.354*np.log(np.maximum(290.0-tv22_a, 1e-6))),
                    -999.99)

    lwp = np.where(alg1 > 0.70, alg1,
           np.where(alg2 > 0.28, alg2,
           np.where(rwvp < 30.0, alg3, alg2)))
    lwp = np.where(lwp > 6.0, 0.0, lwp)
    return np.where(mask, lwp, np.float32(-999.99)).astype(np.float32)


def vapor(ta, tb, is_ocean, sice, add_si):
    """
    Total precipitable water (Alishouse).
    Uses TB for the retrieval, TA for the scattering index check.
    Returns: wvp (...) in mm, -999.99 where invalid
    """
    ch_valid = ((ta[..., 0] != 102.0) & (ta[..., 2] != 102.0) &
                (ta[..., 3] != 102.0) & (ta[..., 5] != 102.0))
    tv19_b, _, tv22_b, tv37_b, _, tv85_b, _ = (tb[..., i] for i in range(7))
    tv19_a, _, tv22_a, _, _, tv85_a, _ = (ta[..., i] for i in range(7))

    # Scattering index (using TA)
    sct = -182.7 + 0.75*tv19_a + 2.543*tv22_a - 0.00543*tv22_a*tv22_a - tv85_a

    no_ice = sice != 100.0
    ocean_only = is_ocean & no_ice & (sct <= 10.0) & ch_valid

    wvp_raw = 232.89 - 0.1486*tv19_b - 0.3695*tv37_b - (1.8291 - 0.006193*tv22_b)*tv22_b
    wvp = -3.753 + 1.507*wvp_raw - 0.01933*wvp_raw**2 + 0.0002191*wvp_raw**3
    wvp = np.where(wvp < 0.0, 0.0, wvp)
    wvp = np.where(wvp > 100.0, 0.0, wvp)
    return np.where(ocean_only, wvp, np.float32(-999.99)).astype(np.float32)


# ---------------------------------------------------------------------------
# Output compositing functions
# ---------------------------------------------------------------------------

def mean_precip(rrmon, M, N, NT, ndays, iflg, ichan):
    """
    Compute monthly mean precipitation field.
    rrmon: (M, N, NT) accumulator array
    Returns xp: (N, M) float32 in GrADS output layout
    """
    SNOWT = 99.0
    iyoff = 0 if ichan == 8 else 2  # channel offset (85GHz=0, 37GHz=2)

    land_pix = rrmon[:, :, NT-2]  # index 11 (NT-2): total land pixels
    oce_pix  = rrmon[:, :, NT-1]  # index 12 (NT-1): total ocean pixels
    tpix = land_pix + oce_pix
    ratio = np.where(tpix > 0, land_pix / tpix, -1.0)

    acc  = rrmon[:, :, iyoff]      # rain accumulation
    freq = rrmon[:, :, iyoff + 1]  # rain frequency

    z = np.full((M, N), -999.99, dtype=np.float32)
    valid = tpix > 0
    if iflg == 1:
        z = np.where(valid, ndays * 24.0 * acc / tpix, z)
    elif iflg == 2:
        z = np.where(valid, freq / tpix, z)

    # Remove sea-ice pixels (ocean-dominated, >10% ice)
    ice_frac = np.where(oce_pix > 0, 100.0 * rrmon[:, :, 9] / oce_pix, 0.0)
    z = np.where((oce_pix > 0) & (ratio > -1.0) & (ratio <= 0.05) & (ice_frac > 10.0), 0.0, z)

    if ichan == 8:
        # Remove snow pixels (land-dominated, >99% snow)
        snow_frac = np.where(land_pix > 0, 100.0 * rrmon[:, :, 8] / land_pix, 0.0)
        z = np.where((land_pix > 0) & (ratio >= 0.95) & (snow_frac > SNOWT), 0.0, z)

    return to_grads(z, M, N)


def mean_lwp(rrmon, M, N, NT, iflg):
    """Cloud liquid water path or cloud fraction."""
    land_pix = rrmon[:, :, NT-2]
    oce_pix  = rrmon[:, :, NT-1]
    tpix = land_pix + oce_pix
    ratio = np.where(tpix > 0, land_pix / tpix, 1.0)

    cf_acc  = rrmon[:, :, 4]   # LWP accumulation
    cf_freq = rrmon[:, :, 5]   # cloud frequency

    z = np.full((M, N), -999.99, dtype=np.float32)
    ocean_dom = (cf_freq > 0) & (ratio <= 0.05)
    if iflg == 1:
        # LWP: mean cloud liquid water path in g/m² (stored as 1000×mm).
        # Fortran MEANLWP: Z = 1000*Y(5)/Y(6)  - lwp_sum / cloud_obs_count
        z = np.where(ocean_dom, 1000.0 * cf_acc / cf_freq, z)
    elif iflg == 2:
        # CFR: cloud fraction = cloud_obs / ocean_pixels.
        # Fortran MEANLWP: Z = Y(6)/Y(NT)  where Y(NT) = ocean pixels, NOT total pixels.
        # BUG FIX: was cf_freq / tpix (total pixels); corrected to cf_freq / oce_pix.
        z = np.where(ocean_dom & (oce_pix > 0), cf_freq / oce_pix, z)

    # Remove sea-ice pixels: zero out ocean-dominated, cloud-observed cells where sea-ice > 10%.
    # Fortran MEANLWP:
    #   IF(Y(6).GT.0) Z = Y(6)/Y(NT)      ! only set Z valid when cloud obs exist
    #   IF(Y(NT).GT.0.AND.RATIO.LE.0.05.AND.RATIO.GT.-1.0) THEN
    #     IF(SEAICE.GT.10.0) Z=0.0         ! only zeroes cells that already had a valid Z
    # BUG FIX 1: previously lacked RATIO <= 0.05 and RATIO > -1.0 guards - caused
    #   ~1462 spurious CFR=0.0 values in land-adjacent cells.
    # BUG FIX 2: must also require cf_freq > 0 (i.e., ocean_dom) so that cells with
    #   Z=-999.99 (no cloud obs) are NOT converted to 0.0 by ice masking - those 654
    #   cells have oce_pix>0 and are ocean-dominated but have zero cloud observations;
    #   Fortran leaves them as -999.99, but without this guard Python sets them to 0.0.
    ice_frac = np.where(oce_pix > 0, 100.0 * rrmon[:, :, 9] / oce_pix, 0.0)
    z = np.where(ocean_dom & (oce_pix > 0) & (ratio > -1.0) & (ratio <= 0.05) & (ice_frac > 10.0), 0.0, z)

    return to_grads(z, M, N)


def mean_wvp(rrmon, M, N, NT):
    """Total precipitable water."""
    land_pix = rrmon[:, :, NT-2]
    oce_pix  = rrmon[:, :, NT-1]
    tpix = land_pix + oce_pix
    ratio = np.where(tpix > 0, land_pix / tpix, 1.0)

    wvp_acc  = rrmon[:, :, 6]
    wvp_freq = rrmon[:, :, 7]

    z = np.full((M, N), -999.99, dtype=np.float32)
    ocean_dom = (wvp_freq > 0) & (ratio <= 0.05)
    z = np.where(ocean_dom, wvp_acc / wvp_freq, z)

    # Remove sea-ice pixels: same Fortran condition as MEANLWP -
    # IF(Y(8).GT.0) Z = Y(7)/Y(8)          ! only valid where wvp observations exist
    # IF(Y(NT).GT.0.AND.RATIO.LE.0.05.AND.RATIO.GT.-1.0) IF(SEAICE.GT.10) Z=0
    # BUG FIX 1: previously lacked RATIO <= 0.05 guard - caused spurious WVP zeros.
    # BUG FIX 2: must also require ocean_dom (wvp_freq > 0) so cells with Z=-999.99
    #   (no wvp obs but ocean present) are not spuriously set to 0.0 by ice masking.
    ice_frac = np.where(oce_pix > 0, 100.0 * rrmon[:, :, 9] / oce_pix, 0.0)
    z = np.where(ocean_dom & (oce_pix > 0) & (ratio > -1.0) & (ratio <= 0.05) & (ice_frac > 10.0), 0.0, z)

    return to_grads(z, M, N)


def mean_snw(rrmon, M, N, NT):
    """Snow cover fraction."""
    land_pix = rrmon[:, :, NT-2]
    oce_pix  = rrmon[:, :, NT-1]
    tpix = land_pix + oce_pix
    ratio = np.where(tpix > 0, land_pix / tpix, -1.0)

    z = np.full((M, N), -999.99, dtype=np.float32)
    land_dom = (land_pix > 0) & (ratio >= 0.95)
    z = np.where(land_dom, rrmon[:, :, 8] / land_pix, z)

    return to_grads(z, M, N)


def mean_ice(rrmon, M, N, NT):
    """Sea ice fraction."""
    land_pix = rrmon[:, :, NT-2]
    oce_pix  = rrmon[:, :, NT-1]
    tpix = land_pix + oce_pix
    ratio = np.where(tpix > 0, land_pix / tpix, 1.0)

    z = np.full((M, N), -999.99, dtype=np.float32)
    ocean_dom = (oce_pix > 0) & (ratio <= 0.05)
    z = np.where(ocean_dom, 100.0 * rrmon[:, :, 9] / oce_pix, z)

    return to_grads(z, M, N)


def mean_ssa(rrmon, M, N, NT):
    """Sampling fraction (SSA)."""
    total_pix = rrmon[:, :, NT-3]   # index 10 = NT-3: total 1/3 pixels
    land_pix  = rrmon[:, :, NT-2]
    oce_pix   = rrmon[:, :, NT-1]
    tpix = land_pix + oce_pix

    z = np.full((M, N), -999.99, dtype=np.float32)
    z = np.where(total_pix > 0, tpix / total_pix, z)

    return to_grads(z, M, N)


def to_grads(z, M, N):
    """
    Rearrange (M, N) array [north-first, 180W-first] into
    GrADS binary layout: (N, M) [south-first, 1.25E-first].

    Step 1: Flip latitude south-first: z[M-1..0, :]
    Step 2: Roll longitude so that col 0 = 1.25°E (was at position N//2):
            np.roll(..., -N//2, axis=1) moves col N//2 to position 0.
    Step 3: Transpose to (N, M) for Fortran column-major output
    """
    # z is (M, N): row=lat(north-first), col=lon(180W-first)
    # Flip lat to south-first
    z_flip = z[::-1, :]
    # Roll longitude: col N//2 (=0°E) -> col 0; (GrADS XDEF starts near 0°E = 1.25°E)
    half = N // 2
    z_rolled = np.roll(z_flip, -half, axis=1)
    # Transpose for Fortran-style (lon-first in memory) output
    xp = z_rolled.T.astype(np.float32)  # shape (N, M)
    return xp


def write_grads(fname, xp):
    """Write GrADS binary output file (lon-fastest, lat-slowest row-major layout).

    GrADS expects X (lon) to vary fastest, Y (lat) slowest:
        for lat in 0..M-1: write all N lon values
    xp has shape (N, M) = (lons, lats), so xp.T is (M, N) = (lats, lons).
    Writing xp.T in C row-major order gives the correct GrADS binary layout.
    """
    xp.T.tofile(fname)
    print(f'  Written: {fname}')


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def load_luts(coeff_dir='.'):
    """Load lookup tables from text files."""
    # lut_add_ocean: 161 rows, format: f6.2 7(f7.2)
    oce = np.loadtxt(os.path.join(coeff_dir, 'lut_add_ocean'), usecols=range(8))
    lnd = np.loadtxt(os.path.join(coeff_dir, 'lut_add_land'),  usecols=range(8))
    si  = np.loadtxt(os.path.join(coeff_dir, 'lut_add_si'),    usecols=range(3))
    add_oce = oce[:, 1:].astype(np.float32)   # (161, 7)
    add_lnd = lnd[:, 1:].astype(np.float32)   # (161, 7)
    add_si  = si[:, 1:].astype(np.float32)    # (149, 2)
    return add_oce, add_lnd, add_si


def load_land_tag(fname='NLNDSEA.TAG'):
    """
    Load 1/3-degree land/sea tag file.
    Returns: (NROW, NCOL) int8 array (0=ocean, 1=land)
    """
    data = np.fromfile(fname, dtype=np.int8)
    return data.reshape(NROW, NCOL)


def process_file(fpath, tag_2d, add_oce, add_lnd, add_si,
                 rrmon, ii_map, jj_map, M, N, YYYY, JDAY, equat_half_res):
    """
    Process one gridded TDR file (ascending or descending) and accumulate
    into the rrmon accumulator array.

    fpath:          full path to the binary input file
    tag_2d:         (NROW, NCOL) land/sea tag
    add_oce/lnd:    (161, 7) bias correction LUTs
    add_si:         (149, 2) scattering index bias LUT
    rrmon:          (M, N, NT) accumulator (modified in place)
    ii_map:         (NROW,) int, output latitude bin for each input row
    jj_map:         (NCOL,) int, output longitude bin for each input column
    YYYY, JDAY:     year and day-of-year for temporal checks
    equat_half_res: half the equatorial pixel count (for sea-ice lat check)
    """
    # Read entire file: NROW records, each (NCOL*7) bytes
    raw = np.fromfile(fpath, dtype=np.uint8)
    expected = NROW * NCOL * 7
    if raw.size != expected:
        print(f'  Warning: {fpath} size {raw.size} != expected {expected}, skipping')
        return False

    data = raw.reshape(NROW, NCOL, 7)   # (540, 1080, 7)

    # Convert packed bytes to antenna temperature: TA = byte + 70
    ta = data.astype(np.float32) + 70.0  # (540, 1080, 7)

    # Missing value: byte=32 -> TA=102
    # A pixel is skipped if channels 0 and 2,3,4,6 (1-indexed: 1,3,4,5,7) are all 102
    skip = ((ta[:, :, 0] == 102.0) & (ta[:, :, 2] == 102.0) &
            (ta[:, :, 3] == 102.0) & (ta[:, :, 4] == 102.0) &
            (ta[:, :, 6] == 102.0))  # (540, 1080)

    # Apply bias corrections (per channel, per land/ocean)
    # kindex = round(TA) - 139 -> 0-indexed into LUT (161 entries, Fortran 1-161 -> 0-160)
    kindex = np.clip(np.round(ta).astype(np.int32) - 139, 0, 160)  # (540, 1080, 7)
    tag_3d = tag_2d[:, :, np.newaxis]   # (540, 1080, 1)
    is_ocean = (tag_2d == 0)            # (540, 1080) bool
    is_ocean_3d = is_ocean[:, :, np.newaxis]
    is_missing_ch = (ta == 102.0)       # (540, 1080, 7)

    ta_corr = ta.copy()
    for ch in range(7):
        k_ch = kindex[:, :, ch]   # (540, 1080)
        valid_ch = ~is_missing_ch[:, :, ch]
        ta_corr[:, :, ch] = np.where(valid_ch & is_ocean,
                                      ta[:, :, ch] - add_oce[k_ch, ch],
                             np.where(valid_ch & ~is_ocean,
                                      ta[:, :, ch] - add_lnd[k_ch, ch],
                                      ta[:, :, ch]))

    # Compute TB from TA (corrected)
    tb = ta2tb(ta_corr)              # (540, 1080, 7)

    # Coastal flag: 1 if any of the 5 nearest neighbors is land (within same row)
    lsrain = maximum_filter1d(tag_2d.astype(np.int8), size=5, axis=1,
                              mode='constant', cval=0).astype(bool)  # (540, 1080)

    # Latitude of each row (for snow check and equatorial ice suppression)
    rlat = (89.667 - 0.333 * np.arange(NROW)).astype(np.float32)  # (540,)

    # Snow check flag: apply stricter melting-snow test
    isnchk = np.zeros(NROW, dtype=bool)
    if JDAY <= 181:
        isnchk |= (rlat >= 60.0)
        if JDAY <= 151:
            isnchk |= (rlat >= 40.0)
        if JDAY <= 91:
            isnchk |= (rlat >= 25.0)
    isnchk_2d = isnchk[:, np.newaxis]  # (540, 1) broadcast to (540, 1080)

    # Apply all retrieval algorithms
    rain1 = precip8(ta_corr, tb, lsrain, np.broadcast_to(isnchk_2d, (NROW, NCOL)), add_si)
    rain2 = precip3(ta_corr, lsrain)

    is_land_2d = (tag_2d == 1)
    snow  = snowc(ta_corr, is_land_2d)

    ichan = 3 if (YYYY == 1990 and JDAY >= 181) or YYYY == 1991 else 8
    sice  = seaice(ta_corr, is_ocean, ichan)

    # Suppress sea ice near equator (within 40° lat of equator in input row space)
    # |I - NROW/2| * RES <= 40 -> suppress
    row_idx = np.arange(NROW)
    equat_mask = np.abs(row_idx - NROW // 2) * (360.0 / NCOL) <= 40.0
    equat_mask_2d = equat_mask[:, np.newaxis]
    sice = np.where(equat_mask_2d & (sice == 100.0), 0.0, sice)

    lwp  = cloud(ta_corr, tb, is_ocean, sice)
    wvp  = vapor(ta_corr, tb, is_ocean, sice, add_si)

    # Accumulate into output grid using output bin maps
    # ii_map: (NROW,) -> lat bin 0..M-1
    # jj_map: (NCOL,) -> lon bin 0..N-1
    ii_2d = ii_map[:, np.newaxis]   # (540, 1)
    jj_2d = jj_map[np.newaxis, :]   # (1, 1080)

    # Flatten to 1D for add.at
    ii_flat = np.broadcast_to(ii_2d, (NROW, NCOL)).ravel()
    jj_flat = np.broadcast_to(jj_2d, (NROW, NCOL)).ravel()
    valid_flat = (~skip).ravel()

    # Total pixel counter (all pixels, not just valid)
    np.add.at(rrmon[:, :, NT-3], (ii_flat, jj_flat), 1)  # index 10 = total 1/3° pixels

    # For remaining accumulators, only valid (non-skip) pixels
    ii_v = ii_flat[valid_flat]
    jj_v = jj_flat[valid_flat]

    def acc(k, vals):
        """Add vals to rrmon[:,:,k] at (ii_v, jj_v) positions."""
        np.add.at(rrmon[:, :, k], (ii_v, jj_v), vals)

    # Land/ocean pixel counts
    is_ocean_flat = is_ocean.ravel()[valid_flat]
    acc(NT-1, is_ocean_flat.astype(np.float32))    # total ocean pixels (index 12)
    acc(NT-2, (~is_ocean_flat).astype(np.float32)) # total land pixels  (index 11)

    rain1_flat = rain1.ravel()[valid_flat]
    rain2_flat = rain2.ravel()[valid_flat]
    snow_flat  = snow.ravel()[valid_flat]
    sice_flat  = sice.ravel()[valid_flat]
    lwp_flat   = lwp.ravel()[valid_flat]
    wvp_flat   = wvp.ravel()[valid_flat]

    # PR1 accumulation and frequency
    r1_pos = rain1_flat > 0.0
    acc(0, np.where(r1_pos, rain1_flat, 0.0))
    acc(1, r1_pos.astype(np.float32))

    # PR2 accumulation and frequency
    r2_pos = rain2_flat > 0.0
    acc(2, np.where(r2_pos, rain2_flat, 0.0))
    acc(3, r2_pos.astype(np.float32))

    # LWP accumulation and frequency
    lwp_pos = lwp_flat > 0.02
    acc(4, np.where(lwp_pos, lwp_flat, 0.0))
    acc(5, lwp_pos.astype(np.float32))

    # WVP accumulation and frequency
    wvp_pos = wvp_flat > 0.0
    acc(6, np.where(wvp_pos, wvp_flat, 0.0))
    acc(7, wvp_pos.astype(np.float32))

    # Snow count
    acc(8, (snow_flat == 100.0).astype(np.float32))

    # Ice count
    acc(9, (sice_flat == 100.0).astype(np.float32))

    return True


def write_monthly(rrmon, outpath, sat, year_str, mm_str, M, N, NT, ndays):
    """Write all 10 monthly output binary files."""
    products = {
        'PR1': (mean_precip, (rrmon, M, N, NT, ndays, 1, 8)),
        'PF1': (mean_precip, (rrmon, M, N, NT, ndays, 2, 8)),
        'PR2': (mean_precip, (rrmon, M, N, NT, ndays, 1, 3)),
        'PF2': (mean_precip, (rrmon, M, N, NT, ndays, 2, 3)),
        'LWP': (mean_lwp,    (rrmon, M, N, NT, 1)),
        'CFR': (mean_lwp,    (rrmon, M, N, NT, 2)),
        'WVP': (mean_wvp,    (rrmon, M, N, NT)),
        'SNW': (mean_snw,    (rrmon, M, N, NT)),
        'ICE': (mean_ice,    (rrmon, M, N, NT)),
        'SSA': (mean_ssa,    (rrmon, M, N, NT)),
    }
    res_str = f'{360.0/N:.1f}'
    for name, (func, args) in products.items():
        xp = func(*args)
        fname = os.path.join(outpath, f'{name}{year_str[2:4]}-{mm_str}-{sat}-{res_str}')
        write_grads(fname, xp)


def run(sat, yyyy, jday_start, jday_end, res=2.5, outdir=None, inpath=None, coeff_dir=None):
    """
    Main processing loop.

    Parameters
    ----------
    sat        : satellite name string (e.g. 'f17')
    yyyy       : four-digit year integer
    jday_start : start Julian day (1-based day-of-year)
    jday_end   : end Julian day (inclusive)
    res        : output grid resolution in degrees (2.5 or 1.0)
    outdir     : directory under which {sat}-{res}/ output subdirectory is created.
                 Defaults to '.' (current working directory), giving '{sat}-{res}/'.
                 Pass e.g. 'test_mar2026' to write to 'test_mar2026/{sat}-{res}/'.
    inpath     : path to the 1/3-degree gridded TDR input files.
                 Defaults to INPATH module constant ('../SSMIS_Grid/').
    coeff_dir  : directory containing lut_add_ocean, lut_add_land, lut_add_si, NLNDSEA.TAG.
                 Defaults to '.' (current working directory).
    """
    if res == 2.5:
        M, N, SIZE = 72, 144, 2.5
    elif res == 1.0:
        M, N, SIZE = 180, 360, 1.0
    else:
        raise ValueError(f'Unsupported resolution {res}')

    # Resolve optional path overrides - allows test runs in non-standard directories
    # without hard-coded path assumptions in the source code.
    _inpath    = inpath    if inpath    is not None else INPATH
    _coeff_dir = coeff_dir if coeff_dir is not None else '.'
    _tag_file  = os.path.join(_coeff_dir, 'NLNDSEA.TAG')

    year_str = f'{yyyy:04d}'
    # Build the output path: if outdir is given, create {outdir}/{sat}-{res:.1f}/;
    # otherwise default to {sat}-{res:.1f}/ in the current working directory.
    if outdir is not None:
        outpath = os.path.join(outdir, f'{sat}-{res:.1f}/')
    else:
        outpath = f'{sat}-{res:.1f}/'
    os.makedirs(outpath, exist_ok=True)

    print(f'climalg_ssmis: sat={sat}, year={yyyy}, jdays={jday_start}-{jday_end}, res={res}°')
    print(f'  inpath={_inpath}  coeff={_coeff_dir}  outpath={outpath}')

    # Load auxiliary data (LUTs and land/sea tag) from the coefficient directory
    add_oce, add_lnd, add_si = load_luts(_coeff_dir)
    tag_2d = load_land_tag(_tag_file)

    # Build output bin maps.
    # Use 1-indexed rows/cols - INT((I)*RES/SIZE) where I=1..NROW - to match the Fortran
    # climalg-ssmis-2.5deg.f / climalg-ssmis-1.0deg.f formula:
    #   II = INT(I*RES/SIZE) + 1  (Fortran 1-indexed)
    # which in 0-indexed Python is: floor((row+1)*RES/SIZE).
    # Using 0-indexed (np.arange(NROW) * RES/SIZE) shifts bin boundaries:
    #   at 1.0°: 33% of rows go to wrong bin; at 2.5°: 13% of rows.
    RES = 360.0 / NCOL
    ii_map = np.minimum(((np.arange(NROW) + 1) * RES / SIZE).astype(int), M - 1)  # lat bin per row
    jj_map = np.minimum(((np.arange(NCOL) + 1) * RES / SIZE).astype(int), N - 1)  # lon bin per col

    # Build month boundary lookup (Julian days since Jan 1)
    def julday_offset(m, yyyy):
        """Days from Jan 1 to start of month m (1-12) in year yyyy."""
        return sum(calendar.monthrange(yyyy, mo)[1] for mo in range(1, m))

    mon_boundaries = [julday_offset(m, yyyy) for m in range(1, 13)] + \
                     [sum(calendar.monthrange(yyyy, m)[1] for m in range(1, 13))]

    rrmon = np.zeros((M, N, NT), dtype=np.float32)
    imons = 0  # accumulation count

    prev_month = -1

    for jday in range(jday_start, jday_end + 1):
        # Determine month for this jday (jday is 1-based day of year)
        day_of_year = jday  # 1-indexed
        cum = 0
        curr_month = 12
        for mo in range(1, 13):
            cum += calendar.monthrange(yyyy, mo)[1]
            if day_of_year <= cum:
                curr_month = mo
                break

        # Check for month boundary -> output and reset
        if prev_month != -1 and curr_month != prev_month:
            if imons > 0:
                mm_str = f'{prev_month:02d}'
                ndays = calendar.monthrange(yyyy, prev_month)[1]
                print(f'\nWriting month {mm_str} ({ndays} days, {imons} half-days accumulated)')
                write_monthly(rrmon, outpath, sat, year_str, mm_str, M, N, NT, ndays)
            rrmon[:] = 0
            imons = 0

        prev_month = curr_month
        jday_str = f'{jday:03d}'

        for prefix in ('as', 'ds'):
            fname = f'{prefix}{year_str[2:4]}{jday_str}.{sat}'
            # Use the resolved _inpath (may be overridden via --inpath argument)
            fpath = os.path.join(_inpath, fname)
            if os.path.exists(fpath):
                print(f'  Processing {fname}')
                ok = process_file(fpath, tag_2d, add_oce, add_lnd, add_si,
                                  rrmon, ii_map, jj_map, M, N, yyyy, jday,
                                  NROW // 2)
                if ok:
                    imons += 1
            else:
                print(f'  Warning! Missing file: {fpath}')

        # Write at end of requested range if we're at a month boundary or last day
        if jday == jday_end and imons > 0:
            mm_str = f'{curr_month:02d}'
            ndays = calendar.monthrange(yyyy, curr_month)[1]
            print(f'\nWriting month {mm_str} ({ndays} days, {imons} half-days accumulated)')
            write_monthly(rrmon, outpath, sat, year_str, mm_str, M, N, NT, ndays)
            rrmon[:] = 0
            imons = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SSMIS monthly climate algorithm')
    parser.add_argument('sat',        help='Satellite name (e.g. f16, f17, f18)')
    parser.add_argument('yyyy',       type=int, help='Year (e.g. 2012)')
    parser.add_argument('jday_start', type=int, help='Start Julian day (e.g. 001)')
    parser.add_argument('jday_end',   type=int, help='End Julian day (e.g. 031)')
    parser.add_argument('--res',      type=float, default=2.5,
                        choices=[2.5, 1.0], help='Output resolution in degrees (default: 2.5)')
    # Optional path overrides - primarily for test/validation runs where the working
    # directory is not the standard monthly/ directory.
    parser.add_argument('--outdir',   default=None,
                        help='Parent directory for {sat}-{res}/ output (default: current dir)')
    parser.add_argument('--inpath',   default=None,
                        help='Path to 1/3-degree TDR input files (default: ../SSMIS_Grid/)')
    parser.add_argument('--coeff',    default=None,
                        help='Directory containing LUT and NLNDSEA.TAG files (default: .)')
    args = parser.parse_args()
    run(args.sat, args.yyyy, args.jday_start, args.jday_end, args.res,
        outdir=args.outdir, inpath=args.inpath, coeff_dir=args.coeff)


if __name__ == '__main__':
    main()
