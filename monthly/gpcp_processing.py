#!/usr/bin/env python3
"""
gpcp_processing.py

PURPOSE
    Generates GPCP (Global Precipitation Climatology Project) datasets by
    applying snow/ice masks and producing dual-satellite merged products.
    Replaces gpcp_f08.pro, gpcp_f10.pro, and gpcp_f08_f10_dual.pro (IDL).

SYNOPSIS
    python gpcp_processing.py

NOTES
    - gpcp_f08 (IDL) -> gpcp_late()   : f17-2.5 rain, masked with snow/ice
    - gpcp_f10 (IDL) -> gpcp_early()  : f16-2.5 rain, masked with snow/ice
    - gpcp_f08_f10_dual (IDL) -> gpcp_dual() : merge f17+f16 weighted by sampling

    NAMES. gpcp_late() and gpcp_early() are named after the identifiers of
    satellite_config (LATE_CONSTELLATION_SAT, EARLY_CONSTELLATION_SAT), which hold
    the opposite chains from their words and are kept that way on purpose. They are
    NOT the labels of the product files. The files these functions write carry no
    label, and the labels are applied by netcdf/generate_netcdf.py as follows.

      function       satellite today   chain          binary written           product label
      gpcp_late()    F-17              morning        gpcp_nesdis_pr?.dat      _early_
      gpcp_early()   F-16              late morning   gpcp_nesdis_f10_pr?.dat  _late_
      gpcp_dual()    F-17 with F-16    both           gpcp_nesdis_dual_*.dat   _dual_

    Between 14 July and 18 November 2026 the netCDF writers applied the two
    labels the other way round, on the belief that "late means F-17" was a GPCP
    convention. No such convention exists. The legacy writers and every archived
    file label the F-08 to F-17 chain early and the F-10 to F-16 chain late.
    - Grid: 144 × 72 (2.5°), GrADS format (N cols × M rows per month)
    - The binary files are sequential: jmon months of (144,72) float32 grids.
    - Initial year for f17 (f08 IDL): 1987; for f16 (f10 IDL): 1992
    - Snow climatology files: snow/SN<MM> (144×72 float32), read for July 1990 to
      December 1991 of the morning chain on every run. They are an input kept with
      the data and are not distributed with the code.
    - Land/sea tag:  lndsea.25d (144×72 float32, whole numbers from -1 to 99,
      compared with 25 and 75 by the coastal masks, which read it as percent land)
"""

import os
import sys
import datetime
import numpy as np

# Import satellite start years from the central config so that when the late
# constellation transitions from F-17 to WSF-M (or early from F-16 to F-18),
# only satellite_config.py needs to change.
try:
    from satellite_config import (late_start_year, early_start_year,
                                  LATE_CONSTELLATION_SAT, EARLY_CONSTELLATION_SAT)
    _LATE_INIT_YEAR  = late_start_year()   # 1987 for F-17 record
    _EARLY_INIT_YEAR = early_start_year()  # 1992 for F-16 record
    _LATE_SAT  = LATE_CONSTELLATION_SAT    # morning chain primary (e.g. 'f17')
    _EARLY_SAT = EARLY_CONSTELLATION_SAT   # late-morning chain primary (e.g. 'f16')
except Exception:
    # Fallback if satellite_config is not in path (e.g., test environments)
    _LATE_INIT_YEAR  = 1987
    _EARLY_INIT_YEAR = 1992
    _LATE_SAT  = 'f17'
    _EARLY_SAT = 'f16'

N_LON = 144
N_LAT = 72
GRID_SIZE = N_LON * N_LAT
MISSING = np.float32(-999.99)

# These aliases preserve backward compatibility for any code that imports them
# directly.  gpcp_late() and gpcp_early() use _LATE_INIT_YEAR / _EARLY_INIT_YEAR.
INIT_YEAR_F17 = _LATE_INIT_YEAR    # corresponds to gpcp_f08 (late constellation)
INIT_YEAR_F16 = _EARLY_INIT_YEAR   # corresponds to gpcp_f10 (early constellation)

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _current_year_month():
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.year, now.month


def _months_since(init_year):
    """Count months in the record up through (but not including) current month."""
    cy, cm = _current_year_month()
    n = (cy - init_year + 1) * 12
    if cm == 1:
        n -= 12
    return n


def read_lndsea(fname='lndsea.25d'):
    """Read land/sea tag file (144×72 float32)."""
    data = np.fromfile(fname, dtype=np.float32)
    if data.size != GRID_SIZE:
        raise RuntimeError(f'lndsea.25d: expected {GRID_SIZE} values, got {data.size}')
    return data.reshape(N_LON, N_LAT)   # (144, 72) GrADS layout: lon-major


def read_snow_clim(month):
    """Read monthly snow climatology file: snow/SN<MM>."""
    mm = f'{month:02d}'
    fname = os.path.join('snow', f'SN{mm}')
    data = np.fromfile(fname, dtype=np.float32)
    if data.size != GRID_SIZE:
        raise RuntimeError(f'{fname}: expected {GRID_SIZE} values, got {data.size}')
    return data.reshape(N_LON, N_LAT)


def read_month_slice(fobj, month_idx):
    """Read one month's grid (N_LON × N_LAT float32) from an open file.

    Reads SEQUENTIALLY from the handle's current position; month_idx is
    informational (callers rely on one read per loop iteration for alignment).
    DEFINED short-read behavior: at or past EOF (including a truncated final
    month) this returns a full MISSING (-999.99) grid rather than raising, and
    every subsequent read does the same, so a consumer iterating past a short
    file degrades to fill months. gpcp_dual() depends on this to fall back to
    the F-17-only estimate when an early-chain binary is shorter than the late
    allocation.
    """
    raw = np.fromfile(fobj, dtype=np.float32, count=GRID_SIZE)
    if raw.size != GRID_SIZE:
        return np.full((N_LON, N_LAT), MISSING, dtype=np.float32)
    return raw.reshape(N_LON, N_LAT)


def _build_coastal_masks():
    """
    Build the PR2 coastal masks that match IDL's loop-index convention.

    BACKGROUND - WHY THIS IS NON-TRIVIAL
    =====================================
    The binary files follow GrADS convention: longitude varies fastest in memory
    (IDL Fortran column-major order).  IDL reads them with fltarr(144,72) and
    loops  'for i=0,143 do for j=0,71 do', so IDL element (i=lon, j=lat) sits at
    flat position  k = i + 144*j  (Fortran-order, first index fastest).

    Python reads the same bytes with np.fromfile + reshape(N_LON, N_LAT) in
    C (row-major) order.  In C order the element at Python[a,b] occupies flat
    position  k = a*N_LAT + b = a*72 + b.  Therefore the IDL latitude index for
    the element that Python calls [a,b] is:
        idl_lat_j = k // N_LON = (a*72 + b) // 144
    which is NOT simply b (Python dim-1).  Using dim-1 as the latitude index
    would misplace the polar/coastal mask bands, corrupting ~50% of grid cells.

    The masks are constant (independent of month data), so we compute them once
    here and reuse them inside the per-month loops.
    """
    # Flat k index for every element of the (N_LON, N_LAT) C-order array.
    k_flat = np.arange(GRID_SIZE, dtype=np.int32).reshape(N_LON, N_LAT)

    # IDL latitude index j = k // N_LON  (j=0 -> northernmost row ~88.75°N).
    idl_lat_j = k_flat // N_LON   # shape (N_LON, N_LAT)

    # icoast: near-polar latitude rows  (IDL j <= 11 -> lat >= ~62.5°N,
    #                                    IDL j >= 60 -> lat <= ~-12.5°S)
    icoast = (idl_lat_j <= 11) | (idl_lat_j >= 60)

    # jcoast: extended high-latitude band (IDL j <= 19 -> lat >= ~45°N,
    #                                       IDL j >= 53 -> lat <= ~-7.5°S)
    jcoast = (idl_lat_j <= 19) | (idl_lat_j >= 53)

    return icoast, jcoast


# Pre-compute once at module load - these are fixed geometry masks.
_ICOAST, _JCOAST = _build_coastal_masks()


def gpcp_late(path_in=None, path_out='./gpcp/'):
    """
    Morning-chain (internal 'late' constellation) GPCP dataset, currently F-17.
    Applies snow/ice masking and writes gpcp_nesdis_pr1.dat / gpcp_nesdis_pr2.dat.
    The primary satellite and its input directory come from
    satellite_config.LATE_CONSTELLATION_SAT.
    """
    if path_in is None:
        path_in = f'./{_LATE_SAT}-2.5/'
    os.makedirs(path_out, exist_ok=True)
    n_months = _months_since(_LATE_INIT_YEAR)
    print(f'gpcp_late ({_LATE_SAT}): processing {n_months} months from {_LATE_INIT_YEAR}')

    snw_file = os.path.join(path_in, f'SNW-{_LATE_SAT}-2.5')
    ice_file = os.path.join(path_in, f'ICE-{_LATE_SAT}-2.5')
    tag = read_lndsea()

    # Generate Julian date sequence starting Jan 1987
    # Use datetime for month/year tracking
    start = datetime.date(INIT_YEAR_F17, 1, 1)

    for iproduct in [1, 2]:
        rain_file = os.path.join(path_in, f'PR{iproduct}-{_LATE_SAT}-2.5')
        out_name = os.path.join(path_out, f'gpcp_nesdis_pr{iproduct}.dat')
        print(f'  Writing {out_name}')

        with open(snw_file, 'rb') as fsnw, \
             open(ice_file, 'rb') as fice, \
             open(rain_file, 'rb') as frain, \
             open(out_name, 'wb') as fout:

            for jmon in range(1, n_months + 1):
                mo = ((jmon - 1) % 12) + 1   # calendar month (1-12)

                snow = read_month_slice(fsnw, jmon)
                ice  = read_month_slice(fice, jmon)
                rain = read_month_slice(frain, jmon)

                # Warn on NaN
                if not np.all(np.isfinite(snow)):
                    print(f'  Warning: NaN in snow month {jmon}')
                if not np.all(np.isfinite(ice)):
                    print(f'  Warning: NaN in ice month {jmon}')
                if not np.all(np.isfinite(rain)):
                    print(f'  Warning: NaN in rain month {jmon}')

                # Apply masks (order matches IDL sequential if-then chain)
                rain = np.where(rain < 0.0, MISSING, rain)

                # Snow mask: months 43-60 = July 1990 to December 1991 in the F-17/F-08
                # record (month 1 is January 1987). These span a period of known F-08
                # snow-sensor anomaly, so the climatology is used instead of the observed
                # snow fraction for those 18 months. This comment said "Apr 1990" until
                # 18 November 2026, which month 43 is not.
                # NOTE: this jmon range is F-17-record-specific (starts 1987).
                # If _LATE_INIT_YEAR changes (e.g., WSF-M starts later), recalculate:
                #   months 43-60 relative to the new start year may not be 1990.
                # TODO: make this range dynamic based on _LATE_INIT_YEAR.
                if 43 <= jmon <= 60:
                    snw_ave = read_snow_clim(mo)
                    rain = np.where(snw_ave >= 0.20, MISSING, rain)
                else:
                    rain = np.where(snow >= 0.20, -100.0 * snow, rain)

                # Ice mask
                rain = np.where(ice >= 25.0, -1.0 * ice, rain)

                # PR2 coastal mask - uses geometry masks from _build_coastal_masks().
                # See that function's docstring for the full C-vs-Fortran explanation.
                if iproduct == 2:
                    # Mask mixed coastal cells (25-75% land) at near-polar latitudes
                    rain = np.where(_ICOAST & (tag >= 25.0) & (tag <= 75.0), MISSING, rain)
                    # Mask spuriously high ocean rain (>1000 mm/day) at extended high latitudes
                    rain = np.where(_JCOAST & (tag < 25.0) & (rain > 1000.0), MISSING, rain)

                fout.write(rain.astype(np.float32).tobytes())


def gpcp_early(path_in=None, path_out='./gpcp/'):
    """
    Late-morning-chain (internal 'early' constellation) GPCP dataset, currently
    F-16. Writes gpcp_nesdis_f10_pr1.dat / gpcp_nesdis_f10_pr2.dat. The primary
    satellite and its input directory come from
    satellite_config.EARLY_CONSTELLATION_SAT.
    """
    if path_in is None:
        path_in = f'./{_EARLY_SAT}-2.5/'
    os.makedirs(path_out, exist_ok=True)
    n_months = _months_since(_EARLY_INIT_YEAR)
    print(f'gpcp_early ({_EARLY_SAT}): processing {n_months} months from {_EARLY_INIT_YEAR}')

    snw_file = os.path.join(path_in, f'SNW-{_EARLY_SAT}-2.5')
    ice_file = os.path.join(path_in, f'ICE-{_EARLY_SAT}-2.5')
    tag = read_lndsea()

    for iproduct in [1, 2]:
        rain_file = os.path.join(path_in, f'PR{iproduct}-{_EARLY_SAT}-2.5')
        out_name = os.path.join(path_out, f'gpcp_nesdis_f10_pr{iproduct}.dat')
        print(f'  Writing {out_name}')

        with open(snw_file, 'rb') as fsnw, \
             open(ice_file, 'rb') as fice, \
             open(rain_file, 'rb') as frain, \
             open(out_name, 'wb') as fout:

            for jmon in range(1, n_months + 1):
                snow = read_month_slice(fsnw, jmon)
                ice  = read_month_slice(fice, jmon)
                rain = read_month_slice(frain, jmon)

                rain = np.where(rain < 0.0, MISSING, rain)
                rain = np.where(snow >= 0.20, -100.0 * snow, rain)
                rain = np.where(ice >= 25.0, -1.0 * ice, rain)

                # Same PR2 coastal mask as gpcp_late - uses module-level
                # _ICOAST / _JCOAST built with IDL-correct flat-index latitude bands.
                if iproduct == 2:
                    rain = np.where(_ICOAST & (tag >= 25.0) & (tag <= 75.0), MISSING, rain)
                    rain = np.where(_JCOAST & (tag < 25.0) & (rain > 1000.0), MISSING, rain)

                fout.write(rain.astype(np.float32).tobytes())


def valid_rain(x):
    """True where a masked GPCP rain value is an amount: finite and zero or more.

    Zero is a valid amount. Every negative value is a flag or the fill and is
    not an amount: the snow mask writes -100 times the snow fraction, the ice
    mask writes minus the ice percent, and the fill is -999.99.
    """
    return np.isfinite(x) & (x >= 0.0)


def valid_sampling(s):
    """True where a sampling fraction is usable: finite and from 0 to 1.

    Zero is a valid fraction (the cell was never sampled). The fill -999.99,
    anything else below zero, anything above 1 and any non-finite value are not
    fractions. On 18 November 2026 neither chain's sampling file, in the project
    tree or the operational tree, held a value above 1 or a non-finite value, so
    those two bounds are guards that changed nothing in the record of that day.
    """
    return np.isfinite(s) & (s >= 0.0) & (s <= 1.0)


def dual_merge_mask(rain_primary, samp_primary, rain_secondary, samp_secondary):
    """True where the dual product takes the weighted mean of the two chains:
    both rain values are amounts and both sampling fractions are positive."""
    return (valid_rain(rain_primary) & valid_rain(rain_secondary)
            & valid_sampling(samp_primary) & valid_sampling(samp_secondary)
            & (samp_primary > 0.0) & (samp_secondary > 0.0))


def merge_dual_month(rain_primary, samp_primary, rain_secondary, samp_secondary):
    """One month of the dual product. Returns (rain, sampling), both float32.

    The primary chain is the morning chain (F-17 today) and the secondary chain
    is the late-morning chain (F-16 today).

    RAIN. A cell holds the sampling-weighted mean of the two chains only where
    all four of these hold: the primary rain is an amount, the secondary rain is
    an amount, the primary sampling fraction is positive, and the secondary
    sampling fraction is positive. Everywhere else the primary value passes
    through unchanged, with its snow flag, ice flag or fill. The secondary chain
    is never used on its own, so the dual product has no more coverage than the
    primary chain. A non-finite primary value is neither an amount nor a flag
    and is written as fill.

    SAMPLING. A cell holds the plain mean of the two chains' sampling fractions
    wherever both are valid fractions, and fill elsewhere. It does not depend on
    either rain value. It is a coverage diagnostic of the two chains together.
    It is NOT the weight behind the rain value in a cell where the primary value
    passed through, because that rain value rests on the primary chain alone.

    WHAT THIS CHANGED (18 November 2026).
    The legacy IDL (gpcp_f08_f10_dual.pro) and the Python until that day formed
    the weighted mean wherever the primary rain was zero or more, without asking
    whether the secondary value was an amount. Where the secondary chain held a
    negative snow or ice flag with a positive sampling fraction, the flag entered
    the mean, and a cell with valid primary rain could come out lower or
    negative. That is a defect of the archived version 1 dual product, which is
    left as it is. The Python until that day also wrote the sampling field as
    fill wherever the primary rain of the second rain product was negative,
    which the legacy never did.
    """
    rain_primary = np.asarray(rain_primary, dtype=np.float32)
    rain_secondary = np.asarray(rain_secondary, dtype=np.float32)
    samp_primary = np.asarray(samp_primary, dtype=np.float32)
    samp_secondary = np.asarray(samp_secondary, dtype=np.float32)

    samp_ok = valid_sampling(samp_primary) & valid_sampling(samp_secondary)
    merge = dual_merge_mask(rain_primary, samp_primary, rain_secondary, samp_secondary)

    samp_total = samp_secondary + samp_primary
    # The mean is formed in single precision, as the legacy forms it. That keeps
    # every cell the rule does not change bit for bit what the legacy and the
    # files made before 18 November 2026 hold. Single precision has two limits,
    # both far outside the product. Amounts near 1e38 overflow in the sum, and a
    # mean that is not finite is not used: the primary value passes through.
    # Sampling fractions near 1e-38 underflow, which a ratio of sample counts
    # cannot reach. Cells outside `merge` are replaced below, so a zero or fill
    # denominator there is of no consequence.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        weighted = (rain_primary * samp_primary
                    + rain_secondary * samp_secondary) / samp_total
        mean_samp = samp_total / np.float32(2.0)
    merge = merge & np.isfinite(weighted)

    passthrough = np.where(np.isfinite(rain_primary), rain_primary, MISSING)
    rain = np.where(merge, weighted, passthrough)
    sampling = np.where(samp_ok, mean_samp, MISSING)
    return rain.astype(np.float32), sampling.astype(np.float32)


def gpcp_dual(path_f17=None, path_f16=None, path_out='./gpcp/'):
    """
    Merge the morning-chain and late-morning-chain GPCP estimates (dual-satellite
    product), currently F-17 and F-16. Writes gpcp_nesdis_dual_pr1.dat /
    gpcp_nesdis_dual_pr2.dat / gpcp_nesdis_dual_ssa.dat. Also copies SSA files for
    GPCP archives. The input directories come from satellite_config
    (LATE_CONSTELLATION_SAT for the morning chain, EARLY_CONSTELLATION_SAT for the
    late-morning chain). The path_f17/path_f16 parameter names are kept for
    backward compatibility.

    The rule for one month is merge_dual_month(), which states it in full. In
    short, F-17 is primary. A cell gets the sampling-weighted two-satellite
    mean only where both chains hold a rain amount and a positive sampling
    fraction, and the F-17 value, its flag or its fill passes through
    everywhere else. F-16 is never substituted on its own. That the product is
    F-17-primary is a settled product decision (2026-07-14), because the
    product is delivered as a GPCP input. Do not "improve" it by substituting
    F-16 months.

    This is NOT the version 1 algorithm unchanged. Version 1
    (gpcp_f08_f10_dual.pro) let a negative F-16 flag into the mean, and version
    2 does not, by decision of 18 November 2026. The sampling field is the plain
    mean of the two chains' fractions wherever both are valid, as in version 1.

    The six input files must all exist. A missing one raises before any output
    is opened, because opening an output truncates the dual file of the previous
    run. The time axis starts in January of the morning chain's first year
    (1987). Months before the late-morning chain begins (1992) are fill, and so
    is every month after the last complete one.
    """
    import shutil
    if path_f17 is None:
        path_f17 = f'./{_LATE_SAT}-2.5/'
    if path_f16 is None:
        path_f16 = f'./{_EARLY_SAT}-2.5/'

    f17_ssa = os.path.join(path_f17, f'SSA-{_LATE_SAT}-2.5')
    f16_ssa = os.path.join(path_f16, f'SSA-{_EARLY_SAT}-2.5')
    rain_inputs = {
        i: (os.path.join(path_out, f'gpcp_nesdis_pr{i}.dat'),
            os.path.join(path_out, f'gpcp_nesdis_f10_pr{i}.dat'))
        for i in (1, 2)}
    needed = [f17_ssa, f16_ssa] + [f for pair in rain_inputs.values() for f in pair]
    absent = [f for f in needed if not os.path.isfile(f)]
    if absent:
        raise FileNotFoundError(
            'gpcp_dual: input file(s) missing, nothing written: ' + ', '.join(absent))

    cy, cm = _current_year_month()

    n_months_f17 = _months_since(_LATE_INIT_YEAR)    # total months in late record
    # Write cutoff = the last COMPLETE month (the month before the current system
    # month), expressed as a 1-based index on the F-17 timeline (jmon 1 = Jan 1987).
    # Months after the cutoff are written as fill. The earlier formula,
    # (n_months_f17 - 12) + cm with the cm==1 special case inside _months_since, ran
    # one month high from February through December (harmless, since that month is
    # fill in the source) but in January collapsed to the PRIOR January, writing 11
    # real prior-year months of the dual product as fill (data loss).
    icurrent_month = (cy - _LATE_INIT_YEAR) * 12 + (cm - 1)

    print(f'gpcp_dual: n_months={n_months_f17}, current_month={icurrent_month}')

    # Copy SSA files - satellite suffix reflects current constellation primaries
    # (these names are fixed for GPCP archival compatibility; do not change)
    for src, dst in [
        (f17_ssa, os.path.join(path_out, 'gpcp_nesdis_ssa.dat')),
        (f16_ssa, os.path.join(path_out, 'gpcp_nesdis_f10_ssa.dat')),
    ]:
        shutil.copy2(src, dst)
        print(f'  Copied {src} -> {dst}')

    FILL = np.full((N_LON, N_LAT), MISSING, dtype=np.float32)

    # The late-morning chain's record starts later than the morning chain's.
    f16_offset = (_EARLY_INIT_YEAR - _LATE_INIT_YEAR) * 12   # 60 (1992-1987)

    for iproduct in [1, 2]:
        f17_pr, f16_pr = rain_inputs[iproduct]
        out_pr  = os.path.join(path_out, f'gpcp_nesdis_dual_pr{iproduct}.dat')
        out_ssa = os.path.join(path_out, 'gpcp_nesdis_dual_ssa.dat')

        print(f'  Writing {out_pr}')

        # The sampling field does not depend on the rain product, so both passes
        # compute the same one, and it is written during the second pass only.
        # Outputs are opened in place. An interruption during a pass can leave
        # partial files, and during the second pass a finished first rain file
        # can stand beside a partial sampling file.
        write_ssa = (iproduct == 2)
        merged = nonfinite = 0

        with open(f17_pr, 'rb') as fp17, \
             open(f17_ssa, 'rb') as fs17, \
             open(f16_pr, 'rb') as f16_pr_fh, \
             open(f16_ssa, 'rb') as f16_ssa_fh, \
             open(out_pr, 'wb') as fpr_out:
            fssa_out = open(out_ssa, 'wb') if write_ssa else None
            try:
                for jmon in range(1, n_months_f17 + 1):
                    ssmi_11 = read_month_slice(fp17, jmon)   # f17
                    samp_11 = read_month_slice(fs17, jmon)   # f17 SSA

                    if jmon <= f16_offset or jmon > icurrent_month:
                        # Before the late-morning chain's record, or after the
                        # last complete month: fill. The F-16 handles are not
                        # advanced before their record begins.
                        if jmon > f16_offset:
                            read_month_slice(f16_pr_fh, jmon - f16_offset)
                            read_month_slice(f16_ssa_fh, jmon - f16_offset)
                        fpr_out.write(FILL.tobytes())
                        if fssa_out:
                            fssa_out.write(FILL.tobytes())
                        continue

                    # No upper bound on the F-16 reads is needed. If a
                    # late-morning binary is shorter than the morning chain's
                    # (a lagged or partial upstream run), read_month_slice
                    # returns a fill grid past its end, neither F-16 field is
                    # then valid, and every cell passes the F-17 value through.
                    ssmi_10 = read_month_slice(f16_pr_fh, jmon - f16_offset)
                    samp_10 = read_month_slice(f16_ssa_fh, jmon - f16_offset)

                    out_rain, out_samp = merge_dual_month(ssmi_11, samp_11,
                                                          ssmi_10, samp_10)
                    nonfinite += int((~np.isfinite(ssmi_11)).sum())
                    merged += int(dual_merge_mask(ssmi_11, samp_11,
                                                  ssmi_10, samp_10).sum())

                    fpr_out.write(out_rain.tobytes())
                    if fssa_out:
                        fssa_out.write(out_samp.tobytes())
            finally:
                if fssa_out:
                    fssa_out.close()

        print(f'    PR{iproduct}: cell-months merged from both chains {merged}')
        if nonfinite:
            print(f'    WARNING PR{iproduct}: {nonfinite} non-finite F-17 value(s) '
                  f'written as fill')


def main():
    """Run all three GPCP processing steps (equivalent to idl.sh)."""
    print('=== GPCP Processing (replaces IDL routines) ===')
    print('\n--- gpcp_late (f17, was gpcp_f08.pro) ---')
    gpcp_late()
    print('\n--- gpcp_early (f16, was gpcp_f10.pro) ---')
    gpcp_early()
    print('\n--- gpcp_dual (merge, was gpcp_f08_f10_dual.pro) ---')
    gpcp_dual()
    print('\nDone.')


if __name__ == '__main__':
    main()
