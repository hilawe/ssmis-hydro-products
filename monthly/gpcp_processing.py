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
    - Grid: 144 × 72 (2.5°), GrADS format (N cols × M rows per month)
    - The binary files are sequential: jmon months of (144,72) float32 grids.
    - Initial year for f17 (f08 IDL): 1987; for f16 (f10 IDL): 1992
    - Snow climatology files: snow/SN<MM> (144×72 float32)
    - Land/sea tag:  lndsea.25d (144×72 float32, 0=ocean, values=% land)
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
    """Read one month's grid (N_LON × N_LAT float32) from an open file."""
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

                # Snow mask: months 43-60 = Apr 1990-Dec 1991 in F-17/F-08 record
                # (these span a period of known F-08 snow-sensor anomaly; use climatology
                #  instead of the actual observed snow fraction for those 18 months).
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


def gpcp_dual(path_f17=None, path_f16=None, path_out='./gpcp/'):
    """
    Merge the morning-chain and late-morning-chain GPCP estimates (dual-satellite
    product), currently F-17 and F-16. Writes gpcp_nesdis_dual_pr1.dat /
    gpcp_nesdis_dual_pr2.dat / gpcp_nesdis_dual_ssa.dat. Also copies SSA files for
    GPCP archives. The input directories come from satellite_config
    (LATE_CONSTELLATION_SAT for the morning chain, EARLY_CONSTELLATION_SAT for the
    late-morning chain). The path_f17/path_f16 parameter names are kept for
    backward compatibility.
    """
    import shutil
    if path_f17 is None:
        path_f17 = f'./{_LATE_SAT}-2.5/'
    if path_f16 is None:
        path_f16 = f'./{_EARLY_SAT}-2.5/'
    os.makedirs(path_out, exist_ok=True)

    now = datetime.datetime.now(datetime.timezone.utc)
    cy, cm = now.year, now.month
    increment_month = cm if cm != 0 else 12

    n_months_f17 = _months_since(_LATE_INIT_YEAR)    # total months in late record
    icurrent_month = (n_months_f17 - 12) + increment_month

    print(f'gpcp_dual: n_months={n_months_f17}, current_month={icurrent_month}')

    # Copy SSA files - satellite suffix reflects current constellation primaries
    # (these names are fixed for GPCP archival compatibility; do not change)
    for src, dst in [
        (os.path.join(path_f17, f'SSA-{_LATE_SAT}-2.5'), os.path.join(path_out, 'gpcp_nesdis_ssa.dat')),
        (os.path.join(path_f16, f'SSA-{_EARLY_SAT}-2.5'), os.path.join(path_out, 'gpcp_nesdis_f10_ssa.dat')),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'  Copied {src} -> {dst}')

    FILL = np.full((N_LON, N_LAT), MISSING, dtype=np.float32)

    for iproduct in [1, 2]:
        f17_pr = os.path.join(path_out, f'gpcp_nesdis_pr{iproduct}.dat')
        f17_ssa = os.path.join(path_f17, f'SSA-{_LATE_SAT}-2.5')
        f16_pr = os.path.join(path_out, f'gpcp_nesdis_f10_pr{iproduct}.dat')
        f16_ssa = os.path.join(path_f16, f'SSA-{_EARLY_SAT}-2.5')

        out_pr  = os.path.join(path_out, f'gpcp_nesdis_dual_pr{iproduct}.dat')
        out_ssa = os.path.join(path_out, 'gpcp_nesdis_dual_ssa.dat')

        print(f'  Writing {out_pr}')

        with open(f17_pr, 'rb') as fp17, \
             open(f17_ssa, 'rb') as fs17, \
             open(out_pr, 'wb') as fpr_out, \
             open(out_ssa, 'wb') as fssa_out:

            # The early-constellation record starts later than the late-constellation.
            # Calculate the offset in months between the two start years.
            f16_offset = (_EARLY_INIT_YEAR - _LATE_INIT_YEAR) * 12   # typically 60 (1992-1987)
            n_months_f16 = _months_since(_EARLY_INIT_YEAR)

            # Open f16 files separately
            f16_pr_fh   = open(f16_pr,  'rb') if os.path.exists(f16_pr)  else None
            f16_ssa_fh  = open(f16_ssa, 'rb') if os.path.exists(f16_ssa) else None

            for jmon in range(1, n_months_f17 + 1):
                ssmi_11 = read_month_slice(fp17, jmon)   # f17
                samp_11 = read_month_slice(fs17, jmon)   # f17 SSA

                if jmon > f16_offset and f16_pr_fh and f16_ssa_fh:
                    # f16 data available from month 61 onward in f17 timeline
                    ssmi_10 = read_month_slice(f16_pr_fh, jmon - f16_offset)
                    samp_10 = read_month_slice(f16_ssa_fh, jmon - f16_offset)

                    out_rain = FILL.copy()
                    out_samp = FILL.copy()

                    samp_total = samp_10 + samp_11

                    # Guard against missing SSA fill values (-999.99) corrupting the
                    # weighted average.  When either satellite's SSA is missing, its
                    # fill value (-999.99) would appear in both the numerator and the
                    # denominator, producing nonsensical results like -1000.86 mm/day
                    # (observed in IDL reference for ~15 months in 2006-2008 when F-16
                    # SSA had data gaps at season transition).  Python correctly falls
                    # back to the F-17-only estimate (ssmi_11) when samp_total <= 0,
                    # which is a deliberate IMPROVEMENT over the IDL reference.
                    valid = (ssmi_11 >= 0.0) & (samp_total > 0.0)
                    # Suppress numpy divide-by-zero for cells where valid=False;
                    # those cells are overwritten by the np.where fallback anyway.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        weighted = (ssmi_11 * samp_11 + ssmi_10 * samp_10) / samp_total
                    out_rain = np.where(valid, weighted, ssmi_11)
                    out_samp = np.where(valid, samp_total / 2.0, FILL)

                    if jmon <= icurrent_month:
                        fpr_out.write(out_rain.astype(np.float32).tobytes())
                        fssa_out.write(out_samp.astype(np.float32).tobytes())
                    else:
                        fpr_out.write(FILL.tobytes())
                        fssa_out.write(FILL.tobytes())
                else:
                    # Before f16 record: write fill
                    fpr_out.write(FILL.tobytes())
                    fssa_out.write(FILL.tobytes())

            if f16_pr_fh:
                f16_pr_fh.close()
            if f16_ssa_fh:
                f16_ssa_fh.close()


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
