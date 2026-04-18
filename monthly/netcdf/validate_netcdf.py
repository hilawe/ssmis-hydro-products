#!/usr/bin/env python3
"""
validate_netcdf.py

PURPOSE
    Validates the Python generate_netcdf.py output against the IDL-era reference
    NetCDF files from the ssmis_monthly_q1 archive for March 2026 (the primary
    validation month used throughout the SSMIS Python conversion effort).

    Validation covers:
      1. Late-constellation 2.5° products (all 10: CFR, LWP, PF1, PF2, PR1, PR2,
         SSA, ICE, SNW, WVP) - Python reads Q1 25deg-bin/*.MON, IDL reference is
         Q1 netcdf/2.5-deg/*_early_202603.nc (note: IDL labels were inverted;
         IDL "early" file = F17/late constellation data).
      2. Early-constellation 2.5° products (all 10) - Python reads Q1 f10-bin/*.F10,
         IDL reference is Q1 netcdf/2.5-deg/*_late_202603.nc (IDL "late" = F16/early).
      3. Coordinate arrays: lat, lon, time (days-since-1987-01-01).
      4. Key CF global attributes: constellation, platform, sensor, units, etc.
      5. GPCP late and early PR1/PR2 NetCDF products.
      6. Structural checks: dimensions, variable names, fill value.

    Reference month: March 2026 (year=2026, month=3).
      Late binary offset: imonth = (2026-1987)*12 + (3-1) = 470
      Early binary offset: imonth = (2026-1992)*12 + (3-1) = 410

IMPORTANT NOTE ON EARLY/LATE LABEL INVERSION IN IDL
    The IDL scripts products_early_twohalfdeg_netcdf.pro and
    products_late_twohalfdeg_netcdf.pro had their "early" and "late" labels SWAPPED
    relative to the scientifically correct definition (equatorial crossing time):
      - IDL "early" file -> contains F17 (late/6pm) constellation data
      - IDL "late"  file -> contains F16 (early/morning) constellation data
    Python generate_netcdf.py corrects this inversion.
    Therefore, for data value comparison:
      - Python late output  ↔  IDL early file (same F17 binary source)
      - Python early output ↔  IDL late file  (same F16 binary source)
    See the project documentation for full background.

CALLED BY
    Manually: python validate_netcdf.py
    No arguments needed; all paths are defined as constants below.

CALLS / IMPORTS
    generate_netcdf (imported for read_month, make_coords, get_n_months, write_netcdf)
    netCDF4, numpy, os, sys

OUTPUT
    Prints a per-product statistical comparison table to stdout.
    Writes Python-generated test NetCDF files to TESTOUT_DIR (created if absent).
    Does NOT modify any operational files.

AUTHOR
    Hilawe Semunegus, NOAA/NCEI
    Validation date: 2026-04-17
"""

import os
import sys
import numpy as np
from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Add parent directory so we can import generate_netcdf helper functions
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_netcdf as gn   # gives us read_month, make_coords, write_netcdf, etc.

# ---------------------------------------------------------------------------
# Path constants - adjust if directory layout changes
# ---------------------------------------------------------------------------

# Q1 archive root (the frozen Fortran-era reference dataset)
Q1_ROOT = '/path/to/ssmis_data/monthly'

# Q1 combined binary files used as input to both the IDL (reference) and Python (test)
Q1_LATE_BIN  = os.path.join(Q1_ROOT, '25deg-bin')   # *.MON files (F17/late chain)
Q1_EARLY_BIN = os.path.join(Q1_ROOT, 'f10-bin')     # *.F10 files (F16/early chain)
Q1_GPCP_BIN  = os.path.join(Q1_ROOT, 'gpcp')        # gpcp_nesdis_*.dat files

# Q1 IDL-generated reference NetCDF files (March 2026)
Q1_NC_25     = os.path.join(Q1_ROOT, 'netcdf', '2.5-deg')

# Output directory for Python-generated test NetCDF files (isolated from operational output)
TESTOUT_DIR  = '/tmp/netcdf_validate_test/'

# Validation target: March 2026
VAL_YEAR  = 2026
VAL_MONTH = 3
VAL_MM    = f'{VAL_MONTH:02d}'

# Month index within each multi-year binary file (0-based)
# Late constellation starts Jan 1987:  March 2026 = (2026-1987)*12 + (3-1) = 470
# Early constellation starts Jan 1992: March 2026 = (2026-1992)*12 + (3-1) = 410
IMONTH_LATE  = (VAL_YEAR - 1987) * 12 + (VAL_MONTH - 1)   # = 470
IMONTH_EARLY = (VAL_YEAR - 1992) * 12 + (VAL_MONTH - 1)   # = 410

# GPCP late starts Jan 1987, early starts Jan 1992 (same as above)
IMONTH_GPCP_LATE  = IMONTH_LATE
IMONTH_GPCP_EARLY = IMONTH_EARLY

# Expected days-since-1987-01-01 for 2026-03-01
# (2026-03-01 - 1987-01-01) = 14,304 days
EXPECTED_TIME_DAYS = 14304.0

# Product keys and their netCDF4 variable names
PRODUCTS = ['CFR', 'LWP', 'PF1', 'PF2', 'PR1', 'PR2', 'SSA', 'ICE', 'SNW', 'WVP']

VAR_NAMES = {
    'CFR': 'cloud_fraction',
    'LWP': 'liquid_water_path',
    'PF1': 'rain_fraction_alg1',
    'PF2': 'rain_fraction_alg2',
    'PR1': 'rainfall_alg1',
    'PR2': 'rainfall_alg2',
    'SSA': 'sampling_fraction',
    'ICE': 'sea_ice_cover',
    'SNW': 'snow_cover_fraction',
    'WVP': 'total_precipitable_water',
}

# Acceptable absolute tolerance for "exact" match (float32 round-trip precision)
EXACT_TOL = 0.001

# ---------------------------------------------------------------------------
# Helper: read one month's data directly from a Q1 binary file
# ---------------------------------------------------------------------------

def read_binary_month(binary_file, imonth, grid):
    """
    Read one month of float32 data from a multi-year combined binary file.

    Parameters
    ----------
    binary_file : str
        Full path to the combined binary file (e.g. Q1_LATE_BIN/PR1.MON).
    imonth : int
        0-based month index within the file (e.g. 470 for March 2026 in the late record).
    grid : dict
        Grid definition dict (e.g. gn.GRID_25).

    Returns
    -------
    data : np.ndarray, shape (nlat, nlon), dtype float32
        Monthly mean field, or None if read fails.
    """
    offset = imonth * grid['nlat'] * grid['nlon'] * 4  # bytes; float32 = 4 bytes each
    with open(binary_file, 'rb') as fobj:
        fobj.seek(offset)
        data = gn.read_month(fobj, grid)
    return data


# ---------------------------------------------------------------------------
# Helper: run the Python generate_netcdf.py logic for one month and one product
# ---------------------------------------------------------------------------

def run_python_netcdf_for_product(prod_key, bin_file, imonth, year, month,
                                  constellation, title, history, summary,
                                  out_prefix, out_dir, grid):
    """
    Generate a single Python NetCDF file for one product/month using Q1 binary data.

    This mirrors the inner loop of generate_netcdf.run_late_25deg() /
    run_early_25deg(), but reads from an arbitrary binary path and writes to an
    arbitrary output directory so the validation does not touch operational directories.

    Called by:   validate_late_25deg(), validate_early_25deg()
    Calls:       gn.read_month(), gn.write_netcdf(), gn.make_coords()
    """
    lats, lons = gn.make_coords(grid)
    mm = f'{month:02d}'

    with open(bin_file, 'rb') as fobj:
        fobj.seek(imonth * grid['nlat'] * grid['nlon'] * 4)
        data = gn.read_month(fobj, grid)

    if data is None or np.all(data <= -999.0):
        return None  # No data for this month/product - skip

    os.makedirs(out_dir, exist_ok=True)
    ncfile = os.path.join(out_dir, f'{out_prefix}{year:04d}{mm}.nc')

    gn.write_netcdf(
        ncfile, data, lats, lons, year, month, prod_key,
        title=gn.PRODUCT_TITLES_25.get(prod_key, prod_key),
        initial_year=(1987 if 'late' in out_prefix else 1992),
        constellation=constellation,
        history=history,
        summary=summary,
        dataset_name=os.path.basename(bin_file),
    )
    return ncfile


# ---------------------------------------------------------------------------
# Helper: compare Python vs IDL NetCDF data for one product
# ---------------------------------------------------------------------------

def compare_nc_product(py_nc_path, idl_nc_path, prod_key, label):
    """
    Open two NetCDF files for the same product/month and compute comparison statistics.

    Returns a dict with:
      n_valid_py, n_valid_idl, n_joint_valid,
      exact_pct, within5pct, mean_py, mean_idl, bias, rmse, max_abs_diff,
      coords_ok (bool), time_ok (bool)

    Called by:   validate_late_25deg(), validate_early_25deg(), validate_gpcp_*()
    Calls:       netCDF4.Dataset
    """
    results = {
        'prod': prod_key, 'label': label,
        'n_valid_py': 0, 'n_valid_idl': 0, 'n_joint': 0,
        'exact_pct': 0.0, 'within5pct': 0.0,
        'mean_py': np.nan, 'mean_idl': np.nan,
        'bias': np.nan, 'rmse': np.nan, 'max_abs_diff': np.nan,
        'coords_ok': False, 'time_ok': False,
        'error': None,
    }

    if py_nc_path is None or not os.path.exists(py_nc_path):
        results['error'] = 'Python NC not generated'
        return results
    if not os.path.exists(idl_nc_path):
        results['error'] = f'IDL reference not found: {idl_nc_path}'
        return results

    try:
        py_ds  = Dataset(py_nc_path,  'r')
        idl_ds = Dataset(idl_nc_path, 'r')

        var_name = VAR_NAMES[prod_key]

        # --- data arrays (squeeze time dim to get 2D) ---
        py_data  = np.array(py_ds.variables[var_name][0, :, :], dtype=np.float32)
        idl_data = np.array(idl_ds.variables[var_name][0, :, :], dtype=np.float32)

        # --- coordinate check ---
        py_lats  = np.array(py_ds.variables['lat'][:])
        py_lons  = np.array(py_ds.variables['lon'][:])
        idl_lats = np.array(idl_ds.variables['lat'][:])
        idl_lons = np.array(idl_ds.variables['lon'][:])
        results['coords_ok'] = (
            np.allclose(py_lats, idl_lats, atol=1e-4) and
            np.allclose(py_lons, idl_lons, atol=1e-4)
        )

        # --- time check ---
        py_time  = float(py_ds.variables['time'][0])
        idl_time = float(idl_ds.variables['time'][0])
        results['time_ok'] = abs(py_time - idl_time) < 1.0  # within 1 day

        # --- valid-cell masks (fill = -999.99; use -999 as threshold) ---
        py_valid  = py_data  > -999.0
        idl_valid = idl_data > -999.0
        joint_valid = py_valid & idl_valid

        results['n_valid_py']  = int(np.sum(py_valid))
        results['n_valid_idl'] = int(np.sum(idl_valid))
        results['n_joint']     = int(np.sum(joint_valid))

        if results['n_joint'] == 0:
            results['error'] = 'No jointly valid cells'
            py_ds.close(); idl_ds.close()
            return results

        py_vals  = py_data[joint_valid]
        idl_vals = idl_data[joint_valid]
        diff     = py_vals - idl_vals

        results['mean_py']       = float(np.mean(py_vals))
        results['mean_idl']      = float(np.mean(idl_vals))
        results['bias']          = float(np.mean(diff))
        results['rmse']          = float(np.sqrt(np.mean(diff**2)))
        results['max_abs_diff']  = float(np.max(np.abs(diff)))
        results['exact_pct']     = float(100.0 * np.sum(np.abs(diff) < EXACT_TOL) / len(diff))

        # <5% relative difference over non-zero IDL cells
        nonzero_mask = np.abs(idl_vals) > 0
        if np.any(nonzero_mask):
            rel_diff = np.abs(diff[nonzero_mask]) / np.abs(idl_vals[nonzero_mask])
            results['within5pct'] = float(100.0 * np.sum(rel_diff < 0.05) / np.sum(nonzero_mask))

        py_ds.close(); idl_ds.close()

    except Exception as exc:
        results['error'] = str(exc)

    return results


# ---------------------------------------------------------------------------
# Helper: check key CF global attributes in a Python-generated NC file
# ---------------------------------------------------------------------------

def check_attributes(py_nc_path, expected_constellation_fragment,
                     expected_product_version='v01r00'):
    """
    Verify that key CF and NOAA CDR global attributes are present and correct
    in a Python-generated NetCDF file.

    Checks:
      - Metadata_Conventions contains 'CF-1.5'
      - product_version == expected_product_version
      - naming_authority == 'gov.noaa.ncdc'
      - constellation contains expected_constellation_fragment
      - time variable has units 'days since 1987-01-01'
      - lat, lon variables exist and have correct standard_name
      - fill_value attribute on data variable == -999.99 (float32)

    Called by:   validate_late_25deg(), validate_early_25deg()
    Returns:     list of (check_name, pass/fail, detail) tuples
    """
    checks = []
    try:
        ds = Dataset(py_nc_path, 'r')

        def chk(name, condition, detail=''):
            checks.append((name, bool(condition), detail))

        # Global attributes
        chk('CF-1.5 in Metadata_Conventions',
            'CF-1.5' in getattr(ds, 'Metadata_Conventions', ''),
            getattr(ds, 'Metadata_Conventions', 'MISSING'))

        chk('product_version == v01r00',
            getattr(ds, 'product_version', '') == expected_product_version,
            getattr(ds, 'product_version', 'MISSING'))

        chk('naming_authority = gov.noaa.ncdc',
            getattr(ds, 'naming_authority', '') == 'gov.noaa.ncdc',
            getattr(ds, 'naming_authority', 'MISSING'))

        chk('constellation contains expected fragment',
            expected_constellation_fragment in getattr(ds, 'constellation', ''),
            getattr(ds, 'constellation', 'MISSING')[:80])

        chk('processing_level = NOAA Level 3',
            getattr(ds, 'processing_level', '') == 'NOAA Level 3',
            getattr(ds, 'processing_level', 'MISSING'))

        chk('time_coverage_duration = P1M',
            getattr(ds, 'time_coverage_duration', '') == 'P1M',
            getattr(ds, 'time_coverage_duration', 'MISSING'))

        # time variable
        if 'time' in ds.variables:
            t_units = getattr(ds.variables['time'], 'units', '')
            chk('time units = days since 1987-01-01',
                t_units == 'days since 1987-01-01', t_units)
            chk('time calendar = gregorian',
                getattr(ds.variables['time'], 'calendar', '') == 'gregorian',
                getattr(ds.variables['time'], 'calendar', 'MISSING'))
            chk('time value = 14304 (2026-03-01)',
                abs(float(ds.variables['time'][0]) - EXPECTED_TIME_DAYS) < 1.0,
                str(float(ds.variables['time'][0])))
        else:
            chk('time variable exists', False, 'MISSING')

        # lat/lon
        if 'lat' in ds.variables:
            chk('lat standard_name = latitude',
                getattr(ds.variables['lat'], 'standard_name', '') == 'latitude',
                getattr(ds.variables['lat'], 'standard_name', 'MISSING'))
            chk('lat[0] = -88.75 (south pole start)',
                abs(float(ds.variables['lat'][0]) - (-88.75)) < 0.01,
                str(float(ds.variables['lat'][0])))
        else:
            chk('lat variable exists', False, 'MISSING')

        if 'lon' in ds.variables:
            chk('lon standard_name = longitude',
                getattr(ds.variables['lon'], 'standard_name', '') == 'longitude',
                getattr(ds.variables['lon'], 'standard_name', 'MISSING'))
            chk('lon[0] = 1.25 (1.25E first)',
                abs(float(ds.variables['lon'][0]) - 1.25) < 0.01,
                str(float(ds.variables['lon'][0])))
        else:
            chk('lon variable exists', False, 'MISSING')

        # Fill value on first data variable
        data_vars = [v for v in ds.variables if v not in ('time', 'time_iso8601', 'lat', 'lon')]
        if data_vars:
            dv = ds.variables[data_vars[0]]
            fv = getattr(dv, '_FillValue', None)
            chk('fill_value = -999.99 (float32)',
                fv is not None and abs(float(fv) - (-999.99)) < 0.01,
                str(fv))

        ds.close()
    except Exception as exc:
        checks.append(('EXCEPTION', False, str(exc)))

    return checks


# ---------------------------------------------------------------------------
# Main validation sections
# ---------------------------------------------------------------------------

def validate_late_25deg():
    """
    Validate Python late-constellation 2.5° NetCDF against IDL reference for Mar 2026.

    Python reads from Q1 25deg-bin/*.MON (F17/late, starts Jan 1987).
    IDL reference is Q1 netcdf/2.5-deg/*_early_202603.nc (F17 data, misnamed 'early').

    Called by:  main()
    Calls:      run_python_netcdf_for_product(), compare_nc_product(), check_attributes()
    """
    print('\n' + '='*70)
    print('SECTION A: Late-constellation 2.5° - Python vs IDL (Mar 2026)')
    print('  Python source: Q1/25deg-bin/*.MON (F17/6pm chain)')
    print('  IDL reference: Q1/netcdf/2.5-deg/*_early_202603.nc (F17, mislabeled)')
    print('='*70)

    constellation = (
        'SSM/I F-08: July 1987-December 1991; '
        'SSM/I F-11: January 1992-April 1995; '
        'SSM/I F-13: May 1995-December 2008; '
        'SSMIS F-17: January 2009-present'
    )
    history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF file.'
    summary = 'NOAA STAR-EESIC-NCDC SSMI-SSMIS Hydrological Products from 1987-present.'

    attr_checked = False
    stats_rows = []

    for prod_key in PRODUCTS:
        bin_file = os.path.join(Q1_LATE_BIN, f'{prod_key}.MON')
        # IDL reference for F17 data is the "early" labeled file (inversion!)
        idl_ref  = os.path.join(Q1_NC_25,
                                f'mw-hydro_v01_2.5-deg_{prod_key.lower()}_early_{VAL_YEAR:04d}{VAL_MM}.nc')

        out_prefix = f'mw-hydro_v01_2.5-deg_{prod_key.lower()}_late_'
        out_dir    = os.path.join(TESTOUT_DIR, 'late25')

        py_nc = run_python_netcdf_for_product(
            prod_key, bin_file, IMONTH_LATE, VAL_YEAR, VAL_MONTH,
            constellation, gn.PRODUCT_TITLES_25.get(prod_key, prod_key),
            history, summary, out_prefix, out_dir, gn.GRID_25
        )

        # Attribute check - only for first successful product
        if py_nc and not attr_checked:
            print('\n--- CF Global Attribute Checks (first late product: PR1) ---')
            attr_checks = check_attributes(py_nc, 'SSMIS F-17')
            for cname, passed, detail in attr_checks:
                status = 'PASS' if passed else 'FAIL'
                print(f'  [{status}] {cname}: {detail}')
            attr_checked = True

        row = compare_nc_product(py_nc, idl_ref, prod_key, 'late25')
        stats_rows.append(row)

    _print_stats_table(stats_rows, 'Late 2.5°')
    return stats_rows


def validate_early_25deg():
    """
    Validate Python early-constellation 2.5° NetCDF against IDL reference for Mar 2026.

    Python reads from Q1 f10-bin/*.F10 (F16/early, starts Jan 1992).
    IDL reference is Q1 netcdf/2.5-deg/*_late_202603.nc (F16 data, misnamed 'late').

    Called by:  main()
    Calls:      run_python_netcdf_for_product(), compare_nc_product(), check_attributes()
    """
    print('\n' + '='*70)
    print('SECTION B: Early-constellation 2.5° - Python vs IDL (Mar 2026)')
    print('  Python source: Q1/f10-bin/*.F10 (F16/morning chain)')
    print('  IDL reference: Q1/netcdf/2.5-deg/*_late_202603.nc (F16, mislabeled)')
    print('='*70)

    constellation = (
        'SSM/I F-10: January 1992-September 1997; '
        'SSM/I F-14: October 1997-December 2001; '
        'SSM/I F-15: January 2002-June 2006; '
        'SSMIS F-16: July 2006-present'
    )
    history = (
        '1) 2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF file converted '
        'from the original gridded binary format. '
        '2) On October 18, 2017, netCDF files were revised due to a file encoding error '
        '(dates were incorrectly encoded).'
    )
    summary = 'NOAA STAR-EESIC-NCDC SSMI-SSMIS Hydrological Products from 1992-present.'

    ext_map = {p: 'F10' for p in PRODUCTS}   # all early files end in .F10

    attr_checked = False
    stats_rows = []

    for prod_key in PRODUCTS:
        bin_file = os.path.join(Q1_EARLY_BIN, f'{prod_key}.{ext_map[prod_key]}')
        # IDL reference for F16 data is the "late" labeled file (inversion!)
        idl_ref  = os.path.join(Q1_NC_25,
                                f'mw-hydro_v01_2.5-deg_{prod_key.lower()}_late_{VAL_YEAR:04d}{VAL_MM}.nc')

        out_prefix = f'mw-hydro_v01_2.5-deg_{prod_key.lower()}_early_'
        out_dir    = os.path.join(TESTOUT_DIR, 'early25')

        py_nc = run_python_netcdf_for_product(
            prod_key, bin_file, IMONTH_EARLY, VAL_YEAR, VAL_MONTH,
            constellation, gn.PRODUCT_TITLES_25.get(prod_key, prod_key),
            history, summary, out_prefix, out_dir, gn.GRID_25
        )

        if py_nc and not attr_checked:
            print('\n--- CF Global Attribute Checks (first early product: PR1) ---')
            attr_checks = check_attributes(py_nc, 'SSMIS F-16')
            for cname, passed, detail in attr_checks:
                status = 'PASS' if passed else 'FAIL'
                print(f'  [{status}] {cname}: {detail}')
            attr_checked = True

        row = compare_nc_product(py_nc, idl_ref, prod_key, 'early25')
        stats_rows.append(row)

    _print_stats_table(stats_rows, 'Early 2.5°')
    return stats_rows


def validate_gpcp():
    """
    Validate Python GPCP NetCDF (late and early) for Mar 2026.

    The GPCP output files carry only PR1 and PR2.  No IDL-generated GPCP NetCDF
    reference files exist in the Q1 archive (the IDL pipeline never archived GPCP
    NetCDF files separately from the main 2.5-deg bundle).

    Validation strategy (closed-chain):
      1. Generate Python GPCP NetCDF from Q1 GPCP binary (gpcp_nesdis_pr1.dat etc.).
      2. Read the generated NetCDF back and compare its data values against
         numpy-direct reads of the same binary file (ground truth).
      3. If NetCDF data == binary data (100% exact), then NetCDF I/O is correct.
      4. Since the GPCP binary was independently validated 100% exact vs IDL
         (Section 6.10 of the project documentation), the chain is:
            binary == IDL  AND  NetCDF == binary  ⟹  NetCDF == IDL.
      5. Structural/attribute checks (CF metadata, coordinates, time) are run
         on the generated files.

    GPCP CDR labels were CORRECT in both IDL and Python (no inversion):
      late -> F17 (6pm chain), reads gpcp_nesdis_pr1.dat
      early -> F16 (morning chain), reads gpcp_nesdis_f10_pr1.dat

    Called by:  main()
    Calls:      run_python_netcdf_for_product(), check_attributes(),
                read_binary_month(), netCDF4.Dataset
    """
    print('\n' + '='*70)
    print('SECTION C: GPCP NetCDF - Python vs Q1 binary round-trip (Mar 2026)')
    print('  (No IDL GPCP NetCDF reference in Q1 archive; chain validated via binary)')
    print('  GPCP labels NEVER inverted - late=F17 6pm, early=F16 morning in both')
    print('='*70)

    gpcp_configs = [
        {
            'label': 'GPCP Late (F17)',
            'products': {
                'PR1': ('gpcp_nesdis_pr1.dat',    IMONTH_GPCP_LATE),
                'PR2': ('gpcp_nesdis_pr2.dat',    IMONTH_GPCP_LATE),
            },
            'constellation': 'SSMIS F-17: January 1987-present',
            'out_prefix_tpl': 'mw-hydro_v01_gpcp_late_{prod}_',
            'out_subdir': 'gpcp_late',
            'initial_year': 1987,
        },
        {
            'label': 'GPCP Early (F16)',
            'products': {
                'PR1': ('gpcp_nesdis_f10_pr1.dat', IMONTH_GPCP_EARLY),
                'PR2': ('gpcp_nesdis_f10_pr2.dat', IMONTH_GPCP_EARLY),
            },
            'constellation': 'SSMIS F-16: January 1992-present',
            'out_prefix_tpl': 'mw-hydro_v01_gpcp_early_{prod}_',
            'out_subdir': 'gpcp_early',
            'initial_year': 1992,
        },
    ]

    all_rows = []
    for cfg in gpcp_configs:
        print(f'\n  -- {cfg["label"]} --')
        history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF GPCP file.'
        summary = 'NOAA STAR-EESIC-NCDC GPCP SSMI-SSMIS Hydrological Products.'
        attr_checked = False
        rows = []

        for prod_key, (bin_fname, imonth) in cfg['products'].items():
            bin_file   = os.path.join(Q1_GPCP_BIN, bin_fname)
            out_prefix = cfg['out_prefix_tpl'].format(prod=prod_key.lower())
            out_dir    = os.path.join(TESTOUT_DIR, cfg['out_subdir'])

            # Step 1: Generate Python NetCDF from Q1 GPCP binary
            py_nc = run_python_netcdf_for_product(
                prod_key, bin_file, imonth, VAL_YEAR, VAL_MONTH,
                cfg['constellation'],
                f'GPCP Monthly 2.5 Degree Rainfall (mm)',
                history, summary, out_prefix, out_dir, gn.GRID_25
            )

            # Step 2: Attribute checks (once per constellation)
            if py_nc and not attr_checked:
                sat_fragment = 'F-17' if 'Late' in cfg['label'] else 'F-16'
                print(f'\n  --- CF Attribute Checks ({cfg["label"]}) ---')
                attr_checks = check_attributes(py_nc, sat_fragment)
                for cname, passed, detail in attr_checks:
                    status = 'PASS' if passed else 'FAIL'
                    print(f'    [{status}] {cname}: {detail}')
                attr_checked = True

            # Step 3: Round-trip check - compare NetCDF values against binary ground truth
            row = _compare_nc_vs_binary(py_nc, bin_file, imonth, prod_key, cfg['label'])
            rows.append(row)
            all_rows.append(row)

        _print_roundtrip_table(rows, cfg['label'])

    return all_rows


def _compare_nc_vs_binary(py_nc_path, bin_file, imonth, prod_key, label):
    """
    Compare a Python-generated GPCP NetCDF data array against the raw binary.

    This is the 'closed-chain' validation for GPCP:
      NetCDF data == binary data  (round-trip fidelity)
    Combined with the separately verified  binary == IDL  (Section 6.10),
    this establishes  NetCDF == IDL.

    Called by:  validate_gpcp()
    Returns:    dict with comparison statistics
    """
    results = {
        'prod': prod_key, 'label': label,
        'n_valid_nc': 0, 'n_valid_bin': 0, 'n_joint': 0,
        'exact_pct': 0.0, 'mean_nc': np.nan, 'mean_bin': np.nan,
        'bias': np.nan, 'max_abs_diff': np.nan,
        'coords_ok': False, 'time_ok': False,
        'error': None,
    }

    if py_nc_path is None or not os.path.exists(py_nc_path):
        results['error'] = 'Python NC not generated'
        return results
    if not os.path.exists(bin_file):
        results['error'] = f'Binary file not found: {bin_file}'
        return results

    try:
        # Read binary ground truth
        bin_data = read_binary_month(bin_file, imonth, gn.GRID_25)
        if bin_data is None:
            results['error'] = 'Binary read returned None'
            return results

        # Read NetCDF data
        nc_ds = Dataset(py_nc_path, 'r')
        var_name = VAR_NAMES[prod_key]
        nc_data = np.array(nc_ds.variables[var_name][0, :, :], dtype=np.float32)

        # Coordinate checks
        lats_nc = np.array(nc_ds.variables['lat'][:])
        lons_nc = np.array(nc_ds.variables['lon'][:])
        lats_exp, lons_exp = gn.make_coords(gn.GRID_25)
        results['coords_ok'] = (
            np.allclose(lats_nc, lats_exp, atol=1e-4) and
            np.allclose(lons_nc, lons_exp, atol=1e-4)
        )

        # Time check
        results['time_ok'] = abs(float(nc_ds.variables['time'][0]) - EXPECTED_TIME_DAYS) < 1.0
        nc_ds.close()

        # Comparison
        nc_valid  = nc_data  > -999.0
        bin_valid = bin_data > -999.0
        joint     = nc_valid & bin_valid

        results['n_valid_nc']  = int(np.sum(nc_valid))
        results['n_valid_bin'] = int(np.sum(bin_valid))
        results['n_joint']     = int(np.sum(joint))

        if results['n_joint'] == 0:
            results['error'] = 'No jointly valid cells'
            return results

        nc_vals  = nc_data[joint]
        bin_vals = bin_data[joint]
        diff     = nc_vals - bin_vals

        results['mean_nc']      = float(np.mean(nc_vals))
        results['mean_bin']     = float(np.mean(bin_vals))
        results['bias']         = float(np.mean(diff))
        results['max_abs_diff'] = float(np.max(np.abs(diff)))
        results['exact_pct']    = float(100.0 * np.sum(np.abs(diff) < EXACT_TOL) / len(diff))

    except Exception as exc:
        results['error'] = str(exc)

    return results


def _print_roundtrip_table(rows, section_label):
    """
    Print a round-trip (NetCDF vs binary) comparison table for GPCP products.

    Called by:  validate_gpcp()
    """
    HDR = (f"{'Product':<6} {'ValidNC':>7} {'ValidBin':>8} {'Joint':>6} "
           f"{'Exact%':>7} {'MeanNC':>9} {'MeanBin':>9} "
           f"{'Bias':>9} {'MaxDiff':>9} {'Coords':>6} {'Time':>5}")
    SEP = '-' * len(HDR)

    print(f'\n  Round-trip Validation - {section_label} (Mar 2026) [NetCDF vs Q1 binary]')
    print(f'  {HDR}')
    print(f'  {SEP}')

    for r in rows:
        if r.get('error'):
            print(f"  {r['prod']:<6}  ERROR: {r['error']}")
            continue
        coord_str = 'OK' if r['coords_ok'] else 'FAIL'
        time_str  = 'OK' if r['time_ok']   else 'FAIL'
        print(
            f"  {r['prod']:<6} {r['n_valid_nc']:>7} {r['n_valid_bin']:>8} "
            f"{r['n_joint']:>6} "
            f"{r['exact_pct']:>7.2f} {r['mean_nc']:>9.4f} {r['mean_bin']:>9.4f} "
            f"{r['bias']:>9.5f} {r['max_abs_diff']:>9.5f} "
            f"{coord_str:>6} {time_str:>5}"
        )


# ---------------------------------------------------------------------------
# Print table helper
# ---------------------------------------------------------------------------

def _print_stats_table(rows, section_label):
    """
    Print a formatted comparison statistics table for a set of products.

    Called by:  validate_late_25deg(), validate_early_25deg(), validate_gpcp()
    """
    HDR = (f"{'Product':<6} {'ValidPy':>7} {'ValidIDL':>8} {'Joint':>6} "
           f"{'Exact%':>7} {'<5%rel':>7} "
           f"{'MeanPy':>9} {'MeanIDL':>9} {'Bias':>9} {'RMSE':>8} {'MaxDiff':>9} "
           f"{'Coords':>6} {'Time':>5}")
    SEP = '-' * len(HDR)

    print(f'\n  Statistical Summary - {section_label} (March 2026)')
    print(f'  {HDR}')
    print(f'  {SEP}')

    for r in rows:
        if r.get('error'):
            print(f"  {r['prod']:<6}  ERROR: {r['error']}")
            continue
        coord_str = 'OK' if r['coords_ok'] else 'FAIL'
        time_str  = 'OK' if r['time_ok']   else 'FAIL'
        print(
            f"  {r['prod']:<6} {r['n_valid_py']:>7} {r['n_valid_idl']:>8} "
            f"{r['n_joint']:>6} "
            f"{r['exact_pct']:>7.2f} {r['within5pct']:>7.2f} "
            f"{r['mean_py']:>9.4f} {r['mean_idl']:>9.4f} "
            f"{r['bias']:>9.5f} {r['rmse']:>8.5f} {r['max_abs_diff']:>9.5f} "
            f"{coord_str:>6} {time_str:>5}"
        )

    # Highlight SSA - the definitive grid-layout correctness anchor
    ssa = next((r for r in rows if r['prod'] == 'SSA' and not r.get('error')), None)
    if ssa:
        exact_str = f"{ssa['exact_pct']:.2f}%"
        anchor_msg = '✓ PERFECT' if ssa['exact_pct'] >= 99.9 else f'⚠ {exact_str}'
        print(f'\n  SSA (grid-layout correctness anchor): Exact% = {exact_str}  {anchor_msg}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run the full NetCDF validation suite for March 2026.

    Sections:
      A - Late-constellation 2.5° (all 10 products)
      B - Early-constellation 2.5° (all 10 products)
      C - GPCP late and early (PR1/PR2 each)

    Writes test NetCDF files to TESTOUT_DIR (/tmp/netcdf_validate_test/) and
    prints comparison tables to stdout.
    """
    print('SSMIS Python NetCDF Validation - March 2026')
    print(f'  Q1 binary root : {Q1_ROOT}')
    print(f'  IDL reference  : {Q1_NC_25}')
    print(f'  Python test out: {TESTOUT_DIR}')
    print(f'  imonth late    : {IMONTH_LATE}  (Jan 1987 -> Mar 2026)')
    print(f'  imonth early   : {IMONTH_EARLY} (Jan 1992 -> Mar 2026)')
    print(f'  Expected time  : {EXPECTED_TIME_DAYS} days since 1987-01-01')

    late_rows  = validate_late_25deg()
    early_rows = validate_early_25deg()
    gpcp_rows  = validate_gpcp()

    # --- Overall summary ---
    all_rows = late_rows + early_rows + gpcp_rows
    passed   = [r for r in all_rows if not r.get('error')]
    failed   = [r for r in all_rows if r.get('error')]

    print('\n' + '='*70)
    print('OVERALL VALIDATION SUMMARY')
    print('='*70)
    print(f'  Products validated successfully : {len(passed)}')
    print(f'  Products with errors           : {len(failed)}')
    if failed:
        for r in failed:
            print(f'    - {r["label"]} / {r["prod"]}: {r["error"]}')

    # Coords / time pass rate
    coords_pass = sum(1 for r in passed if r.get('coords_ok', False))
    time_pass   = sum(1 for r in passed if r.get('time_ok',   False))
    print(f'  Coordinates correct (lat/lon)  : {coords_pass}/{len(passed)}')
    print(f'  Time values correct            : {time_pass}/{len(passed)}')

    # SSA anchor across both standard constellations
    ssa_rows = [r for r in passed if r['prod'] == 'SSA']
    if ssa_rows:
        avg_ssa_exact = np.mean([r['exact_pct'] for r in ssa_rows])
        print(f'  SSA exact% (grid-layout anchor): {avg_ssa_exact:.2f}% avg '
              f'over {len(ssa_rows)} constellation(s)')

    # GPCP round-trip summary
    gpcp_passed = [r for r in gpcp_rows if not r.get('error')]
    if gpcp_passed:
        avg_gpcp_exact = np.mean([r['exact_pct'] for r in gpcp_passed])
        print(f'  GPCP round-trip exact%         : {avg_gpcp_exact:.2f}% avg '
              f'(NetCDF vs binary; {len(gpcp_passed)} products validated)')

    print('\nValidation complete.')


if __name__ == '__main__':
    main()
