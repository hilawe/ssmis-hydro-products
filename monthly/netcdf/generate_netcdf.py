#!/usr/bin/env python3
"""
generate_netcdf.py

PURPOSE
    Converts SSMIS binary product files to NetCDF-CF format.
    Replaces: generate_netcdf.sh + all IDL .pro files in netcdf/:
      - products_late_twohalfdeg_netcdf.pro   (late constellation: F08/F11/F13/F17 - 6pm ascending)
      - products_early_twohalfdeg_netcdf.pro  (early constellation: F10/F14/F15/F16 - morning ascending)
      - products_early_onedeg_netcdf.pro
      - gpcp_late_netcdf.pro
      - gpcp_early_netcdf.pro
      - gpcp_dual_netcdf.pro
      - products_pentad_netcdf.pro

SYNOPSIS
    python generate_netcdf.py [--dataset late25|early25|late10|gpcplate|gpcpearly|dual|all]

    With no arguments: runs all enabled datasets (same as generate_netcdf.sh default).

CONSTELLATION TERMINOLOGY - IMPORTANT
    "Late" constellation  = satellites with ~6pm local ascending equatorial crossing time.
                            Historical chain: F-08 -> F-11 -> F-13 -> F-17 (current); future: WSF-M.
                            Source directory: ../25deg-bin/ (currently fed by f17-2.5/ products).
                            Output filename suffix: _late_

    "Early" constellation = satellites with ~morning local ascending equatorial crossing time.
                            Historical chain: F-10 -> F-14 -> F-15 -> F-16 (current).
                            Source directory: ../f10-bin/ (currently fed by f16-2.5/ products).
                            Output filename suffix: _early_

    KNOWN HISTORICAL BUG IN IDL PIPELINE (discovered 2026-04-16):
    The operational IDL scripts products_late_twohalfdeg_netcdf.pro and
    products_early_twohalfdeg_netcdf.pro had these labels SWAPPED - the IDL "late" script
    processed f10-bin (F16/early constellation data) and the IDL "early" script processed
    25deg-bin (F17/late constellation data).  All v01 NetCDF files published to NCEI and
    distributed to customers therefore carry inverted constellation labels.

    This Python implementation corrects the inversion. The corrected files are still
    labeled v01 for now; a v02 re-publication under corrected filenames is planned once
    coordination with NCEI/downstream users is completed. See the project documentation
    Section 5.4 for full details.

    GPCP CDR products (run_gpcp_late_netcdf / run_gpcp_early_netcdf) were NEVER affected
    by this bug - gpcp_processing.py correctly labeled F17 as late and F16 as early from
    the start, and this file's GPCP functions match that convention.

CALLED BY
    run_ssmis.sh (via cd $hydroMONTHLY_NETCDF && python3 generate_netcdf.py)

NOTES
    - Binary input files are GrADS format: (N_LAT, N_LON) float32 per month,
      south-first latitude, 1.25°E-first longitude (for 2.5°).
      Lon (X) varies fastest within each lat row.
    - Output: individual per-month NetCDF4-CF files.
    - Days-since reference: 1987-01-01
"""

import os
import sys
import argparse
import datetime
import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    raise ImportError('netCDF4 package required: pip install netCDF4')

# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

GRID_25 = dict(nlat=72, nlon=144, res=2.5,
               lat_start=-88.75, lon_start=1.25)
GRID_10 = dict(nlat=180, nlon=360, res=1.0,
               lat_start=-89.5, lon_start=0.5)

REF_DATE = datetime.date(1987, 1, 1)
FILL_VALUE = np.float32(-999.99)

PRODUCT_NAMES = ['CFR', 'LWP', 'PF1', 'PF2', 'PR1', 'PR2', 'SSA', 'ICE', 'SNW', 'WVP']

# IDL index order was: CFR=0,LWP=1,PF1=2,PF2=3,PR1=4,PR2=5,SSA=6,ICE=7,SNW=8,WVP=9
CF_STANDARD = {
    'CFR': ('cloud_fraction',            '1',        'Monthly mean cloud fraction (0-1.0)'),
    'LWP': ('liquid_water_path',         '1000*mm',  'Monthly mean liquid water path'),
    'PF1': ('rain_fraction_alg1',        '1',        'Monthly mean rain fraction Algorithm #1'),
    'PF2': ('rain_fraction_alg2',        '1',        'Monthly mean rain fraction Algorithm #2'),
    'PR1': ('rainfall_alg1',             'mm',       'Monthly rainfall Algorithm #1'),
    'PR2': ('rainfall_alg2',             'mm',       'Monthly rainfall Algorithm #2'),
    'SSA': ('sampling_fraction',         '1',        'Monthly mean sampling fraction'),
    'ICE': ('sea_ice_cover',             'percent',  'Monthly mean sea-ice cover'),
    'SNW': ('snow_cover_fraction',       '1',        'Monthly mean snow cover fraction'),
    'WVP': ('total_precipitable_water',  'mm',       'Monthly mean total precipitable water'),
}

PRODUCT_TITLES_25 = {
    'CFR': 'GPCP Monthly 2.5 Degree Mean Cloud Fraction (0-1.0)',
    'LWP': 'GPCP Monthly 2.5 Degree Mean Liquid Water Path (1000*mm)',
    'PF1': 'GPCP Monthly 2.5 Degree Mean Rain Fraction Algorithm #1 (0-1.0)',
    'PF2': 'GPCP Monthly 2.5 Degree Mean Rain Fraction Algorithm #2 (0-1.0)',
    'PR1': 'GPCP Monthly 2.5 Degree Rainfall (mm) Algorithm #1',
    'PR2': 'GPCP Monthly 2.5 Degree Rainfall (mm) Algorithm #2',
    'SSA': 'GPCP Monthly 2.5 Degree Mean Sampling Fraction (0-1.0)',
    'ICE': 'GPCP Monthly 2.5 Degree Mean Sea-Ice Cover (0-100%)',
    'SNW': 'GPCP Monthly 2.5 Degree Mean Snow Cover Fraction (0-1.0)',
    'WVP': 'GPCP Monthly 2.5 Degree Mean Total Precipitable Water (mm)',
}

# 1.0-degree products.  PF2 and PR2 are not produced at 1.0-degree resolution -
# the 1.0-deg algorithm outputs only these 8 variables.
PRODUCT_NAMES_10 = ['CFR', 'LWP', 'PF1', 'PR1', 'SSA', 'ICE', 'SNW', 'WVP']

PRODUCT_TITLES_10 = {
    'CFR': 'NCDC Monthly 1.0 Degree Mean Cloud Fraction (0-1.0)',
    'LWP': 'NCDC Monthly 1.0 Degree Mean Liquid Water Path (1000*mm)',
    'PF1': 'NCDC Monthly 1.0 Degree Mean Rain Fraction Algorithm #1 (0-1.0)',
    'PR1': 'NCDC Monthly 1.0 Degree Mean Rainfall (mm) Algorithm #1',
    'SSA': 'NCDC Monthly 1.0 Degree Mean Sampling Fraction (0-1.0)',
    'ICE': 'NCDC Monthly 1.0 Degree Mean Sea-Ice Cover (0-100%)',
    'SNW': 'NCDC Monthly 1.0 Degree Mean Snow Cover Fraction (0-1.0)',
    'WVP': 'NCDC Monthly 1.0 Degree Mean Total Precipitable Water (mm)',
}


def make_coords(grid):
    """Return lat and lon coordinate arrays for a grid definition."""
    lats = np.array([grid['lat_start'] + i * grid['res'] for i in range(grid['nlat'])],
                    dtype=np.float32)
    lons = np.array([grid['lon_start'] + j * grid['res'] for j in range(grid['nlon'])],
                    dtype=np.float32)
    return lats, lons


def days_since_ref(year, month):
    """Days from 1987-01-01 to first day of (year, month)."""
    d = datetime.date(year, month, 1)
    return (d - REF_DATE).days


def iso8601(year, month):
    return f'{year:04d}-{month:02d}-01T00:00:00Z'


def get_n_months(binary_file, grid):
    """Infer number of months from file size."""
    if not os.path.exists(binary_file):
        return 0
    sz = os.path.getsize(binary_file)
    month_bytes = grid['nlat'] * grid['nlon'] * 4
    return sz // month_bytes


def read_month(fobj, grid):
    """Read one month of data from open binary file."""
    n = grid['nlat'] * grid['nlon']
    raw = np.fromfile(fobj, dtype=np.float32, count=n)
    if raw.size != n:
        return None
    # GrADS binary layout: lon (X) varies fastest within each lat row.
    # File stores rows south-to-north, each row is all 144 lons in order.
    # = C row-major (nlat, nlon) layout - reshape directly.
    return raw.reshape(grid['nlat'], grid['nlon'])  # (nlat, nlon)


def write_netcdf(ncfile, data_2d, lats, lons, year, month,
                 prod_key, title, initial_year, constellation,
                 history, summary, dataset_name):
    """
    Write one month of one product to a CF-compliant NetCDF4 file.

    data_2d: (nlat, nlon) float32 array
    """
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    var_name, units, description = CF_STANDARD[prod_key]

    with Dataset(ncfile, 'w', format='NETCDF4') as nc:
        # Dimensions
        nc.createDimension('time', 1)
        nc.createDimension('lat', len(lats))
        nc.createDimension('lon', len(lons))
        nc.createDimension('num_char', 20)

        # Global attributes (CF + NOAA CDR metadata)
        nc.Metadata_Conventions = 'CF-1.5, Unidata Dataset Discovery v1.0, NOAA CDR v1.0, GDS v2.0'
        nc.standard_name_vocabulary = 'CF Standard Name Table (v16, 11 October 2010)'
        nc.id = os.path.basename(ncfile).replace('.nc', '')
        nc.naming_authority = 'gov.noaa.ncdc'
        nc.metadata_link = 'gov.noaa.ncdc:C00790'
        nc.title = title
        nc.product_version = 'v01r00'
        nc.date_issued = '2012-07-30'
        nc.summary = summary
        nc.keywords = ('ATMOSPHERE > CLOUDS > CLOUD AMOUNT/FREQUENCY, '
                       'ATMOSPHERE > CLOUDS > CLOUD LIQUID WATER/ICE, '
                       'ATMOSPHERE > ATMOSPHERIC WATER VAPOR > PRECIPITABLE WATER, '
                       'ATMOSPHERE > PRECIPITATION > RAIN, '
                       'OCEANS > SEA ICE > ICE EXTENT')
        nc.keywords_vocabulary = ('NASA Global Change Master Directory (GCMD) Earth Science '
                                  'Keywords, Version 6.0')
        nc.platform = 'DMSP 5D-2 > Defense Meteorological Satellite Program'
        nc.sensor = ('SSM/I > Special Sensor Microwave/Imager; '
                     'SSMIS > Special Sensor Microwave Imager/Sounder;')
        nc.constellation = constellation
        nc.cdm_data_type = 'Grid'
        nc.source = dataset_name
        nc.date_created = now_str
        nc.creator_name = 'Hilawe Semunegus'
        nc.creator_url = 'http://www.ncdc.noaa.gov/oa/rsad/ssmi/gridded/index.php'
        nc.creator_email = 'NCDC.Satorder@noaa.gov'
        nc.institution = ('DOC/NOAA/NESDIS/NCDC > National Climatic Data Center, '
                          'NESDIS, NOAA, U.S. Department of Commerce')
        nc.processing_level = 'NOAA Level 3'
        nc.references = ('http://dx.doi.org/10.1175/1520-0477(1996)077%3C0891:'
                         'AEYTSO%3E2.0.CO;2, '
                         'http://dx.doi.org/10.1175/2009JAMC2294.1')
        nc.history = history
        nc.geospatial_lat_min = np.float32(-90.0)
        nc.geospatial_lat_max = np.float32(90.0)
        nc.geospatial_lon_min = np.float32(0.0)
        nc.geospatial_lon_max = np.float32(360.0)
        nc.geospatial_lat_units = 'degrees_north'
        nc.geospatial_lon_units = 'degrees_east'
        res_deg = abs(lats[1] - lats[0])
        nc.spatial_resolution = f'{res_deg:.1f} deg. X {res_deg:.1f} deg'
        nc.time_coverage_duration = 'P1M'
        nc.license = 'No restrictions on access or use'
        nc.contributor_name = 'Ralph Ferraro, Hilawe Semunegus'
        nc.contributor_role = ('Principal investigator and originator of input/source or antenna '
                               'temperature data, Processor and author of netCDF-4 CF version')

        # Variables
        t_var = nc.createVariable('time', 'f8', ('time',))
        t_var.standard_name = 'time'
        t_var.long_name = 'First day of the whole month that was measured or monthly-averaged'
        t_var.units = 'days since 1987-01-01'
        t_var.calendar = 'gregorian'
        t_var[:] = days_since_ref(year, month)

        iso_var = nc.createVariable('time_iso8601', 'S1', ('num_char', 'time'))
        iso_var.long_name = 'ISO8601 date/time'
        iso_str = iso8601(year, month)
        for ci, ch in enumerate(iso_str.ljust(20)):
            iso_var[ci, 0] = ch

        lat_var = nc.createVariable('lat', 'f4', ('lat',))
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var[:] = lats

        lon_var = nc.createVariable('lon', 'f4', ('lon',))
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var[:] = lons

        prod_var = nc.createVariable(var_name, 'f4', ('time', 'lat', 'lon'),
                                     fill_value=FILL_VALUE)
        prod_var.long_name = PRODUCT_TITLES_25.get(prod_key, description)
        prod_var.units = units
        prod_var.valid_min = np.float32(0.0)
        prod_var.Note = description
        prod_var.coordinates = 'time lat lon'
        prod_var[0, :, :] = data_2d

    print(f'  Written: {ncfile}')


def process_dataset(input_files, output_prefix, grid, initial_year, products,
                    constellation, title_prefix, history, summary, out_dir):
    """
    Generic NetCDF conversion loop for a set of product binary files.

    input_files: dict mapping product key -> binary file path
    output_prefix: prefix for output nc filenames (e.g. 'mw-hydro_v01_2.5-deg_cfr_late_')
    """
    os.makedirs(out_dir, exist_ok=True)
    lats, lons = make_coords(grid)
    month_names = ['01','02','03','04','05','06','07','08','09','10','11','12']

    # Determine record length from first available file
    ref_file = next((f for f in input_files.values() if os.path.exists(f)), None)
    if ref_file is None:
        print('  No input files found, skipping dataset')
        return

    n_months = get_n_months(ref_file, grid)
    print(f'  {n_months} months in record')

    for imonth in range(n_months):
        year  = initial_year + imonth // 12
        month = (imonth % 12) + 1
        iso = iso8601(year, month)
        mm  = month_names[month - 1]

        for prod_key in products:
            bin_file = input_files.get(prod_key)
            if not bin_file or not os.path.exists(bin_file):
                continue

            with open(bin_file, 'rb') as f:
                # Seek to the correct month
                offset = imonth * grid['nlat'] * grid['nlon'] * 4
                f.seek(offset)
                data_lat_lon = read_month(f, grid)

            if data_lat_lon is None:
                continue

            # Skip all-missing months
            if np.all(data_lat_lon <= -999.0):
                continue

            ncfile = os.path.join(out_dir, f'{output_prefix}{year:04d}{mm}.nc')

            write_netcdf(
                ncfile, data_lat_lon, lats, lons, year, month,
                prod_key,
                title=f'{title_prefix} {PRODUCT_TITLES_25.get(prod_key, prod_key)}',
                initial_year=initial_year,
                constellation=constellation,
                history=history,
                summary=summary,
                dataset_name=os.path.basename(bin_file),
            )


def run_late_25deg(out_dir='2.5-deg/'):
    """
    Late-constellation (6pm ascending equatorial crossing) 2.5° monthly products.

    Source directory: ../25deg-bin/  (populated by run_ssmis.sh from f17-2.5/).
    Satellite chain: SSM/I F-08 (Jul 1987) -> F-11 -> F-13 -> SSMIS F-17 (current).
    All satellites in this chain cross the equator on the ascending node at ~6pm local time.

    Replaces: products_early_twohalfdeg_netcdf.pro (IDL) - NOTE: the IDL script was
    misnamed "early" but contained the late (6pm) constellation data.  The Python
    implementation uses the scientifically correct name.  See module docstring for details.

    CALLED BY: main() via --dataset late25 or 'all'
    CALLS: write_netcdf(), make_coords(), get_n_months(), read_month()
    """
    # ../25deg-bin/ is populated each month by run_ssmis.sh:
    #   cp f17-2.5/{CFR,LWP,...}-f17-2.5  25deg-bin/{CFR,LWP,...}.MON
    # It accumulates the full multi-year late-constellation record
    # (originally F08, then F11, F13, and currently F17).
    path_in = '../25deg-bin/'
    initial_year = 1987   # F-08 data begins July 1987
    input_files = {p: os.path.join(path_in, f'{p}.MON') for p in PRODUCT_NAMES}

    # Satellite chain for the late (6pm ascending) equatorial crossing constellation.
    # F-08: July 1987 - December 1991
    # F-11: January 1992 - April 1995
    # F-13: May 1995 - December 2008
    # F-17: January 2009 - present  (future: WSF-M will continue this series)
    constellation = (
        'SSM/I F-08: July 1987-December 1991; '
        'SSM/I F-11: January 1992-April 1995; '
        'SSM/I F-13: May 1995-December 2008; '
        'SSMIS F-17: January 2009-present'
    )
    title_prefix = 'SSMI-SSMIS Hydrological 2.5 Degree Gridded Monthly Products (late constellation)'
    history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF file.'
    summary = f'NOAA STAR-EESIC-NCDC SSMI-SSMIS Hydrological Products from {initial_year}-present.'

    for prod_key in PRODUCT_NAMES:
        bin_file = input_files.get(prod_key)
        if not bin_file or not os.path.exists(bin_file):
            continue
        out_prefix = f'mw-hydro_v01_2.5-deg_{prod_key.lower()}_late_'
        lats, lons = make_coords(GRID_25)
        n_months = get_n_months(bin_file, GRID_25)

        for imonth in range(n_months):
            year  = initial_year + imonth // 12
            month = (imonth % 12) + 1
            mm = f'{month:02d}'

            with open(bin_file, 'rb') as f:
                f.seek(imonth * GRID_25['nlat'] * GRID_25['nlon'] * 4)
                data = read_month(f, GRID_25)
            if data is None or np.all(data <= -999.0):
                continue

            ncfile = os.path.join(out_dir, f'{out_prefix}{year:04d}{mm}.nc')
            os.makedirs(out_dir, exist_ok=True)
            write_netcdf(ncfile, data, lats, lons, year, month, prod_key,
                         title=PRODUCT_TITLES_25.get(prod_key, prod_key),
                         initial_year=initial_year, constellation=constellation,
                         history=history, summary=summary,
                         dataset_name=os.path.basename(bin_file))


def run_early_25deg(out_dir='2.5-deg/'):
    """
    Early-constellation (morning ascending equatorial crossing) 2.5° monthly products.

    Source directory: ../f10-bin/  (populated by run_ssmis.sh from f16-2.5/).
    Satellite chain: SSM/I F-10 (Jan 1992) -> F-14 -> F-15 -> SSMIS F-16 (current).
    All satellites in this chain cross the equator on the ascending node in the morning.
    The directory is named "f10-bin" after the first satellite in this chain (F-10).

    Replaces: products_late_twohalfdeg_netcdf.pro (IDL) - NOTE: the IDL script was
    misnamed "late" but contained the early (morning) constellation data.  The Python
    implementation uses the scientifically correct name.  See module docstring for details.

    CALLED BY: main() via --dataset early25 or 'all'
    CALLS: write_netcdf(), make_coords(), get_n_months(), read_month()
    """
    # ../f10-bin/ is populated each month by run_ssmis.sh:
    #   cp f16-2.5/{CFR,LWP,...}-f16-2.5  f10-bin/{CFR,LWP,...}.F10
    # It accumulates the full multi-year early-constellation record
    # (originally F10, then F14, F15, and currently F16).
    path_in = '../f10-bin/'
    initial_year = 1992   # F-10 data begins January 1992
    input_files = {
        'CFR': os.path.join(path_in, 'CFR.F10'),
        'LWP': os.path.join(path_in, 'LWP.F10'),
        'PF1': os.path.join(path_in, 'PF1.F10'),
        'PF2': os.path.join(path_in, 'PF2.F10'),
        'PR1': os.path.join(path_in, 'PR1.F10'),
        'PR2': os.path.join(path_in, 'PR2.F10'),
        'SSA': os.path.join(path_in, 'SSA.F10'),
        'ICE': os.path.join(path_in, 'ICE.F10'),
        'SNW': os.path.join(path_in, 'SNW.F10'),
        'WVP': os.path.join(path_in, 'WVP.F10'),
    }

    # Satellite chain for the early (morning ascending) equatorial crossing constellation.
    # F-10: January 1992 - September 1997
    # F-14: October 1997 - December 2001
    # F-15: January 2002 - June 2006
    # F-16: July 2006 - present
    constellation = (
        'SSM/I F-10: January 1992-September 1997; '
        'SSM/I F-14: October 1997-December 2001; '
        'SSM/I F-15: January 2002-June 2006; '
        'SSMIS F-16: July 2006-present'
    )
    title_prefix = 'SSMI-SSMIS Hydrological 2.5 Degree Gridded Monthly Products (early constellation)'
    history = (
        '1) 2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF file converted '
        'from the original gridded binary format. '
        '2) On October 18, 2017, netCDF files were revised due to a file encoding error '
        '(dates were incorrectly encoded).'
    )
    summary = f'NOAA STAR-EESIC-NCDC SSMI-SSMIS Hydrological Products from {initial_year}-present.'

    for prod_key in PRODUCT_NAMES:
        bin_file = input_files.get(prod_key)
        if not bin_file or not os.path.exists(bin_file):
            continue
        out_prefix = f'mw-hydro_v01_2.5-deg_{prod_key.lower()}_early_'
        lats, lons = make_coords(GRID_25)
        n_months = get_n_months(bin_file, GRID_25)

        for imonth in range(n_months):
            year  = initial_year + imonth // 12
            month = (imonth % 12) + 1
            mm    = f'{month:02d}'

            with open(bin_file, 'rb') as f:
                f.seek(imonth * GRID_25['nlat'] * GRID_25['nlon'] * 4)
                data = read_month(f, GRID_25)
            if data is None or np.all(data <= -999.0):
                continue

            ncfile = os.path.join(out_dir, f'{out_prefix}{year:04d}{mm}.nc')
            os.makedirs(out_dir, exist_ok=True)
            write_netcdf(ncfile, data, lats, lons, year, month, prod_key,
                         title=title_prefix,
                         initial_year=initial_year, constellation=constellation,
                         history=history, summary=summary,
                         dataset_name=os.path.basename(bin_file))


def run_gpcp_late_netcdf(out_dir='2.5-deg/'):
    """GPCP late constellation (f17/gpcp) NetCDF output."""
    path_in = '../gpcp/'
    initial_year = 1987
    gpcp_files = {
        'PR1': os.path.join(path_in, 'gpcp_nesdis_pr1.dat'),
        'PR2': os.path.join(path_in, 'gpcp_nesdis_pr2.dat'),
    }
    constellation = 'SSMIS F-17: January 1987-present'
    title = 'GPCP Monthly 2.5 Degree Rainfall (mm)'
    history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF GPCP file.'
    summary = f'NOAA STAR-EESIC-NCDC GPCP SSMI-SSMIS Hydrological Products from {initial_year}-present.'

    lats, lons = make_coords(GRID_25)
    for prod_key, bin_file in gpcp_files.items():
        if not os.path.exists(bin_file):
            continue
        out_prefix = f'mw-hydro_v01_gpcp_late_{prod_key.lower()}_'
        n_months = get_n_months(bin_file, GRID_25)

        for imonth in range(n_months):
            year  = initial_year + imonth // 12
            month = (imonth % 12) + 1
            mm    = f'{month:02d}'

            with open(bin_file, 'rb') as f:
                f.seek(imonth * GRID_25['nlat'] * GRID_25['nlon'] * 4)
                data = read_month(f, GRID_25)
            if data is None or np.all(data <= -999.0):
                continue

            ncfile = os.path.join(out_dir, f'{out_prefix}{year:04d}{mm}.nc')
            os.makedirs(out_dir, exist_ok=True)
            write_netcdf(ncfile, data, lats, lons, year, month, prod_key,
                         title=title, initial_year=initial_year,
                         constellation=constellation, history=history,
                         summary=summary,
                         dataset_name=os.path.basename(bin_file))


def run_gpcp_early_netcdf(out_dir='2.5-deg/'):
    """GPCP early constellation (f16) NetCDF output."""
    path_in = '../gpcp/'
    initial_year = 1992
    gpcp_files = {
        'PR1': os.path.join(path_in, 'gpcp_nesdis_f10_pr1.dat'),
        'PR2': os.path.join(path_in, 'gpcp_nesdis_f10_pr2.dat'),
    }
    constellation = 'SSMIS F-16: January 1992-present'
    title = 'GPCP Monthly 2.5 Degree Rainfall (mm) - Early'
    history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF GPCP file.'
    summary = f'NOAA STAR-EESIC-NCDC GPCP SSMI-SSMIS Hydrological Products from {initial_year}-present.'

    lats, lons = make_coords(GRID_25)
    for prod_key, bin_file in gpcp_files.items():
        if not os.path.exists(bin_file):
            continue
        out_prefix = f'mw-hydro_v01_gpcp_early_{prod_key.lower()}_'
        n_months = get_n_months(bin_file, GRID_25)

        for imonth in range(n_months):
            year  = initial_year + imonth // 12
            month = (imonth % 12) + 1
            mm    = f'{month:02d}'

            with open(bin_file, 'rb') as f:
                f.seek(imonth * GRID_25['nlat'] * GRID_25['nlon'] * 4)
                data = read_month(f, GRID_25)
            if data is None or np.all(data <= -999.0):
                continue

            ncfile = os.path.join(out_dir, f'{out_prefix}{year:04d}{mm}.nc')
            os.makedirs(out_dir, exist_ok=True)
            write_netcdf(ncfile, data, lats, lons, year, month, prod_key,
                         title=title, initial_year=initial_year,
                         constellation=constellation, history=history,
                         summary=summary,
                         dataset_name=os.path.basename(bin_file))


def run_gpcp_dual_netcdf(out_dir='2.5-deg/'):
    """GPCP dual-satellite merged NetCDF output."""
    path_in = '../gpcp/'
    initial_year = 1987
    gpcp_files = {
        'PR1': os.path.join(path_in, 'gpcp_nesdis_dual_pr1.dat'),
        'PR2': os.path.join(path_in, 'gpcp_nesdis_dual_pr2.dat'),
    }
    constellation = 'SSMIS F-17 and F-16 merged; January 1992-present'
    title = 'GPCP Monthly 2.5 Degree Rainfall (mm) - Dual Satellite'
    history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF dual GPCP file.'
    summary = f'NOAA STAR-EESIC-NCDC GPCP SSMI-SSMIS Dual Satellite Products from {initial_year}-present.'

    lats, lons = make_coords(GRID_25)
    for prod_key, bin_file in gpcp_files.items():
        if not os.path.exists(bin_file):
            continue
        out_prefix = f'mw-hydro_v01_gpcp_dual_{prod_key.lower()}_'
        n_months = get_n_months(bin_file, GRID_25)

        for imonth in range(n_months):
            year  = initial_year + imonth // 12
            month = (imonth % 12) + 1
            mm    = f'{month:02d}'

            with open(bin_file, 'rb') as f:
                f.seek(imonth * GRID_25['nlat'] * GRID_25['nlon'] * 4)
                data = read_month(f, GRID_25)
            if data is None or np.all(data <= -999.0):
                continue

            ncfile = os.path.join(out_dir, f'{out_prefix}{year:04d}{mm}.nc')
            os.makedirs(out_dir, exist_ok=True)
            write_netcdf(ncfile, data, lats, lons, year, month, prod_key,
                         title=title, initial_year=initial_year,
                         constellation=constellation, history=history,
                         summary=summary,
                         dataset_name=os.path.basename(bin_file))


def run_late_1deg(out_dir='1.0-deg/'):
    """
    Late-constellation (6pm ascending equatorial crossing) 1.0° monthly products.

    Source directory: ../ncdc-bin/  (populated by run_ssmis.sh from f17-1.0/).
    Satellite chain: SSM/I F-08 (Jul 1987) -> F-11 -> F-13 -> SSMIS F-17 (current).
    All satellites in this chain cross the equator on the ascending node at ~6pm local time.

    Binary file layout in ncdc-bin/:
        Filename pattern: {PROD}.{YY}  where YY is a 2-digit calendar year suffix.
        Year disambiguation:  YY 87-99 -> 1987-1999;  YY 00-86 -> 2000-2086.
        Each annual file holds exactly 12 consecutive months of data:
            360 (nlon) × 180 (nlat) × 4 bytes (float32) × 12 months = 3,110,400 bytes.
        Months are in chronological order (January first).

    Products available at 1.0-degree resolution (8 total - PF2 and PR2 are NOT produced
    by the 1.0-degree algorithm and are therefore absent from ncdc-bin/):
        CFR, LWP, PF1, PR1, SSA, ICE, SNW, WVP

    Replaces: products_early_onedeg_netcdf.pro (IDL).
    IMPORTANT HISTORICAL NOTE: That IDL script was MISNAMED "early" but processed data
    from ncdc-bin/ which is the late-constellation (F-08 -> F-11 -> F-13 -> F-17) record.  This
    Python implementation corrects the label to "late" - consistent with the 2.5-deg
    correction documented in the project documentation

    The IDL NetCDF files produced before November 2012 (when production was halted) also
    contained an incorrect date encoding bug.  The Python write_netcdf() function
    computes dates correctly, so that error is automatically corrected here.

    Output NetCDF files written to out_dir (default: 1.0-deg/ relative to this script):
        mw-hydro_v01_1.0-deg_{prod}_late_{YYYY}{MM}.nc
    for every month and product where the source binary is present and non-missing.

    CALLED BY: main() via --dataset late10 or 'all'
    CALLS: write_netcdf(), make_coords(), read_month()
    """
    path_in = '../ncdc-bin/'
    initial_year = 1987   # SSM/I F-08 data begins July 1987

    constellation = (
        'SSM/I F-08: July 1987-December 1991; '
        'SSM/I F-11: January 1992-April 1995; '
        'SSM/I F-13: May 1995-December 2008; '
        'SSMIS F-17: January 2009-present'
    )
    title_prefix = (
        'SSMI-SSMIS Hydrological 1.0 Degree Gridded Monthly Products '
        '(late constellation)'
    )
    history = (
        '1) 2012-07-01, Hilawe Semunegus, NOAA/NCDC, created netCDF file converted '
        'from the original 1.0-degree gridded binary format. '
        '2) 2026-04-22, Hilawe Semunegus, NOAA/NCEI, restarted production under Python '
        'pipeline; corrected constellation label from erroneous "early" (IDL script '
        'products_early_onedeg_netcdf.pro) to correct "late"; corrected date encoding.'
    )
    summary = (
        f'NOAA STAR-EESIC-NCDC SSMI-SSMIS Hydrological 1.0-Degree Products '
        f'from {initial_year}-present.'
    )

    os.makedirs(out_dir, exist_ok=True)
    lats, lons = make_coords(GRID_10)

    # -----------------------------------------------------------------------
    # Build a chronologically sorted list of (calendar_year, yy_string) pairs
    # from the files actually present in ncdc-bin/.  The reference product is
    # CFR - if CFR.{YY} exists, the full set of product files for that year
    # should also be present (they are written together by run_ssmis.sh).
    # 2-digit year rule: 87-99 -> 1987-1999; 00-86 -> 2000-2086.
    # -----------------------------------------------------------------------
    available_years = []
    if not os.path.isdir(path_in):
        print(f'  ERROR: ncdc-bin/ not found at {path_in} - aborting 1.0-deg NetCDF generation')
        return

    for entry in sorted(os.listdir(path_in)):
        if not entry.startswith('CFR.'):
            continue
        yy_str = entry.split('.', 1)[1]          # e.g. '87', '99', '00', '26'
        if not yy_str.isdigit() or len(yy_str) != 2:
            continue
        yy_int = int(yy_str)
        cal_year = (1900 + yy_int) if yy_int >= 87 else (2000 + yy_int)
        available_years.append((cal_year, yy_str))

    available_years.sort(key=lambda t: t[0])     # chronological order

    if not available_years:
        print('  No annual binary files found in ncdc-bin/ - nothing to convert')
        return

    print(f'  Found {len(available_years)} year-files in ncdc-bin/ '
          f'({available_years[0][0]}-{available_years[-1][0]})')

    # -----------------------------------------------------------------------
    # Main loop: for each available year, open each product's annual binary
    # file, read all 12 months, and write individual per-month NetCDF files.
    # -----------------------------------------------------------------------
    for cal_year, yy_str in available_years:

        # Read all 12 months for each available product into memory for this year.
        # Storing as a dict of lists avoids re-opening files per month.
        year_data = {}   # product_key -> list of 12 (nlat, nlon) float32 arrays

        for prod_key in PRODUCT_NAMES_10:
            bin_file = os.path.join(path_in, f'{prod_key}.{yy_str}')
            if not os.path.exists(bin_file):
                continue
            with open(bin_file, 'rb') as fobj:
                months_list = []
                for _ in range(12):
                    arr = read_month(fobj, GRID_10)
                    months_list.append(arr)          # None if file truncated
            year_data[prod_key] = months_list

        if not year_data:
            print(f'  {cal_year}: no product files found in ncdc-bin/, skipping')
            continue

        # Write one NetCDF file per (month, product) - skip all-missing months.
        written = 0
        for month_idx in range(12):
            month = month_idx + 1    # 1-based calendar month number
            mm    = f'{month:02d}'

            for prod_key, months_list in year_data.items():
                data_2d = months_list[month_idx]
                if data_2d is None:
                    continue
                # Skip months where every grid cell is missing (fill = -999.99).
                # This is common for Jan-Jun 1987 (F-08 launched Jul 1987).
                if np.all(data_2d <= -999.0):
                    continue

                ncfile = os.path.join(
                    out_dir,
                    f'mw-hydro_v01_1.0-deg_{prod_key.lower()}_late_{cal_year:04d}{mm}.nc'
                )
                write_netcdf(
                    ncfile, data_2d, lats, lons, cal_year, month,
                    prod_key,
                    title=f'{title_prefix} - {PRODUCT_TITLES_10.get(prod_key, prod_key)}',
                    initial_year=initial_year,
                    constellation=constellation,
                    history=history,
                    summary=summary,
                    dataset_name=f'{prod_key}.{yy_str}',
                )
                written += 1

        print(f'  {cal_year}: {written} NetCDF files written')


def main():
    parser = argparse.ArgumentParser(description='Generate NetCDF-CF files from SSMIS binary products')
    parser.add_argument('--dataset', choices=[
        'late25', 'early25', 'late10', 'gpcplate', 'gpcpearly', 'dual', 'all'
    ], default='all', help='Which dataset to convert (default: all)')
    args = parser.parse_args()

    ds = args.dataset
    print('=== NetCDF-CF Generation ===')

    if ds in ('late25', 'all'):
        # Late constellation: 6pm ascending node chain (F08 -> F11 -> F13 -> F17), reads 25deg-bin/
        # NOTE: previously misnamed "early" in IDL products_early_twohalfdeg_netcdf.pro
        print('\n--- Late constellation 2.5° (corrected from IDL products_early_twohalfdeg_netcdf) ---')
        run_late_25deg()

    if ds in ('early25', 'all'):
        # Early constellation: morning ascending node chain (F10 -> F14 -> F15 -> F16), reads f10-bin/
        # NOTE: previously misnamed "late" in IDL products_late_twohalfdeg_netcdf.pro
        print('\n--- Early constellation 2.5° (corrected from IDL products_late_twohalfdeg_netcdf) ---')
        run_early_25deg()

    if ds in ('late10', 'all'):
        # Late constellation 1.0-degree products from ncdc-bin/ (F08 -> F11 -> F13 -> F17).
        # Replaces IDL products_early_onedeg_netcdf.pro (misnamed "early"; corrected here).
        # Production was halted in Nov 2012 due to NetCDF date encoding errors; restarted
        # under Python pipeline 2026-04-22 with correct date encoding and label.
        #
        # Files are generated into netcdf/1.0-deg/ each monthly run but are NOT tarred
        # or transferred to NCEI ingest - tar_mw-hydro_netcdf.sh and transfer_netcdf.sh
        # only loop over '2.5-deg', 'gpcp-input', and 'imagery', so 1.0-deg is
        # automatically excluded from archive until activated for v02.
        # See the project documentation for the v02 activation plan.
        print('\n--- Late constellation 1.0° (replaces IDL products_early_onedeg_netcdf) ---')
        run_late_1deg()

    if ds in ('gpcplate', 'all'):
        # GPCP late: F17 (6pm ascending) - was correctly labeled in IDL and Python
        print('\n--- GPCP late constellation (gpcp_late_netcdf, F17) ---')
        run_gpcp_late_netcdf()

    if ds in ('gpcpearly', 'all'):
        # GPCP early: F16 (morning ascending) - was correctly labeled in IDL and Python
        print('\n--- GPCP early constellation (gpcp_early_netcdf, F16) ---')
        run_gpcp_early_netcdf()

    if ds in ('dual', 'all'):
        print('\n--- GPCP dual (gpcp_dual_netcdf) ---')
        run_gpcp_dual_netcdf()

    print('\nDone.')


if __name__ == '__main__':
    main()
