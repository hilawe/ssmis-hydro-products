#!/usr/bin/env python3
"""
generate_netcdf.py

PURPOSE
    Converts SSMIS binary product files to NetCDF-CF format.
    Replaces: generate_netcdf.sh + all IDL .pro files in netcdf/:
      - products_late_twohalfdeg_netcdf.pro
      - products_early_twohalfdeg_netcdf.pro
      - products_early_onedeg_netcdf.pro
      - gpcp_late_netcdf.pro
      - gpcp_early_netcdf.pro
      - gpcp_dual_netcdf.pro
      - products_pentad_netcdf.pro

SYNOPSIS
    python generate_netcdf.py [--dataset late25|early25|early10|gpcplate|gcpcparly|dual|pentad]

    With no arguments: runs all enabled datasets (same as generate_netcdf.sh default).

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

TITLES_25_LATE = {
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
        prod_var.long_name = TITLES_25_LATE.get(prod_key, description)
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
                title=f'{title_prefix} {TITLES_25_LATE.get(prod_key, prod_key)}',
                initial_year=initial_year,
                constellation=constellation,
                history=history,
                summary=summary,
                dataset_name=os.path.basename(bin_file),
            )


def run_late_25deg(out_dir='2.5-deg/'):
    """Late constellation (f10-bin) 2.5° monthly products."""
    path_in = '../f10-bin/'
    initial_year = 1992
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
    constellation = (
        'SSM/I F-10: January 1992-September 1997; '
        'SSM/I F-14: October 1997-December 2001; '
        'SSM/I F-15: January 2002-June 2006; '
        'SSMIS F-16: July 2006-present'
    )
    title_prefix = 'SSMI-SSMIS Hydrological 2.5 Degree Gridded Monthly Products (late constellation)'
    history = (
        '1) 2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF file converted '
        'from the original gridded binary format '
        '2) On October 18, 2017, netCDF files were revised due to a file encoding error '
        '(dates were incorrectly encoded).'
    )
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
                         title=TITLES_25_LATE.get(prod_key, prod_key),
                         initial_year=initial_year, constellation=constellation,
                         history=history, summary=summary,
                         dataset_name=os.path.basename(bin_file))


def run_early_25deg(out_dir='2.5-deg/'):
    """Early constellation (25deg-bin) 2.5° monthly products."""
    path_in = '../25deg-bin/'
    initial_year = 1987
    files = {p: os.path.join(path_in, f'{p}.MON') for p in PRODUCT_NAMES}
    constellation = (
        'SSM/I F-08: July 1987-December 1991; '
        'SSM/I F-10: January 1992-September 1997; '
        'SSMIS F-17: present'
    )
    title_prefix = 'SSMI-SSMIS Hydrological 2.5 Degree Gridded Monthly Products (early constellation)'
    history = '2012-07-30, Hilawe Semunegus, NOAA/NCDC, created netCDF file.'
    summary = f'NOAA STAR-EESIC-NCDC SSMI-SSMIS Hydrological Products from {initial_year}-present.'

    for prod_key in PRODUCT_NAMES:
        bin_file = files.get(prod_key)
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


def main():
    parser = argparse.ArgumentParser(description='Generate NetCDF-CF files from SSMIS binary products')
    parser.add_argument('--dataset', choices=[
        'late25', 'early25', 'gpcplate', 'gpcpearly', 'dual', 'all'
    ], default='all', help='Which dataset to convert (default: all)')
    args = parser.parse_args()

    ds = args.dataset
    print('=== NetCDF-CF Generation ===')

    if ds in ('late25', 'all'):
        print('\n--- Late constellation 2.5° (products_late_twohalfdeg_netcdf) ---')
        run_late_25deg()

    if ds in ('early25', 'all'):
        print('\n--- Early constellation 2.5° (products_early_twohalfdeg_netcdf) ---')
        run_early_25deg()

    if ds in ('gpcplate', 'all'):
        print('\n--- GPCP late (gpcp_late_netcdf) ---')
        run_gpcp_late_netcdf()

    if ds in ('gpcpearly', 'all'):
        print('\n--- GPCP early (gpcp_early_netcdf) ---')
        run_gpcp_early_netcdf()

    if ds in ('dual', 'all'):
        print('\n--- GPCP dual (gpcp_dual_netcdf) ---')
        run_gpcp_dual_netcdf()

    print('\nDone.')


if __name__ == '__main__':
    main()
