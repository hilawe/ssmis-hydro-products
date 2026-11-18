#!/usr/bin/env python3
"""
pentad_ssmis.py

PURPOSE
    Generates 5-day averaged (pentad) products at 2.5-degree resolution
    for the full year up through jday_end.
    Replaces pentad-ssmis-2.5deg.f

SYNOPSIS
    python pentad_ssmis.py <sat> <yyyy> <jday_start> <jday_end>

EXAMPLE
    python pentad_ssmis.py f16 2012 001 031

NOTES
    - Always start from jday=1 (pentad 1) and run through jday_end.
    - Pentad boundaries: every 5 days except pentad 12 (Feb 26-28/29).
    - Output: one binary file per product, (N, M, 73) float32 in GrADS layout.
    - Each pentad "page" in the output is N*M floats (one per grid cell).
"""

import sys
import os
import argparse
import calendar
import hashlib
import numpy as np

# The per-file processing, the averaging functions and the table loaders are the
# monthly processor's own. Nothing of the retrieval is restated in this module.
from climalg_ssmis import (
    NCOL, NROW, NT,
    mean_precip, mean_lwp, mean_wvp, mean_snw, mean_ice, mean_ssa,
    load_luts, load_land_tag, process_file, resolve_si_path,
)

M, N, SIZE = 72, 144, 2.5
INPATH = '../SSMIS_Grid/'
N_PENTADS = 73


def build_pentad_boundaries(yyyy):
    """
    Build array IPEN of size 73: cumulative day-of-year at end of each pentad.
    Pentad 12 (Feb) is extended to include leap-day if applicable.
    """
    ipen = np.zeros(N_PENTADS, dtype=int)
    for i in range(11):
        ipen[i] = (i + 1) * 5
    ipen[11] = ipen[10] + 6 if calendar.isleap(yyyy) else ipen[10] + 5
    for i in range(12, N_PENTADS):
        ipen[i] = ipen[i - 1] + 5
    return ipen


def process_file_pentad(fpath, tag_2d, add_oce, add_lnd, add_si,
                         rrmon, ii_map, jj_map, YYYY, JDAY):
    """
    Process one TDR file and accumulate it into rrmon, on the 2.5-degree grid.

    This calls climalg_ssmis.process_file, the monthly processor's function. The
    legacy pentad and monthly Fortran programs have the same per-pixel code
    (pentad-ssmis-2.5deg.f and climalg-ssmis-2.5deg.f do not differ between the
    start of the day loop and the output step), so one function serves both.

    Until 18 November 2026 this function was a second copy of that code, kept
    here "to avoid circular imports", although no import between the two modules
    is circular. The copy did not receive two repairs the monthly processor got
    on 9 October 2026: it tested five channels for a missing pixel where the
    Fortran tests six, and it masked equatorial sea ice one row off. Sharing the
    function removes the duplicated per-file algorithm. This wrapper must still
    pass the same grid, date and table arguments, in the order below.
    """
    return process_file(fpath, tag_2d, add_oce, add_lnd, add_si,
                        rrmon, ii_map, jj_map, M, N, YYYY, JDAY, NROW // 2)


def run(sat, yyyy, jday_start, jday_end, outdir=None, inpath=None, coeff_dir=None):
    """
    Run pentad (5-day) averaging for the given satellite and year.

    outdir    : parent directory for {sat}-2.5/ output (default: current working dir)
    inpath    : path to the 1/3-degree TDR input files (default: ../SSMIS_Grid/)
    coeff_dir : directory containing LUT and NLNDSEA.TAG files (default: .)
    """
    year_str = f'{yyyy:04d}'
    _inpath    = inpath    if inpath    is not None else INPATH
    _coeff_dir = coeff_dir if coeff_dir is not None else '.'

    if outdir is not None:
        outpath = os.path.join(outdir, f'{sat}-2.5/')
    else:
        outpath = f'{sat}-2.5/'
    os.makedirs(outpath, exist_ok=True)

    print(f'pentad_ssmis: sat={sat}, year={yyyy}, jdays=1-{jday_end}')
    print(f'  inpath={_inpath}  coeff={_coeff_dir}  outpath={outpath}')

    # The scattering-index table is the satellite's own when satellite_config names
    # one and no coefficient directory was given, as in the monthly processor. Until
    # 18 November 2026 the pentad processor always read the standard table.
    _si_path = resolve_si_path(sat, coeff_dir)
    print(f'  lut_add_si={_si_path}  sha256={hashlib.sha256(open(_si_path, "rb").read()).hexdigest()}')
    add_oce, add_lnd, add_si = load_luts(_coeff_dir, si_path=_si_path)
    tag_2d = load_land_tag(os.path.join(_coeff_dir, 'NLNDSEA.TAG'))

    # 1-indexed binning to match Fortran INT(I*RES/SIZE)+1 convention;
    # see the same fix in climalg_ssmis.py for detailed explanation.
    RES = 360.0 / NCOL
    ii_map = np.minimum(((np.arange(NROW) + 1) * RES / SIZE).astype(int), M - 1)
    jj_map = np.minimum(((np.arange(NCOL) + 1) * RES / SIZE).astype(int), N - 1)

    ipen = build_pentad_boundaries(yyyy)
    print(f'Pentad boundaries: {ipen}')

    # Output arrays: (N, M, N_PENTADS) for each product - GrADS layout
    xp = {name: np.full((N, M, N_PENTADS), -999.99, dtype=np.float32)
          for name in ['PR1', 'PF1', 'PR2', 'PF2', 'LWP', 'CFR', 'WVP', 'SNW', 'ICE', 'SSA']}

    rrmon = np.zeros((M, N, NT), dtype=np.float32)
    imons = 0

    # Always start from jday 1 for pentad
    for jday in range(1, jday_end + 1):
        jday_str = f'{jday:03d}'

        for prefix in ('as', 'ds'):
            fname = f'{prefix}{year_str[2:4]}{jday_str}.{sat}'
            fpath = os.path.join(_inpath, fname)
            if os.path.exists(fpath):
                print(f'  Processing {fname}')
                ok = process_file_pentad(fpath, tag_2d, add_oce, add_lnd, add_si,
                                          rrmon, ii_map, jj_map, yyyy, jday)
                if ok:
                    imons += 1
            else:
                print(f'  Warning! Missing file: {fpath}')

        # Check if this jday is a pentad boundary
        for ip, pend in enumerate(ipen):
            if jday == pend:
                print(f'  Pentad {ip + 1} boundary at jday={jday}')

                # Compute and store output for each product
                xp['PR1'][:, :, ip] = mean_precip(rrmon, M, N, NT, 5, 1, 8)
                xp['PF1'][:, :, ip] = mean_precip(rrmon, M, N, NT, 5, 2, 8)
                xp['PR2'][:, :, ip] = mean_precip(rrmon, M, N, NT, 5, 1, 3)
                xp['PF2'][:, :, ip] = mean_precip(rrmon, M, N, NT, 5, 2, 3)
                xp['LWP'][:, :, ip] = mean_lwp(rrmon, M, N, NT, 1)
                xp['CFR'][:, :, ip] = mean_lwp(rrmon, M, N, NT, 2)
                xp['WVP'][:, :, ip] = mean_wvp(rrmon, M, N, NT)
                xp['SNW'][:, :, ip] = mean_snw(rrmon, M, N, NT)
                xp['ICE'][:, :, ip] = mean_ice(rrmon, M, N, NT)
                xp['SSA'][:, :, ip] = mean_ssa(rrmon, M, N, NT)

                # Reset accumulator for next pentad
                rrmon[:] = 0
                imons = 0
                break

    # Write output files: one per product.
    # xp[name] has shape (N, M, N_PENTADS) = (144, 72, 73) with dim0=lon, dim1=lat.
    # GrADS binary expects (pentad, lat, lon) = (73, 72, 144) C row-major order,
    # so transpose to (N_PENTADS, M, N) = (73, 72, 144) before writing.
    yr2 = year_str[2:4]
    for name, data in xp.items():
        fname = os.path.join(outpath, f'{name}{yr2}-{sat}-2.5.pen')
        data.transpose(2, 1, 0).tofile(fname)
        print(f'Written: {fname}')


def main():
    parser = argparse.ArgumentParser(description='SSMIS pentad (5-day) products')
    parser.add_argument('sat',        help='Satellite name (e.g. f16)')
    parser.add_argument('yyyy',       type=int)
    parser.add_argument('jday_start', type=int)
    parser.add_argument('jday_end',   type=int)
    # Optional path overrides - mirrors the same arguments in climalg_ssmis.py
    # to support validation test runs in non-standard directories.
    parser.add_argument('--outdir',   default=None,
                        help='Parent directory for {sat}-2.5/ output (default: .)')
    parser.add_argument('--inpath',   default=None,
                        help='Path to 1/3-degree TDR input files (default: ../SSMIS_Grid/)')
    parser.add_argument('--coeff',    default=None,
                        help='Directory with LUT and NLNDSEA.TAG files (default: .)')
    args = parser.parse_args()
    run(args.sat, args.yyyy, args.jday_start, args.jday_end,
        outdir=args.outdir, inpath=args.inpath, coeff_dir=args.coeff)


if __name__ == '__main__':
    main()
