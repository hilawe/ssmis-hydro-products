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
import numpy as np
from scipy.ndimage import maximum_filter1d

# Re-use all algorithm functions from climalg_ssmis
from climalg_ssmis import (
    NCOL, NROW, NT,
    ta2tb, precip8, precip3, snowc, seaice, cloud, vapor,
    mean_precip, mean_lwp, mean_wvp, mean_snw, mean_ice, mean_ssa,
    to_grads, load_luts, load_land_tag,
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
    Process one TDR file, accumulate into rrmon.
    Identical logic to climalg_ssmis.process_file but imported here inline
    to avoid circular imports.
    """
    raw = np.fromfile(fpath, dtype=np.uint8)
    expected = NROW * NCOL * 7
    if raw.size != expected:
        print(f'  Warning: size mismatch for {fpath}, skipping')
        return False

    data = raw.reshape(NROW, NCOL, 7)
    ta = data.astype(np.float32) + 70.0

    skip = ((ta[:, :, 0] == 102.0) & (ta[:, :, 2] == 102.0) &
            (ta[:, :, 3] == 102.0) & (ta[:, :, 4] == 102.0) &
            (ta[:, :, 6] == 102.0))

    kindex = np.clip(np.round(ta).astype(np.int32) - 139, 0, 160)
    is_ocean = (tag_2d == 0)
    is_missing_ch = (ta == 102.0)

    ta_corr = ta.copy()
    for ch in range(7):
        k_ch = kindex[:, :, ch]
        valid_ch = ~is_missing_ch[:, :, ch]
        ta_corr[:, :, ch] = np.where(valid_ch & is_ocean,
                                      ta[:, :, ch] - add_oce[k_ch, ch],
                             np.where(valid_ch & ~is_ocean,
                                      ta[:, :, ch] - add_lnd[k_ch, ch],
                                      ta[:, :, ch]))

    tb = ta2tb(ta_corr)

    lsrain = maximum_filter1d(tag_2d.astype(np.int8), size=5, axis=1,
                              mode='constant', cval=0).astype(bool)

    rlat = (89.667 - 0.333 * np.arange(NROW)).astype(np.float32)
    isnchk = np.zeros(NROW, dtype=bool)
    if JDAY <= 181:
        isnchk |= (rlat >= 60.0)
        if JDAY <= 151:
            isnchk |= (rlat >= 40.0)
        if JDAY <= 91:
            isnchk |= (rlat >= 25.0)
    isnchk_2d = isnchk[:, np.newaxis]

    rain1 = precip8(ta_corr, tb, lsrain,
                    np.broadcast_to(isnchk_2d, (NROW, NCOL)), add_si)
    rain2 = precip3(ta_corr, lsrain)
    is_land_2d = (tag_2d == 1)
    snow  = snowc(ta_corr, is_land_2d)

    ichan = 3 if (YYYY == 1990 and JDAY >= 181) or YYYY == 1991 else 8
    sice  = seaice(ta_corr, is_ocean, ichan)

    row_idx = np.arange(NROW)
    equat_mask = (np.abs(row_idx - NROW // 2) * (360.0 / NCOL) <= 40.0)[:, np.newaxis]
    sice = np.where(equat_mask & (sice == 100.0), 0.0, sice)

    lwp = cloud(ta_corr, tb, is_ocean, sice)
    wvp = vapor(ta_corr, tb, is_ocean, sice, add_si)

    ii_2d = ii_map[:, np.newaxis]
    jj_2d = jj_map[np.newaxis, :]
    ii_flat = np.broadcast_to(ii_2d, (NROW, NCOL)).ravel()
    jj_flat = np.broadcast_to(jj_2d, (NROW, NCOL)).ravel()
    valid_flat = (~skip).ravel()

    np.add.at(rrmon[:, :, NT - 3], (ii_flat, jj_flat), 1)

    ii_v = ii_flat[valid_flat]
    jj_v = jj_flat[valid_flat]

    def acc(k, vals):
        np.add.at(rrmon[:, :, k], (ii_v, jj_v), vals)

    is_ocean_flat = is_ocean.ravel()[valid_flat]
    acc(NT - 1, is_ocean_flat.astype(np.float32))
    acc(NT - 2, (~is_ocean_flat).astype(np.float32))

    rain1_flat = rain1.ravel()[valid_flat]
    rain2_flat = rain2.ravel()[valid_flat]
    snow_flat  = snow.ravel()[valid_flat]
    sice_flat  = sice.ravel()[valid_flat]
    lwp_flat   = lwp.ravel()[valid_flat]
    wvp_flat   = wvp.ravel()[valid_flat]

    r1_pos = rain1_flat > 0.0
    acc(0, np.where(r1_pos, rain1_flat, 0.0))
    acc(1, r1_pos.astype(np.float32))

    r2_pos = rain2_flat > 0.0
    acc(2, np.where(r2_pos, rain2_flat, 0.0))
    acc(3, r2_pos.astype(np.float32))

    lwp_pos = lwp_flat > 0.02
    acc(4, np.where(lwp_pos, lwp_flat, 0.0))
    acc(5, lwp_pos.astype(np.float32))

    wvp_pos = wvp_flat > 0.0
    acc(6, np.where(wvp_pos, wvp_flat, 0.0))
    acc(7, wvp_pos.astype(np.float32))

    acc(8, (snow_flat == 100.0).astype(np.float32))
    acc(9, (sice_flat == 100.0).astype(np.float32))

    return True


def run(sat, yyyy, jday_start, jday_end):
    year_str = f'{yyyy:04d}'
    outpath = f'{sat}-2.5/'
    os.makedirs(outpath, exist_ok=True)

    print(f'pentad_ssmis: sat={sat}, year={yyyy}, jdays=1-{jday_end}')

    add_oce, add_lnd, add_si = load_luts('.')
    tag_2d = load_land_tag('NLNDSEA.TAG')

    RES = 360.0 / NCOL
    ii_map = np.minimum((np.arange(NROW) * RES / SIZE).astype(int), M - 1)
    jj_map = np.minimum((np.arange(NCOL) * RES / SIZE).astype(int), N - 1)

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
            fpath = os.path.join(INPATH, fname)
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
    args = parser.parse_args()
    run(args.sat, args.yyyy, args.jday_start, args.jday_end)


if __name__ == '__main__':
    main()
