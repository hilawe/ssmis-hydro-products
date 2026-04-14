#!/usr/bin/env python3
"""
combine.py

PURPOSE
    Combines monthly per-year product files into a single multi-year binary
    file for each product type.  Replaces Combine-2.5deg.f and Combine-1.0deg.f.

SYNOPSIS
    python combine.py <sat> <start_year> <end_year> <current_month> [--res 2.5|1.0]

EXAMPLE
    python combine.py f16 1992 2012 01 --res 2.5
    python combine.py f17 1987 2012 01 --res 2.5
    python combine.py f16 2012 2012 06 --res 1.0

INPUTS
    sat            : satellite name (f16, f17, f18, …)
    start_year     : first year in the combined record
    end_year       : current (last) year
    current_month  : two-digit month being processed (e.g. '06')
    --res          : grid resolution (2.5 or 1.0)

OUTPUT
    <sat>-<res>/<PROD>-<sat>-<res>   for 2.5-degree
    <sat>-<res>/<PROD>.<YY>-<sat>-<res>  for 1.0-degree (single-year style)

NOTES
    - Files missing from disk are filled with -999.99.
    - Binary layout matches GrADS (N cols × M rows, float32 per month).
    - For 2.5-deg output: one full record per month row (N float32 per row,
      M rows per month) -> multi-year time-series.
    - For 1.0-deg output: single-year time-series (12 months × M rows × N cols).
"""

import sys
import os
import argparse
import numpy as np

PRODUCTS = ['CFR', 'LWP', 'PR1', 'PR2', 'PF1', 'PF2', 'SSA', 'WVP', 'SNW', 'ICE']
MISSING = np.float32(-999.99)


def get_dims(res):
    if res == 2.5:
        return 72, 144    # M rows, N cols
    elif res == 1.0:
        return 180, 360
    else:
        raise ValueError(f'Unsupported resolution {res}')


def combine_25deg(sat, start_year, end_year, current_month):
    """Combine all years into a single multi-year binary per product (2.5°)."""
    M, N = get_dims(2.5)
    path = f'{sat}-2.5/'
    bad = np.full((N, M), MISSING, dtype=np.float32)  # GrADS: (N cols, M rows)
    res_str = '2.5'

    for prod in PRODUCTS:
        out_file = os.path.join(path, f'{prod}-{sat}-{res_str}')
        print(f'\nProcessing {out_file}')
        rows_written = 0

        with open(out_file, 'wb') as fout:
            for yr in range(start_year, end_year + 1):
                yr2 = f'{yr % 100:02d}'
                for mo in range(1, 13):
                    mm = f'{mo:02d}'
                    in_file = os.path.join(path, f'{prod}{yr2}-{mm}-{sat}-{res_str}')

                    if yr < end_year:
                        # Previous years: skip (seek forward)
                        rows_written += M
                        fout.seek(rows_written * N * 4)
                    elif yr == end_year and mo < current_month:
                        # Current year, previous months: read existing file
                        if os.path.exists(in_file):
                            print(f'  Appending from {in_file}')
                            data = np.fromfile(in_file, dtype=np.float32)
                            if data.size == N * M:
                                # data is (N, M) in Fortran column-major layout
                                # Write M rows of N values each
                                for k in range(M):
                                    row = data[k * N:(k + 1) * N]
                                    fout.write(row.tobytes())
                                rows_written += M
                            else:
                                print(f'  Warning: unexpected size for {in_file}')
                                for k in range(M):
                                    fout.write(bad[:, k].tobytes())
                                rows_written += M
                        else:
                            print(f'  Missing {in_file}, filling with -999.99')
                            for k in range(M):
                                fout.write(bad[:, k].tobytes())
                            rows_written += M
                    elif yr == end_year and mo <= current_month:
                        # Current month
                        if os.path.exists(in_file):
                            print(f'  Appending current month from {in_file}')
                            data = np.fromfile(in_file, dtype=np.float32)
                            if data.size == N * M:
                                for k in range(M):
                                    fout.write(data[k * N:(k + 1) * N].tobytes())
                                rows_written += M
                            else:
                                for k in range(M):
                                    fout.write(bad[:, k].tobytes())
                                rows_written += M
                        else:
                            print(f'  Missing {in_file}, filling with -999.99')
                            for k in range(M):
                                fout.write(bad[:, k].tobytes())
                            rows_written += M
                    else:
                        # Future months
                        print(f'  Filling future month {mm} with -999.99')
                        for k in range(M):
                            fout.write(bad[:, k].tobytes())
                        rows_written += M

        print(f'  Total rows written: {rows_written}')


def combine_25deg_efficient(sat, start_year, end_year, current_month):
    """
    Efficient version: rebuild the combined file by reading/copying/filling
    month-by-month. The output format is: for each month, M rows of N float32.
    Previous-year records are copied if existing, else re-filled.
    """
    M, N = get_dims(2.5)
    path = f'{sat}-2.5/'
    bad_row = np.full(N, MISSING, dtype=np.float32)
    res_str = '2.5'

    for prod in PRODUCTS:
        out_file = os.path.join(path, f'{prod}-{sat}-{res_str}')
        print(f'\nProcessing {out_file}')

        with open(out_file, 'wb') as fout:
            for yr in range(start_year, end_year + 1):
                yr2 = f'{yr % 100:02d}'
                for mo in range(1, 13):
                    mm = f'{mo:02d}'
                    in_file = os.path.join(path, f'{prod}{yr2}-{mm}-{sat}-{res_str}')

                    if yr < end_year:
                        # For previous years, we just write placeholders
                        # (in practice the previous-year data is already in the file
                        #  and we just advance past it; here we re-read if available)
                        if os.path.exists(in_file):
                            data = np.fromfile(in_file, dtype=np.float32)
                            if data.size == N * M:
                                fout.write(data.tobytes())
                            else:
                                for _ in range(M):
                                    fout.write(bad_row.tobytes())
                        else:
                            for _ in range(M):
                                fout.write(bad_row.tobytes())
                    elif yr == end_year and mo <= current_month:
                        if os.path.exists(in_file):
                            print(f'  Appending {in_file}')
                            data = np.fromfile(in_file, dtype=np.float32)
                            if data.size == N * M:
                                fout.write(data.tobytes())
                            else:
                                for _ in range(M):
                                    fout.write(bad_row.tobytes())
                        else:
                            print(f'  Missing {in_file}, filling')
                            for _ in range(M):
                                fout.write(bad_row.tobytes())
                    else:
                        for _ in range(M):
                            fout.write(bad_row.tobytes())


def combine_10deg(sat, start_year, end_year, yy):
    """
    Build per-year combined 1.0-degree binary (single year, all 12 months).
    Output filename: <sat>-1.0/<PROD>.<YY>-<sat>-1.0
    Each month = one record of N*M float32.
    """
    M, N = get_dims(1.0)
    path = f'{sat}-1.0/'
    bad = np.full(N * M, MISSING, dtype=np.float32)
    os.makedirs(path, exist_ok=True)

    for prod in PRODUCTS:
        out_file = os.path.join(path, f'{prod}.{yy}-{sat}-1.0')
        log_file = os.path.join(path, f'{prod}.{yy}-{sat}-1.0-LOG')
        print(f'\nProcessing {out_file}')

        with open(out_file, 'wb') as fout, open(log_file, 'w') as flog:
            for yr in range(start_year, end_year + 1):
                yr2 = f'{yr % 100:02d}'
                for mo in range(1, 13):
                    mm = f'{mo:02d}'
                    in_file = os.path.join(path, f'{prod}{yr2}-{mm}-{sat}-1.0')
                    if os.path.exists(in_file):
                        data = np.fromfile(in_file, dtype=np.float32)
                        if data.size == N * M:
                            # data is (M, N) = (180, 360) lat-major from write_grads.
                            # Write directly - the flat byte order is already correct GrADS layout.
                            fout.write(data.tobytes())
                            flog.write(f'{in_file} GOOD\n')
                            print(f'  Reading {in_file}')
                        else:
                            fout.write(bad.tobytes())
                            flog.write(f'{in_file} BAD\n')
                    else:
                        fout.write(bad.tobytes())
                        flog.write(f'{in_file} BAD\n')
                        print(f'  Adding missing {in_file}')


def main():
    parser = argparse.ArgumentParser(description='Combine SSMIS monthly products')
    parser.add_argument('sat',         help='Satellite (e.g. f16)')
    parser.add_argument('start_year',  type=int, help='Start year of record')
    parser.add_argument('end_year',    type=int, help='Current (end) year')
    parser.add_argument('month_or_yy', help='Current month (MM) for 2.5° or YY for 1.0°')
    parser.add_argument('--res', type=float, default=2.5,
                        choices=[2.5, 1.0], help='Resolution (default 2.5)')
    args = parser.parse_args()

    if args.res == 2.5:
        current_month = int(args.month_or_yy)
        print(f'Combining 2.5° for {args.sat}, {args.start_year}-{args.end_year}, '
              f'through month {current_month:02d}')
        combine_25deg_efficient(args.sat, args.start_year, args.end_year, current_month)
    else:
        yy = args.month_or_yy.zfill(2)
        print(f'Combining 1.0° for {args.sat}, {args.start_year}-{args.end_year}, yy={yy}')
        combine_10deg(args.sat, args.start_year, args.end_year, yy)


if __name__ == '__main__':
    main()
