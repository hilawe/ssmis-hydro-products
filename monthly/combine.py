#!/usr/bin/env python3
"""
combine.py

PURPOSE
    Updates each satellite's multi-year 2.5-degree combined file from the
    per-month files of the processing year, keeping every earlier year as it
    was, and builds the per-year 1.0-degree combined file. Replaces
    Combine-2.5deg.f and Combine-1.0deg.f.

SYNOPSIS
    python combine.py <sat> <start_year> <end_year> <month_or_yy> [--res 2.5|1.0]
    python combine.py <sat> <start_year> <end_year> <month_or_yy> --cold-start

EXAMPLE
    python combine.py f16 1992 2012 01 --res 2.5
    python combine.py f17 1987 2012 01 --res 2.5
    python combine.py f16 2012 2012 12 --res 1.0

INPUTS
    sat            : satellite name (f16, f17, f18, ...)
    start_year     : first year in the combined record, which must equal the
                     satellite's start_year in satellite_config.py for 2.5
    end_year       : the processing year
    month_or_yy    : processing month for 2.5-degree output or two-digit output year for 1.0-degree output
    --res          : grid resolution (2.5 or 1.0)
    --cold-start   : 2.5 only, build a record that does not exist yet

OUTPUT
    <sat>-<res>/<PROD>-<sat>-<res>   for 2.5-degree
    <sat>-<res>/<PROD>.<YY>-<sat>-<res>  for 1.0-degree (single-year style)

NOTES
    - The 2.5-degree rule is set by docs/combine_25deg_specification_2026-11-22.md.
      The existing combined file is the seed. A fresh seeded update preserves covered records outside the
      selected monthly inputs. It requires every target, uses available
      earlier inputs of the processing year, and fills uncovered slots.
      Structural failures exit 2 before staging. The first published marker
      reserves the request and mode. Until the second publication, a retry
      restages from current inputs. After the second publication, recovery
      checks the installed products against all ten recorded digests. A
      failed product check leaves the marker in place. An error syncing the
      directory after marker removal can still end the command with a
      failure.
    - For 1.0-deg output, files missing from disk are filled with -999.99.
    - Binary layout matches GrADS (N cols x M rows, float32 per month).
    - For 1.0-deg output: single-year time-series (12 months x M rows x N cols).

FUNCTIONS
    combine_25deg_efficient - updates the 2.5° multi-year combined binary (used by main)
    combine_10deg           - builds the per-year 1.0° combined binary (used by main)
    main                    - CLI entry point called by run_ssmis.sh

CALLED BY
    run_ssmis.sh (via: python combine.py <sat> <start> <end> <month> [--res])
"""

import sys
import os
import re
import stat
import json
import argparse
import hashlib
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


RECORD_BYTES = 72 * 144 * 4
MARKER_NAME = 'combine-2.5.install-in-progress'
PRODUCT_DIGEST = re.compile(r'[0-9a-f]{64}')


class CombineRefusal(Exception):
    """The request, or the installation its marker records, cannot proceed."""


def _write_part(path, data):
    """Write the new contents of one product, or of the marker, to its partial file and sync them."""
    with open(path, 'xb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _install(part, out):
    """Replace one combined file by its checked partial file."""
    os.replace(part, out)


def _sync_folder(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _month_file(path, prod, sat, index, start_year):
    year, month = start_year + index // 12, index % 12 + 1
    return os.path.join(path, f'{prod}{year % 100:02d}-{month:02d}-{sat}-2.5')


def _check_month_file(name, required):
    """Return True for a usable per-month file, False for an absent optional one, and refuse otherwise."""
    if not os.path.lexists(name):
        if required:
            raise CombineRefusal(f'{name} is the requested month and does not exist')
        return False
    if not os.path.isfile(name):
        raise CombineRefusal(f'{name} is not a regular file')
    size = os.path.getsize(name)
    if size != RECORD_BYTES:
        raise CombineRefusal(f'{name} holds {size} bytes, and a 2.5-degree month holds {RECORD_BYTES}')
    return True


def _publish_marker(path, marker, request, mode, digests):
    """Write the marker's record through its own partial file and sync the folder before anything else changes."""
    part = marker + '.part'
    if os.path.lexists(part):
        os.remove(part)            # left by a run stopped before its marker was in place
    record = dict(request=request, mode=mode, sha256=digests)
    _write_part(part, json.dumps(record, sort_keys=True).encode('ascii'))
    os.replace(part, marker)
    _sync_folder(path)


def _read_marker(marker):
    """Return the request, mode and digests the marker records, refusing anything that is not such a record."""
    if os.path.islink(marker) or not os.path.isfile(marker):
        raise CombineRefusal(f'{marker} is not a regular file, so the installation it stands for cannot be checked')
    try:
        record = json.loads(_read(marker).decode('ascii'))
        request, mode, digests = record['request'], record['mode'], record['sha256']
        if not isinstance(request, str) or mode not in ('update', 'cold-start'):
            raise ValueError('no request or mode')
        if digests is not None:
            if not isinstance(digests, dict) or set(digests) != set(PRODUCTS):
                raise ValueError('not one digest for each of the ten products')
            if not all(isinstance(d, str) and PRODUCT_DIGEST.fullmatch(d) for d in digests.values()):
                raise ValueError('a digest that is not 64 hexadecimal digits')
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as err:
        raise CombineRefusal(f'{marker} holds no usable record of an installation ({err}). Check the ten combined '
                             f'files against a verified copy before removing it')
    return request, mode, digests


def _finish_install(path, sat, marker, digests, cold_start):
    """Install the recorded set. Each product's partial file or its installed file must hold the recorded bytes,
    so that no installed file is ever taken as a new seed, and the marker stays if neither does."""
    def matches(name, digest):
        return (not os.path.islink(name) and os.path.isfile(name)
                and hashlib.sha256(_read(name)).hexdigest() == digest)

    files = []
    for prod in PRODUCTS:
        out = os.path.join(path, f'{prod}-{sat}-2.5')
        part, digest = out + '.part', digests[prod]
        try:
            ready = matches(part, digest)
        except OSError:
            if not matches(out, digest):   # a readable installed file with the recorded bytes is proof enough
                raise
            ready = False
        if cold_start and os.path.lexists(out) and not matches(out, digest):
            raise CombineRefusal(f'{out} differs from the recorded cold start. Check it before recovery, and the '
                                 f'marker stays')
        if not ready and not matches(out, digest):
            raise CombineRefusal(f'neither {out} nor {part} holds the checked bytes the marker records. Restore a '
                                 f'verified copy of the set, and the marker stays')
        files.append((out, part, digest, ready))
    _sync_folder(path)             # the marker's record is on disk before the first rename, on a retry as well
    for out, part, digest, ready in files:
        if ready:
            _install(part, out)
    _sync_folder(path)
    for out, part, digest, ready in files:
        if not matches(out, digest):
            raise RuntimeError(f'{out} does not match its checked partial file after installation')
    for out, part, digest, ready in files:
        if os.path.lexists(part):
            os.remove(part)        # a partial file beside a product already verified
        os.utime(out, None)        # the driver's copy check reads the time of each product it copies
    os.remove(marker)
    _sync_folder(path)


def combine_25deg_efficient(sat, start_year, end_year, current_month, basedir='.', cold_start=False):
    """
    Update the 2.5-degree combined file of each product for the requested month,
    by the rule of docs/combine_25deg_specification_2026-11-22.md.

    The existing combined file is the seed. A fresh seeded update copies all records
    before January of end_year and all seed records after current_month byte for byte.
    It fills later months through December of end_year wherever the seed ends before
    them. The requested month is
    written from its per-month file, which must exist, and each earlier month of
    end_year from its per-month file when present, otherwise kept from the seed
    (or filled where the seed does not reach). The new file is as long as the seed
    or as long as December of end_year, whichever is longer.

    The marker, published after the checks and before the first product partial, records
    the request, the mode and, once the ten partial files are checked, their digests.
    A run that finds it accepts only the same request and mode: before the digests are recorded
    it repeats the work, since no combined file has been replaced, and after that it
    finishes the recorded set against the digests without reading any monthly file.

    cold_start builds a record that does not exist yet, from the per-month files up to
    the requested month and fill, and refuses a new cold start if any combined file exists.

    Raises CombineRefusal when a check fails. A refusal of a new request comes before
    anything is written. A refusal of a recorded installation leaves its marker in place.

    basedir : root directory under which {sat}-2.5/ subdirectory is found.
              Allows test runs to point at a non-standard output directory.
    """
    if not 1 <= current_month <= 12:
        raise CombineRefusal(f'month {current_month} is outside 1 to 12')
    if end_year < start_year:
        raise CombineRefusal(f'the processing year {end_year} is before the start year {start_year}')
    path = os.path.join(basedir, f'{sat}-2.5')
    marker = os.path.join(path, MARKER_NAME)
    request = f'{sat} {start_year} {end_year} {current_month:02d}'
    mode = 'cold-start' if cold_start else 'update'
    january = (end_year - start_year) * 12
    target = january + current_month - 1
    through_december = january + 12
    fill_record = np.full(RECORD_BYTES // 4, MISSING, dtype='<f4').tobytes()

    resuming = False
    if os.path.lexists(marker):
        held_request, held_mode, digests = _read_marker(marker)
        if (held_request, held_mode) != (request, mode):
            raise CombineRefusal(
                f'{marker} records an interrupted {held_mode} of "{held_request}". Only that same request may run '
                f'until it completes, and this one is a {mode} of "{request}"')
        if digests is not None:
            print(f'Finishing the installation recorded in {marker}')
            _finish_install(path, sat, marker, digests, cold_start)
            print(f'\nInstalled the {len(PRODUCTS)} recorded combined files for {sat} {end_year}-{current_month:02d}')
            return
        resuming = True
        print(f'Repeating the work stopped before its files were checked, recorded in {marker}')

    # Every check before anything is written.
    plans = []
    for prod in PRODUCTS:
        out = os.path.join(path, f'{prod}-{sat}-2.5')
        if cold_start:
            if os.path.lexists(out):
                raise CombineRefusal(f'{out} exists, and a cold start does not replace a record')
            seed_months, first = 0, 0
        else:
            if not os.path.lexists(out):
                raise CombineRefusal(f'{out}, the seed, does not exist, and an update starts from the existing record')
            if os.path.islink(out) or not os.path.isfile(out):
                raise CombineRefusal(f'{out}, the seed, is not a regular file')
            seed_stat = os.stat(out)
            if os.geteuid() != 0 and not seed_stat.st_mode & stat.S_IRUSR:
                raise CombineRefusal(f'{out} needs owner-read permission for its replacement to be verified, since '
                                     f'the replacement belongs to the writer and takes the seed\'s permission bits')
            size = seed_stat.st_size
            if size == 0 or size % (12 * RECORD_BYTES):
                raise CombineRefusal(f'{out}, the seed, holds {size} bytes, which is not a whole number of years of '
                                     f'2.5-degree months, one year at least')
            seed_months = size // RECORD_BYTES
            if seed_months < january:
                raise CombineRefusal(f'{out}, the seed, holds {seed_months} months from {start_year} and must reach '
                                     f'December {end_year - 1}, {january} months')
            first = january
        sources = {}
        for index in range(first, target + 1):
            name = _month_file(path, prod, sat, index, start_year)
            if _check_month_file(name, required=index == target):
                sources[index] = name
        plans.append(dict(prod=prod, out=out, part=out + '.part', seed_months=seed_months, first=first,
                          sources=sources, months=max(seed_months, through_december)))
    if len({plan['seed_months'] for plan in plans}) != 1:
        raise CombineRefusal('the ten seeds hold different numbers of months, '
                             + ', '.join(f'{plan["out"]} {plan["seed_months"]}' for plan in plans))

    # From here only the same request may run until the set is installed.
    if not resuming:
        _publish_marker(path, marker, request, mode, None)

    # Write each product to its partial file and check it against sources read again from disk.
    for plan in plans:
        print(f'\nProcessing {plan["out"]}')
        if os.path.lexists(plan['part']):
            os.remove(plan['part'])        # left by an interrupted run, and never installed
        data = bytearray(plan['months'] * RECORD_BYTES)
        if plan['seed_months']:
            data[:plan['seed_months'] * RECORD_BYTES] = _read(plan['out'])
        for index in range(plan['seed_months'], plan['months']):
            data[index * RECORD_BYTES:(index + 1) * RECORD_BYTES] = fill_record
        for index, name in sorted(plan['sources'].items()):
            print(f'  Appending {name}')
            data[index * RECORD_BYTES:(index + 1) * RECORD_BYTES] = _read(name)
        for index in range(plan['first'], target):
            if index not in plan['sources']:
                kept = 'kept from the seed' if index < plan['seed_months'] else 'filled'
                print(f'  No {_month_file(path, plan["prod"], sat, index, start_year)}, {kept}')
        _write_part(plan['part'], bytes(data))
        if plan['seed_months']:
            os.chmod(plan['part'], stat.S_IMODE(os.stat(plan['out']).st_mode))   # keep the seed's access mode

        written = _read(plan['part'])
        seed = _read(plan['out']) if plan['seed_months'] else b''
        if len(written) != plan['months'] * RECORD_BYTES:
            raise RuntimeError(f'{plan["part"]} holds {len(written)} bytes after writing')
        for index in range(plan['months']):
            got = written[index * RECORD_BYTES:(index + 1) * RECORD_BYTES]
            if index in plan['sources']:
                want = _read(plan['sources'][index])
            elif index < plan['seed_months']:
                want = seed[index * RECORD_BYTES:(index + 1) * RECORD_BYTES]
            else:
                want = fill_record
            if got != want:
                raise RuntimeError(f'{plan["part"]} month {index} does not match its source after writing')
        plan['sha256'] = hashlib.sha256(written).hexdigest()

    # Record the ten checked digests, then install the set against them.
    digests = {plan['prod']: plan['sha256'] for plan in plans}
    _publish_marker(path, marker, request, mode, digests)
    _finish_install(path, sat, marker, digests, cold_start)
    print(f'\nInstalled {len(plans)} combined files of {plans[0]["months"]} months for {sat} '
          f'{end_year}-{current_month:02d}')


def combine_10deg(sat, start_year, end_year, yy, basedir='.'):
    """
    Build per-year combined 1.0-degree binary (single year, all 12 months).
    Output filename: <sat>-1.0/<PROD>.<YY>-<sat>-1.0
    Each month = one record of N*M float32.

    basedir : root directory under which {sat}-1.0/ subdirectory is found.
    """
    M, N = get_dims(1.0)
    path = os.path.join(basedir, f'{sat}-1.0/')
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
    parser.add_argument('--res',       type=float, default=2.5,
                        choices=[2.5, 1.0], help='Resolution (default 2.5)')
    # Optional path override - allows combine to operate in a test directory that
    # mirrors the structure of monthly/ without touching the real product files.
    parser.add_argument('--basedir',   default='.',
                        help='Root directory containing {sat}-{res}/ subdirectories (default: .)')
    parser.add_argument('--cold-start', action='store_true',
                        help='2.5 only: build a record that does not exist yet (refuses an existing one)')
    args = parser.parse_args()

    if args.cold_start and args.res != 2.5:
        parser.error('--cold-start applies to the 2.5-degree record only')

    if args.res == 2.5:
        try:
            try:
                current_month = int(args.month_or_yy)
            except ValueError:
                raise CombineRefusal(f'the month {args.month_or_yy!r} is not a number')
            import satellite_config
            try:
                declared = satellite_config.get_start_year(args.sat)
            except (KeyError, ValueError) as err:
                raise CombineRefusal(f'satellite_config.py gives no start year for {args.sat}: {err}')
            if args.start_year != declared:
                raise CombineRefusal(f'the start year {args.start_year} differs from {declared}, the start_year of '
                                     f'{args.sat} in satellite_config.py, so every record would be misplaced')
            print(f'Combining 2.5° for {args.sat}, {args.start_year}-{args.end_year}, '
                  f'through month {current_month:02d}  basedir={args.basedir}'
                  + ('  cold start' if args.cold_start else ''))
            combine_25deg_efficient(args.sat, args.start_year, args.end_year, current_month,
                                    basedir=args.basedir, cold_start=args.cold_start)
        except CombineRefusal as err:
            print(f'combine.py refused: {err}', file=sys.stderr)
            sys.exit(2)
    else:
        yy = args.month_or_yy.zfill(2)
        print(f'Combining 1.0° for {args.sat}, {args.start_year}-{args.end_year}, '
              f'yy={yy}  basedir={args.basedir}')
        combine_10deg(args.sat, args.start_year, args.end_year, yy, basedir=args.basedir)

if __name__ == '__main__':
    main()
