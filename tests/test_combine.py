"""combine.py: the 2.5-degree update keeps the accepted record, and the 1.0-degree record order.

The 2.5-degree combined file of each satellite holds one record of 72 x 144 float32 values per month from January of
the satellite's start year. It carries history that no per-month file holds, the predecessor satellites of each
chain, so an update must start from the existing file (the seed) and change only the processing year's months up to
the requested one. docs/combine_25deg_specification_2026-11-22.md sets the rule these tests check, with prior years
and later months copied byte for byte. Fresh structural refusals precede staging. The ten outputs are checked before
installation. While the marker exists, only its request and mode can resume. A refused recovery keeps the marker.

Every case runs on synthetic files in a temporary folder. The satellite name has three letters because the Fortran
program the equivalence cases compile builds its paths from a three-character name.
"""
import hashlib
import json
import os
import stat
import shutil
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
import combine

FILL = np.float32(-999.99)
CELLS = 72 * 144
RECORD = CELLS * 4
SAT = 'f99'
PRODUCTS = ['CFR', 'LWP', 'PR1', 'PR2', 'PF1', 'PF2', 'SSA', 'WVP', 'SNW', 'ICE']
MARKER = 'combine-2.5.install-in-progress'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORTRAN = os.path.join(REPO, 'monthly', 'Combine-2.5deg.f')


class Interrupted(BaseException):
    """Stands for a stop the program does not handle, as KeyboardInterrupt would."""


def folder(base):
    d = os.path.join(str(base), f'{SAT}-2.5')
    os.makedirs(d, exist_ok=True)
    return d


def seed_records(p, months):
    """A seed whose every record is distinct: product p, record i holds 1000 * (p + 1) + i."""
    values = 1000.0 * (p + 1) + np.arange(months, dtype=np.float64)
    return np.repeat(values[:, None], CELLS, axis=1).astype('<f4')


def write_seed(base, months, overrides=None):
    """Write the ten seeds. overrides maps a record index to a value written in every product."""
    d = folder(base)
    for p, prod in enumerate(PRODUCTS):
        arr = seed_records(p, months)
        for idx, value in (overrides or {}).items():
            arr[idx] = value
        arr.tofile(os.path.join(d, f'{prod}-{SAT}-2.5'))


def month_value(p, year, month):
    return 500000.0 + 1000.0 * p + (year % 100) * 12 + month


def write_month(base, year, month, value=None, products=PRODUCTS, cells=CELLS):
    """Write the per-month files of one month. value None gives each product its own month_value."""
    d = folder(base)
    for prod in products:
        p = PRODUCTS.index(prod)
        v = month_value(p, year, month) if value is None else value
        np.full(cells, v, dtype='<f4').tofile(os.path.join(d, f'{prod}{year % 100:02d}-{month:02d}-{SAT}-2.5'))


def read(base, prod):
    return np.fromfile(os.path.join(folder(base), f'{prod}-{SAT}-2.5'), dtype='<f4').reshape(-1, CELLS)


def raw(base, prod):
    with open(os.path.join(folder(base), f'{prod}-{SAT}-2.5'), 'rb') as fh:
        return fh.read()


def snapshot(base):
    """Every file under base with its digest, symbolic links by their target."""
    out = {}
    for root, dirs, files in os.walk(str(base)):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, str(base))
            if os.path.islink(path):
                out[rel] = 'link -> ' + os.readlink(path)
            else:
                with open(path, 'rb') as fh:
                    out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def marker_record(base):
    with open(os.path.join(folder(base), MARKER)) as fh:
        return json.load(fh)


def leftovers(base):
    return sorted(n for n in os.listdir(folder(base)) if n.endswith('.part') or n == MARKER)


# --- prior years, gaps and overlap months ----------------------------------------------------------------------

def test_prior_years_kept_where_per_month_files_hold_other_values(tmp_path):
    """The review's control: the seed holds 7, fill and 41 in three prior-year months whose per-month files hold
    99 and 88 or are absent. The legacy program kept 7, fill and 41. The code before the repair wrote 99, 88 and
    fill."""
    write_seed(tmp_path, 24, overrides={0: 7.0, 1: FILL, 2: 41.0})
    write_month(tmp_path, 2025, 1, 99.0)
    write_month(tmp_path, 2025, 2, 88.0)
    write_month(tmp_path, 2026, 1)
    write_month(tmp_path, 2026, 3)
    seed = {prod: raw(tmp_path, prod) for prod in PRODUCTS}

    combine.combine_25deg_efficient(SAT, 2025, 2026, 3, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        out = read(tmp_path, prod)
        assert out.shape[0] == 24
        assert np.all(out[0] == 7.0) and np.all(out[1] == FILL) and np.all(out[2] == 41.0)
        assert raw(tmp_path, prod)[:12 * RECORD] == seed[prod][:12 * RECORD]
        assert np.all(out[12] == month_value(p, 2026, 1))
        assert np.all(out[14] == month_value(p, 2026, 3))
    assert leftovers(tmp_path) == []


def test_gap_and_overlap_month_kept_from_the_seed(tmp_path):
    """A prior-year month of fill stays fill and a month of a predecessor satellite stays as it was, although this
    satellite's per-month files exist for both."""
    write_seed(tmp_path, 36, overrides={3: FILL, 5: 1234.5})
    write_month(tmp_path, 2020, 4)
    write_month(tmp_path, 2020, 6)
    write_month(tmp_path, 2022, 1)

    combine.combine_25deg_efficient(SAT, 2020, 2022, 1, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        out = read(tmp_path, prod)
        assert np.all(out[3] == FILL)
        assert np.all(out[5] == np.float32(1234.5))
        assert np.all(out[24] == month_value(p, 2022, 1))


# --- refusals ------------------------------------------------------------------------------------------------

def _valid_tree(base):
    """Start 2020, processing March 2022: a seed through December 2022 and per-month files for January to March."""
    write_seed(base, 36)
    for m in (1, 2, 3):
        write_month(base, 2022, m)


def _break(base, case):
    d = folder(base)
    seed = os.path.join(d, f'PR1-{SAT}-2.5')
    if case == 'one seed of ten missing':
        os.remove(seed)
    elif case == 'every seed missing':
        for prod in PRODUCTS:
            os.remove(os.path.join(d, f'{prod}-{SAT}-2.5'))
    elif case == 'seed not a whole number of months':
        with open(seed, 'ab') as fh:
            fh.write(b'\0\0\0\0')
    elif case == 'seed not a whole number of years':
        with open(seed, 'r+b') as fh:
            fh.truncate(35 * RECORD)
    elif case == 'every seed a month short of whole years':
        for prod in PRODUCTS:
            with open(os.path.join(d, f'{prod}-{SAT}-2.5'), 'r+b') as fh:
                fh.truncate(35 * RECORD)
    elif case == 'seed ends before December of the year before':
        with open(seed, 'r+b') as fh:
            fh.truncate(12 * RECORD)
    elif case == 'every seed ends before December of the year before':
        for prod in PRODUCTS:
            with open(os.path.join(d, f'{prod}-{SAT}-2.5'), 'r+b') as fh:
                fh.truncate(12 * RECORD)
    elif case == 'seed is a symbolic link':
        os.rename(seed, seed + '.real')
        os.symlink(os.path.basename(seed) + '.real', seed)
    elif case == 'requested month missing':
        os.remove(os.path.join(d, f'SSA22-03-{SAT}-2.5'))
    elif case == 'requested month short':
        write_month(base, 2022, 3, products=['WVP'], cells=CELLS - 1)
    elif case == 'requested month long':
        write_month(base, 2022, 3, products=['WVP'], cells=CELLS + 1)
    elif case == 'earlier month malformed':
        write_month(base, 2022, 2, products=['ICE'], cells=CELLS - 144)
    elif case == 'one seed longer than the others':
        seed_records(PRODUCTS.index('PR1'), 48).tofile(seed)
    elif case == 'one seed cut to the minimum':
        with open(seed, 'r+b') as fh:
            fh.truncate(24 * RECORD)
    else:
        raise AssertionError(case)


REFUSALS = ['one seed of ten missing', 'every seed missing', 'seed not a whole number of months',
            'seed not a whole number of years', 'every seed a month short of whole years',
            'seed ends before December of the year before', 'every seed ends before December of the year before',
            'seed is a symbolic link', 'requested month missing', 'requested month short', 'requested month long',
            'earlier month malformed', 'one seed longer than the others', 'one seed cut to the minimum']


@pytest.mark.parametrize('case', REFUSALS)
def test_refused_before_anything_is_written(tmp_path, case):
    _valid_tree(tmp_path)
    _break(tmp_path, case)
    before = snapshot(tmp_path)
    with pytest.raises(combine.CombineRefusal):
        combine.combine_25deg_efficient(SAT, 2020, 2022, 3, basedir=str(tmp_path))
    assert snapshot(tmp_path) == before
    assert leftovers(tmp_path) == []


# --- the processing year -------------------------------------------------------------------------------------

def test_earlier_month_without_a_file_kept_from_the_seed(tmp_path):
    """February 2022 has no per-month file, so its record is kept from the seed. January and March are replaced."""
    write_seed(tmp_path, 36)
    write_month(tmp_path, 2022, 1)
    write_month(tmp_path, 2022, 3)

    combine.combine_25deg_efficient(SAT, 2020, 2022, 3, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        out = read(tmp_path, prod)
        seed = seed_records(p, 36)
        assert np.all(out[24] == month_value(p, 2022, 1))
        assert np.array_equal(out[25], seed[25])
        assert np.all(out[26] == month_value(p, 2022, 3))
        assert np.array_equal(out[27:], seed[27:])


def test_january_adds_a_year_and_keeps_the_rest(tmp_path):
    write_seed(tmp_path, 24)
    write_month(tmp_path, 2020, 1)
    write_month(tmp_path, 2021, 6)
    write_month(tmp_path, 2022, 1)
    seed = {prod: raw(tmp_path, prod) for prod in PRODUCTS}

    combine.combine_25deg_efficient(SAT, 2020, 2022, 1, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        assert raw(tmp_path, prod)[:24 * RECORD] == seed[prod]
        out = read(tmp_path, prod)
        assert out.shape[0] == 36
        assert np.all(out[24] == month_value(p, 2022, 1))
        assert np.all(out[25:] == FILL)


def test_december_then_january_keeps_december(tmp_path):
    """December 2021 is written from its file. Its file is then removed, and January 2022 keeps the record."""
    fill_from_december = {i: FILL for i in range(23, 24)}
    write_seed(tmp_path, 24, overrides=fill_from_december)
    for m in range(1, 13):
        write_month(tmp_path, 2021, m)
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(tmp_path))
    after_december = {prod: raw(tmp_path, prod) for prod in PRODUCTS}
    for p, prod in enumerate(PRODUCTS):
        assert np.all(read(tmp_path, prod)[23] == month_value(p, 2021, 12))

    for prod in PRODUCTS:
        os.remove(os.path.join(folder(tmp_path), f'{prod}21-12-{SAT}-2.5'))
    write_month(tmp_path, 2022, 1)
    combine.combine_25deg_efficient(SAT, 2020, 2022, 1, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        assert raw(tmp_path, prod)[:24 * RECORD] == after_december[prod]
        assert np.all(read(tmp_path, prod)[24] == month_value(p, 2022, 1))


def test_rerun_of_an_earlier_month_keeps_the_later_months(tmp_path):
    """The seed holds 2022 through October. A rerun of August keeps September and October."""
    write_seed(tmp_path, 36, overrides={34: FILL, 35: FILL})
    for m in range(1, 9):
        write_month(tmp_path, 2022, m)

    combine.combine_25deg_efficient(SAT, 2020, 2022, 8, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        out = read(tmp_path, prod)
        seed = seed_records(p, 36)
        assert np.all(out[31] == month_value(p, 2022, 8))
        assert np.array_equal(out[32], seed[32]) and np.array_equal(out[33], seed[33])
        assert np.all(out[34:] == FILL)


def test_rerun_in_an_earlier_year_keeps_the_later_years(tmp_path):
    """The seed reaches December 2022. A rerun of December 2021 keeps 2022 and 2020 as they were."""
    write_seed(tmp_path, 36)
    for m in range(1, 13):
        write_month(tmp_path, 2021, m)
    seed = {prod: raw(tmp_path, prod) for prod in PRODUCTS}

    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(tmp_path))

    for p, prod in enumerate(PRODUCTS):
        data = raw(tmp_path, prod)
        assert len(data) == 36 * RECORD
        assert data[:12 * RECORD] == seed[prod][:12 * RECORD]
        assert data[24 * RECORD:] == seed[prod][24 * RECORD:]
        assert np.all(read(tmp_path, prod)[23] == month_value(p, 2021, 12))


# --- interruption and retry ----------------------------------------------------------------------------------

def _december_tree(base):
    """Start 2020, processing December 2021, the seed through December 2021 with December still fill."""
    write_seed(base, 24, overrides={23: FILL})
    for m in range(1, 13):
        write_month(base, 2021, m)
    write_month(base, 2022, 1)


def _uninterrupted(tmp_path):
    clean = tmp_path / 'clean'
    _december_tree(clean)
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(clean))
    return clean


def _stop_on_call(monkeypatch, name, call):
    real = getattr(combine, name)
    count = {'n': 0}

    def wrapper(*args, **kwargs):
        count['n'] += 1
        if count['n'] == call:
            raise Interrupted(f'{name} call {call}')
        return real(*args, **kwargs)

    monkeypatch.setattr(combine, name, wrapper)


def _retry_completes(work, clean, monkeypatch):
    """A run of January 2022 is refused while the marker stands, and the same December run completes the set."""
    monkeypatch.undo()
    if os.path.exists(os.path.join(folder(work), MARKER)):
        before = snapshot(work)
        with pytest.raises(combine.CombineRefusal):
            combine.combine_25deg_efficient(SAT, 2020, 2022, 1, basedir=str(work))
        assert snapshot(work) == before
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    for prod in PRODUCTS:
        assert raw(work, prod) == raw(clean, prod)
    assert leftovers(work) == []


def test_interruption_while_writing_partial_files(tmp_path, monkeypatch):
    clean = _uninterrupted(tmp_path)
    work = tmp_path / 'work'
    _december_tree(work)
    seed = {prod: raw(work, prod) for prod in PRODUCTS}
    _stop_on_call(monkeypatch, '_write_part', 4)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    for prod in PRODUCTS:
        assert raw(work, prod) == seed[prod]
    assert marker_record(work) == dict(request=f'{SAT} 2020 2021 12', mode='update', sha256=None)
    _retry_completes(work, clean, monkeypatch)


@pytest.mark.parametrize('call', [1, 5])
def test_interruption_during_installation(tmp_path, monkeypatch, call):
    """Call 1 stops before the first rename, and call 5 after four of the ten."""
    clean = _uninterrupted(tmp_path)
    work = tmp_path / 'work'
    _december_tree(work)
    seed = {prod: raw(work, prod) for prod in PRODUCTS}
    _stop_on_call(monkeypatch, '_install', call)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    for i, prod in enumerate(PRODUCTS):
        expected = raw(clean, prod) if i < call - 1 else seed[prod]
        assert raw(work, prod) == expected
    assert marker_record(work) == dict(request=f'{SAT} 2020 2021 12', mode='update',
                                       sha256={prod: hashlib.sha256(raw(clean, prod)).hexdigest() for prod in PRODUCTS})
    _retry_completes(work, clean, monkeypatch)


def test_abrupt_stop_after_four_renames_in_a_separate_process(tmp_path):
    """The same state when the process ends at once, with no Python clean-up, as after signal 9."""
    clean = _uninterrupted(tmp_path)
    work = tmp_path / 'work'
    _december_tree(work)
    code = (
        'import os, sys\n'
        f'sys.path.insert(0, {os.path.join(REPO, "monthly")!r})\n'
        'import combine\n'
        'real = combine._install\n'
        'count = [0]\n'
        'def stop(*a, **k):\n'
        '    count[0] += 1\n'
        '    if count[0] == 5:\n'
        '        os._exit(9)\n'
        '    return real(*a, **k)\n'
        'combine._install = stop\n'
        f'combine.combine_25deg_efficient({SAT!r}, 2020, 2021, 12, basedir={str(work)!r})\n'
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True)
    assert result.returncode == 9
    assert os.path.exists(os.path.join(folder(work), MARKER))
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    for prod in PRODUCTS:
        assert raw(work, prod) == raw(clean, prod)
    assert leftovers(work) == []


def test_retry_after_a_failed_digest_check_keeps_the_marker(tmp_path, monkeypatch):
    """A word of an installed prior year is damaged after its rename. The digest check stops the run, and a retry
    of the same month refuses, keeps the marker and leaves the damage for a verified copy to repair, where the
    first repair verified the damaged file against itself and removed the marker."""
    _december_tree(tmp_path)
    real = combine._install

    def damage(part, out):
        real(part, out)
        if out.endswith(f'CFR-{SAT}-2.5'):
            with open(out, 'r+b') as fh:
                fh.write(b'OOPS')

    monkeypatch.setattr(combine, '_install', damage)
    with pytest.raises(RuntimeError, match='after installation'):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(tmp_path))
    monkeypatch.undo()
    with pytest.raises(combine.CombineRefusal, match='marker stays'):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(tmp_path))
    assert os.path.exists(os.path.join(folder(tmp_path), MARKER))
    assert raw(tmp_path, 'CFR')[:4] == b'OOPS'


def test_recovery_reads_no_monthly_file_again(tmp_path, monkeypatch):
    """After four of ten renames every monthly file is removed. The retry finishes the recorded set from the
    digests, where the first repair read the monthly files again and mixed new and old months."""
    clean = _uninterrupted(tmp_path)
    work = tmp_path / 'work'
    _december_tree(work)
    _stop_on_call(monkeypatch, '_install', 5)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    monkeypatch.undo()
    for n in os.listdir(folder(work)):
        if '-' in n and n.split('-')[1].isdigit():
            os.remove(os.path.join(folder(work), n))
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    for prod in PRODUCTS:
        assert raw(work, prod) == raw(clean, prod)
    assert leftovers(work) == []


def _cold_tree(base):
    write_month(base, 2010, 2)
    write_month(base, 2011, 3)


@pytest.mark.parametrize('call', [1, 5, 10])
def test_interrupted_cold_start_completes_with_the_same_request(tmp_path, monkeypatch, call):
    """A cold start stopped before the first rename, after four and after nine is finished by the same cold-start
    request. An ordinary update of the same month is refused while the marker stands."""
    clean = tmp_path / 'clean'
    _cold_tree(clean)
    combine.combine_25deg_efficient(SAT, 2010, 2011, 3, basedir=str(clean), cold_start=True)
    work = tmp_path / 'work'
    _cold_tree(work)
    _stop_on_call(monkeypatch, '_install', call)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2010, 2011, 3, basedir=str(work), cold_start=True)
    monkeypatch.undo()
    assert marker_record(work)['mode'] == 'cold-start'
    before = snapshot(work)
    with pytest.raises(combine.CombineRefusal):
        combine.combine_25deg_efficient(SAT, 2010, 2011, 3, basedir=str(work))
    assert snapshot(work) == before
    combine.combine_25deg_efficient(SAT, 2010, 2011, 3, basedir=str(work), cold_start=True)
    for prod in PRODUCTS:
        assert raw(work, prod) == raw(clean, prod)
    assert leftovers(work) == []


def test_cold_start_recovery_refuses_a_foreign_file(tmp_path, monkeypatch):
    """After four of ten renames a file appears under the name of a product not yet installed. The cold-start retry
    refuses rather than install the checked partial file over it."""
    _cold_tree(tmp_path)
    _stop_on_call(monkeypatch, '_install', 5)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2010, 2011, 3, basedir=str(tmp_path), cold_start=True)
    monkeypatch.undo()
    with open(os.path.join(folder(tmp_path), f'ICE-{SAT}-2.5'), 'wb') as fh:
        fh.write(b'not this record')
    before = snapshot(tmp_path)
    with pytest.raises(combine.CombineRefusal, match='recorded cold start'):
        combine.combine_25deg_efficient(SAT, 2010, 2011, 3, basedir=str(tmp_path), cold_start=True)
    assert snapshot(tmp_path) == before


def test_recovery_refreshes_the_time_of_every_product(tmp_path, monkeypatch):
    """The driver's copy check reads each product's time, so a product installed before the interruption is given
    the time of the run that verified the set."""
    _december_tree(tmp_path)
    _stop_on_call(monkeypatch, '_install', 5)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(tmp_path))
    monkeypatch.undo()
    for prod in PRODUCTS[:4]:
        os.utime(os.path.join(folder(tmp_path), f'{prod}-{SAT}-2.5'), (1e9, 1e9))
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(tmp_path))
    for prod in PRODUCTS:
        assert os.stat(os.path.join(folder(tmp_path), f'{prod}-{SAT}-2.5')).st_mtime > 1.5e9


def test_empty_seed_refused_in_the_first_year(tmp_path):
    """In the satellite's first year no earlier month is required, and ten empty seeds were accepted and filled.
    All ten are empty, so that the check of equal lengths cannot be what refuses."""
    write_month(tmp_path, 2020, 3)
    for prod in PRODUCTS:
        open(os.path.join(folder(tmp_path), f'{prod}-{SAT}-2.5'), 'wb').close()
    before = snapshot(tmp_path)
    with pytest.raises(combine.CombineRefusal):
        combine.combine_25deg_efficient(SAT, 2020, 2020, 3, basedir=str(tmp_path))
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('kind', ['folder', 'record of the first repair', 'digests of nine products'])
def test_marker_that_is_not_a_usable_record_is_refused(tmp_path, kind):
    _valid_tree(tmp_path)
    m = os.path.join(folder(tmp_path), MARKER)
    if kind == 'folder':
        os.mkdir(m)
    elif kind == 'record of the first repair':
        with open(m, 'w') as fh:
            fh.write(f'{SAT} 2020 2022 03\n')
    else:
        with open(m, 'w') as fh:
            json.dump(dict(request=f'{SAT} 2020 2022 03', mode='update',
                           sha256={p: '0' * 64 for p in PRODUCTS[:9]}), fh)
    before = snapshot(tmp_path)
    with pytest.raises(combine.CombineRefusal):
        combine.combine_25deg_efficient(SAT, 2020, 2022, 3, basedir=str(tmp_path))
    assert snapshot(tmp_path) == before


def test_installed_files_keep_the_mode_of_the_seed(tmp_path):
    """A new file takes the writer's creation mask, which turned mode 0664 into 0600 under a mask of 077."""
    _valid_tree(tmp_path)
    for prod in PRODUCTS:
        os.chmod(os.path.join(folder(tmp_path), f'{prod}-{SAT}-2.5'), 0o664)
    old = os.umask(0o077)
    try:
        combine.combine_25deg_efficient(SAT, 2020, 2022, 3, basedir=str(tmp_path))
    finally:
        os.umask(old)
    for prod in PRODUCTS:
        assert stat.S_IMODE(os.stat(os.path.join(folder(tmp_path), f'{prod}-{SAT}-2.5')).st_mode) == 0o664


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a file of mode 0')
def test_unreadable_stale_partial_does_not_block_a_verified_installed_file(tmp_path, monkeypatch):
    """After nine of ten renames a stale partial file of an installed product cannot be read. The installed file
    holds the recorded digest, so the retry finishes, where it stopped with a permission error before."""
    clean = _uninterrupted(tmp_path)
    work = tmp_path / 'work'
    _december_tree(work)
    _stop_on_call(monkeypatch, '_install', 10)
    with pytest.raises(Interrupted):
        combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    monkeypatch.undo()
    stale = os.path.join(folder(work), f'CFR-{SAT}-2.5.part')
    with open(stale, 'wb') as fh:
        fh.write(b'stale')
    os.chmod(stale, 0)
    combine.combine_25deg_efficient(SAT, 2020, 2021, 12, basedir=str(work))
    for prod in PRODUCTS:
        assert raw(work, prod) == raw(clean, prod)
    assert leftovers(work) == []


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a file without owner read')
def test_seed_without_owner_read_refused_before_the_marker(tmp_path):
    """The replacement belongs to the writer and takes the seed's permission bits, so a seed the writer reads only
    through its group or others would give a replacement the writer cannot verify."""
    _valid_tree(tmp_path)
    seed = os.path.join(folder(tmp_path), f'ICE-{SAT}-2.5')
    os.chmod(seed, 0o044)
    before = snapshot_names(tmp_path)
    try:
        with pytest.raises(combine.CombineRefusal, match='owner-read'):
            combine.combine_25deg_efficient(SAT, 2020, 2022, 3, basedir=str(tmp_path))
        assert snapshot_names(tmp_path) == before
    finally:
        os.chmod(seed, 0o644)


def snapshot_names(base):
    return sorted(os.listdir(folder(base)))


# --- the legacy program --------------------------------------------------------------------------------------

EXCLUDED_SOURCES = os.path.join(REPO, 'tests', 'public_export_excluded_sources.txt')


def _left_out_of_this_tree(relative):
    """True when this tree is the public export and its exporter recorded the file as intentionally left out. The
    exporter writes the list. The private repository has no such list, so a missing file there is never excused."""
    try:
        with open(EXCLUDED_SOURCES) as f:
            names = {line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')}
    except FileNotFoundError:
        return False
    return relative in names


def _fortran(tmp_path):
    if not os.path.exists(FORTRAN):
        relative = os.path.relpath(FORTRAN, REPO)
        if _left_out_of_this_tree(relative):
            pytest.skip(f'{relative} is intentionally left out of the public export; this comparison runs in the '
                        'private suite')
        pytest.fail(f'{FORTRAN} is missing and this tree does not record it as intentionally left out')
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('no Fortran compiler on this host')
    exe = str(tmp_path / 'combine.x')
    subprocess.run([compiler, '-o', exe, FORTRAN], check=True, capture_output=True)
    return exe


@pytest.mark.parametrize('end_year,month,seed_months', [(2021, 3, 24), (2022, 1, 24)])
def test_same_bytes_as_the_fortran_program_in_a_forward_month(tmp_path, end_year, month, seed_months):
    """In a month that moves forward, with every earlier per-month file of the year present and fill after the last
    month written, both programs write the same ten files byte for byte. Prior-year per-month files with other values
    are present and are ignored by both."""
    exe = _fortran(tmp_path)
    trees = []
    for name in ('fortran', 'python'):
        base = tmp_path / name
        last_written = (end_year - 2020) * 12 + month - 2
        write_seed(base, seed_months, overrides={i: FILL for i in range(last_written + 1, seed_months)})
        write_month(base, 2020, 5, 77.0)
        for m in range(1, month + 1):
            write_month(base, end_year, m)
        trees.append(base)
    subprocess.run([exe, SAT, '2020', str(end_year), f'{month:02d}'], cwd=str(trees[0]), check=True,
                   capture_output=True)
    combine.combine_25deg_efficient(SAT, 2020, end_year, month, basedir=str(trees[1]))
    for prod in PRODUCTS:
        assert raw(trees[0], prod) == raw(trees[1], prod)


# --- cold start and the command line -------------------------------------------------------------------------

def test_cold_start_offsets_and_fill(tmp_path):
    write_month(tmp_path, 2000, 2, 5.0)
    write_month(tmp_path, 2001, 1, 7.0)
    write_month(tmp_path, 2001, 3, 9.0)

    combine.combine_25deg_efficient(SAT, 2000, 2001, 3, basedir=str(tmp_path), cold_start=True)

    for prod in PRODUCTS:
        out = read(tmp_path, prod)
        assert out.shape[0] == 24
        assert np.all(out[1] == 5.0) and np.all(out[12] == 7.0) and np.all(out[14] == 9.0)
        assert np.all(out[0] == FILL) and np.all(out[13] == FILL) and np.all(out[15:] == FILL)
    assert leftovers(tmp_path) == []


def test_cold_start_refuses_an_existing_record(tmp_path):
    _valid_tree(tmp_path)
    os.remove(os.path.join(folder(tmp_path), f'CFR-{SAT}-2.5'))
    before = snapshot(tmp_path)
    with pytest.raises(combine.CombineRefusal):
        combine.combine_25deg_efficient(SAT, 2020, 2022, 3, basedir=str(tmp_path), cold_start=True)
    assert snapshot(tmp_path) == before


def test_cold_start_requires_the_requested_month(tmp_path):
    write_month(tmp_path, 2000, 2, 5.0)
    before = snapshot(tmp_path)
    with pytest.raises(combine.CombineRefusal):
        combine.combine_25deg_efficient(SAT, 2000, 2001, 3, basedir=str(tmp_path), cold_start=True)
    assert snapshot(tmp_path) == before


@pytest.fixture
def registry_start_2020(monkeypatch):
    """The registry gives the synthetic satellite the start year 2020 and leaves the real ones as they are."""
    import satellite_config
    real = satellite_config.get_start_year
    monkeypatch.setattr(satellite_config, 'get_start_year', lambda name: 2020 if name == SAT else real(name))


def _main(monkeypatch, base, argv):
    monkeypatch.setattr(sys, 'argv', ['combine.py'] + argv + ['--res', '2.5', '--basedir', str(base)])
    combine.main()


@pytest.mark.parametrize('argv,remove', [
    ([SAT, '2019', '2022', '03'], None),     # a seed through 2022 from 2020 reads as one through 2021 from 2019
    ([SAT, '2020', '2022', '00'], None),     # month 0 would name no month of 2022 and change nothing
    ([SAT, '2020', '2022', '13'], None),
    ([SAT, '2020', '2022', 'March'], None),
    ([SAT, '2020', '2022', '03'], f'LWP-{SAT}-2.5'),
    (['f17', '1992', '2026', '10'], None),   # F-17's record starts in 1987
])
def test_command_line_refuses_with_status_2(tmp_path, monkeypatch, capsys, registry_start_2020, argv, remove):
    _valid_tree(tmp_path)
    if remove:
        os.remove(os.path.join(folder(tmp_path), remove))
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _main(monkeypatch, tmp_path, argv)
    assert stop.value.code == 2
    assert 'refused' in capsys.readouterr().err
    assert snapshot(tmp_path) == before


def test_command_line_updates_with_the_registered_start(tmp_path, monkeypatch, registry_start_2020):
    _valid_tree(tmp_path)
    _main(monkeypatch, tmp_path, [SAT, '2020', '2022', '03'])
    for p, prod in enumerate(PRODUCTS):
        assert np.all(read(tmp_path, prod)[26] == month_value(p, 2022, 3))
    assert leftovers(tmp_path) == []


# --- the 1.0-degree record -----------------------------------------------------------------------------------

def _write_grid(path, n_elem, value):
    np.full(n_elem, value, dtype=np.float32).tofile(path)


def test_combine_10deg_record_order():
    import tempfile
    M, N = combine.get_dims(1.0)  # 180, 360
    sat, prod, yy = 'ftest', 'PR1', '01'
    with tempfile.TemporaryDirectory() as base:
        d = os.path.join(base, f'{sat}-1.0')
        os.makedirs(d)
        _write_grid(os.path.join(d, f'{prod}01-05-{sat}-1.0'), M * N, 3.0)  # May

        combine.combine_10deg(sat, 2001, 2001, yy, basedir=base)

        out = np.fromfile(os.path.join(d, f'{prod}.{yy}-{sat}-1.0'), dtype=np.float32)
        assert out.size == 12 * M * N
        out = out.reshape(12, M * N)
        assert np.all(out[4] == 3.0)             # May -> slot 4
        assert np.all(out[0] == FILL)            # Jan missing -> fill
