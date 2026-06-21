"""Golden-month regression: a reproducible, bitwise anchor on real data.

This runs climalg for a fixed, small, already-processed case (F-17, 2026, Julian
days 121-123, 2.5 degree) reading the operational 1/3-degree grids, and compares
the output byte-for-byte (sha256) plus summary statistics against a committed
golden snapshot. Because the input grids are static and the algorithm is
deterministic, the same inputs must always produce identical bytes. Any change
that perturbs the numbers fails here loudly.

The test is data-gated: it skips cleanly when the operational grids are not
mounted (for example on the Mac) or when the golden snapshot has not been
generated yet, so it never produces a false failure off the pipeline host.

Regenerate the golden snapshot on a host that has the grids:

    python tests/test_golden_month.py --write
"""
import hashlib
import json
import os
import sys
import tempfile

import pytest

# climalg lives under monthly/ (added to sys.path by conftest). For the __main__
# generator path, add it explicitly too.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONTHLY_DIR = os.path.join(_REPO, 'monthly')
if MONTHLY_DIR not in sys.path:
    sys.path.insert(0, MONTHLY_DIR)

CASE = dict(sat='f17', year=2026, j0=121, j1=123, res=2.5)
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'golden', 'climalg_f17_2026_121-123_2.5.json')

# Operational 1/3-degree grid archive (overridable for portability).
DEFAULT_INPATH = os.environ.get(
    'GOLDEN_INPATH',
    '/path/to/ssmis_data/SSMIS_Grid/')

FILL = -999.99


def _grids_available():
    p = DEFAULT_INPATH
    if not os.path.isdir(p):
        return False
    # Need at least the first and last day of the case for the requested sat.
    f0 = os.path.join(p, f"as26{CASE['j0']:03d}.{CASE['sat']}")
    f1 = os.path.join(p, f"ds26{CASE['j1']:03d}.{CASE['sat']}")
    return os.path.exists(f0) and os.path.exists(f1)


def _run_case(out_root):
    """Run climalg for the fixed case into out_root; return the output dir."""
    import climalg_ssmis as ca
    cwd = os.getcwd()
    try:
        os.chdir(MONTHLY_DIR)  # LUTs / land tag are resolved relative to here
        ca.run(CASE['sat'], CASE['year'], CASE['j0'], CASE['j1'],
               res=CASE['res'], inpath=DEFAULT_INPATH, outdir=out_root)
    finally:
        os.chdir(cwd)
    return out_root


def _summarize(out_root):
    """Return {relpath: {sha256, bytes, count, mean, min, max, fill_frac}}."""
    import numpy as np
    summary = {}
    for root, _dirs, files in os.walk(out_root):
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, out_root)
            raw = open(full, 'rb').read()
            arr = np.frombuffer(raw, dtype=np.float32)
            valid = arr[arr != np.float32(FILL)]
            summary[rel] = {
                'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw),
                'count': int(arr.size),
                'mean': float(valid.mean()) if valid.size else None,
                'min': float(valid.min()) if valid.size else None,
                'max': float(valid.max()) if valid.size else None,
                'fill_frac': float((arr == np.float32(FILL)).mean()),
            }
    return summary


def test_golden_month():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(f"golden snapshot not generated yet: {GOLDEN_PATH}")
    if not _grids_available():
        pytest.skip(f"operational grids not available at {DEFAULT_INPATH}")

    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)['products']

    with tempfile.TemporaryDirectory() as d:
        _run_case(d)
        fresh = _summarize(d)

    assert set(fresh) == set(golden), (
        f"output file set changed: "
        f"added={set(fresh) - set(golden)} removed={set(golden) - set(fresh)}")
    mismatches = [rel for rel in golden
                  if fresh[rel]['sha256'] != golden[rel]['sha256']]
    assert not mismatches, f"bitwise drift in: {mismatches}"


def _write_golden():
    with tempfile.TemporaryDirectory() as d:
        _run_case(d)
        summary = _summarize(d)
    payload = {'case': CASE, 'inpath': DEFAULT_INPATH, 'products': summary}
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"wrote {GOLDEN_PATH} with {len(summary)} product files")


if __name__ == '__main__':
    if '--write' in sys.argv:
        _write_golden()
    else:
        print("pass --write to (re)generate the golden snapshot")
