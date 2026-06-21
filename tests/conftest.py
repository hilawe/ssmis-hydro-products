"""Shared pytest setup.

The pipeline modules live under monthly/ and monthly/netcdf/ and are run as
scripts (not an installed package), so the test suite puts those directories on
sys.path here rather than relying on the working directory. Tests that need a
heavy dependency (netCDF4, scipy) or real input data guard themselves with
pytest.importorskip / skip markers, so the suite degrades gracefully on a host
that lacks them (for example the Mac base interpreter) instead of erroring at
collection time.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, 'monthly'),
           os.path.join(_REPO, 'monthly', 'netcdf')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO_ROOT = _REPO
MONTHLY_DIR = os.path.join(_REPO, 'monthly')
