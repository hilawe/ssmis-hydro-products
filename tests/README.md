# Test suite

A `pytest` suite that locks in the validated behavior of the Python pipeline.
It is the safety net described in `the project documentation` section 6.9 step 1:
before any later refactor, this is what proves a change did not perturb the
science or the file formats.

## Layout

- `test_product_version.py` - the version token and NetCDF `product_version`
  format, and the coupling between them (the v01 to v02 cutover switch). No
  third-party dependencies; runs anywhere.
- `test_ta2tb.py` - the antenna-to-brightness-temperature conversion: locks the
  calibration constants and checks the vectorized form against the explicit
  per-channel formula.
- `test_io_contract.py` - the binary format contract: the GrADS layout
  transform round-trips, the on-disk byte order is lon-fastest / south-first,
  and the fill sentinel is exactly `float32(-999.99)`.
- `test_geometry_and_dates.py` - grid coordinates (2.5 and 1.0 degree), the
  date helpers, file-size month inference, and the resolution dimensions.
- `test_retrievals.py` - structural invariants for the retrieval algorithms
  (shape, missing-channel sentinel, finiteness, non-negative precipitation).
  Not exact magnitudes; those are covered by the golden month and the Fortran
  cross-validation.
- `test_golden_month.py` - a reproducible bitwise regression: runs climalg for a
  fixed small case (F-17, 2026, Julian days 121-123, 2.5 degree) against the
  operational grids and compares sha256 plus statistics to a committed snapshot.
- `test_product_definitions.py` - the ten monthly products worked by hand from
  their definitions, one grid cell at a time (rainfall as mean rate times 24
  times the days of the month, the ocean and land cell rules, the ice and snow
  screens). Runs anywhere `numpy` and `scipy` are installed.
- `test_gpcp_masking.py` - the snow, ice and coastal masking of the GPCP input
  rain for both chains, and the morning chain's snow climatology window of July
  1990 to December 1991, on small real files with known cells.
- `test_gpcp_dual.py` - the dual-satellite merge: rain validity and sampling
  validity separately, the F-17 fallback, a cell-by-cell oracle, the time
  origin, the cutoff, the year boundary, and short or missing inputs.
- `test_gpcp_netcdf_labels.py` - each GPCP-input label against its input file,
  first year, chain and dates, with the expected values written out from the
  legacy writers and not taken from the module.
- `test_pentad.py` - the pentad processor against the monthly processor on the
  same synthetic days. Eight products must be equal value for value and the two
  rainfall totals must scale as 5 days to 31.
- `test_archive_scripts.py` - the driver and the two archive scripts run inside
  disposable trees: the archive step packages the tree that was processed, and
  a wrong or missing configuration stops it. Needs bash and the GNU tools. It
  travels with the driver and the archive scripts, so a copy of the repository
  without those scripts does not carry this file.

The files from `test_product_definitions.py` on were added on 18 November 2026.
They run in the continuous-integration job, which has no operational grids, so
that job now exercises the numbers and not only the structure. They do not
compare any retrieval with the legacy Fortran. That comparison is still the
golden month, on a host with the grids.

## Running

From the repo root, on a host with the scientific stack (a host with numpy, scipy, and netCDF4 installed):

    python -m pytest            # whole suite
    python -m pytest tests/test_ta2tb.py -v

Tests that need `netCDF4`, `scipy`, or the operational grids skip cleanly where
those are unavailable (for example the Mac base interpreter), so the suite never
produces a false failure off the pipeline host.

## Golden snapshot

The golden snapshot under `tests/golden/` must be generated on a host that has
the operational 1/3-degree grids mounted:

    python tests/test_golden_month.py --write

Regenerate it deliberately only when an output change is intended and reviewed.
Override the grid location with `GOLDEN_INPATH` if the archive moves.
