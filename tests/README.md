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
