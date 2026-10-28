#!/usr/bin/env python3
"""
wsfm_adapter.py

PURPOSE
    Convert WSF-M (Weather System Follow-on - Microwave) NetCDF orbit files
    containing predicted SSMIS antenna temperatures into the 1/3-degree binary
    format expected by climalg_ssmis.py and pentad_ssmis.py.  Once this adapter
    has produced the binary files, the entire downstream science pipeline (climate
    algorithm, pentad, combine, GPCP processing, NetCDF export, imagery) runs
    completely unchanged.

    This is the ONLY module that knows the WSF-M input format.  All other modules
    see 'wsfm' as just another satellite name, identical in usage to 'f17'.

SYNOPSIS
    python wsfm_adapter.py <netcdf_dir> <output_dir> <yyyy> <jday>
    python wsfm_adapter.py /data/wsfm/2030/001 ../SSMIS_Grid/ 2030 001

INPUT FILE FORMAT (WSF-M NetCDF orbit files)
    Variable: Predicted_SSMIS_Tas
      Shape  : (nscan, npixel, 11)  float32
      Units  : Kelvin
      Missing: NaN (float32 IEEE NaN)
      Channels:
        [0] 19V   [1] 19H   [2] 22V   [3] 37V   [4] 37H
        [5] 91V   [6] 91H   [7] 150H* [8] 183±1H* [9] 183±3H* [10] 183±7H*
        (* channels 7-10 contain NaN-only Ta in the current example file;
           these 4 channels are NOT used by the climate algorithm.)

    Other variables used:
      lat     : (nscan, npixel) float32  degrees_north, NaN=missing
      lon     : (nscan, npixel) float32  degrees_east,  NaN=missing
      scan_datetime : (nscan,) str  ISO8601 UTC, e.g. "2025-01-01T00:37:21.320Z"

    NOTE: The current example file is a GMI-to-SSMIS simulation:
      1C-R.GPM.GMI.XCAL2016-C.20250101-S003720-E021032.061565.V07B_Predicted_SSMIS.nc
    The operational WSF-M files will have a different naming convention but the
    same variable layout.  Update _parse_filename() below when the naming scheme
    is known.

OUTPUT FILE FORMAT (1/3-degree binary TDR)
    Identical to the files produced by the IDL ssmis_tdr_decoder.pro for F-17/F-16.
    Shape   : (NROW=540, NCOL=1080, 7)  uint8
    Encoding: uint8 = Ta_K - 70.0  (so Ta=139K -> uint8=69, Ta=299K -> uint8=229)
    Missing : uint8 = 32  (Ta_K would be 102.0 = skip/fill sentinel in algorithm)
    Valid   : only bytes 33..255 survive as data. The encoder maps NaN, inf, and
              everything outside (102, 325] K to the fill byte; because the byte
              is int(Ta - 70) (truncating), surviving data begins at Ta = 103.0 K.
              See _encode_ta for the two-stage operational rule this reproduces.
    Grid    : 1/3-degree, 540 rows, nearest-node centers 89.667N (row 0) to 90S
              (row 539), 1080 cols (180W = col 0)

    Filename convention (same as F-17):
      as{yy}{jday}.wsfm   - ascending-pass composite for day jday of year yyyy
      ds{yy}{jday}.wsfm   - descending-pass composite

    Ascending vs descending: a pass is "ascending" when the spacecraft is moving
    northward (lat[scan+1] > lat[scan] at the sub-satellite point).

CALLED BY
    run_ssmis.sh (via cron or manual invocation, after WSF-M is activated)

USES
    satellite_config.get_sat('wsfm')
    numpy, netCDF4

NOTES
    - Multiple orbit files per day are expected.  All files for a given date are
      gridded together into the single daily as/ds binary files.
    - Drop-in-the-bucket gridding: each swath pixel is assigned to its nearest
      1/3-degree bin (round(lat / RES) and round(lon / RES)).  When multiple
      pixels fall in the same bin, the last write wins (same behavior as IDL
      TDR decoder).  TODO: upgrade to min-distance or median if QC requires it.
    - The 'skip' condition in climalg_ssmis.py checks channels 0, 2, 3, 4, 5
      and 6 all equal 102.0 K.  To flag a pixel as missing, set ALL 7 channels to the
      fill byte (32), not just some.
    - This module is written but NOT YET ACTIVATED.  Activate by:
        1.  setting  wsfm.active = True  in satellite_config.py
        2.  setting  wsfm.start_year    in satellite_config.py
        3.  configuring WSF-M file ingestion directory in HydroCode.properties:
              hydroWSFM_INPUT=/path/to/wsfm/netcdf/files/
        4.  adding this script to the daily cron analogous to
              third_degree/src/cron_tdr_conversion_ssmis.sh
"""

import os
import sys
import glob
import re
import datetime
import argparse
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    raise ImportError(
        "netCDF4 is required for WSF-M input processing. "
        "Install it with: pip install netCDF4"
    )

from satellite_config import get_sat

# ---------------------------------------------------------------------------
# 1/3-degree grid constants (must match NROW/NCOL in climalg_ssmis.py)
# ---------------------------------------------------------------------------
NROW = 540        # latitude rows: nearest-node centers, 89.667°N (row 0) to 90°S (row 539)
NCOL = 1080       # longitude cols: 180°W (col 0) to 179.667°E (col 1079)
N_CHAN = 7        # channels used by algorithm: 19V,19H,22V,37V,37H,91V,91H
RES = 360.0 / NCOL   # 1/3 degree per cell
SCALE = 3.0          # nodes per degree, matches the IDL decoder SCALE
FILL_BYTE = 32    # uint8 value -> Ta = 102.0 K = algorithm skip sentinel

# Ta range representable in uint8 with offset 70:  [70, 325] K -> uint8 [0, 255]
TA_MIN = 70.0
TA_MAX = 325.0

# Ta at or below this is MISSING in the operational product, because
# ssmis_tdr_format.pro folds every byte <= 32 to the fill byte and byte 32
# decodes to 102 K. Valid data therefore requires Ta > 102, not Ta > 70.
TA_VALID_MIN = 102.0

# Channel map: WSF-M NetCDF channel index -> algorithm channel index
# get_sat('wsfm').netcdf_channels = [0, 1, 2, 3, 4, 5, 6]
# These map 1:1 (19V -> ch0, 19H -> ch1, 22V -> ch2, 37V -> ch3, 37H -> ch4, 91V -> ch5, 91H -> ch6)


def _idl_round(x: np.ndarray) -> np.ndarray:
    """Round half AWAY FROM ZERO, which is what IDL's ROUND does.

    numpy's rint rounds half to EVEN, so the two disagree on exactly-.5 values:
    rint(268.5) = 268 while IDL ROUND(268.5) = 269. That reaches the grid at any
    coordinate landing on a half node, for example latitude -0.5 (3*(90-0.5) =
    268.5) or longitude -178.5 (3*(180-178.5) = 4.5), placing the footprint one
    cell away from where the operational decoder puts it.

    On a real delivered orbit this hits about 2 rows and 6 columns per million
    footprints, so it does not move any published statistic. It is fixed anyway
    because it is the same class as the floor-versus-nearest-node defect and
    costs nothing to close (fixed 2026-07-29).
    """
    return np.trunc(x + np.copysign(0.5, x)).astype(np.int32)


def _lat_to_row(lat_deg: np.ndarray) -> np.ndarray:
    """
    Convert geodetic latitude (degrees north) to 1/3-degree row index, matching
    the OPERATIONAL F-17 decoder exactly.

    The operational chain does this in two steps
    (third_degree/src/ssmis_tdr_decoder.pro:229-233 then ssmis_tdr_format.pro):
        jj  = ROUND(3 * (90 + lat))     nearest node, SOUTH-first, clipped
        row = (NROW - 1) - jj           formatter reverses to NORTH-first storage

    It is NEAREST-NODE, not floor. An earlier version of this adapter used
    floor((90 - lat) / RES), which places samples half a cell away from where the
    operational decoder places them: the two longitude formulas disagree on 50.2
    percent of positions and row-and-column agreement on a real orbit was only
    25.0 percent. Every WSF-M-versus-F-17 comparison built on the floor mapping
    compared cells that were not the same cells (found 2026-07-29).

    Row 0 is therefore centered near 89.667 degrees north, which is also what
    climalg_ssmis.py assumes, not the 89.833 the floor version documented.
    """
    jj = _idl_round(SCALE * (90.0 + lat_deg))
    jj = np.clip(jj, 0, NROW - 1)
    return (NROW - 1) - jj


def _lon_to_col(lon_deg: np.ndarray) -> np.ndarray:
    """
    Convert longitude (degrees east) to 1/3-degree column index, matching the
    operational F-17 decoder: ii = ROUND(3 * (180 + lon)), with out-of-range
    values folded to column 0 exactly as the IDL does
    (`if (ii ge LONLEN or ii lt 0) then ii=0`), which is the dateline wrap.

    Col 0 = 180 degrees west. NEAREST-NODE, not floor; see _lat_to_row.
    """
    ii = _idl_round(SCALE * (180.0 + lon_deg))
    return np.where((ii >= NCOL) | (ii < 0), 0, ii).astype(np.int32)


def _parse_filename(nc_path: str) -> dict:
    """
    Extract date/time information from a WSF-M NetCDF filename.

    TODO: Update this function once the operational WSF-M filename convention
    is known.  The current example filename is:
      1C-R.GPM.GMI.XCAL2016-C.20250101-S003720-E021032.061565.V07B_Predicted_SSMIS.nc

    The operational convention will be different - this function will parse
    YYYYMMDD, start time, and end time from the filename.  For now, the date
    is read from the scan_datetime variable inside the file itself.

    Returns a dict with keys: yyyy, mm, dd, jday, start_hhmmss, end_hhmmss
    """
    # Fallback: read date from scan_datetime in the file
    # (This is always valid and avoids filename-parsing fragility)
    f = nc.Dataset(nc_path)
    try:
        dt_str = str(f.variables['scan_datetime'][0])   # e.g. "2025-01-01T00:37:21.320Z"
        dt = datetime.datetime.strptime(dt_str[:19], '%Y-%m-%dT%H:%M:%S')
        day_of_year = dt.timetuple().tm_yday
        return {
            'yyyy': dt.year,
            'mm':   dt.month,
            'dd':   dt.day,
            'jday': day_of_year,
            'start_hhmmss': dt_str[11:17].replace(':', ''),
        }
    finally:
        f.close()


class UnclassifiableGranule(ValueError):
    """A delivered granule with fewer than two scans. The delivery carries 28
    zero-scan granules as orbit-level data gaps; they read cleanly and carry
    nothing. This is a permitted, visible skip. Every other read failure is
    fatal for the day. The first fatal-input version treated this
    refusal as unreadable and failed 13 March, which holds three such gaps."""


def _is_ascending(lat: np.ndarray) -> np.ndarray:
    """
    Determine ascending/descending flag per scan line.

    A scan is ascending if the center-pixel latitude increases from the previous
    scan line (satellite moving northward).

    Returns a boolean array of shape (nscan,).  First scan takes the same flag
    as the second scan (edge case).
    """
    nscan = lat.shape[0]
    mid_col = lat.shape[1] // 2
    center_lat = lat[:, mid_col]          # latitude at center pixel per scan
    ascending = np.empty(nscan, dtype=bool)
    ascending[1:] = center_lat[1:] > center_lat[:-1]
    # A granule with fewer than 2 scans has no latitude trend, so its pass class
    # is UNKNOWABLE from the data. Defaulting it to
    # ascending invents a classification that could contaminate the ascending
    # composite. Fail closed instead: raise, so process_day skips the granule
    # with a visible warning and counts it. Audited on the full delivery
    # (2 x 1,264 orbit files): zero 1-scan granules exist, and the 28 empty
    # 0-scan granules carry no data, so this path is latent and the change is
    # byte-identical on all assessed data.
    if ascending.size < 2:
        raise UnclassifiableGranule(
            f"granule has {ascending.size} scan(s); pass direction is "
            "unclassifiable from a latitude trend, skipping (fail-closed)")
    ascending[0] = ascending[1]
    return ascending


def _encode_ta(ta_float: np.ndarray) -> np.ndarray:
    """
    Encode float32 Ta (Kelvin) array to uint8 with offset 70.

    Encoding: uint8 = Ta_K - 70.0, truncated by the int cast, matching IDL byte().

    VALID RANGE IS 102 < Ta <= 325 K, and everything else encodes to FILL_BYTE.
    That comes from two stages of the operational chain, and the second is easy
    to miss:

      ssmis_tdr_decoder.pro:351-395
          index = where(Ta gt 70 and Ta le 325)   -> byte = Ta - 70
          complement                               -> byte 0
        so the decoder's own interval is (70, 325], exclusive at 70.

      ssmis_tdr_format.pro:135-137
          index = where(out_chan le 32)  ->  out_chan(index) = 32
        so EVERY byte at or below 32 becomes the fill byte, not only byte 0.

    Byte b decodes to Ta = b + 70, so byte <= 32 means Ta <= 102 K. Combining
    both stages, the operational product treats any Ta <= 102 K as missing, and
    because the byte is int(Ta - 70) (truncating, matching IDL byte()), values in
    (102, 103) also truncate to byte 32 and become fill. Surviving data therefore
    begins at Ta = 103.0 K. The `<= TA_VALID_MIN` test below plus truncation
    reproduces that combined rule exactly.

    Until 2026-07-29 this function clipped into [70, 325] and only mapped NaN to
    fill, so a delivered value of, say, 85 K became byte 15 and decoded back to a
    valid-looking 85 K. climalg's skip test fires only when the gate channels read
    exactly 102 K, so such a footprint entered the retrievals as real data, and a
    spuriously cold 91V inflates the scattering index and fabricates rain. That is
    the same failure class as the byte-0 fill bug fixed in b69d992.

    Audited on the delivered ResidualCNN set (`analysis/audit_ta_range.py`, 24
    orbits): 81,446 PointCloud and 13,114 ENV values fall in (70, 102], about
    0.02 and 0.05 percent, with none at or below 70 K and none above 325 K.
    Small but real, and concentrated in the cold tail that drives precipitation.

    The format stage applies its rule to the GRIDDED array, so enforcing it here
    is equivalent as long as the grid is initialized to FILL_BYTE, which it is;
    doing it at encode time also catches the value before it can win a cell.

    Parameters
    ----------
    ta_float : np.ndarray, shape (..., N_CHAN), float32, Kelvin

    Returns
    -------
    np.ndarray, same leading shape (..., N_CHAN), dtype uint8
    """
    # Anything the operational chain would call missing must encode to FILL_BYTE,
    # not to a valid-looking byte. See the docstring for the two-stage IDL rule;
    # the combined effect is that valid data requires 102 < Ta <= 325.
    bad = ~(np.isfinite(ta_float)) | (ta_float <= TA_VALID_MIN) | (ta_float > TA_MAX)
    # Substitute an in-range value before the cast so uint8() never sees NaN or
    # inf (that is platform-dependent and warns). It is overwritten below.
    safe = np.where(bad, TA_MAX, ta_float)
    encoded = (safe - 70.0).astype(np.uint8)
    encoded[bad] = FILL_BYTE
    return encoded


# A real daily pass grid holds about 450,000 valid cells. Below this count a
# west-to-east ratio says nothing, and unit fixtures place a handful of cells.
MIN_CELLS_FOR_BALANCE = 20000
# Fourteen or fifteen granules make a full day; ten or more span enough
# longitude for a west-to-east ratio to mean something.
FULL_DAY_MIN_ORBITS = 10


def crossing_granules(prev_dir: str, pattern: str = '*.nc') -> list:
    """
    Granules in the preceding day's directory whose filename end time is
    earlier than its start time, which means the orbit crosses midnight.
    The scan-day filter then keeps only the scans past midnight, so listing
    a non-crossing granule here would be harmless but slow.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(prev_dir, pattern))):
        m = re.search(r'-S(\d{6})-E(\d{6})\.', os.path.basename(path))
        if m and m.group(2) < m.group(1):
            out.append(path)
    return out


def read_scan_days(f, nscan: int, label: str = '') -> np.ndarray:
    """
    Per-scan UTC day as yyyy*1000 + jday, from the granule's `scan_datetime`
    (ISO strings, one per scan, e.g. 2026-02-28T23:04:23.045Z).

    THE DEFECT THIS SERVES (fixed 2026-09-08). A granule is named for
    the day its first scan falls on, and about one granule a day crosses
    midnight; 009775 starts 28 Feb 23:04 and ends 1 Mar 00:45. The first
    version gridded whole granules into the filename day, so the last 45
    minutes of every such granule landed on the wrong day, and at a month
    boundary in the wrong month. The adapter therefore routes every scan by
    its own UTC scan date.

    WHAT THE F-17 REFERENCE DOES, stated from the decoder source and not
    from execution: ssmis_tdr_decoder.pro selects
    whole NESDIS TDR orbit files by the day tag in their names (line 131)
    and routes each file to the day in its header (line 171). It computes a
    per-scan time (line 200) but does not use it for routing. So the
    reference is FILE-DAY routed unless NESDIS cuts orbit files at UTC
    midnight upstream, which this repository does not establish. The size
    of the resulting difference in the daily composites, and in any product
    built on them, is NOT measured here: tails run from 3
    to 91 minutes in the v5 delivery, a day both gains the preceding tail and
    loses its own, and last-write merging with a nonlinear retrieval means
    reassigned scan time does not bound the product difference. This is not
    claimed as parity and no bound is claimed. An empty granule carries no
    scan_datetime and yields an empty array.
    """
    if nscan == 0:
        return np.zeros(0, dtype=np.int64)
    if 'scan_datetime' not in f.variables:
        raise ValueError(f'{os.path.basename(label)}: no scan_datetime; '
                         f'cannot assign scans to a UTC day')
    raw = f.variables['scan_datetime'][:]
    if len(raw) != nscan:
        raise ValueError(f'{os.path.basename(label)}: scan_datetime has '
                         f'{len(raw)} entries for {nscan} scans')
    days = np.empty(nscan, dtype=np.int64)
    for i, stamp in enumerate(raw):
        d = datetime.date.fromisoformat(str(stamp)[:10])
        days[i] = d.year * 1000 + d.timetuple().tm_yday
    return days


def select_scans_on_day(swath: dict, yyyy: int, jday: int) -> dict:
    """Keep only the scans whose UTC day is the target day."""
    keep = swath['scan_day'] == yyyy * 1000 + jday
    out = dict(swath)
    for key in ('lat', 'lon', 'ta', 'ascending', 'scan_day'):
        out[key] = swath[key][keep]
    out['n_scans_total'] = int(keep.size)
    out['n_scans_kept'] = int(keep.sum())
    return out


def normalize_longitude(lon: np.ndarray, label: str = '') -> np.ndarray:
    """
    Bring a delivered longitude array onto the signed convention the node
    mapping assumes, [-180, 180], and refuse anything else.

    THE DEFECT THIS CLOSES (fixed 2026-09-08). The delivered PointCloud
    `lon` variable is 0 to 360 degrees east, while every other geolocation
    variable in the same files (lon_IMG, lon_ENV, lon_LAS, lon_UAS) is signed.
    `_lon_to_col` folds any index outside the grid to column 0, which is the
    operational dateline wrap for a SIGNED input. Fed an unsigned input, every
    longitude above 180 (the whole western hemisphere) folded onto column 0.
    Measured: a v5 pass grid held 119 valid cells in columns 0 to 539 against
    177,577 in 540 to 1079; the IMG+ENV grid of the same pass held 118,773 west.
    Every PointCloud grid from July to September 2026 carried this.

    Rule: values in (180, 360] are shifted by -360; exactly 360 becomes 0 and
    exactly 180 stays 180 (which the node mapping wraps to column 0, the same
    node as -180). Values outside [-180, 360] are not a convention and raise.
    NaN passes through as missing. The detected convention is logged so a
    reader of the run log can see which one each granule used.
    """
    finite = np.isfinite(lon)
    if not finite.any():
        return lon
    lo = float(np.nanmin(lon))
    hi = float(np.nanmax(lon))
    if lo < -180.0 or hi > 360.0:
        raise ValueError(
            f'{os.path.basename(label)}: longitude range [{lo:.3f}, {hi:.3f}] is '
            f'neither signed [-180, 180] nor unsigned [0, 360]; refusing to grid')
    if hi > 180.0:
        convention = 'unsigned 0-360, shifted to signed'
        out = lon.copy()
        out[finite & (out > 180.0)] -= 360.0
    else:
        convention = 'signed -180-180'
        out = lon
    print(f'  longitude convention {convention} '
          f'(range {lo:.2f} to {hi:.2f}) for {os.path.basename(label)}')
    return out


def hemisphere_balance(grid: np.ndarray, label: str, n_orbits: int) -> None:
    """
    Refuse a pass grid whose valid cells sit almost entirely in one hemisphere.

    A single ascending or descending composite of a polar-orbiting imager
    covers both hemispheres every day. Measured on real F-17 and IMG+ENV
    grids the west-to-east ratio of valid cells is between roughly 0.6 and
    1.7. The longitude-convention defect produced 119 against 177,577, a ratio
    below 0.001. This gate is the check that would have caught it on day one;
    a 5,204-cell monthly mask against 10,365 on the other track was visible for
    six weeks and explained away as "different masks". Bounds are wide on
    purpose: they catch a collapsed hemisphere, not a sampling asymmetry.

    Judged only on a day with at least FULL_DAY_MIN_ORBITS granules. One
    orbit's ascending half lies on one side of the globe and its descending
    half on the other, so a one-granule day (28 January 2026, 28,553 west
    against 1 east) legitimately has a collapsed hemisphere in each pass grid.
    """
    if n_orbits < FULL_DAY_MIN_ORBITS:
        print(f'  {label}: hemisphere balance not judged, {n_orbits} orbit(s)')
        return
    valid = grid[:, :, 0] != FILL_BYTE
    west = int(valid[:, :NCOL // 2].sum())
    east = int(valid[:, NCOL // 2:].sum())
    if west + east < MIN_CELLS_FOR_BALANCE:
        return   # a sparse or empty grid has no hemisphere balance to judge
    ratio = west / max(east, 1)
    if ratio < 0.2 or ratio > 5.0:
        raise ValueError(
            f'{label}: hemisphere balance failed, west {west} east {east} '
            f'(ratio {ratio:.4f}); a collapsed hemisphere means a coordinate '
            f'convention defect, refusing to write')


def read_wsfm_orbit(nc_path: str) -> dict:
    """
    Read one WSF-M orbit NetCDF file and return swath arrays.

    Returns a dict with keys:
      'lat'   : (nscan, npixel) float32 degrees_north, NaN=missing
      'lon'   : (nscan, npixel) float32 degrees_east,  NaN=missing
      'ta'    : (nscan, npixel, 7) float32 K, NaN=missing
                  channels: [0]=19V [1]=19H [2]=22V [3]=37V [4]=37H
                             [5]=91V [6]=91H
      'ascending' : (nscan,) bool  True=northbound pass
      'date_info' : dict from _parse_filename()
    """
    cfg = get_sat('wsfm')
    ch_idx = cfg.netcdf_channels   # [0,1,2,3,4,5,6]

    f = nc.Dataset(nc_path, 'r')
    try:
        lat = np.array(f.variables['lat'][:],  dtype=np.float32)
        lon = np.array(f.variables['lon'][:],  dtype=np.float32)
        tas_all = np.array(f.variables[cfg.netcdf_variable][:], dtype=np.float32)
        scan_day = read_scan_days(f, lat.shape[0], nc_path)
    finally:
        f.close()

    lon = normalize_longitude(lon, nc_path)

    # Extract only the 7 algorithm channels (ignore 150H, 183±1/3/7H which are NaN)
    ta = tas_all[:, :, ch_idx]    # (nscan, npixel, 7)

    return {
        'lat':       lat,
        'lon':       lon,
        'ta':        ta,
        'ascending': _is_ascending(lat),
        'scan_day':  scan_day,
        'date_info': _parse_filename(nc_path),
    }


def grid_swath_to_binary(swath: dict, ascending_only: bool):
    """
    Grid one set of swath pixels (ascending or descending) to the 1/3° binary grid.

    Uses drop-in-the-bucket gridding: each pixel is mapped to its nearest
    1/3° bin.  When multiple pixels fall in the same bin, the last write wins
    (same behavior as the IDL TDR decoder for SSMIS).

    Parameters
    ----------
    swath : dict
        Output of read_wsfm_orbit().
    ascending_only : bool
        True -> use only ascending (northbound) scan lines.
        False -> use only descending (southbound) scan lines.

    Returns
    -------
    np.ndarray, shape (NROW, NCOL, N_CHAN), dtype uint8
        Fill value FILL_BYTE (32) for bins with no data.
    """
    grid = np.full((NROW, NCOL, N_CHAN), FILL_BYTE, dtype=np.uint8)
    # Which cells this half-orbit actually OBSERVED, as distinct from which ones
    # hold a non-fill byte. The two differ whenever an observed Ta is out of
    # range, and process_day needs the distinction to merge the way the IDL does.
    touched = np.zeros((NROW, NCOL), dtype=bool)

    lat    = swath['lat']       # (nscan, npixel)
    lon    = swath['lon']       # (nscan, npixel)
    ta     = swath['ta']        # (nscan, npixel, 7)
    asc    = swath['ascending'] # (nscan,)

    # Select scan lines for this half-orbit
    scan_mask = asc if ascending_only else ~asc
    lat_sel = lat[scan_mask]    # (n_selected_scans, npixel)
    lon_sel = lon[scan_mask]
    ta_sel  = ta[scan_mask]     # (n_selected_scans, npixel, 7)

    # Flatten to 1D pixel lists
    lat_flat = lat_sel.ravel()
    lon_flat = lon_sel.ravel()
    ta_flat  = ta_sel.reshape(-1, N_CHAN)   # (n_pixels, 7)

    # Remove pixels with missing lat/lon
    valid_geo = ~(np.isnan(lat_flat) | np.isnan(lon_flat))
    lat_flat = lat_flat[valid_geo]
    lon_flat = lon_flat[valid_geo]
    ta_flat  = ta_flat[valid_geo]

    if len(lat_flat) == 0:
        return grid, touched     # no data on this half-orbit

    # Convert to grid indices
    row = _lat_to_row(lat_flat)
    col = _lon_to_col(lon_flat)

    # Encode Ta to uint8 (NaN -> FILL_BYTE=32)
    ta_encoded = _encode_ta(ta_flat)   # (n_pixels, 7)

    # Drop-in-the-bucket: scatter pixels into grid (last write wins)
    # Use a loop over channels to allow numpy advanced indexing
    for ch in range(N_CHAN):
        grid[row, col, ch] = ta_encoded[:, ch]
    touched[row, col] = True

    return grid, touched


def process_day(nc_file_list: list, output_dir: str, yyyy: int, jday: int) -> None:
    """
    Process all orbit files for a single day and write as/ds binary output.

    Accumulates all orbits into one ascending and one descending grid, then
    writes binary files:
      {output_dir}/as{yy}{jday:03d}.wsfm
      {output_dir}/ds{yy}{jday:03d}.wsfm

    Parameters
    ----------
    nc_file_list : list of str
        Paths to all WSF-M NetCDF files for the target day.
    output_dir : str
        Directory to write the binary TDR files (equivalent to SSMIS_Grid/).
    yyyy : int
        Four-digit year.
    jday : int
        Day of year (1-365/366).
    """
    os.makedirs(output_dir, exist_ok=True)
    yy = f'{yyyy % 100:02d}'
    jday_str = f'{jday:03d}'

    asc_grid  = np.full((NROW, NCOL, N_CHAN), FILL_BYTE, dtype=np.uint8)
    desc_grid = np.full((NROW, NCOL, N_CHAN), FILL_BYTE, dtype=np.uint8)

    n_bad = 0
    n_gap = 0
    bad_names = []
    for nc_path in nc_file_list:
        print(f'  Reading {os.path.basename(nc_path)} ...')
        try:
            swath = read_wsfm_orbit(nc_path)
        except UnclassifiableGranule as exc:
            print(f'    skipping granule (permitted gap): {exc}')
            n_gap += 1
            continue
        except Exception as exc:
            # An unreadable granule is recorded and makes the day FAIL at the
            # end. A first version skipped it and wrote a correctly sized,
            # possibly all-fill grid with exit 0, which the driver recorded as
            # success. A delivered zero-scan granule is a permitted
            # gap and reads cleanly; a read exception is not a gap.
            print(f'  ERROR: unreadable {os.path.basename(nc_path)} - {exc}')
            n_bad += 1
            bad_names.append(os.path.basename(nc_path))
            continue

        swath = select_scans_on_day(swath, yyyy, jday)
        print(f'    scans on {yyyy}-{jday_str}: {swath["n_scans_kept"]} '
              f'of {swath["n_scans_total"]}')
        if swath['n_scans_kept'] == 0:
            continue

        asc_part,  asc_touched  = grid_swath_to_binary(swath, ascending_only=True)
        desc_part, desc_touched = grid_swath_to_binary(swath, ascending_only=False)

        # Merge on OBSERVED, not on non-fill. The operational decoder assigns
        # every footprint unconditionally (ssmis_tdr_decoder.pro:235-241,
        # `ssmis_91v_as(ii,jj) = a91v`) and only applies the valid-range test
        # afterward, to the whole accumulated grid. So a later footprint whose
        # Ta is out of range MUST still overwrite an earlier valid one, and the
        # cell ends up fill.
        #
        # Merging on `!= FILL_BYTE` instead made a fill fail to overwrite, so the
        # cell silently kept a stale value from an earlier orbit. That was
        # invisible before the encoding fix, because out-of-range Ta was being
        # clipped to a non-fill byte rather than set to fill; the two defects
        # concealed each other. The companion adapter for the IMG+ENV delivery
        # writes footprints straight into the day grid and was always correct here.
        asc_grid[asc_touched]   = asc_part[asc_touched]
        desc_grid[desc_touched] = desc_part[desc_touched]

    if n_gap:
        print(f'  {n_gap} zero-scan gap granule(s) skipped on {yyyy}-{jday_str}')
    if n_bad:
        raise RuntimeError(
            f'{n_bad} unreadable granule(s) for {yyyy}-{jday_str}: '
            f'{", ".join(bad_names)}; day not written')

    # Write binary files: shape (NROW, NCOL, 7) uint8, GrADS-compatible
    for prefix, grid in [('as', asc_grid), ('ds', desc_grid)]:
        out_path = os.path.join(output_dir, f'{prefix}{yy}{jday_str}.wsfm')
        hemisphere_balance(grid, os.path.basename(out_path), len(nc_file_list))
        grid.tofile(out_path)
        n_filled = (grid[:, :, 0] == FILL_BYTE).sum()
        n_total  = NROW * NCOL
        print(f'  Written {out_path}  '
              f'({100*(1-n_filled/n_total):.1f}% coverage)')


def main():
    parser = argparse.ArgumentParser(
        description='Convert WSF-M NetCDF orbit files to 1/3-degree binary TDR format'
    )
    parser.add_argument('netcdf_dir',  help='Directory containing WSF-M NetCDF files for the day')
    parser.add_argument('output_dir',  help='Output directory for binary files (e.g. ../SSMIS_Grid/)')
    parser.add_argument('yyyy',        type=int, help='Year (YYYY)')
    parser.add_argument('jday',        type=int, help='Day of year (1-365)')
    parser.add_argument('--prev-dir',  default=None,
                        help='Directory of the PRECEDING day; its midnight-crossing '
                             'granules contribute their scans on the target day')
    parser.add_argument('--pattern',   default='*.nc',
                        help='Glob pattern to match NetCDF files (default: *.nc)')
    args = parser.parse_args()

    nc_files = sorted(glob.glob(os.path.join(args.netcdf_dir, args.pattern)))
    if args.prev_dir:
        crossing = crossing_granules(args.prev_dir, args.pattern)
        print(f'  preceding-day crossing granules: {len(crossing)}')
        nc_files = crossing + nc_files
    else:
        print('  WARNING: no --prev-dir; scans before the first granule of the '
              'day are missing (first day of a window only)')
    if not nc_files:
        print(f'WARNING: No NetCDF files found in {args.netcdf_dir} '
              f'matching pattern "{args.pattern}"')
        sys.exit(1)

    print(f'wsfm_adapter: yyyy={args.yyyy} jday={args.jday:03d} '
          f'files={len(nc_files)} -> {args.output_dir}')
    process_day(nc_files, args.output_dir, args.yyyy, args.jday)
    print('Done.')


if __name__ == '__main__':
    main()
