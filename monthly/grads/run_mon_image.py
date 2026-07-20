#!/usr/bin/env python3
"""
run_mon_image.py

PURPOSE
    Generate monthly GIF/PNG imagery for SSMIS hydrological products using
    matplotlib, replicating the color scales, contour levels, lat/lon extents,
    titles, and output filenames produced by the operational GrADS scripts:
      pr1.gs -> rainfall (PR1), output filename  Mon{YY}-ra-25prod.gif
      lwp.gs -> cloud LWP,      output filename  Mon{YY}-lw-25prod.gif
      wvp.gs -> TPW (var=ta),   output filename  Mon{YY}-ta-25prod.gif
      snow_4.gs -> snow cover,   output filename  Mon{YY}-sn-25prod.gif

    Replaces:
      run_mon_image.sh  (Bash driver)
      pr1.gs, snw.gs, lwp.gs, wvp.gs, ice.gs, pf1.gs  (GrADS plot scripts)
      cbarmf.gs, cpccol.gs  (color/colorbar utilities)

SYNOPSIS
    python run_mon_image.py [--yy YY] [--mm MM] [--indir DIR] [--outdir DIR]

EXAMPLE
    python run_mon_image.py --yy 26 --mm 03
    python run_mon_image.py --yy 26 --mm 03 --indir ../test_mar2026 --outdir /tmp/img_test
    python run_mon_image.py       # auto-detects previous month

ARGUMENTS
    --yy     Two-digit year (e.g. 26 for 2026). Default: previous month's year.
    --mm     Two-digit month (e.g. 03 for March). Default: previous month.
    --indir  Root directory containing {sat}-2.5/ binary subdirectories.
             Default: '..' (one level up from grads/, i.e., monthly/).
    --outdir Directory to write GIF/PNG output files.
             Default: 'img/' subdirectory under the grads/ directory.

INPUT BINARY FILES (GrADS format)
    Per-satellite monthly binaries in {indir}/{sat}-2.5/:
      PR1{YY}-{MM}-{sat}-2.5  - rainfall accumulator, stored as ndays*24*mm/hr
      LWP{YY}-{MM}-{sat}-2.5  - cloud LWP, stored as 1000×kg/m² (= g/m²)
      WVP{YY}-{MM}-{sat}-2.5  - total precipitable water (mm), stored directly
      SNW{YY}-{MM}-{sat}-2.5  - snow cover fraction (0-1), stored directly
    Binary layout: float32, south-first lat, 1.25°E-first lon, lon-fastest.
    Grid: N_LAT=72 × N_LON=144 = 10,368 elements per file.

    1.0-degree yearly binaries (12 months per file, 360 lon × 180 lat,
    south-first, 0.5°E-first, lon-fastest), used by the snow panels and
    the CFR field map and the ICE / WVP / CFR anomaly diagnostics:
      {indir}/{sat}-1.0/{PROD}.{yy}-{sat}-1.0   (current month + f16/f18 baseline)
      {indir}/ncdc-bin/{PROD}.{YY}              (morning-chain baseline, f17)
    An isolated --indir that lacks these renders the snow panels via their
    2.5-degree fallback and SKIPS the ICE, CFR-field, WVP-anomaly, and
    CFR-anomaly images
    (they have no 2.5-degree fallback by design; watch stdout for the decline
    messages).

OUTPUT
    GIF files (PNG fallback if Pillow not installed) in --outdir/{sat}/, named:
      Mon{YY}-ra-25prod.gif  (e.g., img/f17/Mar26-ra-25prod.gif)
      Mon{YY}-lw-25prod.gif
      Mon{YY}-ta-25prod.gif
      Mon{YY}-sn-25prod.gif       (4-panel: NH+NH-anom top, SH+SH-anom bottom)
      Mon{YY}-ic-25prod.gif       (4-panel sea ice + anomaly, ±50° panels,
                                   SSMIS-era 2009-2020 baseline)
      Mon{YY}-ta-anom-25prod.gif  (global WVP anomaly vs WMO 1991-2020 baseline)
    Each satellite (f17, f16, f18) writes to its own subdirectory, mirroring
    the operational structure at grads/img/{sat}/.  The ic and ta-anom images
    are 1.0-degree diagnostics and are NOT part of the NCEI archive tar below.

    If --archive-dir is given, f17 imagery is ALSO written to that directory
    using the NCEI archive naming convention (used by tar_mw-hydro_netcdf.sh):
      mw-hydro_v01_imagery_pr1_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_lwp_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_wvp_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_snow-color_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_snow-color_{YYYY}{MM}.ps  (PostScript, same content)
    Typically --archive-dir should be set to $hydroMONTHLY_NETCDF/imagery/ so
    that tar_mw-hydro_netcdf.sh can find and package the files correctly.

CALLED BY
    run_ssmis.sh  (monthly driver script)

CALLS / IMPORTS
    matplotlib (required), cartopy (optional), numpy, calendar, Pillow (optional)

COLOR TABLES
    All colors are exact RGB values from two sources:
    1. GrADS 1.9b4 built-in default palette (src/gxX.c, after 'set display color white'):
       colors 0-15 from the reds[]/greens[]/blues[] arrays.
    2. cpccol.gs custom CPC palette: colors 21-97 (light-yellow -> dark-red, etc.)

RENDERING METHOD
    All maps use contourf with BoundaryNorm, which interpolates smoothly between
    the discrete 2.5° grid-cell values.  This matches the visual appearance of
    the operational GrADS output better than a flat cell-fill (pcolormesh) at
    this coarse resolution.  The smooth_field() helper (Gaussian sigma) is
    retained in the code but is not currently applied to the rendered data.

NOTES
    - Data scaling applied before plotting:
        PR1: divide by ndays to convert raw accumulator to mm/day
        LWP: divide by 1000 to convert g/m² -> kg/m² (GrADS: 'display lw/1000')
        WVP: no scaling (stored in mm; GrADS: 'display ta' directly)
        SNW: no scaling (stored as fraction 0-1)
    - Lat range for global products: -50° to 50° (matching 'set lat -50 50' in GrADS)
    - Snow uses NPS/SPS polar stereographic projections, matching 'set mproj nps/sps'
    - Snow anomaly (panels 2 and 4) is 100×(current − 1991-2020 baseline mean),
      following the WMO Climate Normals standard period (updated May 2021).
      Field and baseline are both at 1.0° from the same series: the morning-chain
      ncdc-bin archive (F-08 -> F-17, back to 1987) for the morning primary, else the
      per-satellite {sat}-1.0 yearly files. This replaces the earlier 2.5° anomaly
      whose combined-bin baseline only reached ~2008. If the 1.0° field or baseline
      is unavailable it falls back to the 2.5° anomaly with a non-WMO label.
      Replicates:
        GrADS: define meanval = ave(maskout(sn,sn), t=startmon, t=endmon, 1yr)
               define percent = 100*(current - meanval)
    - Title spacing: matplotlib constrained_layout + fig.suptitle() are used for
      global and snow maps so the header/subtitle/map are packed tightly with no
      manual vertical offsets - spacing is automatic, not arbitrary.

MODIFICATION HISTORY
    2026-04-15  H. Semunegus  Initial conversion from GrADS scripts
    2026-04-15  H. Semunegus  Fixed: color tables (exact RGB from gxX.c + cpccol.gs),
                             filename convention (Mon{YY}-ra-25prod.gif capitalized),
                             WVP filename suffix (-ta- not -wv-), data scaling (PR1/ndays,
                             LWP/1000), lat range (-50 to 50), added --indir/--outdir args
    2026-04-16  H. Semunegus  Added --archive-dir: writes NCEI archive-named copies of f17
                             imagery (mw-hydro_v01_imagery_*_{YYYY}{MM}.gif + .ps for snow)
                             to netcdf/imagery/ so tar_mw-hydro_netcdf.sh can package them.
                             Fixed per-satellite subdirectory output (img/{sat}/).
    2026-04-17  H. Semunegus  (1) Replaced contourf with pcolormesh throughout - GrADS-style
                             flat cell fill, no smoothing/interpolation between grid cells.
                             (2) Switched to constrained_layout + fig.suptitle() so title/
                             subtitle/map spacing is auto-computed with no blank gap.
                             (3) Footer reduced to 'DD Mon YYYY  NOAA NCEI' only.
                             (4) Removed duplicate LWP subtitle (units shown on colorbar).
                             (5) Snow image upgraded to true 4-panel layout (snow_4.gs):
                             NH snow | NH anomaly (top), SH snow | SH anomaly (bottom);
                             anomaly computed from 1987-2010 baseline of combined binary.
    2026-04-18  H. Semunegus  BASELINE UPGRADE: Snow anomaly reference period updated from
                             1987-2010 to 1991-2020, conforming to the WMO Climate Normals
                             standard period (WMO-No. 1203, updated May 2021). This aligns
                             the SSMI/SSMIS anomaly fields with ERA5, GPCP v3, IMERG and
                             all major modern climate datasets. F-18 baseline coverage
                             improves from 1 year (March 2010 only) to 10 years (2010-2020),
                             making F-18 snow anomalies scientifically meaningful for the
                             first time. All baseline_start/baseline_end defaults, docstrings,
                             labels, and annotation strings updated throughout.
                             (6) Added NaN-aware smooth_field() (Gaussian sigma) helper -
                             retained for optional use but rendering reverted to contourf
                             after visual comparison showed contourf better matches the
                             operational GrADS output appearance at 2.5° resolution.
    2026-04-17  H. Semunegus  (7) Removed satellite prefixes (SSM/I, SSMI/S) from all
                             map titles - product will incorporate WSF-M data, making
                             per-sensor labels obsolete.
                             (8) Snow panels switched to 1.0° yearly combined binary
                             (f17-1.0/SNW.{yy}-f17-1.0) matching ops snow_mon_10.ctl -
                             eliminates large holes over Canada/Russia and oversmoothing.
                             (9) Polar projections corrected: central_longitude=-90
                             (90°W at bottom), matching GrADS 'set mpvals -270 90'.
                             (10) Geographic detail upgraded to 50m (from 110m) with
                             admin_0 country borders and admin_1 US/Canada state lines,
                             matching GrADS 'set mpdset hires'.
                             (11) Bottom text overlap fixed: constrained_layout rect
                             reserves 6-7% at figure bottom so colorbar label does not
                             collide with footer.
                             (12) 4-panel snow spacing tightened with h_pad/wspace pads.
    2026-06-22  H. Semunegus  Footer/colorbar overlap regression fixed on the global
                             maps (rainfall/LWP/WVP).  The rect reservation noted in
                             item 11 was not present in the code, so plot_global's
                             constrained_layout packed the colorbar units label
                             ('mm/day') into the same band as the fig.text() footer
                             ('DD Mon YYYY  NOAA NCEI') and the two strings overlapped.
                             The global-map footer is now drawn with fig.supxlabel(),
                             which the constrained_layout engine reserves space for
                             (unlike fig.text), so it stacks cleanly below the colorbar
                             label.  The snow 4-panel (subplots_adjust, constrained_layout
                             OFF) keeps its fig.text() footer and is unchanged.  The
                             global-map figure height was raised 4.0->4.6in: once the
                             footer correctly reserves space, the old 4.0in squeezed the
                             fixed-aspect map until it became height-limited and shrank
                             (~690 px wide); 4.6in keeps it width-limited (~880 px) with
                             the footer clear.
"""

import os
import sys
import argparse
import datetime
import calendar
import numpy as np

# Point GDAL's PROJ at the conda environment's PROJ data directory before cartopy
# (and its GDAL) are imported. The pipeline invokes python directly rather than via
# conda activate, so GDAL_DATA/PROJ_DATA are unset and GDAL's PROJ context fails its
# first database probe with "ERROR 1: PROJ: proj_create_from_database: Open of ...
# failed" before silently falling back. pyproj resolves the correct, version-matched
# directory; reuse it so the probe succeeds and the message is not emitted. Harmless
# if pyproj is unavailable.
try:
    import pyproj as _pyproj
    _proj_data_dir = _pyproj.datadir.get_data_dir()
    os.environ.setdefault('PROJ_DATA', _proj_data_dir)
    os.environ.setdefault('PROJ_LIB', _proj_data_dir)
except Exception:
    pass

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (no X server needed)
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import matplotlib.path as mpath
    from matplotlib.gridspec import GridSpec
except ImportError:
    raise ImportError('matplotlib is required: pip install matplotlib')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    print('Warning: cartopy not found - using simple lat/lon maps for all products')
    HAS_CARTOPY = False

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print('Warning: scipy not found - light smoothing will be skipped')

import warnings

# Single source of truth for the version token in imagery filenames
# (see monthly/product_version.py). Kept dependency-free so importing it
# does not pull netCDF4 into the imagery path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_version import PRODUCT_VERSION


# ---------------------------------------------------------------------------
# 2.5-degree grid definition
# GrADS .ctl files: XDEF 144 linear 1.25 2.5 (lon 1.25°E-358.75°E)
#                   YDEF  72 linear -88.75 2.5 (lat 88.75°S-88.75°N)
# ---------------------------------------------------------------------------
N_LON = 144
N_LAT = 72
LON_START = 1.25
LAT_START = -88.75
RES = 2.5

# Grid cell centers (used for axis labels and meshgrid construction)
LONS = np.array([LON_START + j * RES for j in range(N_LON)])  # 1.25 to 358.75
LATS = np.array([LAT_START + i * RES for i in range(N_LAT)])  # -88.75 to 88.75

# Grid cell EDGES for pcolormesh (used for dummy colorbar stubs only;
# primary rendering uses contourf with the cell-center LONS/LATS above).
LON_EDGES = np.linspace(0.0, 360.0, N_LON + 1)   # 145 values, 2.5° boundaries
LAT_EDGES = np.linspace(-90.0, 90.0, N_LAT + 1)  #  73 values, 2.5° boundaries

# ---------------------------------------------------------------------------
# 1.0-degree grid definition
# The operational snow_4.gs reads snow from snow_mon_10.ctl which points to
# the 1.0-degree combined yearly file (f17-1.0/SNW.{yy}-f17-1.0).
# Using 1.0° for the snow NH/SH panels gives 4× the spatial density vs 2.5°,
# eliminating the large cell-edge holes and oversmoothing seen at 2.5°.
# The first 1.0° column is centered at 0.5°E (to_grads() in climalg_ssmis.py
# rolls the 0°E-side bin to column 0, same convention as the 2.5° grid's
# 1.25°E). The legacy snow_mon_10.ctl declares XDEF -0.5, which mislabels the
# field one column (1°) west; verified against the writer and by coastline
# mask alignment (2026-07-12). Do not copy the ctl value here.
# ---------------------------------------------------------------------------
N_LON_1 = 360
N_LAT_1 = 180
LONS_1 = np.array([0.5 + j * 1.0 for j in range(N_LON_1)])   # 0.5 to 359.5
LATS_1 = np.array([-89.5 + i * 1.0 for i in range(N_LAT_1)]) # -89.5 to 89.5

# Three-letter month abbreviations matching the operational filename convention
# (Mon{YY}-ra-25prod.gif uses the capitalized abbreviation from this list)
MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

SATELLITES = ['f17', 'f16', 'f18']

# Start year for each satellite's combined multi-year binary (matches combine.py)
SAT_START_YEAR = {'f17': 1987, 'f16': 1992, 'f18': 2010}

# The morning (early) constellation, F-08 -> F-11 -> F-13 -> F-17, has a full
# 1.0-degree monthly record on disk in ncdc-bin/{PROD}.{YY} (1987-present),
# read by generate_netcdf.py run_early_1deg(). This is the ONLY chain with a
# multi-decadal 1.0-degree archive, so it is the source for a genuine WMO
# 1991-2020 snow-anomaly baseline. The current morning primary is F-17; if it
# flips (e.g. to WSF-M), update this alongside the f17 archive-imagery logic.
NCDC_BIN_SAT = 'f17'


def _year4_from_yy(yy):
    """Expand a 2-digit year to 4 digits using the pipeline's calendar rule
    (same as ncdc-bin/{PROD}.{YY} and _compute_climatology_1deg):
    87-99 -> 1987-1999, 00-86 -> 2000-2086. For every currently renderable
    month (2001 onward) this returns exactly what the former '20'+yy hardcode
    did; it only differs for pre-2000 years, which the fixed '20' prefix would
    have mislabeled (e.g. yy='98' -> '2098' on the image caption).
    """
    n = int(yy)
    return f'19{n:02d}' if n >= 87 else f'20{n:02d}'

# Default base paths (overridden by --indir / --outdir arguments)
DEFAULT_INDIR  = '..'    # monthly/ directory (one level above grads/)
DEFAULT_OUTDIR = 'img'   # img/ under grads/

# ---------------------------------------------------------------------------
# Light spatial smoothing
#
# GrADS 'set csmooth on' convolves the gridded field with a 3×3 kernel before
# shading - equivalent to a very mild Gaussian blur (sigma ≈ 0.5 grid cells).
# A pure cell-fill (pcolormesh, sigma=0) looks blocky at 2.5° resolution;
# a heavy smooth (sigma≥2, or contourf) over-interpolates and misrepresents
# the actual data resolution.  sigma=0.7 sits comfortably in between:
#   - softens the hard rectangular cell edges so the eye reads the pattern,
#     not the grid
#   - does not push gradients across more than ~1 grid cell
#   - preserves fill/missing (NaN) boundaries - NaN cells are not blurred
#     into neighbouring valid cells (see smooth_field() implementation)
#
# To disable smoothing entirely, set SMOOTH_SIGMA = 0.0.
# To increase smoothing, try 1.0 (≈ two-cell blur radius).
# ---------------------------------------------------------------------------
SMOOTH_SIGMA = 1.0   # Gaussian standard deviation in grid-cell units


# ===========================================================================
# GrADS color lookup table
# Sources:
#   Colors  0-15: GrADS 1.9b4 src/gxX.c reds[]/greens[]/blues[] arrays,
#                 as set after 'set display color white' (swaps indices 0 and 1
#                 so 0=white/background, 1=black/foreground).
#   Colors 21-97: cpccol.gs custom CPC palette, exact RGB values from the
#                 'set rgb' commands in that script.
# ===========================================================================

# Maps GrADS color index -> (R, G, B) as integers 0-255
_GRADS_RGB = {
    # Built-in default colors (from gxX.c, after 'set display color white')
    #   Comment in source: "0-black 1-white 2-red 3-green 4-blue 5-cyan
    #                        6-magenta 7-yellow 8-orange 9-purple
    #                       10-yell-grn 11-lt.blue"
    #   Arrays (original indices 0=black, 1=white; swapped by 'set display color white'):
    0:  (255, 255, 255),  # white  - background after 'set display color white'
    1:  (  0,   0,   0),  # black  - foreground (coastlines, text)
    2:  (250,  60,  60),  # coral red          (reds[2]=250, greens[2]=60,  blues[2]=60)
    3:  (  0, 220,   0),  # bright green       (reds[3]=0,   greens[3]=220, blues[3]=0)
    4:  ( 30,  60, 255),  # blue               (reds[4]=30,  greens[4]=60,  blues[4]=255)
    5:  (  0, 200, 200),  # cyan               (reds[5]=0,   greens[5]=200, blues[5]=200)
    6:  (240,   0, 130),  # magenta/hot-pink   (reds[6]=240, greens[6]=0,   blues[6]=130)
    7:  (230, 220,  50),  # yellow             (reds[7]=230, greens[7]=220, blues[7]=50)
    8:  (240, 130,  40),  # orange             (reds[8]=240, greens[8]=130, blues[8]=40)
    9:  (160,   0, 200),  # purple             (reds[9]=160, greens[9]=0,   blues[9]=200)
    10: (160, 230,  50),  # yellow-green       (reds[10]=160,greens[10]=230,blues[10]=50)
    11: (  0, 160, 255),  # sky blue           (reds[11]=0,  greens[11]=160,blues[11]=255)
    12: (230, 175,  45),  # amber/gold         (reds[12]=230,greens[12]=175,blues[12]=45)
    13: (  0, 210, 140),  # sea-foam/teal      (reds[13]=0,  greens[13]=210,blues[13]=140)
    14: (130,   0, 220),  # violet             (reds[14]=130,greens[14]=0,  blues[14]=220)
    15: (170, 170, 170),  # grey               (reds[15]=170,greens[15]=170,blues[15]=170)
    # ---------------------------------------------------------------------------
    # cpccol.gs - light yellow -> dark red (oranges/reds)
    21: (255, 250, 170), 22: (255, 232, 120), 23: (255, 192,  60),
    24: (255, 160,   0), 25: (255,  96,   0), 26: (255,  50,   0),
    27: (225,  20,   0), 28: (192,   0,   0), 29: (165,   0,   0),
    # cpccol.gs - light green -> dark green
    31: (230, 255, 225), 32: (200, 255, 190), 33: (180, 250, 170),
    34: (150, 245, 140), 35: (120, 245, 115), 36: ( 80, 240,  80),
    37: ( 55, 210,  60), 38: ( 30, 180,  30), 39: ( 15, 160,  15),
    # cpccol.gs - light blue -> dark blue
    41: (225, 255, 255), 42: (180, 240, 250), 43: (150, 210, 250),
    44: (120, 185, 250), 45: ( 80, 165, 245), 46: ( 60, 150, 245),
    47: ( 40, 130, 240), 48: ( 30, 110, 235), 49: ( 20, 100, 210),
    # cpccol.gs - light purple -> dark purple
    51: (220, 220, 255), 52: (192, 180, 255), 53: (160, 140, 255),
    54: (128, 112, 235), 55: (112,  96, 220), 56: ( 72,  60, 200),
    57: ( 60,  40, 180), 58: ( 45,  30, 165), 59: ( 40,   0, 160),
    # cpccol.gs - light pink -> dark rose
    61: (255, 230, 230), 62: (255, 200, 200), 63: (245, 160, 160),
    64: (230, 130, 130), 65: (225, 100, 100), 66: (215,  80,  80),
    67: (200,  60,  60), 68: (180,  40,  40), 69: (164,  32,  32),
    # cpccol.gs - light beige -> dark brown
    71: (250, 240, 230), 72: (240, 220, 210), 73: (225, 190, 180),
    74: (200, 160, 150), 75: (180, 140, 130), 76: (160, 120, 110),
    77: (140, 100,  90), 78: (120,  80,  70), 79: (100,  60,  50),
    # cpccol.gs - light grey -> dark grey
    81: (240, 240, 240), 82: (225, 225, 225), 83: (210, 210, 210),
    84: (195, 195, 195), 85: (180, 180, 180), 86: (165, 165, 165),
    87: (150, 150, 150), 88: (135, 135, 135), 89: (120, 120, 120),
    91: (105, 105, 105), 92: ( 90,  90,  90), 93: ( 75,  75,  75),
    94: ( 60,  60,  60), 95: ( 45,  45,  45), 96: ( 30,  30,  30),
    97: ( 15,  15,  15),
}


def _rgb_f(gidx):
    """Return float (0-1) RGB tuple for a GrADS color index.
    Falls back to grey (0.5, 0.5, 0.5) for unknown indices."""
    r, g, b = _GRADS_RGB.get(gidx, (128, 128, 128))
    return (r / 255.0, g / 255.0, b / 255.0)


def build_cmap(interval_ccols, over_ccol):
    """
    Build a matplotlib ListedColormap from GrADS ccols convention.

    GrADS convention (N clevs, N+1 ccols):
      ccols[0] -> color for values BELOW clevs[0]  (typically 0 = white)
      ccols[1..N-1] -> N-1 colors for intervals between consecutive clevs
      ccols[N] -> color for values ABOVE clevs[-1]

    For PR1 (N clevs, N ccols - no explicit below-minimum color):
      interval_ccols = ccols[0..N-1]  (colors for the N-1 intervals + overflow)
      over_ccol      = ccols[-1]

    Parameters
    ----------
    interval_ccols : list of int
        GrADS color indices for the filled intervals (between consecutive levels).
        The number of colors here = number of level intervals = len(levels) - 1.
    over_ccol : int
        GrADS color index for values above the maximum level.

    Returns
    -------
    cmap : ListedColormap
        Colormap with N-1 colors for the intervals.
        cmap.set_over() is called with the over_ccol RGB value.
    """
    colors = [_rgb_f(c) for c in interval_ccols]
    cmap = mcolors.ListedColormap(colors, name='grads_custom')
    cmap.set_over(_rgb_f(over_ccol))
    cmap.set_under((1.0, 1.0, 1.0))  # white for below-minimum
    return cmap


# ---------------------------------------------------------------------------
# Per-product color maps and level specifications
# Exact reproduction of GrADS script settings:
#   pr1.gs:   'set clevs 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24'
#             'set ccols 53 55 4 11 5 13 3 38 10 7 12 8 2 27 6 29'
#   lwp.gs:   'set clevs 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5'
#             'set ccols 0 62 21 32 34 43 44 47 54 56 57'
#   wvp.gs:   'set clevs 5 10 15 20 25 30 35 40 45 50 55 60'
#             'set ccols 0 72 62 21 31 32 34 43 45 47 54 56 59'
#   snow_4.gs:'set clevs .1 .2 .3 .4 .5 .6 .7 .8 .9 1'
#             'set ccols 0 22 31 32 34 43 45 47 54 56 59'
# ---------------------------------------------------------------------------

# PR1: 16 clevs -> 1 below-minimum + 15 intervals, GrADS uses last ccol as overflow
# GrADS pr1.gs ccols (16 total for 16 clevs):
#   ccols[0]  = 53 -> below 1 mm/day  (land and zero-rain ocean - lavender purple)
#   ccols[1]  = 55 -> 1-2 mm/day
#   ccols[2]  = 4 -> 2-3 mm/day
#   ccols[3]  = 11 -> 3-4 mm/day
#   ccols[4]  = 5 -> 4-5 mm/day
#   ccols[5]  = 13 -> 5-6 mm/day
#   ccols[6]  = 3 -> 6-7 mm/day
#   ccols[7]  = 38 -> 7-8 mm/day
#   ccols[8]  = 10 -> 8-10 mm/day
#   ccols[9]  = 7 -> 10-12 mm/day
#   ccols[10] = 12 -> 12-14 mm/day
#   ccols[11] = 8 -> 14-16 mm/day
#   ccols[12] = 2 -> 16-18 mm/day
#   ccols[13] = 27 -> 18-20 mm/day
#   ccols[14] = 6 -> 20-22 mm/day
#   ccols[15] = 29 -> 22-24 mm/day and overflow >24 mm/day
PR1_LEVELS  = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24]
PR1_CCOLS   = [55, 4, 11, 5, 13, 3, 38, 10, 7, 12, 8, 2, 27, 6, 29]  # 15 interval colors
PR1_OVER    = 29
PR1_CMAP    = build_cmap(PR1_CCOLS, PR1_OVER)
PR1_CMAP.set_under(_rgb_f(53))  # lavender-purple (GrADS color 53) for land and < 1 mm/day

# LWP: 10 clevs -> 9 intervals + 1 overflow = 10 ccols (+1 under = 11 total)
#   ccols[0] = 0 (white) -> below 0.05 kg/m²
#   ccols[1..9] -> 9 interval colors
#   ccols[10] = 57 -> overflow (>0.5 kg/m²)
LWP_LEVELS  = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
LWP_CCOLS   = [62, 21, 32, 34, 43, 44, 47, 54, 56]  # 9 interval colors
LWP_OVER    = 57
LWP_CMAP    = build_cmap(LWP_CCOLS, LWP_OVER)

# WVP: 12 clevs -> 11 intervals + 1 overflow = 12 ccols (+1 under = 13 total)
#   ccols[0] = 0 (white) -> below 5 mm
#   ccols[1..11] -> 11 interval colors
#   ccols[12] = 59 -> overflow (>60 mm)
WVP_LEVELS  = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
WVP_CCOLS   = [72, 62, 21, 31, 32, 34, 43, 45, 47, 54, 56]  # 11 interval colors
WVP_OVER    = 59
WVP_CMAP    = build_cmap(WVP_CCOLS, WVP_OVER)

# SNW: 10 clevs -> 9 intervals + 1 overflow = 10 ccols (+1 under = 11 total)
#   ccols[0] = 0 (white) -> below 0.1 fraction
#   ccols[1..9] -> 9 interval colors
#   ccols[10] = 59 -> overflow (>1.0)
SNW_LEVELS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SNW_CCOLS   = [22, 31, 32, 34, 43, 45, 47, 54, 56]  # 9 interval colors
SNW_OVER    = 59
SNW_CMAP    = build_cmap(SNW_CCOLS, SNW_OVER)

# CFR field: cloud fraction is a 0-1 fraction like snow, and the approved
# design (eyes-only draft, 2026-07-19, promoted to permanent) reuses the snow
# field's GrADS levels and colors. Named separately so the CFR image does not
# silently change if the snow palette is ever retuned.
CFR_LEVELS  = list(SNW_LEVELS)
CFR_CMAP    = build_cmap(SNW_CCOLS, SNW_OVER)

# ---------------------------------------------------------------------------
# Snow ANOMALY colormap - replicates snow_4.gs panels 3 and 4
# GrADS: 'set clevs -35 -25 -15 -5 5 15 25 35'
#        'set ccols 79 77 75 73 0 32 34 36 38'
#
# GrADS ccols convention for 8 clevs (9 ccols):
#   ccols[0]=79 (dark brown) -> below -35 %pts
#   ccols[1]=77 -> -35 to -25
#   ccols[2]=75 -> -25 to -15
#   ccols[3]=73 (light brown) -> -15 to  -5
#   ccols[4]=0  (white) -> -5 to  +5  (near-normal band)
#   ccols[5]=32 (light green) -> +5 to +15
#   ccols[6]=34 -> +15 to +25
#   ccols[7]=36 -> +25 to +35
#   ccols[8]=38 (dark green) -> above +35
#
# Anomaly definition (from snow_4.gs):
#   meanval = ave(maskout(sn, sn), t=startmon, t=endmon, 1yr)   [1987-2010 baseline]
#   percent = 100 * (current - meanval)                          [% point departure]
# ---------------------------------------------------------------------------
ANOM_LEVELS = [-35, -25, -15, -5, 5, 15, 25, 35]
# interval_ccols covers the 7 intervals between the 8 clevs:
_ANOM_INTERVAL_CCOLS = [77, 75, 73, 0, 32, 34, 36]
_ANOM_OVER  = 38
ANOM_CMAP   = build_cmap(_ANOM_INTERVAL_CCOLS, _ANOM_OVER)
ANOM_CMAP.set_under(_rgb_f(79))   # dark brown for departures below -35 %pts
                                   # (overrides the default white set_under)

# ---------------------------------------------------------------------------
# ICE: 10 clevs -> 9 intervals + 1 overflow = 10 ccols (+1 under = 11 total)
# GrADS reference (ice.gs): 'set clevs 10 20 30 40 50 60 70 80 90 100'
#                           'set ccols 0 21 31 32 34 43 45 47 54 56 59'
# ice.gs colors the first interval 21 on the NH panel and 22 on the SH panel.
# This figure deliberately uses the NH value in both hemispheres, so the two
# panels share one colorbar convention: the per-hemisphere difference is a
# legacy quirk of two hand-written GrADS blocks, and this is a new figure that
# replicates no existing archive deliverable, so nothing depends on matching it.
# The 1.0-degree ICE archive stores percent directly (observed range 0 to ~103,
# the >100 tail being cells where the pass-count normalization slightly exceeds
# unity), so these levels are read in percent with no scaling.
# ---------------------------------------------------------------------------
ICE_LEVELS  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ICE_CCOLS   = [21, 31, 32, 34, 43, 45, 47, 54, 56]   # 9 interval colors
ICE_OVER    = 59
ICE_CMAP    = build_cmap(ICE_CCOLS, ICE_OVER)

# ---------------------------------------------------------------------------
# WVP anomaly levels are in mm, not % points, so they cannot reuse ANOM_LEVELS
# (±35 %pts), which would render a WVP anomaly field almost entirely within the
# single near-normal band.  Measured over the 1.0-degree ncdc-bin record, a
# monthly WVP departure from the 1991-2020 normal spans about -12 to +20 mm,
# with a 1st-to-99th-percentile range near -5 to +11 mm, so ±8 mm resolves the
# bulk of the field and leaves the tails to the under/over colors.  The colormap
# is shared with the snow and ice anomaly (brown below normal, green above).
# ---------------------------------------------------------------------------
WVP_ANOM_LEVELS = [-8, -6, -4, -2, 2, 4, 6, 8]

# ---------------------------------------------------------------------------
# Anomaly display scaling, per product.  The three 1.0-degree anomaly products
# do NOT share a storage scale, and this table is the one place that records it:
#   SNW  stored as a 0-1 fraction -> ×100 yields % points, matching snow_4.gs
#                                     ('percent = 100*(current-meanval)')
#   ICE  stored as percent already -> ×1, the departure is already in % points
#   WVP  stored in mm -> ×1, the departure is in mm
# Applying the snow factor to ICE or WVP would inflate their anomalies 100-fold.
# ---------------------------------------------------------------------------
ANOM_SCALE = {'SNW': 100.0, 'CFR': 100.0, 'ICE': 1.0, 'WVP': 1.0}

# ---------------------------------------------------------------------------
# Known-bad years in the 1.0-degree archive, excluded from any climatology.
#
# ncdc-bin/WVP.98 holds ~8.9 mm in all twelve months where 1997 and 1999 hold
# ~25 mm, with a normal valid-cell count, so its cells carry wrong values
# rather than fill and no emptiness check catches them.  The archived
# 2.5-degree NetCDF for the same months is correct (May 1998: 25.26 mm early,
# 25.38 mm late), so the defect is specific to this 1.0-degree file and does
# not affect the published 2.5-degree record or the paper's WVP comparison.
# Left in the average, it drags the 1991-2020 WVP normal down by roughly
# 0.55 mm at every ocean cell, which would surface as a spurious +0.55 mm
# anomaly in every month rendered against it.  The 1.0-degree archive cannot be
# regenerated (the pre-1998 antenna-temperature inputs no longer exist on
# disk), so the year is excluded here rather than repaired.
#
# This is a deliberate, per-year list backed by that evidence, NOT an automatic
# outlier filter.  A genuinely extreme year (a strong ENSO, a volcanic signal)
# belongs in a climate normal, so only a defect this well characterized is
# removed, and removing one is a code change that gets reviewed.
# Only the morning chain reaches 1998 (the per-satellite F-16/F-18 1.0-degree
# files begin in 2006), so in practice this bites only the ncdc-bin baseline.
# ---------------------------------------------------------------------------
BAD_1DEG_YEARS = {'WVP': frozenset({1998})}

# Minimum fraction of grid cells a 1.0-degree month must fill to be treated as a
# real observation rather than a partially-written or near-empty slot.  Applied
# symmetrically: to each baseline year AND to the current month, so a current
# field is never differenced against a baseline it would not itself qualify for.
# 2% of the 64,800-cell grid is ~1,296 cells, comfortably below the smallest
# real product domain (snow ~29%, WVP ~51%, ice ~62%) and above a stray handful
# of cells in a month that was caught mid-write.
MIN_VALID_FRAC = 0.02


def _month_is_valid(grid):
    """True if a 1.0-degree month grid has at least MIN_VALID_FRAC real cells.

    The single integrity floor used for both the baseline years and the current
    field, so the two are gated on the same rule.  A None grid (missing/short/
    all-fill, already screened by _read_1deg_month) is not valid.
    """
    if grid is None:
        return False
    return np.count_nonzero(~np.isnan(grid)) >= grid.size * MIN_VALID_FRAC

# ---------------------------------------------------------------------------
# Anomaly baseline window, per product.  These are NOT interchangeable.
#
# SNW and WVP use the WMO 1991-2020 normal, which is the point of computing the
# anomaly from the 1.0-degree archive at all: it reaches back to 1987, where the
# 2.5-degree combined bins only reach 2008.  Both products cross the 2008/2009
# SSM/I -> SSMIS sensor transition without a step (WVP holds ~30 mm either side;
# the snow series declines monotonically across the satellite epochs, a check
# the v02 paper also makes for its snow trend), so averaging across the
# transition is sound for them.
#
# ICE cannot use that window.  Its concentration steps sharply at the same
# transition, and the step grows poleward.  Measured over ncdc-bin May fields,
# SSM/I era (1992-2008) vs SSMIS era (2009-2020) means:
#     50-60N  -0.9    70-75N  -6.1    83-86N  -14.2    88-90N  -32.9
# Every SSM/I year reads exactly 100.00 poleward of 80N while every SSMIS year
# reads 65-92, and the change is a cliff between 2006 and 2009 rather than a
# trend, so it is sensor-dependent, not ice loss.  A 1991-2020 ICE baseline
# would average 18 SSM/I years against 11 SSMIS years into a hybrid normal that
# matches neither sensor, and differencing a current SSMIS month against it
# would paint 20-30 points of pure artifact across the central Arctic that
# reads as ice loss.  Masking a polar cap does not rescue it, because the step
# reaches down into the marginal ice zone.  So ICE is referenced to an
# SSMIS-only window instead: shorter, but sensor-self-consistent and honestly
# labeled.  2009 is the F-17 record start and also the start of the era the v02
# paper's own ICE/NSIDC-0079 validation uses.
#
# Scope of the ICE step: it affects CONCENTRATION anomalies only.  The paper's
# sea-ice EXTENT series (Fig. 5) thresholds at 15% concentration, which is
# insensitive to a 100->65 shift deep in the pack; the extent computed from the
# paper's own 2.5-degree NetCDF runs 10.09 (Mar 2008, SSM/I) to 10.03 (Mar 2009,
# SSMIS), a change well inside the ±0.4 interannual spread, so that trend and
# the paper's SSMIS-era validation are unaffected by this.
#
# Decided by Hilawe 2026-07-16 after the step was quantified.  A genuine 30-year
# ICE normal needs inter-sensor homogenization, which is a research task and is
# a separate research task, not attempted here.
# ---------------------------------------------------------------------------
ANOM_BASELINE = {
    'SNW': (1991, 2020),
    'CFR': (1991, 2020),
    'WVP': (1991, 2020),
    'ICE': (2009, 2020),
}


def _baseline_desc(baseline, n_base):
    """
    Describe the baseline a figure was actually computed against, for colorbar
    labels and titles.

    Takes the (start, end) window the caller ACTUALLY passed to
    _compute_climatology_1deg rather than re-reading ANOM_BASELINE, so a caller
    that overrides the window cannot end up with a label naming the default one.
    The whole point of this figure family is that the label states the baseline
    behind the number, so the label must not be able to drift from it.

    Reports the REALIZED year count rather than the nominal window length: the
    morning primary covers 29 of the 30 WMO years for snow, while the late
    chain covers far fewer, and a label implying a full 30-year normal for
    every satellite would overstate what is behind the number.

    Parameters
    ----------
    baseline : tuple
        (start_year, end_year) as passed to _compute_climatology_1deg.
    n_base : int
        Number of years that actually contributed.

    Called by: gen_snw, gen_ice, gen_wvp_anom
    """
    b0, b1 = baseline
    tag = 'WMO' if (b0, b1) == (1991, 2020) else 'SSMIS era'
    return f'{b0}-{b1} mean ({tag}, {n_base} yr)'


# ---------------------------------------------------------------------------
# Binary file readers
# ---------------------------------------------------------------------------

def read_grads_binary(fpath):
    """
    Read a single-record GrADS binary product file.

    The file contains one month of data stored as N_LON*N_LAT float32 values.
    GrADS binary layout: south-first latitude, 1.25°E-first longitude,
    lon varies fastest (X-fastest convention, i.e., C row-major with lat as outer).
    Written by write_grads() in climalg_ssmis.py as xp.T.tofile() where xp has
    shape (N_LON, N_LAT), making xp.T shape (N_LAT, N_LON) stored C row-major.

    Returns
    -------
    grid : ndarray of shape (N_LAT, N_LON), dtype float32
        Values at grid points, with NaN where data is missing (-999.99 fill).
    None if file is missing or too small.
    """
    if not os.path.exists(fpath):
        print(f'  Missing: {fpath}')
        return None
    data = np.fromfile(fpath, dtype=np.float32)
    if data.size < N_LON * N_LAT:
        print(f'  Warning: {fpath} has only {data.size} values (expected {N_LON*N_LAT})')
        return None
    # Reshape to (N_LAT, N_LON) - lat rows (south -> north), lon columns (1.25°E -> 358.75°E)
    grid = data[:N_LON * N_LAT].reshape(N_LAT, N_LON)
    grid = grid.copy()
    grid[grid <= -999.0] = np.nan
    return grid


def read_current_1deg(indir, sat, prod, yy, mm):
    """
    Read the current month's 1.0-degree field for an anomaly figure, applying
    the SAME gates the baseline applies: the MIN_VALID_FRAC integrity floor and
    the BAD_1DEG_YEARS exclusion.  Returns the grid, or None with a printed
    reason if the month is missing, too sparse, or a documented-bad product-year.

    This is why gen_ice and gen_wvp_anom use it instead of read_prod_1deg: an
    anomaly must never difference a current field that would itself fail the
    baseline's admission test (a partially-written month), and must never render
    a known-corrupt year (e.g. WVP 1998) as the current field against a baseline
    that correctly excluded it.

    Called by: gen_ice, gen_wvp_anom
    """
    yr4 = int(yy) if int(yy) >= 100 else int(_year4_from_yy(yy))
    if yr4 in BAD_1DEG_YEARS.get(prod, ()):
        print(f'  {prod} {yr4} is a documented-bad 1.0° year; no image written')
        return None
    # quiet: this function prints the one decisive line below, so the inner
    # reader stays silent instead of emitting a near-duplicate message first.
    grid = read_prod_1deg(indir, sat, prod, yy, mm, quiet=True)
    if not _month_is_valid(grid):
        print(f'  1.0° {prod} field missing or too sparse for {sat} {yy}-{mm}; '
              f'no image written')
        return None
    return grid


def read_prod_1deg(indir, sat, prod, yy, mm, quiet=False):
    """
    Read one month's field for a product from the 1.0-degree per-year combined
    binary file.  For SNW this replicates what the operational GrADS snow_4.gs
    reads via snow_mon_10.ctl.

    The operational snow CTL (snow_mon_10.ctl) points to:
        f17-1.0/SNW.{yy}-f17-1.0
    This is the yearly combined file written by combine.py (combine_10deg()),
    containing 12 months × (N_LON_1 * N_LAT_1) = 12 × 64,800 float32 values.
    Grid layout: 360 lon × 180 lat, first column 0.5°E, south-first.
    ICE and WVP use the identical layout under the same {sat}-1.0 directory.

    Using 1.0° data for the polar panels eliminates the large blocky cell-edge
    holes and oversmoothing that appear when using the 2.5° per-month binary,
    because there are 4× as many grid cells to interpolate over.

    Parameters
    ----------
    indir : str
        Root monthly directory (one level above {sat}-1.0/).
    sat : str
        Satellite identifier (e.g. 'f17').
    prod : str
        Product code as it appears in the filename ('SNW', 'ICE', 'WVP').
    yy : str
        Two-digit year string (e.g. '26' for 2026).
    mm : str
        Two-digit month string (e.g. '03' for March).

    Returns
    -------
    grid : ndarray (N_LAT_1, N_LON_1) = (180, 360) float32, or None.
        Product field in its stored units (SNW fraction, ICE percent, WVP mm),
        with NaN where fill (-999.99).

    Called by: gen_snw, gen_ice, gen_wvp_anom
    """
    fpath = os.path.join(indir, f'{sat}-1.0', f'{prod}.{yy}-{sat}-1.0')
    grid = _read_1deg_month(fpath, int(mm) - 1)
    if grid is None and not quiet:
        print(f'  1.0° {prod} file missing, too short, or month empty: {fpath}')
    return grid


def _read_1deg_month(fpath, mon_0based):
    """
    Read one calendar month from a 12-month 1.0-degree yearly binary.

    Shared by read_prod_1deg (per-satellite {sat}-1.0/{PROD}.{yy}-{sat}-1.0 files)
    and _compute_climatology_1deg (the ncdc-bin/{PROD}.{YY} morning-chain
    archive).  Both use the identical layout: 12 consecutive months, each
    N_LON_1*N_LAT_1 = 64,800 float32 values in GrADS (N_LAT_1, N_LON_1) order
    (south-first, 0.5°E-first, lon fastest).

    Returns (N_LAT_1, N_LON_1) float32 with NaN at fill (<= -999.0), or None if
    the file is missing or does not contain the requested month.
    """
    cells       = N_LON_1 * N_LAT_1
    byte_offset = mon_0based * cells * 4
    if not os.path.exists(fpath):
        return None
    if os.path.getsize(fpath) < byte_offset + cells * 4:
        return None
    with open(fpath, 'rb') as f:
        f.seek(byte_offset)
        raw = f.read(cells * 4)
    grid = np.frombuffer(raw, dtype=np.float32).copy().reshape(N_LAT_1, N_LON_1)
    grid[grid <= -999.0] = np.nan
    # A month slot present in the file but entirely fill (a not-yet-processed
    # month, e.g. a mid-year gap in the yearly file) is treated as absent, so the
    # current-field read falls back to 2.5-degree and empty baseline years are
    # skipped rather than contributing an all-NaN layer.
    if not np.any(~np.isnan(grid)):
        return None
    return grid


def read_combined_month(fpath, month_idx):
    """
    Read one month's data from a multi-year combined 2.5-degree binary file.

    The combined file is structured as a flat sequence of months, each stored
    as N_LON*N_LAT = 10,368 float32 values in GrADS (N_LAT, N_LON) layout -
    exactly the same layout as the individual monthly files written by
    climalg_ssmis.py and assembled by combine.py.

    Parameters
    ----------
    fpath : str
        Path to the combined multi-year binary (e.g., SNW-f17-2.5).
    month_idx : int
        Zero-based index of the month within the file.
        Index 0 = first month in the file (e.g., January of start_year).
        For f17 (start_year=1987): March 1987 -> idx=2, March 2010 -> idx=278.

    Returns
    -------
    grid : ndarray of shape (N_LAT, N_LON), dtype float32
        Values with NaN where fill (-999.99). None if file missing or too short.

    Called by: _compute_snow_climatology
    """
    cells = N_LON * N_LAT           # 10,368 float32 values per month
    byte_offset = month_idx * cells * 4

    if not os.path.exists(fpath):
        return None
    if os.path.getsize(fpath) < byte_offset + cells * 4:
        return None   # month not yet written (future slot still at fill)

    with open(fpath, 'rb') as f:
        f.seek(byte_offset)
        raw = f.read(cells * 4)

    grid = np.frombuffer(raw, dtype=np.float32).copy().reshape(N_LAT, N_LON)
    grid[grid <= -999.0] = np.nan
    return grid


def smooth_field(data, sigma=SMOOTH_SIGMA):
    """
    Apply a light Gaussian spatial smoothing to a 2-D gridded field,
    preserving NaN (fill/missing) boundaries.

    This replicates GrADS 'set csmooth on', which convolves the field with
    a small kernel before shading, softening the blocky appearance of a
    2.5°-resolution pcolormesh without the heavy cubic-spline overshoot that
    matplotlib contourf produces.

    The NaN-aware algorithm (replacing NaN with 0 for convolution, then
    dividing by a smoothed mask of valid-cell weights) is a standard technique
    that prevents valid data near coastlines or data-void regions from bleeding
    into, or being diluted by, neighbouring NaN cells.

    Parameters
    ----------
    data : ndarray (N_LAT, N_LON), dtype float32
        Input gridded field; NaN marks missing / fill cells.
    sigma : float
        Gaussian standard deviation in grid-cell units.
        0.0 -> no smoothing (pure pcolormesh cell-fill).
        0.7 -> light smoothing (~1 cell radius); default.
        1.0 -> moderate smoothing (~2 cell radius, approaching GrADS csmooth).
        Values above 1.5 are not recommended for 2.5° data.

    Returns
    -------
    smoothed : ndarray (N_LAT, N_LON), dtype float32
        Smoothed field; NaN cells remain NaN.

    Called by: plot_global, plot_polar_4panel
    """
    if not HAS_SCIPY or sigma == 0.0:
        return data

    # NaN-aware Gaussian smooth:
    #   1. Replace NaN with 0 before convolution (NaN would propagate otherwise).
    #   2. Build a binary validity mask (1=valid, 0=NaN).
    #   3. Smooth both the zeroed data and the mask with the same kernel.
    #   4. Divide to get the weighted mean over valid neighbours only.
    #   5. Restore NaN wherever the original cell was NaN.
    nan_mask  = np.isnan(data)
    data_fill = np.where(nan_mask, 0.0, data).astype(np.float64)
    weight    = (~nan_mask).astype(np.float64)

    smooth_data   = _gaussian_filter(data_fill, sigma=sigma)
    smooth_weight = _gaussian_filter(weight,    sigma=sigma)

    # Avoid division by zero at cells that are entirely surrounded by NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(smooth_weight > 0, smooth_data / smooth_weight, np.nan)

    result[nan_mask] = np.nan   # restore original NaN positions
    return result.astype(np.float32)


def _compute_snow_climatology(indir, sat, cal_month_0based,
                               baseline_start=1991, baseline_end=2020):
    """
    2.5-degree monthly snow-cover climatology from the combined multi-year
    binary. FALLBACK ONLY: gen_snw now computes the anomaly at 1.0-degree via
    _compute_climatology_1deg, and only calls this when the 1.0-degree
    field or baseline is unavailable.

    IMPORTANT, why this is a fallback and not the primary: the 2.5-degree
    combined bin SNW-{sat}-2.5 only holds valid morning-chain data from about
    2008 onward (the archived daily Ta grids do not reach earlier). So although
    this asks for the WMO 1991-2020 period, the years it actually finds are
    roughly 2008-2020 for the morning chain, i.e. a ~13-year average, NOT the
    30-year normal. gen_snw therefore labels the fallback anomaly with a
    non-WMO caption. The genuine 1991-2020 baseline lives in the 1.0-degree
    ncdc-bin archive and is used by _compute_climatology_1deg.

    Replicates the GrADS snow_4.gs definition:
      'define meanval = ave(maskout(sn, sn), t=startmon, t=endmon, 1yr)'
    where 't=startmon' and 't=endmon' bound the requested period for the target
    calendar month, and '1yr' tells GrADS to stride by 12 months (accumulate
    only the matching calendar month from each year).

    Parameters
    ----------
    indir : str
        Root input directory (monthly/).  Combined file is at
        {indir}/{sat}-2.5/SNW-{sat}-2.5.
    sat : str
        Satellite identifier ('f17', 'f16', 'f18').
    cal_month_0based : int
        Calendar month index 0-based (0=Jan, 1=Feb, 2=Mar, ..., 11=Dec).
    baseline_start : int
        First year of the WMO climatological baseline period (default 1991).
    baseline_end : int
        Last year of the WMO climatological baseline period (default 2020).

    Returns
    -------
    clim : ndarray (N_LAT, N_LON) float32, or None if insufficient data.
        Climatological mean snow fraction for this calendar month.

    Called by: gen_snw
    """
    combined_path = os.path.join(indir, f'{sat}-2.5', f'SNW-{sat}-2.5')
    if not os.path.exists(combined_path):
        print(f'  Climatology: combined file not found: {combined_path}')
        return None

    sat_start = SAT_START_YEAR.get(sat, 1987)
    # Clamp baseline to years the combined file actually covers for this satellite
    eff_start = max(baseline_start, sat_start)
    eff_end   = min(baseline_end, 2026)

    monthly_grids = []
    for yr in range(eff_start, eff_end + 1):
        # month_idx = offset of this year from the file's start year × 12 + calendar month
        month_idx = (yr - sat_start) * 12 + cal_month_0based
        grid = read_combined_month(combined_path, month_idx)
        if grid is not None:
            monthly_grids.append(grid)

    if not monthly_grids:
        print(f'  Climatology: no baseline months found for {sat}')
        return None

    # Stack all baseline years along axis 0, then compute the nanmean.
    # nanmean ignores fill/missing cells so ocean/land-free grid boxes get NaN.
    stack = np.stack(monthly_grids, axis=0)          # (n_years, N_LAT, N_LON)
    clim  = np.nanmean(stack, axis=0).astype(np.float32)
    print(f'  Climatology: averaged {len(monthly_grids)} years '
          f'({eff_start}-{eff_end}) for {sat} month {cal_month_0based+1:02d}')
    return clim


# A climatology built from very few years is not a normal in any useful sense
# and an anomaly against it is dominated by the interannual noise of those years.
# Require at least this many contributing years, else decline the anomaly (the
# panel then renders its 'Climatology unavailable' placeholder). Six is a
# pragmatic floor: below it the "mean" is barely distinguishable from a single
# year, while the shortest baseline actually shipped (f18, 11 years) clears it.
MIN_BASELINE_YEARS = 6


def _compute_climatology_1deg(indir, sat, prod, cal_month_0based,
                              baseline_start=1991, baseline_end=2020):
    """
    Compute the WMO 1991-2020 monthly climatology for a product at 1.0-degree,
    from the same 1.0-degree series used for the display panels.

    For SNW this supersedes _compute_snow_climatology (which reads the
    2.5-degree combined bins).  Those 2.5-degree combined files only hold valid
    data from 2008 onward (morning chain), so their "1991-2020" baseline was in
    practice a ~2008-2020 average.  Computing the whole anomaly at 1.0-degree
    from a series that reaches back to 1987 gives a genuine WMO normal, and
    keeps the baseline and the current-month field at the same resolution (no
    1.0-vs-2.5 mixing).  ICE and WVP have the same 29-of-30-year coverage in
    ncdc-bin and use this function on the same terms.

    The returned climatology is in the product's STORED units (SNW fraction,
    ICE percent, WVP mm).  Converting a departure to display units is the
    caller's job, via ANOM_SCALE (the three products do not share a scale).

    Baseline source by chain:
      - Morning chain (NCDC_BIN_SAT, currently F-17): the ncdc-bin archive
        ncdc-bin/{PROD}.{YY}, F-08 -> F-17, real data for 29 of the 30 years
        1991-2020 (only Jan 1991 absent, the F-08/F-11 handover).  Verified for
        SNW, ICE and WVP.
      - Other satellites (F-16 late chain, F-18): their own per-year 1.0-degree
        files {sat}-1.0/{PROD}.{yy}-{sat}-1.0, which currently begin in 2006.
        Each satellite uses only its own chain, so no cross-chain calibration
        mixing.

    Years whose file is absent, or whose requested month is fill/empty (fewer
    than 2% valid cells), are skipped; the climatology is the nanmean of the
    remaining years, so a fill cell in some years does not corrupt the mean.
    No per-cell minimum-year requirement is enforced, matching the operational
    GrADS definition ave(maskout(sn,sn), t=start, t=end, 1yr), which averages
    whatever valid years a cell has. A cell valid in only a few baseline years
    therefore still gets a climatology (and a defined anomaly); cells with no
    valid baseline year remain NaN.

    Parameters
    ----------
    indir : str
        Root monthly directory (holds ncdc-bin/ and {sat}-1.0/).
    sat : str
        Satellite identifier ('f17', 'f16', 'f18').
    prod : str
        Product code as it appears in the filename ('SNW', 'ICE', 'WVP').
    cal_month_0based : int
        Calendar-month index (0=Jan ... 11=Dec).
    baseline_start, baseline_end : int
        Inclusive WMO baseline period (default 1991-2020).

    Returns
    -------
    (clim, n_years) : (ndarray (N_LAT_1, N_LON_1) float32, int), or (None, 0)
        Climatological mean in the product's stored units, and the number of
        contributing years.

    Called by: gen_snw, gen_ice, gen_wvp_anom
    """
    grids = []
    excluded = []
    for yr in range(baseline_start, baseline_end + 1):
        yy2 = yr % 100
        if sat == NCDC_BIN_SAT:
            fpath = os.path.join(indir, 'ncdc-bin', f'{prod}.{yy2:02d}')
        else:
            fpath = os.path.join(indir, f'{sat}-1.0', f'{prod}.{yy2:02d}-{sat}-1.0')
        grid = _read_1deg_month(fpath, cal_month_0based)
        if not _month_is_valid(grid):
            continue    # missing, or too sparse to be a real observed month
        if yr in BAD_1DEG_YEARS.get(prod, ()):
            # Documented archive defect, not a data gap; see BAD_1DEG_YEARS.
            # Tested here, after the year is known to hold real data, so the
            # exclusion is only reported when it actually removed something (a
            # satellite whose record starts after the bad year never had it).
            excluded.append(yr)
            continue
        grids.append(grid)

    if len(grids) < MIN_BASELINE_YEARS:
        print(f'  1.0° {prod} climatology: only {len(grids)} valid baseline '
              f'year(s) for {sat} (need {MIN_BASELINE_YEARS}); declining anomaly')
        return None, 0
    with warnings.catch_warnings():
        # Cells with no valid baseline year nanmean to NaN (correct); silence only
        # the routine "Mean of empty slice" notice, leaving any real overflow or
        # invalid-arithmetic RuntimeWarning from corrupt input to surface.
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        clim = np.nanmean(np.stack(grids, axis=0), axis=0).astype(np.float32)
    src = 'ncdc-bin' if sat == NCDC_BIN_SAT else f'{sat}-1.0'
    excl = f', excluding {sorted(excluded)} as known-bad' if excluded else ''
    print(f'  1.0° {prod} climatology: averaged {len(grids)} years '
          f'({baseline_start}-{baseline_end}, {src}) for {sat} '
          f'month {cal_month_0based+1:02d}{excl}')
    return clim, len(grids)


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def save_gif(fig, outpath, dpi=100):
    """
    Save matplotlib figure as GIF via Pillow.
    Falls back to PNG if Pillow is not installed.

    Called by: plot_global, plot_polar_4panel, write_archive_imagery
    """
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img.save(outpath, format='GIF')
        print(f'  Saved: {outpath}')
    except ImportError:
        outpath_png = outpath.replace('.gif', '.png')
        fig.savefig(outpath_png, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'  Saved (PNG - install Pillow for GIF): {outpath_png}')


def save_ps(fig, outpath, dpi=100):
    """
    Save matplotlib figure as PostScript (.ps) file.

    The operational GrADS snow_4.gs produces both a GIF and a .ps file for the
    snow cover panel.  The .ps file is included in the NCEI imagery tar alongside
    the GIF.  matplotlib's PostScript backend writes vector PS directly.

    Called by: write_archive_imagery (for snow-color product only)
    """
    fig.savefig(outpath, format='ps', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'  Saved: {outpath}')


def write_archive_imagery(f17_figs, yyyy4, mm2, archive_dir):
    """
    Write f17 imagery to the NCEI archive directory using the standard
    mw-hydro_v01_imagery_* naming convention that tar_mw-hydro_netcdf.sh expects.

    The archive tar at netcdf/imagery/ must contain exactly these files per month:
      mw-hydro_v01_imagery_pr1_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_lwp_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_wvp_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_snow-color_{YYYY}{MM}.gif
      mw-hydro_v01_imagery_snow-color_{YYYY}{MM}.ps

    Only f17 imagery is archived (f17 is the primary distribution satellite).
    All four figures are already rendered by the time this function is called -
    each was produced by gen_pr1/gen_lwp/gen_wvp/gen_snw during the f17 iteration
    of the satellite loop in run().  Reusing them avoids redundant rendering.
    The GIF files are written via save_gif(); the snow .ps via save_ps().

    After this function returns, run() closes all figures with plt.close().

    Parameters
    ----------
    f17_figs : dict
        Keys: 'pr1', 'lwp', 'wvp', 'snw'.
        Values: matplotlib Figure objects (or None if data was missing).
    yyyy4    : str    Four-digit year string (e.g. '2026')
    mm2      : str    Zero-padded two-digit month string (e.g. '03')
    archive_dir : str Path to netcdf/imagery/ directory.

    Called by: run() when --archive-dir is specified
    """
    os.makedirs(archive_dir, exist_ok=True)
    yyyymm = f'{yyyy4}{mm2}'

    for prod_key, archive_stem in (('pr1', 'pr1'),
                                    ('lwp', 'lwp'),
                                    ('wvp', 'wvp'),
                                    ('snw', 'snow-color')):
        fig = f17_figs.get(prod_key)
        if fig is None:
            print(f'  Archive: skipping {prod_key} - no figure (missing input data?)')
            continue
        gif_name = f'mw-hydro_{PRODUCT_VERSION}_imagery_{archive_stem}_{yyyymm}.gif'
        save_gif(fig, os.path.join(archive_dir, gif_name))
        # Snow cover also requires a PostScript file for the NCEI imagery tar.
        # GrADS historically produced both .gif and .ps from the same render;
        # we replicate this by writing the same figure to PostScript format.
        if prod_key == 'snw':
            ps_name = f'mw-hydro_{PRODUCT_VERSION}_imagery_{archive_stem}_{yyyymm}.ps'
            save_ps(fig, os.path.join(archive_dir, ps_name))


def _footer_str():
    """
    Return the operational footer text: '{D} {Mon} {YYYY}  NOAA NCEI'.
    Example: '17 Apr 2026  NOAA NCEI'.

    Date uses today's actual date with a non-zero-padded day (matching the
    GrADS annotation convention: '17 Apr 2026', not '17 APR 2026').
    strftime('%b') gives 'Apr' (title-case, not uppercase), matching ops style.

    Called by: _add_footer (snow 4-panel); plot_global (via fig.supxlabel).
    """
    today = datetime.date.today()
    return f'{today.day} {today.strftime("%b %Y")}  NOAA NCEI'


def _add_footer(fig):
    """
    Draw the operational footer as a plain fig.text() at the bottom of the figure.

    Used ONLY by the snow 4-panel (plot_polar_4panel), which lays out with
    constrained_layout OFF and an explicit subplots_adjust(bottom=0.09) reserved
    margin, so a fixed fig.text() at y=0.01 sits well clear of the lowest colorbar
    label.  The global maps (plot_global) use constrained_layout, which ignores
    fig.text(); they place the footer with fig.supxlabel() instead so the layout
    engine reserves space for it and it cannot collide with the colorbar label
    (see plot_global).

    Called by: plot_polar_4panel
    """
    fig.text(0.5, 0.01, _footer_str(),
             ha='center', va='bottom', fontsize=7, style='italic', color='#333333')


# ---------------------------------------------------------------------------
# Global equatorial map (PR1, LWP, WVP)
# Matches GrADS: 'set lat -50 50', 'set lon 0 360', PlateCarree projection
# ---------------------------------------------------------------------------

def plot_global(data, product_var, header_title,
                cmap, levels, cbar_label, outpath, lat_range=(-50, 50),
                lons=None, lats=None, subtitle=None, extend='max'):
    """
    Plot a global equatorial-focused shaded map (standard for PR1, LWP, WVP).

    Rendering method: contourf with BoundaryNorm - smoothly interpolates between
    the discrete 2.5° grid-cell centers, matching the visual appearance of the
    operational GrADS output better than a flat cell-fill at this resolution.

    Title layout: a single fig.suptitle() line carries the full title
    (e.g., 'Rainfall for Mar 2026').  There is no separate axes-level title,
    keeping the layout compact and unambiguous.

    Parameters
    ----------
    data : ndarray (N_LAT, N_LON)
        Gridded field to plot (already scaled to display units).
    product_var : str
        Short product code for internal ID (e.g., 'ra', 'lw', 'ta').
    header_title : str
        Single-line figure title (e.g., 'Rainfall for Mar 2026').
    cmap : ListedColormap
        Pre-built colormap from build_cmap().
    levels : list
        Color level boundaries (N values -> N-1 color intervals).
    cbar_label : str
        Units string drawn below the colorbar.
    outpath : str
        Full output file path (.gif or .png).
    lat_range : tuple
        (lat_min, lat_max) for the map extent. Default (-50, 50).
    lons, lats : array-like or None
        1-D coordinate centers for data.  Default: LONS/LATS (2.5°).  Pass
        LONS_1/LATS_1 for a 1.0° field (gen_wvp_anom).
    subtitle : str or None
        Optional second title line, drawn under header_title.  Used by
        gen_wvp_anom to name the baseline the anomaly is actually against.
    extend : str
        contourf/colorbar extend mode.  Default 'max' suits the sequential
        products, whose fields are bounded below by zero; a diverging anomaly
        field passes 'both' so departures beyond either end keep their
        under/over colors instead of being clipped to the end bins.

    Called by: gen_pr1, gen_lwp, gen_wvp, gen_wvp_anom
    """
    if data is None:
        return None
    _lons = lons if lons is not None else LONS
    _lats = lats if lats is not None else LATS

    # BoundaryNorm maps the N-1 intervals between the N level boundaries to
    # the N-1 colors in the ListedColormap.  clip=False lets values outside
    # [levels[0], levels[-1]] use the set_under/set_over colors.
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

    # constrained_layout=True handles title / axes / colorbar spacing automatically.
    # Do NOT call subplots_adjust after this - it disables constrained_layout and
    # causes the colorbar to overlap the map.
    #
    # Figure height: PlateCarree with extent [0,360,-50,50] has a 3.6:1 aspect
    # ratio (360° lon / 100° lat), so the map is widest (spans the full 10" width)
    # only while the axes box stays wider than 3.6:1.  At 4.0" the title, colorbar,
    # and footer reservation squeeze the axes height enough that the fixed-aspect
    # map becomes height-limited and shrinks in BOTH dimensions (~690 px wide after
    # the tight-bbox crop).  4.6" restores enough axes height to keep the map
    # width-limited - it spans the full width (~1000 px) with the footer placed
    # clear of the colorbar label.  (The earlier 4.0" assumed the footer overlapped
    # the colorbar label and therefore needed no space of its own; that was the bug.)
    #
    # h_pad=0.2 (~0.5 cm) is tight but sufficient - keeps title clear of the map
    # and colorbar label clear of the footer without excess dead space.
    if HAS_CARTOPY:
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 4.6), constrained_layout=True,
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180.0)}
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.2, w_pad=0.3)
    fig.patch.set_facecolor('white')

    # Single-line suptitle: the full product + month/year string.
    # No separate axes-level title - constrained_layout packs the suptitle
    # directly above the map with no extra gap.  An anomaly caller passes a
    # subtitle naming the baseline, which is appended as a smaller second line.
    if subtitle:
        fig.suptitle(f'{header_title}\n{subtitle}', fontsize=13,
                     fontweight='bold', color='#1a1a6e', linespacing=1.4)
    else:
        fig.suptitle(header_title, fontsize=13, fontweight='bold',
                     color='#1a1a6e')

    if HAS_CARTOPY:
        ax.set_extent([0, 360, lat_range[0], lat_range[1]],
                      crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.6, color='k')
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                'cultural', 'admin_0_boundary_lines_land', '50m',
                edgecolor='#555555', facecolor='none', linewidth=0.3
            )
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                'cultural', 'admin_1_states_provinces_lines', '50m',
                edgecolor='#888888', facecolor='none', linewidth=0.2
            )
        )
        # Labelled graticule: LABELS only, no lines (linewidth=0).  The former
        # semi-transparent grey dashes blended with the saturated fill colors
        # and the GIF's 256-color quantization snapped those blends to patchy
        # whitish streaks that read as straight-line data artifacts (found on
        # the CFR field diagnostic, 2026-07-19; the underlying data was
        # verified smooth at the graticule latitudes).  The gridliner itself
        # must stay, with draw_labels=True, because it is what draws the
        # lat/lon axis labels - removing it entirely loses them.
        # Longitude ticks: 0, 60E, 120E, 180, 120W, 60W (matching ops pr1/lwp/wvp.gs)
        # Latitude ticks: every 10° across the caller's extent.
        # Labels only on left (lat) and bottom (lon) to avoid crowding the map.
        gl = ax.gridlines(draw_labels=True, linewidth=0,
                          crs=ccrs.PlateCarree())
        gl.xlocator     = mticker.FixedLocator([0, 60, 120, 180, -120, -60])
        # Latitude ticks every 10° across whatever extent the caller asked for,
        # so a wider anomaly map (gen_wvp_anom) still labels its full range
        # instead of stopping at the ±50° default.
        gl.ylocator     = mticker.FixedLocator(
            list(range(int(np.ceil(lat_range[0] / 10.0)) * 10,
                       int(np.floor(lat_range[1] / 10.0)) * 10 + 1, 10)))
        gl.top_labels    = False
        gl.bottom_labels = True
        gl.left_labels   = True
        gl.right_labels  = False  # left-side labels only; right side omitted
        gl.xlabel_style = {'size': 7, 'color': '#333333'}
        gl.ylabel_style = {'size': 7, 'color': '#333333'}
        # contourf interpolates smoothly between the 2.5° cell centers,
        # matching the visual appearance of the operational GrADS output.
        # transform=PlateCarree() tells cartopy the data coords are lon/lat degrees.
        cf = ax.contourf(_lons, _lats, data,
                         levels=levels, cmap=cmap, norm=norm, extend=extend,
                         transform=ccrs.PlateCarree())
    else:
        lat_mask = (_lats >= lat_range[0]) & (_lats <= lat_range[1])
        # Mask NaN so contourf does not attempt to interpolate through missing cells
        data_masked = np.ma.masked_invalid(data[lat_mask, :])
        cf = ax.contourf(_lons, _lats[lat_mask], data_masked,
                         levels=levels, cmap=cmap, norm=norm, extend=extend)
        ax.set_xlabel('Longitude (°E)', fontsize=8)
        ax.set_ylabel('Latitude (°N)', fontsize=8)
        ax.tick_params(labelsize=7)

    # Colorbar (horizontal, matching GrADS cbarmf.gs position).
    # fraction=0.07 makes the bar ~75% thicker than the previous 0.04 so the
    # color gradient bands are clearly distinguishable.
    # shrink=0.92 extends horizontal coverage to ~92% of axes width.
    # Colorbar extend must match the contourf extend so the legend shows the same
    # under/over arrows the map uses. A diverging anomaly (extend='both') needs
    # the lower arrow too; hardcoding 'max' would drop it while the map still
    # colored sub-range cells with the under color.
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                        fraction=0.07, shrink=0.92, extend=extend)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Auto-thin colorbar ticks when there are more than 8 level boundaries.
    # A horizontal colorbar ~8" wide at fontsize 7 fits ≤ 8 labels without
    # overlap.  LWP has 10 levels at 0.05 intervals - skip every other one
    # so labels read as 0.10, 0.20 ... 0.50 instead of all 10 cramming together.
    if len(levels) > 8:
        cbar.set_ticks(levels[::2])

    # Footer (date + attribution).  This figure uses constrained_layout, which
    # does NOT reserve space for a plain fig.text() - placing the footer that way
    # packs it directly under the colorbar's units label (e.g. 'mm/day') and the
    # two strings overlap (the jumbled text under the legend).  fig.supxlabel() IS
    # accounted for by the constrained_layout engine, which stacks it cleanly
    # below the colorbar label with automatic spacing - no manual offsets, no
    # collision.  The snow 4-panel keeps the fig.text() footer (it lays out with
    # constrained_layout OFF and reserves a bottom margin via subplots_adjust).
    fig.supxlabel(_footer_str(), fontsize=7, fontstyle='italic', color='#333333')
    save_gif(fig, outpath)
    # Note: figure is NOT closed here - caller is responsible for plt.close(fig).
    # This allows run() to reuse the f17 figure for archive-named copies without
    # re-rendering the expensive cartopy plot.
    return fig


# ---------------------------------------------------------------------------
# Polar stereographic 4-panel map (SNW)
# Replicates the complete snow_4.gs layout:
#   Panel NW (top-left):  Northern Hemisphere snow cover  (NPS, 30-90°N)
#   Panel NE (top-right): NH anomaly (100×departure from 1991-2020 WMO baseline)
#   Panel SW (bot-left):  Southern Hemisphere snow cover  (SPS, 30-90°S)
#   Panel SE (bot-right): SH anomaly
#
# GrADS script settings reproduced here:
#   Snow:    'set clevs .1 .2 .3 .4 .5 .6 .7 .8 .9 1'
#            'set ccols 0 22 31 32 34 43 45 47 54 56 59'
#   Anomaly: 'set clevs -35 -25 -15 -5 5 15 25 35'
#            'set ccols 79 77 75 73 0 32 34 36 38'
#   Anomaly computed as: 100 * (current_month − climatological_mean)
# ---------------------------------------------------------------------------

def plot_polar_4panel(data_full, anom_precomp, header_title,
                      cmap, levels, anom_cmap, anom_levels,
                      cbar_label, anom_label, outpath,
                      snw_lons=None, snw_lats=None,
                      anom_lons=None, anom_lats=None,
                      lat_cutoff=30, subtitle=None):
    """
    Plot a 4-panel polar stereographic snow-cover figure matching snow_4.gs.

    Layout (2 rows × 2 columns):
      [0,0] NH snow cover         [0,1] NH anomaly (% departure)
      [1,0] SH snow cover         [1,1] SH anomaly

    If clim_full is None (combined file unavailable), the two anomaly panels
    are drawn with a 'Climatology unavailable' label instead of data.

    Parameters
    ----------
    data_full : ndarray (N_LAT, N_LON)
        Current month's global snow fraction field (0-1).
    clim_full : ndarray (N_LAT, N_LON) or None
        1991-2020 WMO baseline climatological mean snow fraction.
        If None, anomaly panels show a placeholder message.
    header_title : str
        Figure suptitle (e.g., 'SSM/I Snow Cover for Mar 2026').
    cmap : ListedColormap
        Snow cover colormap (SNW_CMAP).
    levels : list
        Snow cover level boundaries (SNW_LEVELS).
    anom_cmap : ListedColormap
        Anomaly colormap (ANOM_CMAP).
    anom_levels : list
        Anomaly level boundaries in % points (ANOM_LEVELS).
    cbar_label : str
        Units label for the snow cover colorbar (e.g., 'fraction').
    anom_label : str
        Units label for the anomaly colorbar (e.g., '% departure from mean').
    outpath : str
        Full output file path (.gif or .png).
    snw_lons : array-like or None
        1-D longitude centers for data_full.  Default: LONS (2.5°).
        Pass LONS_1 when data_full is from the 1.0° yearly file.
    snw_lats : array-like or None
        1-D latitude centers for data_full.  Default: LATS (2.5°).
        Pass LATS_1 when data_full is from the 1.0° yearly file.
    anom_lons : array-like or None
        1-D longitude centers for anom_precomp.  Default: LONS (2.5°).
    anom_lats : array-like or None
        1-D latitude centers for anom_precomp.  Default: LATS (2.5°).
    lat_cutoff : int
        Equatorward edge of both hemispheres' polar panels, in degrees.
        Default 30 matches snow_4.gs ('set mpvals -270 90 30 90').  gen_ice
        passes 50 to match ice.gs, whose sea-ice domain stops at ±50°.
    subtitle : str or None
        Second suptitle line describing the baseline.  MUST describe the
        baseline actually used: the caller falling back to a non-WMO baseline
        passes a subtitle that does not claim WMO.  Default None draws the
        header alone.

    Called by: gen_snw, gen_ice
    """
    if data_full is None:
        return None

    # Use caller-supplied coordinate arrays. gen_snw() now supplies 1.0° grids
    # for both the snow field and the anomaly (from ncdc-bin / {sat}-1.0); the
    # 2.5° defaults here apply only to the fallback path and any legacy caller.
    _snw_lons  = snw_lons  if snw_lons  is not None else LONS
    _snw_lats  = snw_lats  if snw_lats  is not None else LATS
    _anom_lons = anom_lons if anom_lons is not None else LONS
    _anom_lats = anom_lats if anom_lats is not None else LATS

    # Anomaly is pre-computed by gen_snw() on the grid given by anom_lons/anom_lats
    # (1.0° from the WMO 1991-2020 ncdc-bin baseline, or 2.5° on the fallback path).
    anom = anom_precomp

    norm_snw  = mcolors.BoundaryNorm(boundaries=levels,      ncolors=cmap.N,      clip=False)
    norm_anom = mcolors.BoundaryNorm(boundaries=anom_levels, ncolors=anom_cmap.N, clip=False)

    # ---------------------------------------------------------------------------
    # Layout: explicit subplots_adjust instead of constrained_layout.
    #
    # constrained_layout cannot be used here because it applies h_pad to EVERY
    # side of EVERY subplot cell - meaning the inter-row gap = 2×h_pad + hspace.
    # With the h_pad needed for outer margins (~0.6"), the center gap balloons to
    # 1.24", completely dominating the figure.  subplots_adjust gives independent
    # control over each of the six margin/spacing values.
    #
    # Figure geometry - 10" wide × 11" tall (portrait-ish):
    #   subplot grid:  left=0.05  right=0.95  top=0.88  bottom=0.09
    #     col width  = (0.95-0.05) × 10" / 2 = 4.5"   per column
    #     row height ≈ (0.88-0.09) × 11" / (2+0.15) ≈ 4.0"  per row
    # -> cells are ~4.5" × 4.0", nearly square -> circles ~4" diameter
    #   left/right outer margins = 0.05 × 10" = 0.5" ≈ 1.3 cm
    #   hspace = 0.15 -> inter-row gap = 0.15 × 4.0" = 0.60" ≈ 1.5 cm
    #     (colorbars steal space from the axes via fraction/pad; the remaining
    #      gap sits between colorbar label and the next row's panel title)
    #   wspace = 0.03 -> minimal column gap (circles provide natural separation)
    #   suptitle y=0.96 -> 0.08 × 11" = 0.88" above grid top ≈ 2.2 cm
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 11), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.09,
                        hspace=0.15, wspace=0.03)
    fig.patch.set_facecolor('white')

    # Two-line suptitle above the subplot grid.
    # y=0.96 places it above top=0.88, giving ~0.9" of clear space.
    # The baseline line is caller-supplied because it is a factual claim about
    # which baseline produced the anomaly panels: this was previously hardcoded
    # to the WMO string, so a figure that fell back to the 2.5-degree ~2008-2020
    # baseline still asserted "1991-2020 (WMO)" in its title while its colorbar
    # label correctly said otherwise.  Callers now pass the matching text.
    fig.suptitle('\n'.join([header_title, subtitle]) if subtitle else header_title,
                 fontsize=12, fontweight='bold', color='#1a1a6e', linespacing=1.4,
                 y=0.96)

    # Circular boundary path - clips each polar axes to a disc, replicating
    # the circular GrADS polar stereo output.  Without this, cartopy renders
    # a rectangular bounding box with white corners.
    _theta = np.linspace(0, 2 * np.pi, 100)
    _circle_path = mpath.Path(
        np.vstack([np.sin(_theta), np.cos(_theta)]).T * 0.5 + 0.5
    )

    # Projection: central_longitude=-90 replicates GrADS 'set mpvals -270 90'
    # whose midpoint = (-270+90)/2 = -90°, placing 90°W (Americas) at the bottom
    # of the circular polar plot - the standard NOAA NPS/SPS orientation.
    if HAS_CARTOPY:
        proj_nps = ccrs.NorthPolarStereo(central_longitude=-90)
        proj_sps = ccrs.SouthPolarStereo(central_longitude=-90)
        axs = np.array([
            [fig.add_subplot(2, 2, 1, projection=proj_nps),
             fig.add_subplot(2, 2, 2, projection=proj_nps)],
            [fig.add_subplot(2, 2, 3, projection=proj_sps),
             fig.add_subplot(2, 2, 4, projection=proj_sps)],
        ])
    else:
        axs = np.array([
            [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)],
            [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)],
        ])

    # Panel specification:
    #   (row, col, lat_s, lat_n, field, norm, cmap_used, field_lons, field_lats,
    #    title, cbar_lbl)
    # Snow and anomaly panels both use caller-supplied coordinates
    # (_snw_lons/_snw_lats and _anom_lons/_anom_lats); gen_snw now supplies 1.0°
    # for both, with 2.5° only on the fallback path.
    panels = [
        (0, 0, lat_cutoff,  90,  data_full, norm_snw,  cmap,
         _snw_lons, _snw_lats, 'Northern Hemisphere',         cbar_label),
        (0, 1, lat_cutoff,  90,  anom,      norm_anom, anom_cmap,
         _anom_lons, _anom_lats, 'Northern Hemisphere Anomaly', anom_label),
        (1, 0, -90, -lat_cutoff, data_full, norm_snw,  cmap,
         _snw_lons,  _snw_lats,  'Southern Hemisphere',         cbar_label),
        (1, 1, -90, -lat_cutoff, anom,      norm_anom, anom_cmap,
         _anom_lons, _anom_lats, 'Southern Hemisphere Anomaly', anom_label),
    ]

    for row, col, lat_s, lat_n, field, norm, cm, f_lons, f_lats, title, cb_lbl \
            in panels:
        ax = axs[row, col]

        if HAS_CARTOPY:
            # Clip the axes to a circle - replicates the disc-shaped GrADS polar
            # stereo plot.  Must be called before set_extent so the circular mask
            # is applied in axes coordinates (transAxes), not data coordinates.
            ax.set_boundary(_circle_path, transform=ax.transAxes)
            ax.set_extent([-180, 180, lat_s, lat_n], crs=ccrs.PlateCarree())
            # 50m resolution matches GrADS 'set mpdset hires'
            ax.coastlines(resolution='50m', linewidth=0.5, color='k')
            # Country borders (50m)
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    'cultural', 'admin_0_boundary_lines_land', '50m',
                    edgecolor='#555555', facecolor='none', linewidth=0.3
                )
            )
            # US state boundaries (matching 'set mpdset hires' which includes states)
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    'cultural', 'admin_1_states_provinces_lines', '50m',
                    edgecolor='#888888', facecolor='none', linewidth=0.2
                )
            )
            # No graticule lines: over the saturated snow/ice fill they suffer
            # the same GIF-quantization streaking as the global maps (see
            # plot_global), and these panels draw no labels, so the gridliner
            # has nothing else to contribute. Coastlines carry the georeference.

        if not HAS_CARTOPY:
            # Degraded (no-cartopy) mode draws plain Cartesian axes; without an
            # equal aspect a 360-lon x 60-lat extent is stretched to the square
            # panel. Ops hosts have cartopy, so this only affects diagnostics.
            ax.set_aspect('equal')

        ax.set_title(title, fontsize=9, fontweight='bold')

        if field is None:
            # Anomaly panels with no climatology - placeholder text + dummy colorbar.
            ax.text(0.5, 0.5, 'Climatology\nunavailable',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='#666666', style='italic')
            dummy = np.full((N_LAT, N_LON), np.nan)
            if HAS_CARTOPY:
                cf = ax.pcolormesh(LON_EDGES, LAT_EDGES, dummy,
                                   cmap=cm, norm=norm,
                                   transform=ccrs.PlateCarree(), shading='flat')
            else:
                cf = ax.pcolormesh(LON_EDGES, LAT_EDGES, dummy,
                                   cmap=cm, norm=norm, shading='flat')
        else:
            # contourf with masked-invalid data.
            # Per-panel f_lons/f_lats let each panel use its own coordinate
            # arrays (snow and anomaly both 1.0°, or 2.5° on the fallback path).
            field_masked = np.ma.masked_invalid(field)
            extend_mode  = 'both' if cm is anom_cmap else 'max'
            if HAS_CARTOPY:
                cf = ax.contourf(f_lons, f_lats, field_masked,
                                 levels=norm.boundaries, cmap=cm, norm=norm,
                                 extend=extend_mode,
                                 transform=ccrs.PlateCarree())
            else:
                lat_mask = (f_lats >= lat_s) & (f_lats <= lat_n)
                cf = ax.contourf(f_lons, f_lats[lat_mask],
                                 field_masked[lat_mask, :],
                                 levels=norm.boundaries, cmap=cm, norm=norm,
                                 extend=extend_mode)

        extend_mode = 'both' if cm is anom_cmap else 'max'
        cbar = fig.colorbar(cf, ax=ax, orientation='horizontal',
                            pad=0.04, fraction=0.05, shrink=0.85,
                            extend=extend_mode)
        cbar.set_label(cb_lbl, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    # Footer placed at y=0.01 (fixed figure coordinates).  With subplots_adjust
    # bottom=0.09, the lowest colorbar label sits at ~9% from the bottom, and the
    # footer text at 1% - leaving ~8% ≈ 0.9" of clear space between them.
    _add_footer(fig)
    save_gif(fig, outpath)
    # Note: figure is NOT closed here - caller is responsible for plt.close(fig).
    return fig


# ---------------------------------------------------------------------------
# Per-product image generation functions
# Each function reads the appropriate binary file, applies the required
# data scaling, builds the title strings exactly matching the GrADS output,
# constructs the output filename following the operational convention
# (Mon{YY}-{var}-25prod.gif, e.g., Mar26-ra-25prod.gif), and calls the
# appropriate plotting routine.
# ---------------------------------------------------------------------------

def gen_pr1(sat, yy, mm, outdir, indir):
    """
    Monthly rainfall (PR1) global map.

    Data scaling: binary stores ndays*24*mean_rain_mmhr; divide by ndays to
    get mm/day (matching GrADS 'display ra/daynum').

    Output filename: {Mon}{YY}-ra-25prod.gif  (e.g., Mar26-ra-25prod.gif)
    GrADS reference: pr1.gs
    """
    mon_idx = int(mm) - 1
    ndays   = calendar.monthrange(2000 + int(yy), int(mm))[1]
    fpath   = os.path.join(indir, f'{sat}-2.5', f'PR1{yy}-{mm}-{sat}-2.5')
    data    = read_grads_binary(fpath)
    if data is None:
        return None
    # Scale: raw accumulator -> mm/day (GrADS: 'display ra/daynum')
    data = data / float(ndays)

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Rainfall for {mon_name} {yr4}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-ra-25prod.gif')
    return plot_global(data, 'ra', header_title,
                       PR1_CMAP, PR1_LEVELS, 'mm/day', outpath,
                       # PR1's set_under lavender (GrADS 53, land / <1 mm/day) is
                       # painted on the map, so the colorbar needs the lower
                       # arrow too or the legend leaves that color undocumented.
                       extend='both',
                       lat_range=(-50, 50))


def gen_lwp(sat, yy, mm, outdir, indir):
    """
    Monthly cloud liquid water path (LWP) global map.

    Data scaling: binary stores mean LWP in g/m² (= 1000×kg/m²); divide
    by 1000 to get kg/m² (matching GrADS 'display lw/1000').

    Output filename: {Mon}{YY}-lw-25prod.gif
    GrADS reference: lwp.gs
    """
    mon_idx = int(mm) - 1
    fpath   = os.path.join(indir, f'{sat}-2.5', f'LWP{yy}-{mm}-{sat}-2.5')
    data    = read_grads_binary(fpath)
    if data is None:
        return None
    # Scale: g/m² -> kg/m² (GrADS: 'display lw/1000')
    data = data / 1000.0

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Liquid Water Path for {mon_name} {yr4}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-lw-25prod.gif')
    return plot_global(data, 'lw', header_title,
                       LWP_CMAP, LWP_LEVELS, 'kg/m²', outpath,
                       lat_range=(-50, 50))


def gen_wvp(sat, yy, mm, outdir, indir):
    """
    Monthly total precipitable water / TPW (WVP) global map.

    Data scaling: none - binary stores WVP directly in mm.

    IMPORTANT: The operational GrADS script wvp.gs uses variable name 'ta'
    (total atmospheric), not 'wv'. The output filename MUST use '-ta-' not '-wv-':
      {Mon}{YY}-ta-25prod.gif  (e.g., Mar26-ta-25prod.gif)

    GrADS reference: wvp.gs
    """
    mon_idx = int(mm) - 1
    fpath   = os.path.join(indir, f'{sat}-2.5', f'WVP{yy}-{mm}-{sat}-2.5')
    data    = read_grads_binary(fpath)
    if data is None:
        return None
    # No scaling needed (already in mm; GrADS: 'display ta' directly)

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Total Precipitable Water for {mon_name} {yr4}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-ta-25prod.gif')
    return plot_global(data, 'ta', header_title,
                       WVP_CMAP, WVP_LEVELS, 'monthly mm average', outpath,
                       lat_range=(-50, 50))


def gen_snw(sat, yy, mm, outdir, indir):
    """
    Monthly snow cover (SNW) 4-panel polar stereographic image.

    Data scaling: none - binary stores snow fraction (0-1) directly.

    Replicates the complete snow_4.gs 4-panel layout:
      Panel 1 (top-left):  NH snow cover (NPS, lat 30-90°N)
      Panel 2 (top-right): NH anomaly (100 × departure from WMO 1991-2020 mean)
      Panel 3 (bot-left):  SH snow cover (SPS, lat 30-90°S)
      Panel 4 (bot-right): SH anomaly

    Both the snow field and the anomaly are computed at 1.0-degree.  The WMO
    1991-2020 climatological mean comes from the same 1.0-degree series
    (_compute_climatology_1deg): the ncdc-bin morning-chain archive
    (F-08 -> F-17, back to 1987) for the morning primary, else the
    per-satellite {sat}-1.0 yearly files.  This replaces the earlier 2.5-degree
    anomaly whose combined-bin baseline only reached 2008.  If the 1.0-degree
    current field or baseline is unavailable, the panel falls back to the
    2.5-degree anomaly so it still renders.

    Output filename: {Mon}{YY}-sn-25prod.gif
    GrADS reference: snow_4.gs
    """
    mon_idx = int(mm) - 1

    # Read snow fraction from the 1.0-degree yearly combined file, matching
    # the operational snow_4.gs which reads snow_mon_10.ctl -> f17-1.0/SNW.{yy}.
    # This gives 4× finer spatial resolution than the 2.5° per-month binary,
    # reducing the large cell-edge holes visible over Canada and Russia.
    data = read_prod_1deg(indir, sat, 'SNW', yy, mm)
    if data is None:
        # Fall back to 2.5° if the 1.0° file is unavailable (e.g., early in a year
        # before combine_10deg has been run, or for a satellite with no 1.0° output).
        print(f'  Falling back to 2.5° snow data for {sat} {yy}-{mm}')
        fpath = os.path.join(indir, f'{sat}-2.5', f'SNW{yy}-{mm}-{sat}-2.5')
        data  = read_grads_binary(fpath)
        snw_lons, snw_lats = LONS, LATS   # 2.5° centers
    else:
        snw_lons, snw_lats = LONS_1, LATS_1   # 1.0° centers

    # Decline if there is no current-month snow field to show. This catches both a
    # missing file (data is None) and a present-but-entirely-fill month, which the
    # 2.5° reader (read_grads_binary) returns as an all-NaN grid rather than None
    # (unlike the 1.0° reader). Without this guard such a month renders as a blank
    # 4-panel figure that falsely reads as "no snow anywhere" (e.g. a not-yet-
    # processed or partially-written month). The 1.0° path already rejects an
    # all-fill month upstream in _read_1deg_month; this makes the 2.5° path match.
    if data is None or not np.any(np.isfinite(data)):
        if data is not None:
            print(f'  Snow field all fill for {sat} {yy}-{mm}; no image written')
        return None

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Snow Cover for {mon_name} {yr4}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-sn-25prod.gif')

    # Anomaly, computed entirely at 1.0-degree.  The current month is the same
    # 1.0-degree field shown in the NH/SH panels (from ncdc-bin for the morning
    # chain, or the per-satellite {sat}-1.0 yearly file).  The baseline is the
    # genuine WMO 1991-2020 normal from the same 1.0-degree series
    # (_compute_climatology_1deg).  This replaces the former 2.5-degree
    # anomaly, whose combined-bin baseline only reached 2008 and so was really a
    # ~13-year average mislabeled as 1991-2020.  Doing both fields at 1.0-degree
    # also removes the previous 1.0-vs-2.5 resolution mismatch between the
    # display panels and the anomaly.
    #
    # GrADS reference (snow_4.gs): 'define percent = 100*(current - meanval)'
    # where meanval = ave(maskout(sn,sn), t=startmon, t=endmon, 1yr).
    anom = None
    anom_lons_use, anom_lats_use = LONS_1, LATS_1
    # Anomaly-panel label tracks the baseline ACTUALLY used, and the WMO claim is
    # asserted only inside the successful 1.0-degree branch. The neutral default
    # holds if no anomaly can be computed (placeholder panel) so the WMO label
    # never appears without the WMO baseline behind it.
    anom_label = '% departure from baseline mean'
    # The suptitle baseline line tracks the same branch as anom_label, so a
    # fallback figure never carries a WMO claim in its title (see
    # plot_polar_4panel's subtitle parameter).
    anom_subtitle = None
    # The anomaly branch applies the same MIN_VALID_FRAC floor the baseline
    # years pass through (_month_is_valid), so a partially-written 1.0-degree
    # month is never differenced against the gated WMO baseline (the round-2
    # symmetric-gating principle; this was the one current-field path that
    # still bypassed it, found by the multi-model review). The FIELD panels
    # still render whatever the month holds - only the anomaly declines, and
    # control then falls to the honestly-labeled 2.5-degree path below.
    if snw_lons is LONS_1 and _month_is_valid(data):   # gated 1.0-degree field
        _b0, _b1 = ANOM_BASELINE['SNW']
        clim_1deg, n_base = _compute_climatology_1deg(
            indir, sat, 'SNW', mon_idx, baseline_start=_b0, baseline_end=_b1)
        if clim_1deg is not None:
            # NaN propagates through the subtraction, so a fill cell in either the
            # current field or the baseline yields NaN in the anomaly (no explicit
            # mask needed). The Python float 100.0 does not upcast the float32 arrays.
            # SNW is stored as a 0-1 fraction, so ANOM_SCALE['SNW'] = 100 converts
            # the departure to % points (ICE and WVP are NOT scaled; see ANOM_SCALE).
            anom = (ANOM_SCALE['SNW'] * (data - clim_1deg)).astype(np.float32)
            # Report the realized baseline length. It is the full WMO window only for
            # the morning primary (f17, 29 of 30 yr); the late chain covers fewer years
            # (f16 ~15, f18 ~11 within 1991-2020), so naming the count keeps the label
            # honest rather than implying a full 30-year normal for every satellite.
            _desc = _baseline_desc((_b0, _b1), n_base)
            anom_label = f'% departure from {_desc}'
            anom_subtitle = f'Anomaly Based on Departure from the {_desc}'

    if anom is None:
        # 1.0-degree current field or baseline unavailable (e.g. 2.5° fallback
        # path). Fall back to the 2.5-degree anomaly so the panel still renders.
        # That 2.5-degree combined baseline only reaches ~2008 for the morning
        # chain, so it is NOT the WMO normal; the label is set accordingly to
        # avoid mislabeling the fallback image.
        #
        # DELIBERATE POLICY: when the 1.0-degree current field rendered but its
        # baseline declined (a satellite's early years, e.g. a future WSF-M with
        # fewer than MIN_BASELINE_YEARS of history), this produces a MIXED figure:
        # 1.0-degree field panels beside a 2.5-degree anomaly, each drawn on its
        # own coordinate arrays (plot_polar_4panel takes them separately for
        # exactly this reason). Snow prefers rendering-with-honest-label over the
        # placeholder ICE uses; pinned by a regression test.
        fpath_25 = os.path.join(indir, f'{sat}-2.5', f'SNW{yy}-{mm}-{sat}-2.5')
        data_25  = read_grads_binary(fpath_25)
        clim_25  = _compute_snow_climatology(indir, sat, mon_idx,
                                             baseline_start=1991, baseline_end=2020)
        if data_25 is not None and clim_25 is not None:
            # NaN propagates through the subtraction (see the 1.0-degree branch).
            anom = (ANOM_SCALE['SNW'] * (data_25 - clim_25)).astype(np.float32)
            anom_lons_use, anom_lats_use = LONS, LATS
            anom_label = '% departure from recent-period mean'
            anom_subtitle = ('Anomaly Based on Departure from the '
                             'Recent-Period Mean (not the WMO baseline)')

    return plot_polar_4panel(
        data, anom, header_title,
        SNW_CMAP,  SNW_LEVELS,
        ANOM_CMAP, ANOM_LEVELS,
        'monthly snow cover fraction', anom_label,
        outpath,
        snw_lons=snw_lons, snw_lats=snw_lats,
        anom_lons=anom_lons_use, anom_lats=anom_lats_use,
        subtitle=anom_subtitle
    )


def gen_ice(sat, yy, mm, outdir, indir):
    """
    Monthly sea ice (ICE) 4-panel polar stereographic image with anomaly.

    Layout mirrors gen_snw:
      Panel 1 (top-left):  NH sea ice        Panel 2 (top-right): NH anomaly
      Panel 3 (bot-left):  SH sea ice        Panel 4 (bot-right): SH anomaly

    Data scaling: none.  ICE is stored in PERCENT (0-100), unlike SNW which is
    a 0-1 fraction, so the departure is already in % points and ANOM_SCALE
    leaves it alone.  Applying the snow ×100 here would inflate it 100-fold.

    Both the ice field and the anomaly are computed at 1.0-degree from the same
    series (_compute_climatology_1deg): the ncdc-bin morning-chain archive for
    the morning primary, else the per-satellite {sat}-1.0 yearly files.

    The baseline is the SSMIS-era window ANOM_BASELINE['ICE'] (2009-2020), NOT
    the WMO 1991-2020 normal that SNW and WVP use, because ICE concentration
    steps at the 2008/2009 SSM/I -> SSMIS transition by -5 points at 60-70N
    growing to -33 points at 88-90N.  See the ANOM_BASELINE comment for the
    measurements and the reasoning; the short version is that a cross-era ICE
    normal would paint the central Arctic with sensor artifact that reads as
    ice loss.  The label names the SSMIS era explicitly so the shorter baseline
    is visible on the figure.

    There is no 2.5-degree fallback: unlike snow, ICE has no operational
    2.5-degree anomaly path to fall back to, and a 2.5-degree combined baseline
    would only reach 2008, so when the 1.0-degree baseline is unavailable the
    anomaly panels render their 'Climatology unavailable' placeholder rather
    than a silently-shorter baseline.

    Panels stop at ±50° latitude, matching ice.gs ('set mpvals -270 90 50 90'),
    which is the sea-ice domain.

    Output filename: {Mon}{YY}-ic-25prod.gif  (var 'ic', per ice.gs)
    NOT part of the NCEI archive imagery tar, whose contents are fixed at
    pr1/lwp/wvp/snow-color (see write_archive_imagery).

    GrADS reference: ice.gs (field panels; the anomaly panels are new)
    """
    mon_idx = int(mm) - 1

    # read_current_1deg applies the MIN_VALID_FRAC floor and the bad-year
    # exclusion, the same gates the baseline uses, so a partially-written or
    # known-bad current month declines rather than rendering a misleading figure.
    data = read_current_1deg(indir, sat, 'ICE', yy, mm)
    if data is None:
        return None

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Sea Ice Cover for {mon_name} {yr4}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-ic-25prod.gif')

    anom       = None
    anom_label = '% point departure from baseline mean'
    anom_subtitle = None
    _b0, _b1 = ANOM_BASELINE['ICE']
    clim, n_base = _compute_climatology_1deg(
        indir, sat, 'ICE', mon_idx, baseline_start=_b0, baseline_end=_b1)
    if clim is not None:
        # NaN propagates through the subtraction, so a fill cell in either field
        # yields NaN in the anomaly.
        anom = (ANOM_SCALE['ICE'] * (data - clim)).astype(np.float32)
        _desc = _baseline_desc((_b0, _b1), n_base)
        anom_label = f'% point departure from {_desc}'
        anom_subtitle = f'Anomaly Based on Departure from the {_desc}'

    return plot_polar_4panel(
        data, anom, header_title,
        ICE_CMAP,  ICE_LEVELS,
        ANOM_CMAP, ANOM_LEVELS,
        'percent of time and area', anom_label,
        outpath,
        snw_lons=LONS_1, snw_lats=LATS_1,
        anom_lons=LONS_1, anom_lats=LATS_1,
        lat_cutoff=50, subtitle=anom_subtitle
    )


def gen_wvp_anom(sat, yy, mm, outdir, indir):
    """
    Monthly total precipitable water (WVP) anomaly global map.

    This is a SEPARATE image from gen_wvp, which keeps rendering the 2.5-degree
    WVP field unchanged.  The archive imagery tar carries that field image
    (mw-hydro_v01_imagery_wvp_*), so its content is a fixed contract; the
    anomaly is additive and is not archived.

    Data scaling: none.  WVP is stored in mm, so the departure is in mm and
    ANOM_SCALE leaves it alone (contrast SNW, a 0-1 fraction scaled ×100).
    Levels are WVP_ANOM_LEVELS (±8 mm), not the ±35 %-point ANOM_LEVELS, which
    would collapse a WVP anomaly into the single near-normal band.

    Both the current field and the baseline are read at 1.0-degree from the same
    series (_compute_climatology_1deg): ncdc-bin for the morning primary
    (29 of 30 baseline years), else the per-satellite {sat}-1.0 yearly files.
    No 2.5-degree fallback, for the same reason as gen_ice.

    The map spans ±60°, wider than the ±50° of the WVP field image, because the
    ocean WVP retrieval reaches into both mid-latitude storm-track regions where
    the moisture anomaly is of interest; it stops short of the poles where the
    retrieval is not defined.

    Output filename: {Mon}{YY}-ta-anom-25prod.gif  (var 'ta', per wvp.gs)
    """
    mon_idx = int(mm) - 1

    # Same gating as gen_ice: the WVP anomaly must not render 1998 (a
    # documented-bad year) as its current field against a baseline that excludes
    # it, nor difference a partially-written month.
    data = read_current_1deg(indir, sat, 'WVP', yy, mm)
    if data is None:
        return None

    _b0, _b1 = ANOM_BASELINE['WVP']
    clim, n_base = _compute_climatology_1deg(
        indir, sat, 'WVP', mon_idx, baseline_start=_b0, baseline_end=_b1)
    if clim is None:
        print(f'  No 1.0° WVP baseline for {sat}; no anomaly image written')
        return None

    anom = (ANOM_SCALE['WVP'] * (data - clim)).astype(np.float32)

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Total Precipitable Water Anomaly for {mon_name} {yr4}'
    subtitle     = f'Departure from the {_baseline_desc((_b0, _b1), n_base)}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-ta-anom-25prod.gif')

    return plot_global(anom, 'ta', header_title,
                       ANOM_CMAP, WVP_ANOM_LEVELS,
                       'monthly mm departure', outpath,
                       lat_range=(-60, 60),
                       lons=LONS_1, lats=LATS_1,
                       subtitle=subtitle, extend='both')


def gen_cfr(sat, yy, mm, outdir, indir):
    """
    Monthly cloud fraction (CFR) global field map.

    Promoted from the 2026-07-19 eyes-only diagnostic draft at Hilawe's
    direction; the render is identical to that approved draft. Unlike the
    PR1/LWP/WVP field maps (2.5-degree per-month binaries, ±50°), CFR reads
    the 1.0-degree yearly file and spans ±60°, matching its anomaly companion.
    There is no 2.5-degree fallback: a missing or too-sparse 1.0-degree month
    declines, consistent with the anomaly family (read_current_1deg gate).

    Data scaling: none - the binary stores the 0-1 fraction directly.

    Output filename: {Mon}{YY}-cf-25prod.gif  (var 'cf', per cfr.gs)
    Not part of the NCEI archive imagery tar (fixed contents; see
    write_archive_imagery).
    """
    mon_idx = int(mm) - 1

    data = read_current_1deg(indir, sat, 'CFR', yy, mm)
    if data is None:
        return None

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Cloud Fraction for {mon_name} {yr4}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-cf-25prod.gif')

    return plot_global(data, 'cf', header_title,
                       CFR_CMAP, CFR_LEVELS,
                       'monthly cloud fraction (0-1)', outpath,
                       lat_range=(-60, 60),
                       lons=LONS_1, lats=LATS_1)


def gen_cfr_anom(sat, yy, mm, outdir, indir):
    """
    Monthly cloud fraction (CFR) anomaly global map.

    Companion to the gen_cfr field map (the legacy cfr.gs is a dead path that
    opens PR1.ctl, so those two are CFR's only images). Enabled once the F-13
    2008 wrong-variant year in ncdc-bin was regenerated (2026-07-19), which had
    been the hold on any CFR climatology from this archive.

    Data scaling: CFR is stored as a 0-1 fraction like SNW, so ANOM_SCALE
    converts the departure to percentage points (contrast ICE, already percent,
    and WVP, mm).  The measured CFR anomaly spans about +/-34 percentage points
    at the 1st/99th percentiles with ~1% of cells beyond +/-35, so the shared
    +/-35 percentage-point ANOM_LEVELS palette fits without CFR-specific levels.

    Baseline: the genuine WMO 1991-2020 normal from ncdc-bin (29 of 30 years,
    morning primary), the same window as SNW and WVP.  No 2.5-degree fallback,
    for the same reason as gen_ice.

    Output filename: {Mon}{YY}-cf-anom-25prod.gif  (var 'cf', per cfr.gs)
    Not part of the NCEI archive imagery tar (fixed contents; see
    write_archive_imagery).
    """
    mon_idx = int(mm) - 1

    # Same gating as gen_ice/gen_wvp_anom: never difference a partially-written
    # month or a documented-bad year against the gated baseline.
    data = read_current_1deg(indir, sat, 'CFR', yy, mm)
    if data is None:
        return None

    _b0, _b1 = ANOM_BASELINE['CFR']
    clim, n_base = _compute_climatology_1deg(
        indir, sat, 'CFR', mon_idx, baseline_start=_b0, baseline_end=_b1)
    if clim is None:
        print(f'  No 1.0° CFR baseline for {sat}; no anomaly image written')
        return None

    anom = (ANOM_SCALE['CFR'] * (data - clim)).astype(np.float32)

    mon_name     = MONTH_ABBR[mon_idx]
    yr4          = _year4_from_yy(yy)
    header_title = f'Cloud Fraction Anomaly for {mon_name} {yr4}'
    subtitle     = f'Departure from the {_baseline_desc((_b0, _b1), n_base)}'
    outpath      = os.path.join(outdir, f'{mon_name}{yy}-cf-anom-25prod.gif')

    return plot_global(anom, 'cf', header_title,
                       ANOM_CMAP, ANOM_LEVELS,
                       'monthly % point departure', outpath,
                       lat_range=(-60, 60),
                       lons=LONS_1, lats=LATS_1,
                       subtitle=subtitle, extend='both')


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(yy, mm, indir=None, outdir=None, archive_dir=None):
    """
    Generate all imagery for a given two-digit year and two-digit month.

    Loops over all three operational satellites (f17, f16, f18), reads the
    corresponding per-satellite monthly binary files from indir/{sat}-2.5/,
    and writes GIF images to outdir/{sat}/.

    If archive_dir is provided, f17 imagery is ALSO written to archive_dir
    using the NCEI mw-hydro_v01_imagery_* naming convention, so that
    tar_mw-hydro_netcdf.sh can find and tar them for the ingest server.
    The snow product additionally produces a .ps PostScript file.

    Parameters
    ----------
    yy          : str   Two-digit year (zero-padded, e.g. '26')
    mm          : str   Two-digit month (zero-padded, e.g. '03')
    indir       : str   Root directory containing {sat}-2.5/ subdirectories.
                        Defaults to DEFAULT_INDIR ('..').
    outdir      : str   Parent output directory. Per-satellite subdirectories
                        ({outdir}/{sat}/) are created automatically, mirroring the
                        operational layout (img/f17/, img/f16/, img/f18/).
                        Defaults to DEFAULT_OUTDIR ('img/').
    archive_dir : str   Path to netcdf/imagery/ (or None to skip archive output).
                        Typically $hydroMONTHLY_NETCDF/imagery/.
    """
    mm_str = f'{int(mm):02d}'
    yy_str = f'{int(yy):02d}'
    yyyy4  = _year4_from_yy(yy_str)
    _indir  = indir  if indir  is not None else DEFAULT_INDIR
    _outdir = outdir if outdir is not None else DEFAULT_OUTDIR

    # f17_figs stores the returned Figure objects from the f17 satellite pass
    # so they can be reused for archive-named copies without re-rendering.
    f17_figs = {'pr1': None, 'lwp': None, 'wvp': None, 'snw': None}

    for sat in SATELLITES:
        # Mirror the operational directory structure: img/f17/, img/f16/, img/f18/.
        # Each satellite writes to its own subdirectory so files are never overwritten.
        sat_outdir = os.path.join(_outdir, sat)
        os.makedirs(sat_outdir, exist_ok=True)
        print(f'\n=== Generating images for {sat.upper()}, {yyyy4}-{mm_str} ===')

        fig_pr1 = gen_pr1(sat, yy_str, mm_str, sat_outdir, _indir)
        fig_lwp = gen_lwp(sat, yy_str, mm_str, sat_outdir, _indir)
        fig_wvp = gen_wvp(sat, yy_str, mm_str, sat_outdir, _indir)
        fig_snw = gen_snw(sat, yy_str, mm_str, sat_outdir, _indir)
        # ICE and the WVP anomaly are 1.0-degree diagnostic images. WVP is
        # referenced to the genuine WMO 1991-2020 normal; ICE is referenced to an
        # SSMIS-only 2009-2020 window instead, because its concentration steps at
        # the SSM/I -> SSMIS transition (see ANOM_BASELINE). Both are additive:
        # neither enters the archive imagery tar, whose contents are fixed
        # (write_archive_imagery), and gen_wvp's own field image is unchanged.
        fig_ice = gen_ice(sat, yy_str, mm_str, sat_outdir, _indir)
        fig_wva = gen_wvp_anom(sat, yy_str, mm_str, sat_outdir, _indir)
        fig_cfr = gen_cfr(sat, yy_str, mm_str, sat_outdir, _indir)
        fig_cfa = gen_cfr_anom(sat, yy_str, mm_str, sat_outdir, _indir)

        if sat == 'f17':
            # Retain f17 archive figures; close the rest immediately. fig_ice and
            # fig_wva are not archived, so they are closed here for every
            # satellite including f17.
            f17_figs = {'pr1': fig_pr1, 'lwp': fig_lwp, 'wvp': fig_wvp, 'snw': fig_snw}
            for fig in (fig_ice, fig_wva, fig_cfr, fig_cfa):
                if fig is not None:
                    plt.close(fig)
        else:
            # Close non-f17 figures to free memory.
            for fig in (fig_pr1, fig_lwp, fig_wvp, fig_snw, fig_ice, fig_wva, fig_cfr, fig_cfa):
                if fig is not None:
                    plt.close(fig)

    # Write NCEI archive-named copies for the ingest server (f17 only).
    if archive_dir is not None:
        print(f'\n=== Writing archive imagery to {archive_dir} ===')
        write_archive_imagery(f17_figs, yyyy4, mm_str, archive_dir)

    # Close all retained f17 figures now that archive output is done.
    for fig in f17_figs.values():
        if fig is not None:
            plt.close(fig)

    print('\nImage generation complete.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate monthly SSMIS hydrological product imagery')
    parser.add_argument('--yy',     type=str, default=None,
                        help='Year: two-digit (e.g. 26) or four-digit (e.g. 2026)')
    parser.add_argument('--mm',     type=str, default=None,
                        help='Two-digit month (e.g. 03 for March)')
    parser.add_argument('--indir',  type=str, default=None,
                        help=f'Root dir containing {{sat}}-2.5/ subdirs '
                             f'(default: {DEFAULT_INDIR})')
    parser.add_argument('--outdir', type=str, default=None,
                        help=f'Output directory for GIF files '
                             f'(default: {DEFAULT_OUTDIR})')
    parser.add_argument('--archive-dir', type=str, default=None, dest='archive_dir',
                        help='Path to netcdf/imagery/ for NCEI archive-named copies '
                             '(mw-hydro_v01_imagery_*_{YYYYMM}.gif + snow .ps). '
                             'Typically $hydroMONTHLY_NETCDF/imagery/. '
                             'If omitted, no archive copies are written.')
    args = parser.parse_args()

    if args.yy and args.mm:
        # Accept either 2-digit ('26') or 4-digit ('2026') year on the CLI;
        # reduce to 2-digit via modulo so filenames and title strings are consistent.
        yy = str(int(args.yy) % 100).zfill(2)
        mm = args.mm.zfill(2)
    else:
        # Auto-detect previous calendar month
        today           = datetime.date.today()
        first_this_mon  = today.replace(day=1)
        prev            = first_this_mon - datetime.timedelta(days=1)
        yy = str(prev.year % 100).zfill(2)
        mm = str(prev.month).zfill(2)
        print(f'Auto-detected previous month: {_year4_from_yy(yy)}-{mm}')

    run(yy, mm, indir=args.indir, outdir=args.outdir, archive_dir=args.archive_dir)


if __name__ == '__main__':
    main()
