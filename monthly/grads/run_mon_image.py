#!/usr/bin/env python3
"""
run_mon_image.py

PURPOSE
    Generate monthly GIF/PNG imagery for SSMIS hydrological products
    using matplotlib. Replaces:
      - run_mon_image.sh  (driver)
      - pr1.gs, snw.gs, lwp.gs, wvp.gs, ice.gs, pf1.gs  (GrADS scripts)
      - cbarmf.gs, cpccol.gs  (color/colorbar utilities)

SYNOPSIS
    python run_mon_image.py [--yy YY] [--mm MM]

EXAMPLE
    python run_mon_image.py --yy 12 --mm 06
    python run_mon_image.py       # auto-detects previous month

NOTES
    - Input binary files are in GrADS format: (N_LAT, N_LON) float32, lon-fastest.
      Data runs S -> N (-88.75° to 88.75°), 1.25°E to 358.75°E (for 2.5°).
    - Output images are written to img/<sat>/ as GIF files.
    - Snow/ice products use polar stereographic projections (N and S hemispheres).
    - Cartopy is used for map projections; PIL for GIF output.
    - Dependencies: matplotlib, cartopy, numpy, Pillow.
"""

import os
import sys
import argparse
import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (no X server needed)
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    from matplotlib.ticker import MaxNLocator
except ImportError:
    raise ImportError('matplotlib is required')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    print('Warning: cartopy not found - using simple lat/lon maps for all products')
    HAS_CARTOPY = False

# ---------------------------------------------------------------------------
# Grid definition (2.5-degree)
# ---------------------------------------------------------------------------
N_LON = 144
N_LAT = 72
LON_START = 1.25
LAT_START = -88.75
RES = 2.5

LONS = np.array([LON_START + j * RES for j in range(N_LON)])
LATS = np.array([LAT_START + i * RES for i in range(N_LAT)])

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

SATELLITES = ['f17', 'f16', 'f18']

IMG_DIR   = 'img'
PARENT_DIR = '..'   # monthly/ relative to grads/


# ---------------------------------------------------------------------------
# CPC color palette (from cpccol.gs, approximated)
# ---------------------------------------------------------------------------

def cpc_rainfall_colors():
    """
    CPC rainfall color scale approximating GrADS cpccol.gs + ccols specification.
    GrADS ccols: 53 55 4 11 5 13 3 38 10 7 12 8 2 27 6 29
    Returns a ListedColormap for 16 levels.
    """
    # NCAR color index approximations -> RGB
    colors = [
        '#aaffaa',  # 53 - light green
        '#55ff55',  # 55 - medium green
        '#0000ff',  # 4  - blue
        '#00ffff',  # 11 - cyan
        '#00ff00',  # 5  - bright green
        '#ff00ff',  # 13 - magenta
        '#ff0000',  # 3  - red
        '#ffff00',  # 38 - yellow
        '#ff8800',  # 10 - orange
        '#8800ff',  # 7  - purple
        '#00ff88',  # 12 - teal
        '#0088ff',  # 8  - light blue
        '#ff0088',  # 2  - pink-red
        '#ffaa00',  # 27 - gold
        '#00ffff',  # 6  - cyan
        '#ff5500',  # 29 - dark orange
    ]
    return mcolors.ListedColormap(colors)


def snow_colors():
    """Snow/ice color scale from GrADS snw.gs: ccols 0 22 31 32 34 43 45 47 54 56 59."""
    colors = [
        '#ffffff',  # 0  - white
        '#aaddff',  # 22 - light blue
        '#6699cc',  # 31 - steel blue
        '#3366aa',  # 32 - dark blue
        '#0033aa',  # 34 - navy
        '#00aa33',  # 43 - green
        '#00cc66',  # 45 - medium green
        '#00ee99',  # 47 - bright green
        '#aaffcc',  # 54 - light mint
        '#66ffaa',  # 56 - mint
        '#00ffaa',  # 59 - cyan-green
    ]
    return mcolors.ListedColormap(colors)


def lwp_wvp_colors():
    """Blue-to-cyan-to-yellow colormap for LWP/WVP."""
    return plt.cm.jet


# ---------------------------------------------------------------------------
# Binary file reader
# ---------------------------------------------------------------------------

def read_grads_binary(fpath):
    """
    Read a GrADS binary file (N_LON × N_LAT float32 per time step).
    Returns 2D array (N_LAT, N_LON) with lats S -> N, lons 1.25°E -> 358.75°E.
    Returns None if file missing.
    """
    if not os.path.exists(fpath):
        print(f'  Missing: {fpath}')
        return None
    data = np.fromfile(fpath, dtype=np.float32)
    if data.size < N_LON * N_LAT:
        return None
    # GrADS binary: lon (X) varies fastest within each lat row, stored S -> N.
    # Layout is (N_LAT, N_LON) in C row-major order - reshape directly.
    grid = data[:N_LON * N_LAT].reshape(N_LAT, N_LON)  # (N_LAT, N_LON)
    grid[grid <= -999.0] = np.nan
    return grid


def read_monthly_slice(fpath, month_idx):
    """
    Read one specific month (0-indexed) from a multi-month binary.
    Returns (N_LAT, N_LON) float32 or None.
    """
    if not os.path.exists(fpath):
        return None
    offset = month_idx * N_LAT * N_LON * 4
    try:
        with open(fpath, 'rb') as f:
            f.seek(offset)
            raw = np.fromfile(f, dtype=np.float32, count=N_LAT * N_LON)
        if raw.size < N_LAT * N_LON:
            return None
        grid = raw.reshape(N_LAT, N_LON)
        grid[grid <= -999.0] = np.nan
        return grid
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plotting routines
# ---------------------------------------------------------------------------

def save_gif(fig, outpath, dpi=100):
    """Save matplotlib figure as GIF via PIL."""
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img.save(outpath, format='GIF')
        print(f'  Saved: {outpath}')
    except ImportError:
        # Fallback: save as PNG if PIL not available
        outpath = outpath.replace('.gif', '.png')
        fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
        print(f'  Saved (PNG fallback): {outpath}')


def plot_global(data, title, cmap, levels, units, outpath, lon2d=None, lat2d=None,
                lat_range=(-50, 50)):
    """
    Plot a global equatorial-focused map (standard for PR1, LWP, WVP).
    """
    if data is None:
        return

    lons2d, lats2d = np.meshgrid(LONS, LATS)
    fig = plt.figure(figsize=(10, 6))

    if HAS_CARTOPY:
        ax = fig.add_axes([0.05, 0.12, 0.88, 0.78],
                          projection=ccrs.PlateCarree(central_longitude=180.0))
        ax.set_extent([0, 360, lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
        cf = ax.contourf(lons2d, lats2d, data, levels=levels, cmap=cmap,
                         extend='both', transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(111)
        cf = ax.contourf(LONS, LATS[np.where((LATS >= lat_range[0]) & (LATS <= lat_range[1]))],
                         data[(LATS >= lat_range[0]) & (LATS <= lat_range[1]), :],
                         levels=levels, cmap=cmap, extend='both')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')

    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                        fraction=0.04, shrink=0.8)
    cbar.set_label(units, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=11, fontweight='bold')

    save_gif(fig, outpath)
    plt.close(fig)


def plot_polar(data_n, data_s, title_n, title_s, main_title, cmap, levels, units,
               outpath_n, outpath_s):
    """
    Plot North and South polar stereographic maps (for SNW, ICE).
    Produces two separate GIF files (NH and SH).
    """
    for data, hemi, title, outpath, lat_lim, pole in [
        (data_n, 'N', title_n, outpath_n, 30, 'N'),
        (data_s, 'S', title_s, outpath_s, -30, 'S'),
    ]:
        if data is None:
            continue
        lons2d, lats2d = np.meshgrid(LONS, LATS)

        fig = plt.figure(figsize=(7, 7))
        if HAS_CARTOPY:
            proj = ccrs.NorthPolarStereo() if pole == 'N' else ccrs.SouthPolarStereo()
            ax = fig.add_axes([0.05, 0.12, 0.84, 0.80], projection=proj)
            if pole == 'N':
                ax.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
            else:
                ax.set_extent([-180, 180, -90, lat_lim], crs=ccrs.PlateCarree())
            ax.coastlines(resolution='110m', linewidth=0.5)
            ax.gridlines(linewidth=0.3, alpha=0.5)
            cf = ax.contourf(lons2d, lats2d, data, levels=levels, cmap=cmap,
                             extend='both', transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(111)
            if pole == 'N':
                mask = LATS >= lat_lim
            else:
                mask = LATS <= lat_lim
            cf = ax.contourf(LONS, LATS[mask], data[mask, :],
                             levels=levels, cmap=cmap, extend='both')
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')

        cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                            fraction=0.04, shrink=0.8)
        cbar.set_label(units, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.text(0.5, 0.01, main_title, ha='center', fontsize=8, style='italic')

        save_gif(fig, outpath)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-product image generation
# ---------------------------------------------------------------------------

def gen_pr1(sat, yy, mm, outdir):
    """Monthly rainfall (PR1) global map."""
    mon_idx = int(mm) - 1
    yr = 2000 + int(yy)
    fpath = os.path.join(PARENT_DIR, f'{sat}-2.5', f'PR1{yy}-{mm}-{sat}-2.5')
    data = read_grads_binary(fpath)
    if data is None:
        return
    mon_name = MONTH_ABBR[mon_idx]
    title = f'SSMI/S Rainfall for {mon_name} 20{yy}\n{sat.upper()}'
    outpath = os.path.join(outdir, f'{mon_name.lower()}{yy}-ra-25prod.gif')
    levels = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    plot_global(data, title, cpc_rainfall_colors(), levels, 'mm/day', outpath)


def gen_snw(sat, yy, mm, outdir):
    """Monthly snow cover (SNW) polar maps."""
    fpath = os.path.join(PARENT_DIR, f'{sat}-2.5', f'SNW{yy}-{mm}-{sat}-2.5')
    data = read_grads_binary(fpath)
    if data is None:
        return
    mon_idx = int(mm) - 1
    mon_name = MONTH_ABBR[mon_idx]
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    base = f'{mon_name.lower()}{yy}-sn-25prod'
    outpath_n = os.path.join(outdir, f'{base}-NH.gif')
    outpath_s = os.path.join(outdir, f'{base}-SH.gif')
    plot_polar(
        data, data,
        title_n='Northern Hemisphere Snow Cover',
        title_s='Southern Hemisphere Snow Cover',
        main_title=f'SSMI/S Snow Cover for {mon_name} 20{yy} | NOAA NCEI',
        cmap=snow_colors(), levels=levels, units='fraction of time and area',
        outpath_n=outpath_n, outpath_s=outpath_s,
    )


def gen_lwp(sat, yy, mm, outdir):
    """Monthly cloud liquid water path (LWP)."""
    fpath = os.path.join(PARENT_DIR, f'{sat}-2.5', f'LWP{yy}-{mm}-{sat}-2.5')
    data = read_grads_binary(fpath)
    if data is None:
        return
    mon_idx = int(mm) - 1
    mon_name = MONTH_ABBR[mon_idx]
    title = f'SSMI/S Cloud Liquid Water Path for {mon_name} 20{yy}\n{sat.upper()}'
    outpath = os.path.join(outdir, f'{mon_name.lower()}{yy}-lw-25prod.gif')
    levels = np.linspace(0, 0.6, 13)
    plot_global(data, title, plt.cm.Blues, levels, '1000·mm (kg/m²)', outpath,
                lat_range=(-60, 60))


def gen_wvp(sat, yy, mm, outdir):
    """Monthly total precipitable water (WVP)."""
    fpath = os.path.join(PARENT_DIR, f'{sat}-2.5', f'WVP{yy}-{mm}-{sat}-2.5')
    data = read_grads_binary(fpath)
    if data is None:
        return
    mon_idx = int(mm) - 1
    mon_name = MONTH_ABBR[mon_idx]
    title = f'SSMI/S Total Precipitable Water for {mon_name} 20{yy}\n{sat.upper()}'
    outpath = os.path.join(outdir, f'{mon_name.lower()}{yy}-wv-25prod.gif')
    levels = np.arange(0, 65, 5)
    plot_global(data, title, plt.cm.RdYlBu, levels, 'mm', outpath,
                lat_range=(-60, 60))


def gen_snow4(sat, yy, mm, outdir):
    """4-panel snow product combining NH and SH snow + ice."""
    fpath_snw = os.path.join(PARENT_DIR, f'{sat}-2.5', f'SNW{yy}-{mm}-{sat}-2.5')
    fpath_ice = os.path.join(PARENT_DIR, f'{sat}-2.5', f'ICE{yy}-{mm}-{sat}-2.5')
    snw = read_grads_binary(fpath_snw)
    ice = read_grads_binary(fpath_ice)

    if snw is None and ice is None:
        return

    mon_idx = int(mm) - 1
    mon_name = MONTH_ABBR[mon_idx]
    levels_snw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    levels_ice = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'SSMI/S Snow and Ice Products for {mon_name} 20{yy}\n{sat.upper()}',
                 fontsize=12, fontweight='bold')

    panels = [
        (snw, 'NH Snow', 'N', levels_snw, snow_colors(), 'fraction', 1),
        (snw, 'SH Snow', 'S', levels_snw, snow_colors(), 'fraction', 2),
        (ice, 'NH Sea Ice', 'N', levels_ice, plt.cm.Blues_r, '%', 3),
        (ice, 'SH Sea Ice', 'S', levels_ice, plt.cm.Blues_r, '%', 4),
    ]

    lons2d, lats2d = np.meshgrid(LONS, LATS)

    for data, panel_title, pole, lvls, cmap, units, panelnum in panels:
        if data is None:
            continue
        if HAS_CARTOPY:
            proj = ccrs.NorthPolarStereo() if pole == 'N' else ccrs.SouthPolarStereo()
            ax = fig.add_subplot(2, 2, panelnum, projection=proj)
            lat_lim = 30 if pole == 'N' else -30
            if pole == 'N':
                ax.set_extent([-180, 180, lat_lim, 90], crs=ccrs.PlateCarree())
            else:
                ax.set_extent([-180, 180, -90, lat_lim], crs=ccrs.PlateCarree())
            ax.coastlines(resolution='110m', linewidth=0.5)
            ax.gridlines(linewidth=0.3, alpha=0.4)
            cf = ax.contourf(lons2d, lats2d, data, levels=lvls, cmap=cmap,
                             extend='both', transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(2, 2, panelnum)
            mask = LATS >= 30 if pole == 'N' else LATS <= -30
            cf = ax.contourf(LONS, LATS[mask], data[mask, :],
                             levels=lvls, cmap=cmap, extend='both')
        ax.set_title(panel_title, fontsize=10)
        fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                     fraction=0.05, shrink=0.8, label=units)

    outpath = os.path.join(outdir, f'{mon_name.lower()}{yy}-snow4-25prod.gif')
    save_gif(fig, outpath, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(yy, mm):
    """Generate all images for a given two-digit year and two-digit month."""
    mm_str = f'{int(mm):02d}'
    yy_str = f'{int(yy):02d}'

    for sat in SATELLITES:
        sat_dir = os.path.join(IMG_DIR, sat)
        os.makedirs(sat_dir, exist_ok=True)

        print(f'\n=== Generating images for {sat.upper()}, 20{yy_str}-{mm_str} ===')

        gen_pr1(sat, yy_str, mm_str, sat_dir)
        gen_snw(sat, yy_str, mm_str, sat_dir)
        gen_wvp(sat, yy_str, mm_str, sat_dir)
        gen_lwp(sat, yy_str, mm_str, sat_dir)

        if sat == 'f17':
            # 4-panel snow product only for primary satellite
            gen_snow4(sat, yy_str, mm_str, sat_dir)

    print('\nImage generation complete.')


def main():
    parser = argparse.ArgumentParser(description='Generate monthly SSMIS imagery')
    parser.add_argument('--yy', type=str, help='Two-digit year (e.g. 12 for 2012)')
    parser.add_argument('--mm', type=str, help='Two-digit month (e.g. 06 for June)')
    args = parser.parse_args()

    if args.yy and args.mm:
        yy = args.yy.zfill(2)
        mm = args.mm.zfill(2)
    else:
        # Auto-detect previous month
        today = datetime.date.today()
        first_of_this_month = today.replace(day=1)
        prev = first_of_this_month - datetime.timedelta(days=1)
        yy = str(prev.year % 100).zfill(2)
        mm = str(prev.month).zfill(2)
        print(f'Auto-detected previous month: 20{yy}-{mm}')

    run(yy, mm)


if __name__ == '__main__':
    main()
