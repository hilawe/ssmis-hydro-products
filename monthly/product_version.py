"""Single source of truth for the product version token.

Every output filename (`mw-hydro_<PRODUCT_VERSION>_...`) and the NetCDF
`product_version` global attribute derive from the values below. This exists
so the eventual v01 to v02 cutover for the NCEI archive is a one-line change
here (plus `hydroPRODUCT_VERSION` in config/HydroCode.properties for the shell
archive scripts), instead of editing ~15 scattered string literals across the
Python and shell pipeline. See the project documentation

Kept deliberately dependency-free (no imports) so any pipeline module can
import it without pulling in heavy packages such as netCDF4.

CUTOVER: change PRODUCT_VERSION to 'v02' below. PRODUCT_VERSION_ATTR then
follows automatically as 'v02r00'. If NCEI assigns a different revision suffix
(for example r01), override PRODUCT_VERSION_ATTR explicitly at that time.
"""

# Filename token, e.g. mw-hydro_v01_2.5-deg_cfr_late_201205.nc
PRODUCT_VERSION = 'v01'

# NetCDF global attribute product_version, version + revision suffix.
PRODUCT_VERSION_ATTR = PRODUCT_VERSION + 'r00'
