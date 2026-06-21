"""Lock the product-version token format.

This is the single source of truth that the v01 to v02 cutover flips. These
tests have no third-party dependencies, so they run anywhere (including a base
interpreter with no numpy). They guard the filename token, the NetCDF
product_version attribute format, and the coupling between the two.
"""
import product_version as pv


def test_token_is_versioned_string():
    # Form must stay 'v' + digits (e.g. v01, v02), never empty or freeform.
    assert pv.PRODUCT_VERSION.startswith('v')
    assert pv.PRODUCT_VERSION[1:].isdigit()
    assert len(pv.PRODUCT_VERSION) >= 3  # 'v' + at least two digits


def test_attr_is_token_plus_revision():
    # product_version attribute is the token plus an r-revision suffix, and the
    # two stay coupled so a cutover that changes the token also moves the attr.
    assert pv.PRODUCT_VERSION_ATTR.startswith(pv.PRODUCT_VERSION)
    suffix = pv.PRODUCT_VERSION_ATTR[len(pv.PRODUCT_VERSION):]
    assert suffix.startswith('r') and suffix[1:].isdigit()


def test_current_value_is_v01():
    # Guard against an accidental premature flip. When the NCEI cutover happens
    # this test is updated deliberately, in lockstep with the flip.
    assert pv.PRODUCT_VERSION == 'v01'
    assert pv.PRODUCT_VERSION_ATTR == 'v01r00'


def test_filename_prefix_construction():
    # The exact shape downstream code builds, e.g. mw-hydro_v01_2.5-deg_cfr_late_
    prefix = f'mw-hydro_{pv.PRODUCT_VERSION}_2.5-deg_cfr_late_'
    assert prefix == 'mw-hydro_v01_2.5-deg_cfr_late_'
