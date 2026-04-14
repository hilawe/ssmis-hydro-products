#!/usr/bin/env python3
"""
satellite_config.py

PURPOSE
    Single-source configuration for the satellites processed by the SSMI(S)
    hydrological product pipeline.  Every science module (climalg_ssmis.py,
    gpcp_processing.py, generate_netcdf.py) imports from here so that adding a
    new satellite or transitioning from one spacecraft to another requires
    changing only this file.

ARCHITECTURE
    The pipeline has two distinct "constellation chains" defined by the
    satellite's equatorial crossing time:

    LATE  (morning crossing ~06h LST): F-08 -> F-11 -> F-13 -> F-17.
          Their combined multi-year record feeds the morning-chain archive.

    EARLY (late-morning crossing ~09-10h LST): F-10 -> F-14 -> F-15 -> F-16.
          Their record feeds the late-morning-chain archive.

SYNOPSIS
    from satellite_config import (
        SATELLITE_REGISTRY,
        LATE_CONSTELLATION_SAT, EARLY_CONSTELLATION_SAT,
        get_active_satellites, get_sat, SatelliteConfig,
    )

USES
    None (no science imports; pure configuration data)

NOTES
    - 'binary_tdr' input_format: IDL-decoded 1/3-degree binary files,
       shape (540x1080x7) uint8 with offset +70 K; fill = byte value 32
       (-> Ta = 102.0 K), named  as{yy}{jday}.{sat}  /  ds{yy}{jday}.{sat}.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SatelliteConfig:
    """
    All science-relevant properties for one DMSP spacecraft.

    Parameters
    ----------
    name : str
        Short identifier (e.g. 'f17').  Used in filenames and output directory
        names throughout the pipeline.
    constellation : str
        'late'  -> morning equatorial crossing (~06h LST).
        'early' -> late-morning equatorial crossing (~09-10h LST).
    start_year : int or None
        Calendar year of the satellite's first full month of data.
        None means "not yet determined".
    input_format : str
        'binary_tdr' -> IDL-decoded 1/3-degree uint8 binary files.
    active : bool
        True  -> this satellite is currently being processed.
        False -> placeholder / retired / not yet launched.
    constellation_history : str
        Human-readable description of the satellite's chain history, used in
        NetCDF global attributes.
    """
    name: str
    constellation: str              # 'late' or 'early'
    start_year: Optional[int]       # None = TBD
    input_format: str               # 'binary_tdr'
    active: bool = False
    constellation_history: str = ''


# ---------------------------------------------------------------------------
# SATELLITE REGISTRY
# ---------------------------------------------------------------------------
# Each key is the short satellite identifier used throughout the pipeline.
# ---------------------------------------------------------------------------

SATELLITE_REGISTRY: dict = {

    # F-16 (DMSP Block 5D-3, SSMIS), late-morning constellation.
    'f16': SatelliteConfig(
        name='f16',
        constellation='early',
        start_year=1992,            # Record combines F-10(1992)->F-14->F-15->F-16
        input_format='binary_tdr',
        active=True,
        constellation_history=(
            'SSM/I F-10: January 1992-September 1997; '
            'SSM/I F-14: October 1997-December 2001; '
            'SSM/I F-15: January 2002-June 2006; '
            'SSMIS F-16: July 2006-present'
        ),
    ),

    # F-17 (DMSP Block 5D-3, SSMIS), morning constellation.
    'f17': SatelliteConfig(
        name='f17',
        constellation='late',
        start_year=1987,            # Record combines F-08(1987)->F-11->F-13->F-17
        input_format='binary_tdr',
        active=True,
        constellation_history=(
            'SSM/I F-08: July 1987-December 1991; '
            'SSM/I F-11: January 1992-April 1995; '
            'SSM/I F-13: May 1995-December 2008; '
            'SSMIS F-17: January 2009-present'
        ),
    ),

    # F-18 (DMSP Block 5D-3, SSMIS), late-morning constellation.
    'f18': SatelliteConfig(
        name='f18',
        constellation='early',
        start_year=2010,
        input_format='binary_tdr',
        active=True,
        constellation_history='SSMIS F-18: January 2010-present',
    ),
}


# ---------------------------------------------------------------------------
# CONSTELLATION ASSIGNMENTS
# ---------------------------------------------------------------------------
# These two variables are the primary control for which operational satellite
# represents each chain. Everything downstream reads them.
# ---------------------------------------------------------------------------

LATE_CONSTELLATION_SAT: str  = 'f17'
EARLY_CONSTELLATION_SAT: str = 'f16'


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_sat(name: str) -> SatelliteConfig:
    """Return the SatelliteConfig for the given satellite name."""
    if name not in SATELLITE_REGISTRY:
        raise KeyError(f"Unknown satellite '{name}'. "
                       f"Known satellites: {list(SATELLITE_REGISTRY)}")
    return SATELLITE_REGISTRY[name]


def get_active_satellites() -> list:
    """Return list of currently active satellite names (active=True)."""
    return [name for name, cfg in SATELLITE_REGISTRY.items() if cfg.active]


def get_start_year(name: str) -> int:
    """Return start_year for the satellite; raises ValueError if not set."""
    cfg = get_sat(name)
    if cfg.start_year is None:
        raise ValueError(
            f"Satellite '{name}' has no start_year set in satellite_config.py. "
            f"Please set it before processing."
        )
    return cfg.start_year


def get_constellation_history(name: str) -> str:
    """Return the constellation history string for NetCDF metadata."""
    return get_sat(name).constellation_history


def late_start_year() -> int:
    """Convenience: start year for the current late-constellation primary."""
    return get_start_year(LATE_CONSTELLATION_SAT)


def early_start_year() -> int:
    """Convenience: start year for the current early-constellation primary."""
    return get_start_year(EARLY_CONSTELLATION_SAT)
