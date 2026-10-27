"""A satellite may carry its own scattering-index table, and only that satellite reads it.

The other coefficient tables and the land/sea tag are always the standard ones. An explicit coefficient directory
passed to the climate algorithm overrides any satellite-specific table, so experiments keep full control. The tables
here are synthetic and written to a temporary directory, so the test needs none of the operational coefficient files.
"""
import os

import numpy as np

import climalg_ssmis as cl
import satellite_config as sc


def write_tables(d, ocean_shift=0.0, si_name="lut_add_si"):
    """Standard-shaped synthetic tables: 161-row ocean and land tables, a 149-row index table (key, ocean, land)."""
    for name in ("lut_add_ocean", "lut_add_land"):
        with open(os.path.join(d, name), "w") as f:
            for k in range(161):
                f.write(f"{140 + k:6.2f}" + "".join(f"{0.01 * k + c:7.2f}" for c in range(7)) + "\n")
    with open(os.path.join(d, si_name), "w") as f:
        for k in range(1, 150):
            f.write(f"{k:5.2f} {0.05 * k + ocean_shift:5.2f} {-0.02 * k:5.2f}\n")
    return os.path.join(d, si_name)


def test_satellites_without_their_own_table_read_the_standard_one():
    for name in ("f16", "f17", "f18"):
        assert sc.get_lut_add_si(name) is None
        assert cl.resolve_si_path(name, None) == os.path.join(".", "lut_add_si")


def test_a_satellite_with_its_own_table_reads_it_and_an_explicit_directory_wins(tmp_path, monkeypatch):
    write_tables(tmp_path)
    own = write_tables(tmp_path, ocean_shift=1.0, si_name="own_si")
    entry = sc.SatelliteConfig(name="xsat", constellation="late", start_year=2026, input_format="binary_tdr",
                               lut_add_si=own)
    monkeypatch.setitem(sc.SATELLITE_REGISTRY, "xsat", entry)
    assert cl.resolve_si_path("xsat", None) == own
    assert cl.resolve_si_path("xsat", str(tmp_path)) == os.path.join(str(tmp_path), "lut_add_si")
    standard = cl.load_luts(str(tmp_path))[2]
    mine = cl.load_luts(str(tmp_path), si_path=cl.resolve_si_path("xsat", None))[2]
    assert np.allclose(mine[:, 0], standard[:, 0] + 1.0, atol=0.006)
    assert np.array_equal(mine[:, 1], standard[:, 1])


def test_a_relative_table_path_is_read_from_the_configuration_directory_not_the_working_directory(tmp_path,
                                                                                                    monkeypatch):
    entry = sc.SatelliteConfig(name="ysat", constellation="late", start_year=2026, input_format="binary_tdr",
                               lut_add_si="lut_add_si")
    monkeypatch.setitem(sc.SATELLITE_REGISTRY, "ysat", entry)
    monkeypatch.chdir(tmp_path)
    assert sc.get_lut_add_si("ysat") == os.path.join(os.path.dirname(os.path.abspath(sc.__file__)), "lut_add_si")
