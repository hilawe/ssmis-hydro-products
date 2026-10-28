"""The WSF-M entry carries the morning chain's history unchanged up to F-17's transition date.

tests/test_constellation_history.py checks the F-17 history against the membership found in the data. This keeps the
WSF-M entry, which repeats that history before its placeholder transition date, from drifting away from it.
"""
import satellite_config as sc


def test_wsfm_history_begins_with_the_morning_chain_history_through_f17():
    f17 = sc.get_constellation_history("f17")
    assert f17.endswith("present"), f17
    prefix = f17[: -len("present")]
    assert sc.get_constellation_history("wsfm").startswith(prefix), sc.get_constellation_history("wsfm")
