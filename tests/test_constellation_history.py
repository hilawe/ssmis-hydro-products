"""The chain histories written into the netCDF `constellation` attribute match the record's satellite membership.

The months below were found from the data, not taken from any stated history. On 27 October 2026 every month of the
operational combined records (25deg-bin for the morning chain, f10-bin for the late-morning chain) was matched byte
for byte to one satellite's monthly product, and the archive's own values changed satellite in the same months. The
stated histories had F-15 from January 2002, F-08 from July 1987 and F-18 from January 2010, and the data give
January 2001, August 1987 and February 2010. They are written here independently of satellite_config.py, so the test
can disagree with it.
"""
import re

import satellite_config as sc

ENTRY = re.compile(r"(?:SSM/I|SSMIS) (F-\d\d): ([A-Z][a-z]+ \d{4})-([A-Z][a-z]+ \d{4}|present)")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

EXPECTED = {
    "f17": [("F-08", "August 1987", "December 1991"), ("F-11", "January 1992", "April 1995"),
            ("F-13", "May 1995", "December 2008"), ("F-17", "January 2009", "present")],
    "f16": [("F-10", "January 1992", "September 1997"), ("F-14", "October 1997", "December 2000"),
            ("F-15", "January 2001", "June 2006"), ("F-16", "July 2006", "present")],
    "f18": [("F-18", "February 2010", "present")],
}


def entries(history):
    """(satellite, first month, last month or 'present') for each entry, refusing any unparsed text."""
    found = ENTRY.findall(history)
    rebuilt = "; ".join(f"{'SSMIS' if int(s[2:]) >= 16 else 'SSM/I'} {s}: {a}-{b}" for s, a, b in found)
    assert rebuilt == history.strip().rstrip(";"), f"history has text the parser did not read: {history!r}"
    return found


def month_index(text):
    name, year = text.split()
    return int(year) * 12 + MONTHS.index(name)


def test_each_chain_history_matches_the_membership_found_in_the_data():
    for sat, expected in EXPECTED.items():
        assert entries(sc.get_constellation_history(sat)) == expected, sat


def test_each_chain_history_runs_month_to_month_without_gap_or_overlap():
    for sat in EXPECTED:
        found = entries(sc.get_constellation_history(sat))
        for (_, _, end), (_, start, _) in zip(found, found[1:]):
            assert month_index(start) == month_index(end) + 1, (sat, end, start)
        assert found[-1][2] == "present", sat
