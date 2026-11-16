"""Checks of the fixed conversion maps as a file: their shape, their days and the digest of their values.

The file read is the public copy beside this test (route_a_maps.json) or, where that is absent, the fitted maps it is
made from (fit_record/maps.json). The two hold the same maps and differ only in how they list the fit's inputs. Either
must hold the 288 maps of the 72 fitting days, with 101 knots for each of the seven channels on each side and AMSR2
knots that strictly increase, and the values must be at the digest fixed here, so a file with other knots, days or
sample counts does not pass as this conversion's maps.
"""
import hashlib
import json
import os

import pytest

pytest.importorskip("h5py")             # the reader's one third-party package beyond numpy: without it these tests skip

import route_a_calendar as Q  # noqa: E402
import route_a_fit as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = next(p for p in (os.path.join(HERE, "route_a_maps.json"), os.path.join(HERE, "fit_record", "maps.json"))
            if os.path.isfile(p))
# SHA-256 of the knots' percentiles, the channels, the days and the 288 maps, written as JSON with sorted keys and no
# spaces. It was taken from the fitted maps at e51c1e0b...80715 on 15 November 2026.
VALUES_SHA256 = "30dbc22ef9db4a7b88a0186ff8b5ce2705ee3ffa143ede38585fdeb3e98a37dd"
FITTED_SHA256 = "e51c1e0bd5cad9f4005b75f8b60980b8f4c094103d67e2ce049f4ab6fdc80715"


def load():
    with open(MAPS, "rb") as fh:
        raw = fh.read()
    return raw, json.loads(raw)


def values_digest(m):
    body = {k: m[k] for k in ("knots_percent", "channels", "days", "maps")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_the_maps_are_the_288_of_the_72_fitting_days():
    _, m = load()
    assert m["days"] == Q.FIT_DAYS and len(m["days"]) == 72
    assert (m["days"][0], m["days"][-1]) == ("2022-04-05", "2024-03-25")
    assert m["knots_percent"] == F.KNOTS and len(F.KNOTS) == 101
    assert m["channels"] == ["19V", "19H", "22V", "37V", "37H", "91V", "91H"]
    want = {"%02d|%s|%s|%d" % (month, pas, cls, zone) for month in range(1, 13) for pas in ("asc", "dsc")
            for cls in ("ocean", "land") for zone in range(6)}
    assert set(m["maps"]) == want and len(want) == 288


def test_every_map_holds_101_knots_a_channel_on_each_side_and_its_sample_counts():
    _, m = load()
    for key, one in m["maps"].items():
        assert sorted(one) == ["amsr2", "f16", "n_amsr2", "n_f16"], key
        assert one["n_amsr2"] >= 101 and one["n_f16"] >= 101, key
        for side in ("amsr2", "f16"):
            assert len(one[side]) == 7 and all(len(knots) == 101 for knots in one[side]), key
        for knots in one["amsr2"]:
            assert all(b > a for a, b in zip(knots, knots[1:])), key          # strictly increasing, as the fit requires
        for knots in one["f16"]:
            assert all(b >= a for a, b in zip(knots, knots[1:])), key


def test_the_values_are_at_the_fixed_digest():
    raw, m = load()
    assert values_digest(m) == VALUES_SHA256
    if os.path.basename(MAPS) == "route_a_maps.json":       # the public copy names the fitted maps it was made from
        assert m["source_sha256"] == FITTED_SHA256
        assert all("/" not in name and "\\" not in name for name in m["inputs_sha256"]) and len(m["inputs_sha256"]) == 217
    else:
        assert hashlib.sha256(raw).hexdigest() == FITTED_SHA256


def test_a_changed_knot_a_changed_day_or_a_changed_count_changes_the_digest():
    _, m = load()
    key = sorted(m["maps"])[0]
    for change in ("knot", "day", "count"):
        other = json.loads(json.dumps(m))
        if change == "knot":
            other["maps"][key]["f16"][0][50] += 0.01
        elif change == "day":
            other["days"][0] = "2022-04-06"
        else:
            other["maps"][key]["n_amsr2"] += 1
        assert values_digest(other) != VALUES_SHA256, change
