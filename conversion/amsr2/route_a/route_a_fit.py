"""Route A relation: per-channel quantile maps fitted on the 72 fitting days, applied once in floats and encoded once.

fit    For each SSMIS channel, surface class (ocean where NLNDSEA.TAG is 0, land where it is 1), latitude zone (six bands
       of 30 degrees by the cell-center latitude, south edge inclusive), pass and calendar month: the AMSR2 brightness
       temperatures of the month's fitting days on cells valid in all seven channels, and F-16's antenna temperatures
       (the byte plus 70.5 K) on cells above byte 32 in all seven, each side counted on its own. The knots are the
       0.1, 1, 2, ..., 98, 99 and 99.9 percentiles of each side (numpy's linear percentile). A map refuses with fewer
       samples on either side than it has knots, or with AMSR2 knots that do not strictly increase. Records the knots,
       the sample counts and every input's SHA-256.
apply  Maps one day's float composite with a committed map file, linear between knots and beyond the end knots by the
       end knot's offset, and writes {as,ds}{yy}{jday}.amsr2 with the adapter's encoding.

usage:
  python route_a_fit.py fit --composites DIR --days D [D ...] --f16-grids DIR --tag NLNDSEA.TAG --out maps.json
  python route_a_fit.py apply --maps maps.json --composite DAY.npz --day YYYY-MM-DD --tag NLNDSEA.TAG --out-dir DIR
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np

import route_a as R

A = R.A
TARGET_OFFSET = 70.5
KNOTS = [0.1] + list(range(1, 100)) + [99.9]
CLASSES = {"ocean": 0, "land": 1}
PASSES = {"asc": "as", "dsc": "ds"}
CHANNELS = ["19V", "19H", "22V", "37V", "37H", "91V", "91H"]
N_ZONES = 6


Refused = R.Refused                           # one refusal class for the reader, the fit and the application


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def zones():
    """Zone index of each grid row: 0 for 90 S to 60 S, ..., 5 for 60 N to 90 N, by the row's center latitude."""
    lat = (269 - np.arange(A.NROW)) / 3.0
    assert np.array_equal(A._lat_to_row(lat), np.arange(A.NROW))
    return np.minimum(((lat + 90.0) // 30.0).astype(int), N_ZONES - 1)


def load_tag(path):
    tag = np.fromfile(path, dtype=np.int8).reshape(A.NROW, A.NCOL)
    if not np.isin(tag, (0, 1)).all():
        raise Refused(f"{path}: land/sea tag values other than 0 and 1")
    return tag


def f16_path(grids, prefix, day):
    return os.path.join(grids, f"{prefix}{day.year % 100:02d}{day.timetuple().tm_yday:03d}.f16")


def map_key(month, pass_key, cls, zone):
    return f"{month:02d}|{pass_key}|{cls}|{zone}"


def fit(composites, days, grids, tag_path, out):
    """The fit's only entry point. It refuses before reading any scientific input unless the days are exactly the
    declared 72 and every fitting composite verifies against its record. The fit runs once: an
    existing output or start marker refuses, and the marker, created atomically, stays after success or interruption
    until it is removed by hand."""
    import route_a_calendar as Q
    for p in (out, out + ".started"):                   # an earlier output or attempt refuses before any input
        if os.path.lexists(p):
            raise FileExistsError(p)
    if list(days) != Q.FIT_DAYS:
        raise Refused("the fitting days are not exactly the declared calendar")
    for s in days:
        try:
            R.verify_composite(os.path.join(composites, f"{s}.npz"), os.path.join(composites, f"{s}.json"), s)
        except R.Refused as exc:
            raise Refused(f"fitting composite {s}: {exc}")
    os.mkdir(out + ".started")
    return fit_kernel(composites, days, grids, tag_path, out)


def fit_kernel(composites, days, grids, tag_path, out):
    """The maps from already verified fitting composites. Called only through fit, except by the numerical tests."""
    tag = load_tag(tag_path)
    zone = np.broadcast_to(zones()[:, None], (A.NROW, A.NCOL))
    inputs = {tag_path: sha(tag_path)}
    by_month = {}
    for s in days:
        by_month.setdefault(datetime.date.fromisoformat(s).month, []).append(s)
    rec = {"protocol": "route_a_protocol.md", "form": "per-channel quantile map, AMSR2 brightness temperature to F-16 "
           "antenna temperature (byte + 70.5 K), per channel, surface class, 30-degree zone, pass and calendar month",
           "knots_percent": KNOTS, "channels": CHANNELS, "days": list(days), "maps": {}, "inputs_sha256": inputs}
    for month, mdays in sorted(by_month.items()):
        side = {k: {"amsr2": [], "f16": []} for k in PASSES}
        cls_zone = {k: {"amsr2": [], "f16": []} for k in PASSES}
        for s in mdays:
            day = datetime.date.fromisoformat(s)
            comp = os.path.join(composites, f"{s}.npz")
            inputs[comp] = sha(comp)
            z = np.load(comp)
            for key, prefix in PASSES.items():
                p = f16_path(grids, prefix, day)
                inputs[p] = sha(p)
                f16 = np.fromfile(p, dtype=np.uint8).reshape(A.NROW, A.NCOL, A.N_CHAN)
                ok_a = np.isfinite(z[key]).all(axis=-1)
                ok_f = (f16 > A.FILL_BYTE).all(axis=-1)
                side[key]["amsr2"].append(z[key][ok_a].astype(np.float64))
                cls_zone[key]["amsr2"].append(np.stack([tag[ok_a], zone[ok_a]], axis=-1))
                side[key]["f16"].append(f16[ok_f].astype(np.float64) + TARGET_OFFSET)
                cls_zone[key]["f16"].append(np.stack([tag[ok_f], zone[ok_f]], axis=-1))
        for key in PASSES:
            vals = {k: np.concatenate(side[key][k]) for k in ("amsr2", "f16")}
            labels = {k: np.concatenate(cls_zone[key][k]) for k in ("amsr2", "f16")}
            for cname, cv in CLASSES.items():
                for zi in range(N_ZONES):
                    sel = {k: vals[k][(labels[k][:, 0] == cv) & (labels[k][:, 1] == zi)] for k in vals}
                    n = {k: len(v) for k, v in sel.items()}
                    name = map_key(month, key, cname, zi)
                    if min(n.values()) < len(KNOTS):
                        raise Refused(f"map {name}: {n} samples, fewer than its {len(KNOTS)} knots")
                    q = {k: np.percentile(v, KNOTS, axis=0).T for k, v in sel.items()}     # (7, knots)
                    if not (np.diff(q["amsr2"], axis=1) > 0).all():
                        raise Refused(f"map {name}: AMSR2 knots do not strictly increase")
                    rec["maps"][name] = {"amsr2": q["amsr2"].tolist(), "f16": q["f16"].tolist(),
                                         "n_amsr2": n["amsr2"], "n_f16": n["f16"]}
        print(f"month {month:02d}: {len(mdays)} days, maps made")
    with open(out, "x") as f:
        json.dump(rec, f)
    return rec


def map_values(x, a, f):
    """One channel's map: linear between knots, and beyond the end knots the end knot's offset."""
    y = np.interp(x, a, f)
    lo, hi = x < a[0], x > a[-1]
    y[lo] = x[lo] + (f[0] - a[0])
    y[hi] = x[hi] + (f[-1] - a[-1])
    return y


def convert(grid, tag, maps, month, pass_key):
    """Map a float composite (NaN where any channel is missing) and return the float result."""
    out = np.full(grid.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(grid).all(axis=-1)
    zone = np.broadcast_to(zones()[:, None], (A.NROW, A.NCOL))
    for cname, cv in CLASSES.items():
        for zi in range(N_ZONES):
            m = ok & (tag == cv) & (zone == zi)
            if not m.any():
                continue
            mp = maps["maps"][map_key(month, pass_key, cname, zi)]
            a, f = np.asarray(mp["amsr2"]), np.asarray(mp["f16"])
            for c in range(A.N_CHAN):
                out[m, c] = map_values(grid[m, c].astype(np.float64), a[c], f[c])
    return out


PLANNED_MIN_MINUTES = 990                     # ten orbit lengths of 99 minutes, the minimum for a planned application


def apply(maps_path, composite, day, tag_path, out_dir, expect_npz_sha256=None, expect_json_sha256=None):
    """Map one verified composite. A planned application names both digests the preflight recorded, and the archive
    and its record must still be those, with the planned minimum of observed minutes, before anything is converted."""
    planned = expect_npz_sha256 is not None or expect_json_sha256 is not None
    ident = R.verify_composite(composite, composite[:-len(".npz")] + ".json", day,
                               min_minutes=PLANNED_MIN_MINUTES if planned else None)
    if planned and (ident["npz_sha256"] != expect_npz_sha256 or ident["json_sha256"] != expect_json_sha256):
        raise Refused(f"{composite}: the archive or its record changed since the preflight")
    maps = json.load(open(maps_path))
    tag = load_tag(tag_path)
    z = np.load(composite)
    d = datetime.date.fromisoformat(day)
    os.makedirs(out_dir, exist_ok=True)
    for key, prefix in PASSES.items():
        enc = A._encode_ta(convert(z[key], tag, maps, d.month, key).astype(np.float32))
        path = os.path.join(out_dir, f"{prefix}{d.year % 100:02d}{d.timetuple().tm_yday:03d}.amsr2")
        A.hemisphere_balance(enc, os.path.basename(path), int(ident["observed_minutes"] // R.ORBIT_MINUTES))
        enc.tofile(path)
    print(f"applied {os.path.basename(maps_path)} to {day}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--composites", required=True)
    f.add_argument("--days", nargs="+", required=True)
    f.add_argument("--f16-grids", required=True)
    f.add_argument("--tag", required=True)
    f.add_argument("--out", required=True)
    a_ = sub.add_parser("apply")
    for n in ("--maps", "--composite", "--day", "--tag", "--out-dir"):
        a_.add_argument(n, required=True)
    a_.add_argument("--expect-npz-sha256")
    a_.add_argument("--expect-json-sha256")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.composites, a.days, a.f16_grids, a.tag, a.out)
    else:
        apply(a.maps, a.composite, a.day, a.tag, a.out_dir, a.expect_npz_sha256, a.expect_json_sha256)


if __name__ == "__main__":
    main()
