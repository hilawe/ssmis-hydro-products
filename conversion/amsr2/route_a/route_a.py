"""Route A reader and gridder: CSU's AMSR2 near-real-time brightness temperatures as daily float composites.

The gridding is that of the adapter in monthly/wsfm_adapter.py, with nearest-node mapping, a last-write merge on
observed cells, and the adapter's encoding applied once after conversion.

usage: python route_a.py composite --day YYYY-MM-DD --files F [F ...] --out FILE.npz
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "monthly")))
import wsfm_adapter as A  # noqa: E402

LOW = [("fcdr_tb18v", "quality_tb18", 0), ("fcdr_tb18h", "quality_tb18", 1), ("fcdr_tb23v", "quality_tb23", 0),
       ("fcdr_tb36v", "quality_tb36", 0), ("fcdr_tb36h", "quality_tb36", 1)]
HIGH = [("fcdr_tb89v_ascan", "quality_tb89_ascan", 0), ("fcdr_tb89h_ascan", "quality_tb89_ascan", 1)]
FILL = -9999.9
VALID_MIN, VALID_MAX = 50.0, 350.0
EPOCH = datetime.datetime(1987, 1, 1)
ORBIT_MINUTES = 99.0          # nominal AMSR2 orbit period, for the full-day test
NAME = re.compile(r"^CSU_AMSR2_ICDR_V02R00_GCOMW1_D(\d{8})_S(\d{4})_E(\d{4})_R(\d+)\.nc$")


class Refused(ValueError):
    """A file that may not be read as data. Fatal for the day."""


def parse_name(name):
    m = NAME.match(os.path.basename(name))
    if not m:
        raise Refused(f"{name}: not a CSU AMSR2 near-real-time file name")
    d, s, e, r = m.groups()
    return {"date": d, "start": s, "end": e, "rev": int(r)}


def crosses_midnight(name):
    p = parse_name(name)
    return p["end"] < p["start"]


def _field(f, name):
    a = np.asarray(f[name][...], dtype=np.float64)
    return a


def read_amsr2_granule(path):
    """One granule as the adapter's swath dict: seven channels on the low-resolution footprints, the A-scan's footprint
    2k+1 paired with low-resolution footprint k, invalid footprints NaN in all seven."""
    with h5py.File(path, "r") as f:
        cols = []
        valid = None
        for var, qvar, q in LOW + HIGH:
            v = _field(f, var)
            qf = np.asarray(f[qvar][..., q])
            if var in [h[0] for h in HIGH]:
                v, qf = v[:, 1::2], qf[:, 1::2]
            ok = np.isfinite(v) & (v != np.float32(FILL)) & (v >= VALID_MIN) & (v <= VALID_MAX) & (qf == 0)
            valid = ok if valid is None else valid & ok
            cols.append(v)
        if len({c.shape for c in cols}) != 1:
            raise Refused(f"{os.path.basename(path)}: channel shapes differ after pairing: {[c.shape for c in cols]}")
        lat = np.asarray(f["lat_tb36"][...], dtype=np.float32)
        lon = np.asarray(f["lon_tb36"][...], dtype=np.float32)
        st = _field(f, "scan_time")
    ta = np.stack(cols, axis=-1).astype(np.float32)
    ta[~valid] = np.nan
    for arr in (lat, lon):
        arr[~np.isfinite(arr) | (arr <= FILL + 1)] = np.nan
    lon = A.normalize_longitude(lon, path)
    nscan = ta.shape[0]
    if nscan < 2:
        raise A.UnclassifiableGranule(f"{os.path.basename(path)}: {nscan} scan(s)")
    scan_day = np.full(nscan, -1, dtype=np.int64)          # a scan without a time is routed to no day
    for k, s in enumerate(st):
        if np.isfinite(s) and s > FILL + 1:
            t = EPOCH + datetime.timedelta(seconds=float(s))
            scan_day[k] = t.year * 1000 + t.timetuple().tm_yday
    return {"lat": lat, "lon": lon, "ta": ta, "ascending": A._is_ascending(lat), "scan_day": scan_day, "scan_time": st,
            "meta": {"file": os.path.basename(path), "scans": nscan, "valid_footprints": int(valid.sum()),
                     "footprints": int(valid.size), "scans_without_time": int((scan_day < 0).sum())}}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_composite(npz, record, day, min_minutes=None):
    """A published composite's identity, or a refusal. The record must name the day, a finite minute count within the
    day and the archive's SHA-256, the archive must hash to it and hold float32 asc and dsc grids of the daily shape,
    and with min_minutes the day must hold at least that many observed minutes."""
    try:
        rec = json.load(open(record))
        minutes = rec["observed_minutes"]
        want = rec["npz_sha256"]
    except (OSError, ValueError, KeyError) as exc:
        raise Refused(f"{record}: no verifiable composite record ({exc})")
    if rec.get("day") != day or not isinstance(minutes, int) or not 0 <= minutes <= 1440:
        raise Refused(f"{record}: the record does not name {day} with a minute count within the day")
    try:
        got = sha256_file(npz)
    except OSError as exc:
        raise Refused(f"{npz}: unreadable ({exc})")
    if got != want:                                      # the digest first, so a damaged archive is never opened
        raise Refused(f"{npz}: SHA-256 {got} is not the record's")
    try:
        with np.load(npz) as z:
            ok = all(z[k].shape == (A.NROW, A.NCOL, A.N_CHAN) and z[k].dtype == np.float32 for k in ("asc", "dsc"))
    except Exception as exc:                             # any unreadable archive, whatever the reader raises
        raise Refused(f"{npz}: unreadable ({exc})")
    if not ok:
        raise Refused(f"{npz}: its grids are not the daily float32 grids")
    if min_minutes is not None and minutes < min_minutes:
        raise Refused(f"{day}: {minutes} observed minutes, fewer than {min_minutes}")
    return {"npz_sha256": got, "json_sha256": sha256_file(record), "observed_minutes": minutes}


def grid_float(swath, ascending_only):
    """The adapter's grid_swath_to_binary in floats."""
    grid = np.full((A.NROW, A.NCOL, A.N_CHAN), np.nan, dtype=np.float32)
    touched = np.zeros((A.NROW, A.NCOL), dtype=bool)
    scan_mask = swath["ascending"] if ascending_only else ~swath["ascending"]
    lat = swath["lat"][scan_mask].ravel()
    lon = swath["lon"][scan_mask].ravel()
    ta = swath["ta"][scan_mask].reshape(-1, A.N_CHAN)
    ok = ~(np.isnan(lat) | np.isnan(lon))
    lat, lon, ta = lat[ok], lon[ok], ta[ok]
    if lat.size == 0:
        return grid, touched
    row, col = A._lat_to_row(lat), A._lon_to_col(lon)
    for c in range(A.N_CHAN):
        grid[row, col, c] = ta[:, c]
    touched[row, col] = True
    return grid, touched


def composite_day(paths, yyyy, jday):
    asc = np.full((A.NROW, A.NCOL, A.N_CHAN), np.nan, dtype=np.float32)
    dsc = np.full_like(asc, np.nan)
    info = {"files": [], "gaps": [], "scans_kept": 0}
    minutes = set()
    for p in paths:
        try:
            sw = read_amsr2_granule(p)
        except A.UnclassifiableGranule as exc:
            info["gaps"].append(str(exc))
            continue
        on_day = sw["scan_day"] == yyyy * 1000 + jday
        minutes.update(np.floor(np.mod(sw["scan_time"][on_day], 86400.0) / 60.0).astype(int).tolist())
        sw = A.select_scans_on_day(sw, yyyy, jday)
        info["files"].append(dict(sw["meta"], scans_on_day=sw["n_scans_kept"]))
        info["scans_kept"] += sw["n_scans_kept"]
        if sw["n_scans_kept"] == 0:
            continue
        for grid, flag in ((asc, True), (dsc, False)):
            part, touched = grid_float(sw, flag)
            grid[touched] = part[touched]
    # The adapter judges hemisphere balance on a day of at least FULL_DAY_MIN_ORBITS orbits. The count here is the
    # distinct minutes of the day holding scans, in orbit lengths, so a
    # preceding day's short tail, a skipped granule or a file without scans on the day adds nothing.
    info["observed_minutes"] = len(minutes)
    info["orbits_on_day"] = len(minutes) / ORBIT_MINUTES
    for label, grid in (("as", asc), ("ds", dsc)):
        A.hemisphere_balance(A._encode_ta(grid), f"{label}{yyyy % 100:02d}{jday:03d}",
                             int(len(minutes) // ORBIT_MINUTES))
    return asc, dsc, info


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("composite")
    c.add_argument("--day", required=True)
    c.add_argument("--files", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    day = datetime.date.fromisoformat(a.day)
    asc, dsc, info = composite_day(sorted(a.files), day.year, day.timetuple().tm_yday)
    if info["scans_kept"] == 0:
        raise SystemExit(f"refused: no scans on {a.day}")
    tmp = a.out + ".part.npz"
    np.savez_compressed(tmp, asc=asc, dsc=dsc)
    os.replace(tmp, a.out)
    info.update(day=a.day, out=os.path.basename(a.out), npz_sha256=sha256_file(a.out),
                valid_cells={k: int(np.isfinite(g[..., 0]).sum()) for k, g in (("as", asc), ("ds", dsc))})
    with open(a.out.replace(".npz", ".json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"composite {a.day}: {len(info['files'])} files, {info['scans_kept']} scans, valid cells {info['valid_cells']}")


if __name__ == "__main__":
    main()
