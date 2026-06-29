"""
Parse WPC coded surface frontal analysis files (codsus*_hr format).

Format: each 7-digit token is LLLGGGG
  LLL  = latitude  × 10  (degrees North)
  GGGG = longitude × 10  (degrees West)

Usage:
    from parse_coded_sfc import parse_coded_sfc, load_period
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

FRONT_TYPES = {
    "COLD":   "CF",
    "WARM":   "WF",
    "STNRY":  "SF",
    "OCFNT":  "OF",
    "TROF":   "TROF",
}

_COORD_RE = re.compile(r"\b(\d{7})\b")
_VALID_RE = re.compile(r"VALID\s+(\d{2})(\d{2})(\d{2})Z", re.IGNORECASE)
_YEAR_RE  = re.compile(r"\b(20\d{2})\b")


def _decode_coord(token: str) -> tuple[float, float]:
    """Return (lat, lon_east) from a 7-digit WPC token."""
    lat = int(token[:3]) / 10.0
    lon_west = int(token[3:]) / 10.0
    lon_east = -lon_west          # convert West → East (negative = West)
    return lat, lon_east


def parse_coded_sfc(path: str | Path) -> dict:
    """Parse one codsus*_hr file.

    Returns:
        {
            "valid_time": datetime | None,
            "fronts": [
                {"type": "CF"|"WF"|"SF"|"OF"|"TROF",
                 "lats": [...], "lons": [...]}   # lons in degrees East
            ]
        }
    """
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    valid_time = None
    year = None
    for line in lines:
        if year is None:
            ym = _YEAR_RE.search(line)
            if ym:
                year = int(ym.group(1))
        m = _VALID_RE.search(line)
        if m and year:
            mo, dd, hh = int(m.group(1)), int(m.group(2)), int(m.group(3))
            valid_time = datetime(year, mo, dd, hh)
            break

    fronts = []
    current_type = None
    current_coords = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a front type keyword
        parts = line.split()
        if parts[0] in FRONT_TYPES:
            if current_type and current_coords:
                fronts.append(_make_front(current_type, current_coords))
            current_type = FRONT_TYPES[parts[0]]
            tokens = parts[1:]
            current_coords = [_decode_coord(t) for t in tokens if _COORD_RE.fullmatch(t)]
        elif current_type and current_coords is not None:
            # Continuation line
            tokens = parts
            current_coords += [_decode_coord(t) for t in tokens if _COORD_RE.fullmatch(t)]

    if current_type and current_coords:
        fronts.append(_make_front(current_type, current_coords))

    return {"valid_time": valid_time, "fronts": fronts}


def _make_front(ftype: str, coords: list[tuple]) -> dict:
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return {"type": ftype, "lats": lats, "lons": lons}


def load_period(coded_sfc_dir: str | Path) -> list[dict]:
    """Load all codsus*_hr files under coded_sfc_dir (recursive)."""
    results = []
    for f in sorted(Path(coded_sfc_dir).rglob("codsus*_hr")):
        parsed = parse_coded_sfc(f)
        parsed["source_file"] = str(f)
        results.append(parsed)
    return results


if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        result = parse_coded_sfc(path)
        print(f"Valid time: {result['valid_time']}")
        for f in result["fronts"]:
            print(f"  {f['type']}: {len(f['lats'])} points  "
                  f"lat {min(f['lats']):.1f}–{max(f['lats']):.1f}  "
                  f"lon {min(f['lons']):.1f}–{max(f['lons']):.1f}")
