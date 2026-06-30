"""
Build hybrid WPC×ERA5 front labels on NASA Discover.

Pipeline (per year):
  1. Download WPC namfntsfc GIFs from NOAA archive
  2. Extract CF/WF/SF/OF front masks by color + morphology
  3. Project to ERA5 0.25° grid (calibrated full affine)
  4. Intersect with ERA5 TFP → hybrid_YYYY.nc

All paths are Discover-specific. Self-contained — no imports from other
project scripts needed.

Usage:
  module load python/GEOSpyD/24.11.3-0/3.12
  pip install --user requests opencv-python-headless scikit-image pyproj

  # Full pipeline 2019-2025
  python build_hybrid_discover.py --years 2019 2020 2021 2022 2023 2024 2025

  # Single year (resume-safe)
  python build_hybrid_discover.py --years 2019

  # Skip GIF download (GIFs already in CACHE_DIR)
  python build_hybrid_discover.py --years 2019 --no-download

Run in screen to survive logout:
  screen -S hybrid
  module load python/GEOSpyD/24.11.3-0/3.12
  python build_hybrid_discover.py --years 2019 2020 2021 2022 2023 2024 2025
  Ctrl-A D to detach
"""

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
import xarray as xr
from PIL import Image
from scipy.ndimage import binary_dilation, label as scipy_label

warnings.filterwarnings("ignore")

# ── Discover paths ─────────────────────────────────────────────────────────
FRONT     = Path("/discover/nobackup/projects/giss/paleofun/hbong/front")
TRAIN_DIR = FRONT / "data" / "training"
WPC_DIR   = FRONT / "data" / "wpc_labels"
OUT_DIR   = FRONT / "data" / "hybrid_labels"
CACHE_DIR = FRONT / "data" / "wpc_gif_cache"

for d in (WPC_DIR, OUT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── ERA5 output grid ───────────────────────────────────────────────────────
LAT = np.arange(70, 14.75, -0.25)    # 70°N → 15°N
LON = np.arange(-170, -49.75, 0.25)  # 170°W → 50°W

# ── WPC front colors (RGB) ─────────────────────────────────────────────────
FRONT_COLORS = {
    "CF": [(0, 0, 255), (0, 178, 238)],
    "WF": [(255, 0, 0)],
    "OF": [(145, 44, 238), (148, 0, 211)],
    "SF": [(255, 0, 255), (238, 0, 238)],
}
COLOR_TOL = 25

# ── Calibrated LCC affine (col,row) → (x_proj, y_proj) in metres ──────────
# Includes ~10° image rotation. Validated against WPC bulletins 2024+2026.
LCC_PARAMS = dict(proj="lcc", lat_1=25, lat_2=25, lat_0=25,
                  lon_0=-100.0, x_0=0, y_0=0, ellps="WGS84")
AFFINE = (
    16113.0630,   # A  x per col
    -2822.9960,   # B  x per row (rotation)
    -5282554.6,   # C  x offset
    -3854.3380,   # D  y per col (rotation)
    -15749.9993,  # E  y per row
     8455168.1,   # F  y offset
)

# ── Hybrid label constants ─────────────────────────────────────────────────
TFP_THRESH  = 0.12   # |TFP| threshold (K/m²×10⁶)
DILATE_CELLS = 2     # WPC dilation ±2 cells ≈ ±50 km
LABEL = {"BG": 0, "CF": 1, "WF": 2, "SF": 3, "OF": 4}


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — GIF download
# ═══════════════════════════════════════════════════════════════════════════

def wpc_url(dt: datetime) -> str:
    return (f"https://www.wpc.ncep.noaa.gov/archives/sfc/{dt.year}/"
            f"namfntsfc{dt.strftime('%Y%m%d%H')}.gif")


def download_gif(dt: datetime) -> Path | None:
    path = CACHE_DIR / f"namfntsfc{dt.strftime('%Y%m%d%H')}.gif"
    if path.exists() and path.stat().st_size > 1000:
        return path
    try:
        r = requests.get(wpc_url(dt), timeout=30)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return path
    except Exception:
        pass
    return None


def download_year(year: int, workers: int = 8) -> dict:
    """Download all GIFs for a year. Returns {datetime: Path}."""
    tasks = []
    dt = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 18)
    while dt <= end:
        for h in (0, 6, 12, 18):
            tasks.append(dt.replace(hour=h))
        dt += timedelta(days=1)
    tasks = [t for t in tasks if t <= datetime.utcnow()]

    results = {}
    ok = err = 0
    print(f"  Downloading {len(tasks)} GIFs for {year}  ({workers} workers)...",
          flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_gif, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            t = futures[fut]
            p = fut.result()
            if p:
                results[t] = p
                ok += 1
            else:
                err += 1
            if (i + 1) % 500 == 0:
                print(f"    [{i+1}/{len(tasks)}]  ok={ok} err={err}", flush=True)
    print(f"  Done: ok={ok} err={err}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Color extraction + morphological cleaning
# ═══════════════════════════════════════════════════════════════════════════

def color_mask(arr, colors, tol=COLOR_TOL):
    mask = np.zeros(arr.shape[:2], bool)
    for r, g, b in colors:
        mask |= (
            (np.abs(arr[:, :, 0].astype(int) - r) < tol) &
            (np.abs(arr[:, :, 1].astype(int) - g) < tol) &
            (np.abs(arr[:, :, 2].astype(int) - b) < tol)
        )
    return mask


def clean_mask(mask, min_size=12, min_extent=16):
    from skimage.morphology import skeletonize
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros_like(closed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            continue
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < min_extent:
            continue
        keep[labels == i] = 1
    return skeletonize(keep.astype(bool))


def detect_stationary(red_raw, blue_raw, prox=18, min_extent=20):
    from skimage.morphology import skeletonize
    k = np.ones((prox, prox), np.uint8)
    corridor = ((cv2.dilate(red_raw.astype(np.uint8), k) > 0) &
                (cv2.dilate(blue_raw.astype(np.uint8), k) > 0))
    sf_raw = (red_raw | blue_raw) & corridor
    closed = cv2.morphologyEx(sf_raw.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros_like(closed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < min_extent:
            continue
        keep[labels == i] = 1
    sf = skeletonize(keep.astype(bool))
    zone = cv2.dilate(sf.astype(np.uint8), np.ones((8, 8), np.uint8)) > 0
    return sf, zone


def extract_masks(gif_path: Path) -> dict:
    img = Image.open(gif_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    legend = np.ones((H, W), bool)
    legend[int(H * 0.70):, :int(W * 0.20)] = False

    raw = {ft: color_mask(arr, colors) & legend
           for ft, colors in FRONT_COLORS.items()}

    sf, zone = detect_stationary(raw["WF"], raw["CF"])
    raw["CF"] = raw["CF"] & ~zone
    raw["WF"] = raw["WF"] & ~zone

    masks = {ft: clean_mask(raw[ft]) for ft in ("CF", "WF", "OF")}
    masks["SF"] = sf
    return masks


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Project to ERA5 grid
# ═══════════════════════════════════════════════════════════════════════════

def rasterize(mask, transform, lcc):
    import pyproj
    out = np.zeros((len(LAT), len(LON)), np.uint8)
    if not mask.any():
        return out
    rows_px, cols_px = np.where(mask)
    a, b, c, d, e, f = transform
    x = a * cols_px + b * rows_px + c
    y = d * cols_px + e * rows_px + f
    lons_pts, lats_pts = lcc(x, y, inverse=True)
    lat_idx = np.round((LAT[0] - lats_pts) / 0.25).astype(int)
    lon_idx = np.round((lons_pts - LON[0]) / 0.25).astype(int)
    valid = ((lat_idx >= 0) & (lat_idx < len(LAT)) &
             (lon_idx >= 0) & (lon_idx < len(LON)))
    out[lat_idx[valid], lon_idx[valid]] = 1
    return out


def process_gif(gif_path: Path, transform, lcc) -> dict | None:
    try:
        masks = extract_masks(gif_path)
        return {ft: rasterize(m, transform, lcc) for ft, m in masks.items()}
    except Exception as e:
        print(f"    ERROR processing {gif_path.name}: {e}", flush=True)
        return None


def extract_year_to_netcdf(year: int, gif_map: dict, workers: int = 4) -> Path | None:
    """Extract all GIFs for a year → wpc_labels_YYYY.nc."""
    import pyproj
    out_path = WPC_DIR / f"wpc_labels_{year}.nc"
    if out_path.exists():
        print(f"  wpc_labels_{year}.nc already exists — skipping extraction", flush=True)
        return out_path

    lcc = pyproj.Proj(**LCC_PARAMS)

    entries = []
    print(f"  Extracting {len(gif_map)} GIFs for {year}...", flush=True)

    def _do(item):
        t, gif = item
        grids = process_gif(gif, AFFINE, lcc)
        return t, grids

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_do, item): item for item in gif_map.items()}
        for i, fut in enumerate(as_completed(futures)):
            t, grids = fut.result()
            if grids is not None:
                entries.append((t, grids))
            if (i + 1) % 400 == 0:
                print(f"    [{i+1}/{len(gif_map)}]", flush=True)

    if not entries:
        print(f"  No data for {year} — skipping", flush=True)
        return None

    entries.sort(key=lambda x: x[0])
    times = [e[0] for e in entries]
    ds = xr.Dataset(
        {k: (["time", "lat", "lon"],
             np.stack([e[1][k] for e in entries], axis=0).astype(np.int8))
         for k in entries[0][1]},
        coords={"time": times, "lat": LAT, "lon": LON},
    )
    ds.attrs["source"] = "NOAA/WPC namfntsfc color extraction"
    enc = {k: {"zlib": True, "complevel": 4, "dtype": "int8"} for k in ds.data_vars}
    ds.to_netcdf(out_path, encoding=enc)
    mb = out_path.stat().st_size / 1e6
    cf = int(ds["CF"].values.sum()); wf = int(ds["WF"].values.sum())
    sf = int(ds["SF"].values.sum()); of = int(ds["OF"].values.sum())
    print(f"  Saved wpc_labels_{year}.nc ({mb:.0f} MB)  "
          f"CF={cf:,} WF={wf:,} SF={sf:,} OF={of:,}", flush=True)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Build hybrid labels
# ═══════════════════════════════════════════════════════════════════════════

def build_hybrid_year(year: int, overwrite: bool = False) -> Path | None:
    out_path = OUT_DIR / f"hybrid_{year}.nc"
    if out_path.exists() and not overwrite:
        print(f"  hybrid_{year}.nc already exists — skipping", flush=True)
        return out_path

    train_path = TRAIN_DIR / f"era5_{year}_training.nc"
    wpc_path   = WPC_DIR   / f"wpc_labels_{year}.nc"

    if not train_path.exists():
        print(f"  ERROR: {train_path} not found", flush=True)
        return None
    if not wpc_path.exists():
        print(f"  ERROR: {wpc_path} not found", flush=True)
        return None

    print(f"  Building hybrid_{year}.nc...", flush=True)
    tr  = xr.open_dataset(train_path)
    wpc = xr.open_dataset(wpc_path)

    tfp    = tr["tfp_850"].values          # (T, lat, lon) — already training domain
    # WPC grid is 70→15°N, 170→50°W (221×481); training grid same dimensions
    wpc_cf = wpc["CF"].values              # (T_wpc, lat, lon)
    wpc_wf = wpc["WF"].values
    wpc_sf = wpc["SF"].values
    wpc_of = wpc["OF"].values

    tr_times  = tr["time"].values
    wpc_times = wpc["time"].values

    T = len(tr_times)
    H, W = tfp.shape[1], tfp.shape[2]
    hybrid = np.zeros((T, H, W), dtype=np.int8)

    struct = np.ones((1, 2 * DILATE_CELLS + 1, 2 * DILATE_CELLS + 1), bool)

    wpc_time_index = {np.datetime64(t, "ns"): i for i, t in enumerate(wpc_times)}

    of_total = sf_total = cf_total = wf_total = 0

    for ti, t in enumerate(tr_times):
        key = np.datetime64(t, "ns")
        if key not in wpc_time_index:
            continue
        wi = wpc_time_index[key]

        tfp_t = np.abs(tfp[ti])          # frontal mask from ERA5
        front_mask = tfp_t > TFP_THRESH

        cf_d = binary_dilation(wpc_cf[wi].astype(bool), struct[0])
        wf_d = binary_dilation(wpc_wf[wi].astype(bool), struct[0])
        sf_d = binary_dilation(wpc_sf[wi].astype(bool), struct[0])
        of_d = binary_dilation(wpc_of[wi].astype(bool), struct[0])

        lbl = np.zeros((H, W), dtype=np.int8)
        lbl[front_mask & cf_d] = LABEL["CF"]
        lbl[front_mask & wf_d] = LABEL["WF"]
        lbl[front_mask & sf_d] = LABEL["SF"]
        # OF: skip TFP requirement — occluded fronts have weak lower-trop gradient
        lbl[of_d] = LABEL["OF"]

        hybrid[ti] = lbl
        cf_total += int((lbl == 1).sum())
        wf_total += int((lbl == 2).sum())
        sf_total += int((lbl == 3).sum())
        of_total += int((lbl == 4).sum())

    ds_out = xr.Dataset(
        {"front_label": (["time", "lat", "lon"], hybrid)},
        coords={"time": tr_times, "lat": tr["lat"].values, "lon": tr["lon"].values},
    )
    ds_out.attrs["tfp_thresh"]   = TFP_THRESH
    ds_out.attrs["dilate_cells"] = DILATE_CELLS
    ds_out.attrs["of_note"]      = "OF uses WPC only (no TFP requirement)"
    ds_out.attrs["label_map"]    = "0=BG 1=CF 2=WF 3=SF 4=OF"

    enc = {"front_label": {"zlib": True, "complevel": 4, "dtype": "int8"}}
    ds_out.to_netcdf(out_path, encoding=enc)
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved hybrid_{year}.nc ({mb:.0f} MB)  "
          f"CF={cf_total:,} WF={wf_total:,} SF={sf_total:,} OF={of_total:,}", flush=True)

    tr.close(); wpc.close()
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="+", type=int,
                   default=[2019, 2020, 2021, 2022, 2023, 2024, 2025],
                   help="Years to process")
    p.add_argument("--no-download", action="store_true",
                   help="Skip GIF download (use existing cache)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing hybrid_YYYY.nc files")
    p.add_argument("--dl-workers", type=int, default=8,
                   help="Parallel GIF download workers (default 8)")
    p.add_argument("--ex-workers", type=int, default=4,
                   help="Parallel GIF extraction workers (default 4)")
    args = p.parse_args()

    print(f"=== build_hybrid_discover.py  years={args.years} ===", flush=True)
    print(f"  WPC cache : {CACHE_DIR}", flush=True)
    print(f"  WPC labels: {WPC_DIR}", flush=True)
    print(f"  Hybrid out: {OUT_DIR}", flush=True)
    print(f"  TFP thresh: {TFP_THRESH}  dilate: {DILATE_CELLS} cells", flush=True)
    print(f"  OF note   : no TFP requirement (occluded fronts have weak TFP)", flush=True)
    print(flush=True)

    for year in args.years:
        print(f"── {year} ────────────────────────────────", flush=True)

        # Step 1: download GIFs
        if not args.no_download:
            gif_map = download_year(year, workers=args.dl_workers)
        else:
            gifs = sorted(CACHE_DIR.glob(f"namfntsfc{year}*.gif"))
            gif_map = {}
            for g in gifs:
                stem = g.stem.replace("namfntsfc", "")
                try:
                    gif_map[datetime.strptime(stem, "%Y%m%d%H")] = g
                except ValueError:
                    pass
            print(f"  Using {len(gif_map)} cached GIFs for {year}", flush=True)

        if not gif_map:
            print(f"  No GIFs for {year} — skipping", flush=True)
            continue

        # Step 2+3: extract to wpc_labels_YYYY.nc
        wpc_path = extract_year_to_netcdf(year, gif_map, workers=args.ex_workers)
        if wpc_path is None:
            continue

        # Step 4: build hybrid_YYYY.nc
        build_hybrid_year(year, overwrite=args.overwrite)

        print(flush=True)

    print("=== all done ===", flush=True)


if __name__ == "__main__":
    main()
