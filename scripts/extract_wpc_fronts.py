"""
Extract weather front labels (CF/WF/SF/OF/TROF) from WPC surface analysis GIF images.

WPC archive URL pattern:
  https://www.wpc.ncep.noaa.gov/archives/sfc/{YEAR}/namfntsfc{YYYYMMDDHH}.gif
  maptype=namfntsfc → "North America (Fronts/Analysis)"
  ✓ Isobars present but drawn in DARK red (139,0,0); fronts use BRIGHT colors
    (CF pure/sky blue, WF pure red 255,0,0, OF purple) → color-separable.
    Verified palette (2024 & 2026): dark-red isobars are NOT confused with WF.
  ✓ North America domain: matches ERA5 training domain
  ✓ Available: 2007-present, 6-hourly (00/06/12/18 UTC)

Pipeline:
  1. Download GIF from WPC archive
  2. Extract front pixels by color (no ML needed — WPC uses fixed color coding)
  3. Apply morphological cleaning to remove thin isobars
  4. Calibrate Lambert Conformal map projection → pixel-to-lat/lon
  5. Rasterize to ERA5 0.25° grid
  6. Save as NetCDF (CF/WF/SF/OF/TROF label arrays)

Usage:
  # Single date
  python extract_wpc_fronts.py --date 2020-06-15 --hours 0 6 12 18

  # Date range
  python extract_wpc_fronts.py --start 2010-01-01 --end 2020-12-31

  # Calibration check (overlay coastline on extracted fronts)
  python extract_wpc_fronts.py --date 2020-06-15 --hours 12 --verify

  # Use local GIF (skip download)
  python extract_wpc_fronts.py --gif /path/to/namussfc2020061512.gif
"""

import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
import xarray as xr
from PIL import Image
from scipy.ndimage import label as scipy_label

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR  = Path("/Volumes/SSD_Hayoung/fronts/wpc_labels")
CACHE_DIR = Path("/Volumes/SSD_Hayoung/fronts/wpc_gif_cache")

# ── ERA5 output grid (North America, matching training domain) ──────────────
LAT = np.arange(70, 14.75, -0.25)   # 70°N → 15°N
LON = np.arange(-170, -49.75, 0.25) # 170°W → 50°W

# ── WPC front colors (RGB) ──────────────────────────────────────────────────
# Verified from GIF palette analysis
FRONT_COLORS = {
    "CF":   [(  0,   0, 255), (  0, 178, 238)],   # Blue / sky-blue front line
    "WF":   [(255,   0,   0)],                     # Bright red ONLY (dark red 139,0,0 = wind barbs)
    "OF":   [(145,  44, 238), (148,   0, 211)],   # Purple occluded front
    "SF":   [(255,   0, 255), (238,   0, 238)],   # Magenta stationary front
    # TROF: not detectable from namfntsfc — orange pixels are H/L markers, not trough lines
}
COLOR_TOL = 25  # Tighter tolerance — namfntsfc has clean distinct colors

# ── Lambert Conformal Conic projection (WPC NAM domain) ────────────────────
# Calibrated by ICP of the GIF's black coastline/border pixels against Natural
# Earth geography (see calibrate_from_coastline).  lon_0 paired with the AFFINE
# below; both must change together.
LCC_PARAMS = dict(proj="lcc", lat_1=25, lat_2=25, lat_0=25,
                  lon_0=-100.0, x_0=0, y_0=0, ellps="WGS84")

# FULL affine (col,row) → LCC (x,y) in metres:
#   x_proj = A*col + B*row + C
#   y_proj = D*col + E*row + F
# The off-diagonal B, D terms capture a real ~-10° rotation between the image
# axes and the LCC axes — consistent across 2024 and 2026 imagery.  The previous
# DIAGONAL model (B=D=0) omitted this rotation and left extracted fronts offset
# ~100-360 km from the analyst bulletins (worse in later years).
AFFINE = (
    16113.0630,    # A  (x per col)
    -2822.9960,    # B  (x per row — rotation)
    -5282554.6,    # C  (x offset)
    -3854.3380,    # D  (y per col — rotation)
    -15749.9993,   # E  (y per row)
     8455168.1,    # F  (y offset)
)

# Legacy diagonal constants — DEPRECATED.  Kept only so older calibration/verify
# tools that import them still load; the pipeline now uses the full AFFINE above.
AFFINE_SX =   9048.0
AFFINE_OX = -3006210.0
AFFINE_SY = -13731.0
AFFINE_OY =  6066288.0

# Fallback GCPs (used only if coastline calibration mode requested)
GCPS = [
    (487, 412, -81.8, 24.5),
    ( 62, 210, -124.7, 48.4),
    (618, 225, -63.6, 44.6),
    (315, 452, -97.4, 25.9),
]


# ── Download ────────────────────────────────────────────────────────────────

def wpc_url(dt: datetime, maptype: str = "namfntsfc") -> str:
    return (f"https://www.wpc.ncep.noaa.gov/archives/sfc/{dt.year}/"
            f"{maptype}{dt.strftime('%Y%m%d%H')}.gif")


def download_gif(dt: datetime, use_cache: bool = True,
                 maptype: str = "namfntsfc") -> Path | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{maptype}{dt.strftime('%Y%m%d%H')}.gif"
    if use_cache and path.exists():
        return path
    url = wpc_url(dt, maptype)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return path
        print(f"  404: {url}")
    except Exception as e:
        print(f"  Download error: {e}")
    return None


# ── Color extraction ────────────────────────────────────────────────────────

def color_mask(arr: np.ndarray, colors: list[tuple], tol: int = COLOR_TOL) -> np.ndarray:
    """Return boolean mask where pixels match any of the given RGB colors."""
    mask = np.zeros(arr.shape[:2], bool)
    for r, g, b in colors:
        match = (
            (np.abs(arr[:,:,0].astype(int) - r) < tol) &
            (np.abs(arr[:,:,1].astype(int) - g) < tol) &
            (np.abs(arr[:,:,2].astype(int) - b) < tol)
        )
        mask |= match
    return mask


def clean_mask(mask: np.ndarray, min_size: int = 12, min_extent: int = 16) -> np.ndarray:
    """
    Remove text glyphs (H/L markers AND coloured pressure numbers) and noise specks,
    then skeletonize to a 1-px centerline.

    H, L and the bold pressure-value digits are painted in the SAME blue/red colour
    as CF/WF lines, so colour alone cannot separate them.  Two earlier filters failed:
      * bbox ASPECT RATIO — real fronts hook back on themselves (near-square bbox),
        so it rejected every front and returned empty masks; and
      * FILL RATIO — an "L" glyph is an L-shape that fills only ~30 % of its bbox, so
        a fill>0.55 test let isolated L's (and digits) through; measurement showed
        ~60 % of the extracted "WF" pixels were actually L markers and red numbers.

    The robust discriminator is simply **extent**: every text glyph is small (≤ ~15 px
    in both dimensions) while a real front spans far more once short gaps are bridged.
    Drop any component whose larger bbox side is below min_extent, regardless of shape
    or fill.  This removes H/L letters and pressure numbers in one rule while keeping
    fronts.  A light morphological CLOSE first bridges 1-2 px gaps so a front is a
    single long component rather than several sub-extent fragments.

    min_extent=16 was tuned against the analyst bulletins: it drops ≤15 px glyphs while
    preserving short genuine fronts (recall held; larger thresholds e.g. 24 started
    erasing real warm-front segments and inflated bulletin→GIF distance).

    Skeletonization eliminates year-to-year line-thickness variation in WPC images
    (WPC changed rendering software ~2013, making lines thicker in later years),
    ensuring each front occupies exactly 1 pixel regardless of how thick it was drawn.
    """
    from skimage.morphology import skeletonize as sk_thin
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean = np.zeros_like(closed)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) < min_extent:
            # Small in both dimensions — H/L glyph, pressure number, or speck, not a front
            continue
        clean[labels == i] = 1
    # Skeletonize: reduce thick lines to 1-px centerline
    skeleton = sk_thin(clean.astype(bool))
    return skeleton


def detect_stationary_fronts(red_raw: np.ndarray, blue_raw: np.ndarray,
                             prox: int = 18, min_extent: int = 20):
    """
    Recover STATIONARY fronts (SF) from the GIF.

    WPC namfntsfc has NO dedicated stationary-front colour (verified: zero magenta
    pixels).  A stationary front is instead drawn as a single line carrying
    ALTERNATING red (warm) semicircles and blue (cold) triangles.  We detect it as
    the corridor where red and blue front pixels coexist within `prox` px of each
    other, then keep the elongated components and skeletonize.

    Returns (sf_skeleton, removal_zone).  removal_zone (the corridor dilated around
    the detected SF line) should be subtracted from the CF and WF masks so a
    stationary front is not ALSO double-counted as both a cold and a warm front —
    which is exactly what happened before (it inflated WF precision error ~2x).
    """
    from skimage.morphology import skeletonize as sk_thin
    k = np.ones((prox, prox), np.uint8)
    corridor = ((cv2.dilate(red_raw.astype(np.uint8),  k) > 0) &
                (cv2.dilate(blue_raw.astype(np.uint8), k) > 0))
    sf_raw = (red_raw | blue_raw) & corridor
    closed = cv2.morphologyEx(sf_raw.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros_like(closed)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < min_extent:
            continue
        keep[labels == i] = 1
    sf = sk_thin(keep.astype(bool))
    zone = cv2.dilate(sf.astype(np.uint8), np.ones((8, 8), np.uint8)) > 0
    return sf, zone


def extract_front_masks(gif_path: Path, recover_sf: bool = True):
    """Return ({front_type: binary mask}, image_shape) from a WPC GIF."""
    img  = Image.open(gif_path).convert("RGB")
    arr  = np.array(img)
    H, W = arr.shape[:2]

    # Mask out legend area (lower-left corner of WPC namfntsfc images).
    # The legend shows colored front symbols at fixed pixel positions that
    # would otherwise be detected as real fronts. Empirically: rows > 70%,
    # cols < 20% consistently contains only the legend, never actual fronts.
    legend_mask = np.ones((H, W), bool)
    legend_mask[int(H * 0.70):, :int(W * 0.20)] = False

    raw = {ftype: color_mask(arr, colors) & legend_mask
           for ftype, colors in FRONT_COLORS.items()}

    # Stationary fronts (alternating red/blue, no dedicated colour) — detect from
    # the red/blue corridor and remove those pixels from CF/WF to avoid the
    # double-count.  recover_sf=False reproduces the old colour-only behaviour.
    if recover_sf:
        sf, zone = detect_stationary_fronts(raw["WF"], raw["CF"])
        raw["CF"] = raw["CF"] & ~zone
        raw["WF"] = raw["WF"] & ~zone
    else:
        sf = np.zeros((H, W), bool)

    masks = {ftype: clean_mask(raw[ftype]) for ftype in ("CF", "WF", "OF")}
    masks["SF"] = sf

    return masks, arr.shape  # shape: (H, W, 3)


# ── Projection calibration ──────────────────────────────────────────────────

def fit_affine_2d(gcps):
    """
    Fit a full 2D affine transform (pixel col,row) → (LCC x,y).

    Solves:  x_proj = a*col + b*row + c
             y_proj = d*col + e*row + f

    This correctly handles LCC where both col AND row jointly
    determine the projected x and y coordinate.
    Returns (A6, lcc) where A6 = [a,b,c,d,e,f].
    """
    import pyproj
    from numpy.linalg import lstsq

    lcc  = pyproj.Proj(**LCC_PARAMS)
    cols = np.array([g[0] for g in gcps], float)
    rows = np.array([g[1] for g in gcps], float)
    lons = np.array([g[2] for g in gcps], float)
    lats = np.array([g[3] for g in gcps], float)

    xs, ys = lcc(lons, lats)

    # Design matrix: [col, row, 1]
    A = np.column_stack([cols, rows, np.ones_like(cols)])
    (a, b, c), _, _, _ = lstsq(A, xs, rcond=None)
    (d, e, f), _, _, _ = lstsq(A, ys, rcond=None)

    return np.array([a, b, c, d, e, f]), lcc


def _line_black_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Return (N,2) [col,row] of black pixels that belong to LINE features
    (coastline, country/state borders) rather than text.

    The previous calibration matched ALL black pixels, but WPC maps are covered
    in black numeric labels and station plots whose pixels have no geographic
    reference, flooring the fit at ~10 px.  Lines are long/thin; text glyphs are
    small and compact — keep components that are either large in extent or have a
    low bounding-box fill ratio.
    """
    H, W = arr.shape[:2]
    black = ((arr[:, :, 0] < 60) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)).astype(np.uint8)
    black[:5, :] = 0; black[-60:, :] = 0; black[:, :5] = 0; black[:, -5:] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    keep = np.zeros_like(black)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) >= 40 or (area / (w * h) < 0.25 and area >= 20):
            keep[lab == i] = 1
    rows_b, cols_b = np.where(keep)
    return np.column_stack([cols_b, rows_b]).astype(float)


def calibrate_from_coastline(arr: np.ndarray, lon0: float = None,
                             iters: int = 40, verbose: bool = False):
    """
    Calibrate the FULL pixel→proj affine by ICP of the GIF's black coastline /
    border pixels against Natural Earth geography.

    Returns (transform6, lon0) where transform6 = (a, b, c, d, e, f) maps
      x_proj = a*col + b*row + c ;  y_proj = d*col + e*row + f
    consistent with an LCC using the returned lon0.  Returns None on failure.

    Unlike the old version this fits a full affine (so the ~-10° image rotation
    is captured) via iterative-closest-point, which is far more robust than
    Nelder-Mead on a 5-parameter diagonal model.
    """
    try:
        import cartopy.io.shapereader as shpreader
        from scipy.spatial import cKDTree
        import pyproj
    except ImportError:
        print("  cartopy/shapely not available — cannot coastline-calibrate")
        return None

    if lon0 is None:
        lon0 = LCC_PARAMS["lon_0"]

    # Reference geography (coastline + national + state borders) in the NA window
    segs = []
    for cat, name in [("physical", "coastline"),
                      ("cultural", "admin_0_boundary_lines_land"),
                      ("cultural", "admin_1_states_provinces_lines")]:
        try:
            path = shpreader.natural_earth(resolution="110m", category=cat, name=name)
            for g in shpreader.Reader(path).geometries():
                for ln in (g.geoms if hasattr(g, "geoms") else [g]):
                    segs.append(np.asarray(ln.coords))
        except Exception:
            continue
    if not segs:
        return None
    ref_ll = np.vstack(segs)
    m = ((ref_ll[:, 0] > -175) & (ref_ll[:, 0] < -50) &
         (ref_ll[:, 1] > 15) & (ref_ll[:, 1] < 75))
    ref_ll = ref_ll[m]

    lcc = pyproj.Proj(proj="lcc", lat_1=25, lat_2=25, lat_0=25,
                      lon_0=lon0, ellps="WGS84")
    rx, ry = lcc(ref_ll[:, 0], ref_ll[:, 1])
    ref_xy = np.column_stack([rx, ry])

    pix = _line_black_pixels(arr)
    if len(pix) < 100:
        return None
    tree = cKDTree(pix)

    def solve(src, dst):                       # src,dst (N,2) → 2x3 affine
        Adesign = np.column_stack([src, np.ones(len(src))])
        mx, _, _, _ = np.linalg.lstsq(Adesign, dst[:, 0], rcond=None)
        my, _, _, _ = np.linalg.lstsq(Adesign, dst[:, 1], rcond=None)
        return np.vstack([mx, my])

    def apply(P, M):
        return np.column_stack([P, np.ones(len(P))]) @ M.T

    # Initialise proj→pixel from the legacy diagonal constants
    P2P = np.array([[1 / AFFINE_SX, 0, -AFFINE_OX / AFFINE_SX],
                    [0, 1 / AFFINE_SY, -AFFINE_OY / AFFINE_SY]])
    for _ in range(iters):
        pp = apply(ref_xy, P2P)
        inb = ((pp[:, 0] > -40) & (pp[:, 0] < arr.shape[1] + 40) &
               (pp[:, 1] > -40) & (pp[:, 1] < arr.shape[0] + 40))
        if inb.sum() < 20:
            break
        d, idx = tree.query(pp[inb], k=1)
        keep = d < np.percentile(d, 70)        # robust to text/outliers
        P2P = solve(ref_xy[inb][keep], pix[idx[keep]])

    # Invert proj→pixel to the pixel→proj affine the pipeline expects
    M, t = P2P[:, :2], P2P[:, 2]
    Minv = np.linalg.inv(M)
    tinv = -Minv @ t
    transform6 = (Minv[0, 0], Minv[0, 1], tinv[0],
                  Minv[1, 0], Minv[1, 1], tinv[1])

    if verbose:
        pp = apply(ref_xy, P2P)
        inb = ((pp[:, 0] > 0) & (pp[:, 0] < arr.shape[1]) &
               (pp[:, 1] > 0) & (pp[:, 1] < arr.shape[0]))
        d, _ = tree.query(pp[inb], k=1)
        rot = np.degrees(np.arctan2(P2P[0, 1], P2P[0, 0]))
        print(f"  Coastline ICP: {len(pix)} line px, residual median {np.median(d):.2f} px, "
              f"rotation {rot:.2f}°")
    return transform6, lon0


def build_pixel_to_latlon(arr: np.ndarray = None, use_coastline_cal: bool = False):
    """
    Return (transform, lcc) using optimized affine calibration.

    transform = (a, b, c, d, e, f) where:
      x_proj = a*col + b*row + c
      y_proj = d*col + e*row + f

    Default uses pre-optimized constants (multi-image coastline matching).
    use_coastline_cal=True re-runs optimization on the given arr.
    """
    import pyproj

    lcc = pyproj.Proj(**LCC_PARAMS)

    if use_coastline_cal and arr is not None:
        result = calibrate_from_coastline(arr, verbose=True)
        if result is not None:
            transform6, lon0 = result
            lcc = pyproj.Proj(proj="lcc", lat_1=25, lat_2=25,
                              lat_0=25, lon_0=lon0, ellps="WGS84")
            return transform6, lcc

    # Use pre-calibrated full affine (includes rotation; see AFFINE definition)
    return AFFINE, lcc


def pixel_to_projected(rows_px, cols_px, transform):
    """Apply 2D affine: (col,row) → (x_proj, y_proj)."""
    a, b, c, d, e, f = transform
    x = a * cols_px + b * rows_px + c
    y = d * cols_px + e * rows_px + f
    return x, y


def rasterize_to_era5(mask: np.ndarray, transform, lcc) -> np.ndarray:
    """
    Project front mask (pixel space) onto ERA5 0.25° grid.
    Returns binary array with shape (len(LAT), len(LON)).
    """
    out = np.zeros((len(LAT), len(LON)), np.uint8)
    if not mask.any():
        return out

    rows_px, cols_px = np.where(mask)
    x_pts, y_pts = pixel_to_projected(rows_px.astype(float),
                                       cols_px.astype(float), transform)
    lons_pts, lats_pts = lcc(x_pts, y_pts, inverse=True)

    lat_idx = np.round((LAT[0] - lats_pts) / 0.25).astype(int)
    lon_idx = np.round((lons_pts - LON[0]) / 0.25).astype(int)

    valid = ((lat_idx >= 0) & (lat_idx < len(LAT)) &
             (lon_idx >= 0) & (lon_idx < len(LON)))
    out[lat_idx[valid], lon_idx[valid]] = 1
    return out


# ── Main extraction for one timestep ───────────────────────────────────────

def process_one(dt: datetime, verify: bool = False, gif_path: Path = None) -> xr.Dataset | None:
    if gif_path is None:
        gif_path = download_gif(dt)
    if gif_path is None:
        return None

    masks, img_shape = extract_front_masks(gif_path)
    arr_rgb = np.array(Image.open(gif_path).convert("RGB"))
    transform, lcc = build_pixel_to_latlon(arr_rgb, use_coastline_cal=False)

    grids = {}
    for ftype, mask in masks.items():
        grids[ftype] = rasterize_to_era5(mask, transform, lcc)

    ds = xr.Dataset(
        {k: (["lat", "lon"], v.astype(np.int8)) for k, v in grids.items()},
        coords={"lat": LAT, "lon": LON, "time": dt},
    )
    ds.attrs["source"]      = "NOAA/WPC surface analysis (image color extraction)"
    ds.attrs["gif_url"]     = wpc_url(dt)
    ds.attrs["projection"]  = str(LCC_PARAMS)
    ds.attrs["created"]     = datetime.utcnow().isoformat()

    if verify:
        _verify_plot(gif_path, masks, grids, dt)

    return ds


# ── Verification plot ───────────────────────────────────────────────────────

def _verify_plot(gif_path, masks, grids, dt):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(20, 8))

    # Left: raw extraction on image
    ax1 = fig.add_subplot(1, 2, 1)
    img_rgb = np.array(Image.open(gif_path).convert("RGB"))
    overlay = img_rgb.copy()
    colors_ov = {"CF": [0,80,255], "WF": [255,50,50],
                 "OF": [160,0,255], "SF": [255,0,220], "TROF": [255,160,0]}
    for ftype, mask in masks.items():
        overlay[mask] = colors_ov[ftype]
    ax1.imshow(overlay)
    ax1.set_title(f"Raw color extraction\n{dt.strftime('%Y-%m-%d %HZ')}", fontsize=11)
    patches = [mpatches.Patch(color=np.array(c)/255, label=k)
               for k, c in colors_ov.items()]
    ax1.legend(handles=patches, loc="lower left", fontsize=9)

    # Right: rasterized on map
    proj = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(25, 25))
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.set_extent([-135, -55, 20, 60], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE, lw=0.8, edgecolor="black")
    ax2.add_feature(cfeature.BORDERS,   lw=0.5, edgecolor="gray")
    ax2.add_feature(cfeature.STATES,    lw=0.3, edgecolor="gray")
    ax2.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax2.add_feature(cfeature.OCEAN, facecolor="#d0e8f0")

    plot_colors = {"CF": "blue", "WF": "red", "OF": "purple",
                   "SF": "magenta", "TROF": "darkorange"}
    for ftype, grid in grids.items():
        if not grid.any():
            continue
        lats_p, lons_p = np.where(grid)
        lat_vals = LAT[lats_p]
        lon_vals = LON[lons_p]
        ax2.scatter(lon_vals, lat_vals, s=1, color=plot_colors[ftype],
                    transform=ccrs.PlateCarree(), label=ftype, alpha=0.7)

    ax2.legend(loc="lower left", fontsize=9, markerscale=8)
    ax2.set_title(f"Rasterized on ERA5 grid (0.25°)\n{dt.strftime('%Y-%m-%d %HZ')}", fontsize=11)

    plt.tight_layout()
    out = Path(f"/Users/hayoungbong/Analysis/Front/figures/"
               f"wpc_extract_{dt.strftime('%Y%m%d%H')}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Verify plot: {out}")
    plt.close()


# ── Batch processing ────────────────────────────────────────────────────────

def process_range(start: datetime, end: datetime, hours=(0, 6, 12, 18),
                  skip_existing: bool = True, n_workers: int = 8):
    """
    Batch extract WPC fronts for a date range.
    Saves one NetCDF per year: wpc_labels_YYYY.nc (time × lat × lon)
    n_workers: parallel download threads (I/O bound → safe to use many)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build task list grouped by year
    tasks = []
    dt = start
    while dt <= end:
        for h in hours:
            tasks.append(dt.replace(hour=h))
        dt += timedelta(days=1)

    # Shared transform (computed once)
    transform, lcc = build_pixel_to_latlon()

    ok = err = skip = 0
    year_datasets = {}   # year → list of (time, grids_dict)

    def _do_one(t):
        gif = download_gif(t)
        if gif is None:
            return t, None
        masks, img_shape = extract_front_masks(gif)
        grids = {ftype: rasterize_to_era5(mask, transform, lcc)
                 for ftype, mask in masks.items()}
        return t, grids

    print(f"Processing {len(tasks):,} timesteps  ({n_workers} workers)...", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_do_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            t, grids = fut.result()
            if grids is None:
                err += 1
                continue
            yr = t.year
            if yr not in year_datasets:
                year_datasets[yr] = []
            year_datasets[yr].append((t, grids))
            ok += 1

            if (i + 1) % 200 == 0:
                done_pct = (i+1)/len(tasks)*100
                print(f"  [{i+1:,}/{len(tasks):,}  {done_pct:.0f}%]  "
                      f"ok={ok} err={err}", flush=True)

    # Save one NetCDF per year
    print("Saving yearly NetCDF files...")
    for yr, entries in sorted(year_datasets.items()):
        out_path = OUT_DIR / f"wpc_labels_{yr}.nc"
        entries.sort(key=lambda x: x[0])
        times = [e[0] for e in entries]
        ds = xr.Dataset(
            {k: (["time","lat","lon"],
                 np.stack([e[1][k] for e in entries], axis=0).astype(np.int8))
             for k in entries[0][1]},
            coords={"time": times, "lat": LAT, "lon": LON},
        )
        ds.attrs["source"] = "NOAA/WPC namfntsfc (Fronts/Analysis only)"
        encoding = {k: {"zlib": True, "complevel": 4, "dtype": "int8"}
                    for k in ds.data_vars}
        if out_path.exists():
            out_path.unlink()   # remove old file to prevent xarray append-mode conflict
        ds.to_netcdf(out_path, encoding=encoding)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {yr}: {len(entries)} timesteps → {out_path.name}  ({size_mb:.1f} MB)")

    print(f"\nDone: {ok} ok, {err} failed, {skip} skipped")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date",   help="Single date YYYY-MM-DD")
    p.add_argument("--start",  help="Start date YYYY-MM-DD")
    p.add_argument("--end",    help="End date   YYYY-MM-DD")
    p.add_argument("--hours",  nargs="+", type=int, default=[0, 6, 12, 18])
    p.add_argument("--verify", action="store_true",
                   help="Generate calibration verification plot")
    p.add_argument("--gif",    help="Use local GIF file (skip download)")
    args = p.parse_args()

    if args.gif:
        gif_path = Path(args.gif)
        stem = gif_path.stem
        dt_str = stem.replace("namussfc", "") if "namussfc" in stem else "2007010112"
        try:
            dt = datetime.strptime(dt_str, "%Y%m%d%H")
        except ValueError:
            dt = datetime(2007, 1, 1, 12)  # fallback for test files
        ds = process_one(dt, verify=True, gif_path=gif_path)
        if ds:
            out = OUT_DIR / f"wpc_labels_{dt.strftime('%Y%m%d%H')}.nc"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out)
            print(f"Saved: {out}")
            for v in ds.data_vars:
                print(f"  {v}: {int(ds[v].values.sum())} front pixels")
    elif args.date:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
        for h in args.hours:
            t  = dt.replace(hour=h)
            ds = process_one(t, verify=args.verify)
            if ds:
                out = OUT_DIR / f"wpc_labels_{t.strftime('%Y%m%d%H')}.nc"
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                ds.to_netcdf(out)
                print(f"Saved {out}")
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end   = datetime.strptime(args.end,   "%Y-%m-%d")
        process_range(start, end, hours=args.hours)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
