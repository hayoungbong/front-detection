"""
calibrate_wpc_projection.py
===========================
Use WPC coded bulletin front positions (exact lat/lon) as GCPs to calibrate
the LCC projection + affine transform used in extract_wpc_fronts.py.

Key improvement over naive KD-tree approach
-------------------------------------------
Instead of nearest-neighbor (discontinuous → Nelder-Mead can't converge), we
precompute a Euclidean distance transform for each timestep:
  distance_map[row, col] = distance in pixels to the nearest extracted CF pixel

During optimization, we project each bulletin lat/lon → (col, row), look up
its value in the precomputed distance_map, and average.  The distance map is
smooth → cost landscape is smooth → Nelder-Mead converges reliably.

Two-stage optimization:
  Stage 1 — 21×21 grid search over (dOX, dOY) ± 3000 km to find global basin
  Stage 2 — Nelder-Mead over all 6 params starting from Stage 1 result

Output
------
  figures/wpc_calibration/calibration_result.json
  figures/wpc_calibration/grid_cost_landscape.png
  figures/wpc_calibration/error_histogram.png
  figures/wpc_calibration/before_after_{dt}.png × 4
  figures/wpc_calibration/geo_comparison_{dt}.png × 4
"""

import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
from PIL import Image
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from parse_coded_sfc import load_period
from extract_wpc_fronts import (
    LCC_PARAMS, AFFINE_SX, AFFINE_OX, AFFINE_SY, AFFINE_OY,
    color_mask, FRONT_COLORS, COLOR_TOL,
)

GIF_DIR   = Path('/Volumes/SSD_Hayoung/fronts/wpc_gif_cache')
CODED_DIR = Path('/Volumes/SSD_Hayoung/fronts/coded_sfc')
OUT_DIR   = Path('/Users/hayoungbong/Analysis/Front/figures/wpc_calibration')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameter vector: [lon_0, lat_1, SX, OX, SY, OY] ─────────────────────────
X0 = np.array([
    LCC_PARAMS['lon_0'],  # -105.0
    LCC_PARAMS['lat_1'],  #   25.0
    AFFINE_SX,            # 9048.0
    AFFINE_OX,            # -3006210.0
    AFFINE_SY,            # -13731.0
    AFFINE_OY,            # 6066288.0
])
SCALE = np.array([1.0, 1.0, 1000.0, 1e6, 1000.0, 1e6])


def make_proj(p):
    lon_0, lat_1, sx, ox, sy, oy = p
    lcc = pyproj.Proj(proj='lcc', lat_1=float(lat_1), lat_2=float(lat_1),
                      lat_0=float(lat_1), lon_0=float(lon_0), datum='WGS84')
    return lcc, (float(sx), float(ox), float(sy), float(oy))


def bul_to_pixel(lats, lons, lcc, sx, ox, sy, oy):
    xs, ys = lcc(np.asarray(lons, float), np.asarray(lats, float))
    return (xs - ox) / sx, (ys - oy) / sy   # cols, rows


def pixel_to_latlon(rows_px, cols_px, lcc, sx, ox, sy, oy):
    xs = sx * np.asarray(cols_px, float) + ox
    ys = sy * np.asarray(rows_px, float) + oy
    lons, lats = lcc(xs, ys, inverse=True)
    return lats, lons


def cf_mask_dense(img_rgb):
    """Dense CF mask: color extraction + morphological clean + aspect ratio filter."""
    raw = color_mask(img_rgb, FRONT_COLORS['CF'], tol=COLOR_TOL)
    kernel = np.ones((3, 3), np.uint8)
    eroded  = cv2.erode(raw.astype(np.uint8), kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    clean = np.zeros_like(dilated)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 15:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / (min(w, h) + 1e-3) >= 2.5:
            clean[labels == i] = 1
    return clean.astype(bool)


# ── Load dataset (precompute distance maps) ───────────────────────────────────
def load_dataset(times, bul_map, sigma_px=15.0):
    """
    For each timestep: extract CF mask → Euclidean distance transform → Gaussian blur.
    sigma_px controls the "pull radius" for the bulletin points.
    """
    records = []
    for dt in times:
        gif_path = GIF_DIR / f'namfntsfc{dt.strftime("%Y%m%d%H")}.gif'
        if not gif_path.exists():
            continue
        bul = bul_map.get(dt)
        if bul is None:
            continue

        img_rgb = np.array(Image.open(gif_path).convert('RGB'))
        H, W = img_rgb.shape[:2]

        # Dense CF mask (legend area excluded)
        cf_mask = cf_mask_dense(img_rgb)
        cf_mask[int(H * 0.70):, :int(W * 0.20)] = False

        if cf_mask.sum() < 20:
            continue

        # Euclidean distance transform: each pixel → distance to nearest CF pixel
        # invert mask: distance_transform_edt operates on background (False pixels)
        dist_map = distance_transform_edt(~cf_mask).astype(np.float32)
        # Gaussian blur → smooth, differentiable landscape
        if sigma_px > 0:
            dist_map = gaussian_filter(dist_map, sigma=sigma_px)

        # CF bulletin points
        bul_lats, bul_lons = [], []
        for front in bul['fronts']:
            if front['type'] == 'CF':
                bul_lats.extend(front['lats'])
                bul_lons.extend(front['lons'])
        if len(bul_lats) < 5:
            continue

        records.append(dict(
            dt       = dt,
            dist_map = dist_map,
            cf_mask  = cf_mask,
            bul_lats = np.array(bul_lats, float),
            bul_lons = np.array(bul_lons, float),
            img      = img_rgb,
            H=H, W=W,
        ))
    return records


# ── Cost function (smooth distance-map lookup) ────────────────────────────────
def cost_fn(p, records):
    try:
        lcc, (sx, ox, sy, oy) = make_proj(p)
    except Exception:
        return 1e9

    total = 0.0; n_pts = 0
    for r in records:
        H, W = r['H'], r['W']
        cols_b, rows_b = bul_to_pixel(r['bul_lats'], r['bul_lons'], lcc, sx, ox, sy, oy)
        # Clip to image bounds (with margin for bilinear interp)
        ci = np.clip(np.round(cols_b).astype(int), 0, W - 1)
        ri = np.clip(np.round(rows_b).astype(int), 0, H - 1)
        # Valid only if projected point is near the image (within ±200 px margin)
        valid = ((cols_b > -200) & (cols_b < W + 200) &
                 (rows_b > -200) & (rows_b < H + 200))
        if valid.sum() < 3:
            total += 500.0 * r['bul_lats'].size
            n_pts += r['bul_lats'].size
            continue
        total  += r['dist_map'][ri[valid], ci[valid]].sum()
        n_pts  += int(valid.sum())

    return total / max(n_pts, 1)


# ── Per-timestep error in km ──────────────────────────────────────────────────
def errors_km(p, records):
    """Return list of median pixel distances (converted to km) per timestep."""
    lcc, (sx, ox, sy, oy) = make_proj(p)
    errs = []
    for r in records:
        H, W = r['H'], r['W']
        cols_b, rows_b = bul_to_pixel(r['bul_lats'], r['bul_lons'], lcc, sx, ox, sy, oy)
        ci = np.clip(np.round(cols_b).astype(int), 0, W - 1)
        ri = np.clip(np.round(rows_b).astype(int), 0, H - 1)
        valid = ((cols_b > -200) & (cols_b < W + 200) &
                 (rows_b > -200) & (rows_b < H + 200))
        if valid.sum() < 3:
            continue
        # Use raw distance map (sigma=0 version doesn't exist here; approx via dist_map)
        d_px = r['dist_map'][ri[valid], ci[valid]]
        errs.append(float(np.median(d_px)) * abs(sx) / 1e3)
    return errs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading bulletins...')
    bulletins = load_period(CODED_DIR)
    bulletins = [b for b in bulletins if b['valid_time'] is not None]
    bul_map   = {b['valid_time']: b for b in bulletins}

    times = sorted(bul_map.keys())
    times = [t for t in times if t.hour in (0, 6, 12, 18)]

    print(f'Loading {len(times)} timestep GIFs + computing distance maps...')
    records = load_dataset(times, bul_map, sigma_px=15.0)
    print(f'  → {len(records)} valid pairs loaded.')

    c0 = cost_fn(X0, records)
    print(f'\nInitial cost: {c0:.3f} px  (~{c0 * abs(X0[2])/1e3:.0f} km)')
    print(f'  lon_0={X0[0]:.2f}  lat_1={X0[1]:.2f}')
    print(f'  SX={X0[2]:.0f}  OX={X0[3]:.0f}  SY={X0[4]:.0f}  OY={X0[5]:.0f}')

    # ── Stage 1: 21×21 grid search over (dOX, dOY) ───────────────────────────
    print('\n[Stage 1] Grid search ±3000 km, step 300 km (21×21=441 evals)...')
    n_grid = 21
    dOX_vals = np.linspace(-3e6, 3e6, n_grid)
    dOY_vals = np.linspace(-3e6, 3e6, n_grid)
    grid_costs = np.zeros((n_grid, n_grid))
    best_c1 = 1e9; best_dOX = 0.0; best_dOY = 0.0

    for i, dOX in enumerate(dOX_vals):
        for j, dOY in enumerate(dOY_vals):
            p = X0.copy(); p[3] += dOX; p[5] += dOY
            c = cost_fn(p, records)
            grid_costs[i, j] = c
            if c < best_c1:
                best_c1 = c; best_dOX = dOX; best_dOY = dOY
        if (i + 1) % 5 == 0:
            print(f'  row {i+1}/{n_grid}  best: {best_c1:.3f} px  '
                  f'dOX={best_dOX/1e3:.0f}km  dOY={best_dOY/1e3:.0f}km')

    # Grid cost heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    clip_max = np.percentile(grid_costs, 95)
    im = ax.contourf(dOX_vals / 1e3, dOY_vals / 1e3,
                     np.clip(grid_costs, 0, clip_max).T,
                     levels=30, cmap='RdYlGn_r')
    ax.contour(dOX_vals / 1e3, dOY_vals / 1e3,
               np.clip(grid_costs, 0, clip_max).T,
               levels=10, colors='k', linewidths=0.4, alpha=0.4)
    plt.colorbar(im, ax=ax, label='Mean distance-map score (px)')
    ax.plot(best_dOX / 1e3, best_dOY / 1e3, '*', c='cyan', ms=14, mew=1.5,
            label=f'Grid best ({best_dOX/1e3:.0f}, {best_dOY/1e3:.0f}) km')
    ax.plot(0, 0, 'wx', ms=10, mew=2, label='Current params')
    ax.set_xlabel('ΔOX (km)', fontsize=11)
    ax.set_ylabel('ΔOY (km)', fontsize=11)
    ax.set_title('Stage 1 Grid Search — Cost landscape\n'
                 'Green = low error, Red = high error', fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(OUT_DIR / 'grid_cost_landscape.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved grid_cost_landscape.png')
    print(f'  Grid best: dOX={best_dOX/1e3:.0f}km  dOY={best_dOY/1e3:.0f}km  '
          f'cost={best_c1:.3f} px (~{best_c1*abs(X0[2])/1e3:.0f} km)')

    # Refine Stage 1 with Nelder-Mead
    print('  Refining with Nelder-Mead...')
    step = 200_000  # 200 km
    def cost_translate(d):
        p = X0.copy(); p[3] += d[0]; p[5] += d[1]
        return cost_fn(p, records)

    res1 = minimize(cost_translate, [best_dOX, best_dOY], method='Nelder-Mead',
                    options={'xatol': 100, 'fatol': 0.001, 'maxiter': 2000,
                             'initial_simplex': np.array([
                                 [best_dOX,        best_dOY],
                                 [best_dOX + step, best_dOY],
                                 [best_dOX,        best_dOY + step],
                             ])})
    X1 = X0.copy()
    X1[3] += res1.x[0]; X1[5] += res1.x[1]
    c1 = cost_fn(X1, records)
    print(f'  → cost={c1:.3f} px  (~{c1 * abs(X1[2])/1e3:.0f} km)')
    print(f'     dOX={res1.x[0]/1e3:+.1f} km  dOY={res1.x[1]/1e3:+.1f} km')

    # ── Stage 2: all 6 parameters ─────────────────────────────────────────────
    print('\n[Stage 2] All 6 parameters (Nelder-Mead)...')
    eval_count = [0]
    x1_norm = X1 / SCALE

    n = len(x1_norm)
    init_simplex = np.tile(x1_norm, (n + 1, 1))
    # lon_0 ±5°, lat_1 ±5°, SX ±20%, OX ±500km, SY ±20%, OY ±500km
    perturbs = np.array([5.0, 5.0, 0.20, 0.50, 0.20, 0.50])
    for i in range(n):
        init_simplex[i + 1, i] += perturbs[i]

    def cost_norm(x_norm):
        eval_count[0] += 1
        c = cost_fn(x_norm * SCALE, records)
        if eval_count[0] % 500 == 0:
            print(f'  eval {eval_count[0]:5d}: cost={c:.4f} px')
        return c

    res2 = minimize(cost_norm, x1_norm, method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-4, 'maxiter': 15000,
                             'adaptive': True, 'initial_simplex': init_simplex})
    X2 = res2.x * SCALE
    c2 = cost_fn(X2, records)
    print(f'  → cost={c2:.4f} px  (~{c2 * abs(X2[2])/1e3:.0f} km)')
    print(f'  lon_0={X2[0]:.4f}  lat_1={X2[1]:.4f}')
    print(f'  SX={X2[2]:.2f}  OX={X2[3]:.2f}  SY={X2[4]:.2f}  OY={X2[5]:.2f}')

    # ── Save result ────────────────────────────────────────────────────────────
    result = dict(
        initial = dict(zip(['lon_0','lat_1','SX','OX','SY','OY'], X0.tolist())),
        stage1  = dict(zip(['lon_0','lat_1','SX','OX','SY','OY'], X1.tolist())),
        final   = dict(zip(['lon_0','lat_1','SX','OX','SY','OY'], X2.tolist())),
        cost_initial_px = float(c0), cost_stage1_px = float(c1), cost_final_px = float(c2),
        cost_initial_km = float(c0 * abs(X0[2]) / 1e3),
        cost_stage1_km  = float(c1 * abs(X1[2]) / 1e3),
        cost_final_km   = float(c2 * abs(X2[2]) / 1e3),
        n_timesteps     = len(records),
    )
    with open(OUT_DIR / 'calibration_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved → {OUT_DIR}/calibration_result.json')

    # ── Figures ────────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    lcc0, (sx0, ox0, sy0, oy0) = make_proj(X0)
    lcc2, (sx2, ox2, sy2, oy2) = make_proj(X2)

    # Figure 1: error histogram (before / after)
    # Reload records WITHOUT Gaussian blur to get raw pixel distances
    print('  Computing raw distance errors...')
    records_raw = load_dataset(times, bul_map, sigma_px=0.0)
    errs_before = errors_km(X0, records_raw)
    errs_stage1 = errors_km(X1, records_raw)
    errs_after  = errors_km(X2, records_raw)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('WPC Projection Calibration — CF Error Distribution (median per timestep)',
                 fontsize=13, fontweight='bold')
    all_vals = errs_before + errs_stage1 + errs_after
    bins = np.linspace(0, min(max(all_vals) * 1.05, 2000), 30)
    for ax, errs, col, ttl in zip(
            axes,
            [errs_before, errs_stage1, errs_after],
            ['#e74c3c', '#f39c12', '#2ecc71'],
            ['Before calibration', 'After Stage 1\n(translation only)', 'After Stage 2\n(all params)']):
        ax.hist(errs, bins=bins, color=col, alpha=0.8, edgecolor='white', lw=0.5)
        med = np.median(errs)
        ax.axvline(med, color='black', lw=2, ls='--', label=f'Median {med:.0f} km')
        ax.axvspan(0, 50,  alpha=0.10, color='green')
        ax.axvline(50,  color='green',  lw=1, ls=':', alpha=0.7, label='50 km')
        ax.axvline(150, color='orange', lw=1, ls=':', alpha=0.7, label='150 km')
        ax.set_title(f'{ttl}\nMedian = {med:.0f} km', fontsize=11)
        ax.set_xlabel('Median CF pixel error per timestep (km)', fontsize=10)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('# timesteps', fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'error_histogram.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  Saved error_histogram.png')

    # Figures 2-5: pixel-space before/after overlay
    n = len(records)
    sample_idx = [0, n // 3, 2 * n // 3, n - 1]
    samples = [records[i] for i in sample_idx]

    for r in samples:
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        fig.suptitle(
            f'Before / After Calibration  —  {r["dt"].strftime("%Y-%m-%d %HZ")}\n'
            'Green = extracted CF pixels  |  × = bulletin CF projected to pixel space',
            fontsize=11, fontweight='bold')

        for ax, (lcc, sx, ox, sy, oy), title, xcol in [
            (axes[0], (lcc0, sx0, ox0, sy0, oy0), 'BEFORE', '#e74c3c'),
            (axes[1], (lcc2, sx2, ox2, sy2, oy2), 'AFTER',  '#27ae60'),
        ]:
            ax.imshow(r['img'])
            ry, rx = np.where(r['cf_mask'])
            ax.scatter(rx, ry, s=1, c='lime', alpha=0.5, zorder=2)
            cols_b, rows_b = bul_to_pixel(r['bul_lats'], r['bul_lons'], lcc, sx, ox, sy, oy)
            valid = ((cols_b > -50) & (cols_b < r['W'] + 50) &
                     (rows_b > -50) & (rows_b < r['H'] + 50))
            ax.plot(cols_b[valid], rows_b[valid], 'x', c=xcol, ms=6, mew=2, zorder=5)
            ax.plot(cols_b[valid], rows_b[valid], '-', c=xcol, lw=0.8, alpha=0.4, zorder=4)
            ax.set_xlim(0, r['W']); ax.set_ylim(r['H'], 0); ax.axis('off')
            if valid.sum() > 3:
                ci = np.clip(np.round(cols_b[valid]).astype(int), 0, r['W']-1)
                ri = np.clip(np.round(rows_b[valid]).astype(int), 0, r['H']-1)
                med_km = np.median(r['dist_map'][ri, ci]) * abs(sx) / 1e3
            else:
                med_km = float('nan')
            ax.set_title(f'{title}\nMedian error ≈ {med_km:.0f} km', fontsize=11)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'before_after_{r["dt"].strftime("%Y%m%d%H")}.png',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved before_after_{r["dt"].strftime("%Y%m%d%H")}.png')

    # Figures 6-9: geographic comparison
    map_proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=38,
                                      standard_parallels=(33, 45))
    for r in samples:
        fig, axes = plt.subplots(1, 2, figsize=(22, 9),
                                  subplot_kw={'projection': map_proj})
        fig.suptitle(
            f'Geographic Alignment  —  {r["dt"].strftime("%Y-%m-%d %HZ")}\n'
            'Black line = bulletin CF (ground truth)  |  Colored = GIF extracted CF (re-projected)',
            fontsize=11, fontweight='bold')

        for ax, (lcc, sx, ox, sy, oy), title, col in [
            (axes[0], (lcc0, sx0, ox0, sy0, oy0), 'BEFORE', '#3498db'),
            (axes[1], (lcc2, sx2, ox2, sy2, oy2), 'AFTER',  '#27ae60'),
        ]:
            ax.set_extent([-130, -60, 20, 60], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND,      facecolor='#f5f5f5')
            ax.add_feature(cfeature.OCEAN,     facecolor='#dce9f5')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=':')
            ax.add_feature(cfeature.STATES,    linewidth=0.3, edgecolor='#bbb')
            gl = ax.gridlines(linewidth=0.3, color='gray', alpha=0.5, draw_labels=True)
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {'size': 8}

            ry, rx = np.where(r['cf_mask'])
            lats_ex, lons_ex = pixel_to_latlon(ry, rx, lcc, sx, ox, sy, oy)
            valid = ((lats_ex > 15) & (lats_ex < 75) &
                     (lons_ex > -175) & (lons_ex < -45))
            ax.scatter(lons_ex[valid][::5], lats_ex[valid][::5], s=4, c=col,
                       alpha=0.55, transform=ccrs.PlateCarree(), zorder=3,
                       label='GIF extracted CF')
            ax.plot(r['bul_lons'], r['bul_lats'], 'k-', lw=2.8,
                    transform=ccrs.PlateCarree(), zorder=5)
            ax.plot(r['bul_lons'], r['bul_lats'], 'o', c='black', ms=4,
                    transform=ccrs.PlateCarree(), zorder=6, label='Bulletin CF')
            ax.set_title(f'{title}', fontsize=11)
            ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f'geo_comparison_{r["dt"].strftime("%Y%m%d%H")}.png',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved geo_comparison_{r["dt"].strftime("%Y%m%d%H")}.png')

    # ── Final summary ─────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('CALIBRATION COMPLETE')
    print('='*60)
    print(f'  Initial:  {np.median(errs_before):.0f} km median CF error (raw dist)')
    print(f'  Stage 1:  {np.median(errs_stage1):.0f} km')
    print(f'  Final:    {np.median(errs_after):.0f} km')
    print()
    print('Paste into extract_wpc_fronts.py:')
    print(f'  LCC_PARAMS = dict(proj="lcc", lat_1={X2[1]:.4f}, lat_2={X2[1]:.4f},')
    print(f'                    lat_0={X2[1]:.4f}, lon_0={X2[0]:.4f}, datum="WGS84")')
    print(f'  AFFINE_SX = {X2[2]:.4f}')
    print(f'  AFFINE_OX = {X2[3]:.4f}')
    print(f'  AFFINE_SY = {X2[4]:.4f}')
    print(f'  AFFINE_OY = {X2[5]:.4f}')


if __name__ == '__main__':
    main()
