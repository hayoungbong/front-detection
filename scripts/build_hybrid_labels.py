"""
Hybrid Front Label Builder
==========================
Combines ERA5-based TFP labels (accurate position) with WPC image-extracted
labels (accurate front TYPE: CF/WF/OF) to produce a superior training target.

Why hybrid labels?
------------------
TFP (Thermal Front Parameter) auto-labels:
  + Perfectly aligned with ERA5 grid (no projection error)
  + Captures every frontal zone in the thermodynamic field
  - Cannot distinguish CF from WF from OF
  - Over-detects: labels frontal ZONES, not discrete front lines
  - No human expert judgment

WPC image-extracted labels (this project):
  + Expert meteorologist judgment (CF vs WF vs OF correctly typed)
  + 20-year archive (2007–2026), 6-hourly
  + Includes occluded fronts (OF) — impossible with TFP alone
  - ~30 km systematic positional offset (LCC projection calibration error)
  - Year-to-year thickness variation (partially fixed by skeletonization)
  - Cannot detect TROF reliably from image colors

Hybrid strategy (intersection):
  ┌─────────────────────────────────────────────────────────────────┐
  │  ERA5 TFP detects a front at grid cell (i,j)                   │
  │          AND                                                     │
  │  WPC shows front type X within ±2 grid cells (~50 km)          │
  │          →  Label (i,j) as type X                               │
  │                                                                  │
  │  ERA5 TFP detects front, but WPC shows nothing nearby           │
  │          →  Label as Background (likely TFP over-detection)      │
  │                                                                  │
  │  WPC shows front, but ERA5 TFP shows no gradient                │
  │          →  Discard (WPC color artifact or calibration error)    │
  └─────────────────────────────────────────────────────────────────┘

This approach gives us:
  - ERA5-grid-accurate POSITIONS  (from TFP, no projection error)
  - Expert-classified TYPES       (from WPC: CF/WF/SF/OF)
  - Reduced false positives       (both must agree)
  - New OF class                  (impossible with TFP alone)

Output label encoding:
  0 = Background
  1 = Cold Front (CF)
  2 = Warm Front (WF)
  3 = Stationary Front (SF)
  4 = Occluded Front (OF)   ← NEW CLASS enabled by WPC labels

Usage:
  python build_hybrid_labels.py --years 2019 2020 2021 2022 2023 2024
  python build_hybrid_labels.py --years 2019 2024 --dilate 2 --tfp-thresh 0.10
  python build_hybrid_labels.py --verify 2024
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_DIR = Path("/Volumes/SSD_Hayoung/fronts/training")
WPC_DIR   = Path("/Volumes/SSD_Hayoung/fronts/wpc_labels")
OUT_DIR   = Path("/Volumes/SSD_Hayoung/fronts/hybrid_labels")

# ── Constants ──────────────────────────────────────────────────────────────
# TFP threshold for front detection (units: K/m²×10⁶, positive = warm side)
# p95 ≈ 0.159, p99 ≈ 0.238 from 2024 data
TFP_THRESH_DEFAULT = 0.12   # conservative: captures clear fronts only

# Dilation radius for WPC labels (grid cells, 0.25° ≈ 28 km each)
# Compensates for ~30 km LCC projection calibration offset
DILATE_CELLS = 2            # ±2 cells = ±50 km search radius

# Label encoding
LABEL = {"BG": 0, "CF": 1, "WF": 2, "SF": 3, "OF": 4}
LABEL_NAMES = {v: k for k, v in LABEL.items()}


def build_hybrid_one_year(year: int,
                           tfp_thresh: float = TFP_THRESH_DEFAULT,
                           dilate: int = DILATE_CELLS,
                           overwrite: bool = False) -> Path:
    """
    Build hybrid labels for a single year.

    Parameters
    ----------
    year       : calendar year (must have both training NC and wpc_labels NC)
    tfp_thresh : |TFP| threshold above which a grid cell is considered frontal
    dilate     : WPC label dilation radius in grid cells (compensates calibration offset)
    overwrite  : re-generate even if output file already exists

    Returns
    -------
    Path to output NetCDF file
    """
    out_path = OUT_DIR / f"hybrid_{year}.nc"
    if out_path.exists() and not overwrite:
        print(f"  {year}: already exists, skipping (use --overwrite to force)")
        return out_path

    train_path = TRAIN_DIR / f"era5_{year}_training.nc"
    wpc_path   = WPC_DIR   / f"wpc_labels_{year}.nc"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not wpc_path.exists():
        raise FileNotFoundError(f"WPC labels not found: {wpc_path}")

    print(f"  {year}: loading ERA5 training + WPC labels...")
    train = xr.open_dataset(train_path)
    wpc   = xr.open_dataset(wpc_path)

    # Align timestamps (inner join on time)
    t_train = set(str(t)[:16] for t in train.time.values)
    t_wpc   = set(str(t)[:16] for t in wpc.time.values)
    common  = sorted(t_train & t_wpc)
    if len(common) < 10:
        raise ValueError(f"{year}: only {len(common)} common timesteps — check data")
    print(f"    Common timesteps: {len(common)} / train={len(t_train)} wpc={len(t_wpc)}")

    # Select matching times
    train_sel = train.sel(time=[t for t in train.time.values
                                if str(t)[:16] in t_wpc])
    wpc_sel   = wpc.sel(time=[t for t in wpc.time.values
                               if str(t)[:16] in t_train])

    # Sort both by time
    train_sel = train_sel.sortby("time")
    wpc_sel   = wpc_sel.sortby("time")

    n_time = len(train_sel.time)
    n_lat  = len(train_sel.lat)
    n_lon  = len(train_sel.lon)

    # ERA5 TFP field and existing TFP-based label
    tfp       = train_sel["tfp_850"].values          # (T, lat, lon)
    tfp_label = train_sel["front_label"].values      # (T, lat, lon) TFP-based

    # WPC binary masks — dilated to handle ~30 km calibration offset
    struct = np.ones((2*dilate+1, 2*dilate+1), bool)   # square dilation kernel

    def get_wpc(var):
        if var in wpc_sel.data_vars:
            return wpc_sel[var].values.astype(bool)
        return np.zeros((n_time, n_lat, n_lon), bool)

    wpc_cf = get_wpc("CF")
    wpc_wf = get_wpc("WF")
    wpc_sf = get_wpc("SF")
    wpc_of = get_wpc("OF")

    print(f"    Dilating WPC masks by {dilate} cells (±{dilate*28:.0f} km)...")
    # 2D dilation per timestep, skip empty frames — much faster than 3D bulk dilation
    def dilate_3d(arr):
        out = np.zeros_like(arr)
        for t in range(arr.shape[0]):
            if arr[t].any():
                out[t] = binary_dilation(arr[t], struct)
        return out
    wpc_cf_d = dilate_3d(wpc_cf)
    wpc_wf_d = dilate_3d(wpc_wf)
    wpc_sf_d = dilate_3d(wpc_sf)
    wpc_of_d = dilate_3d(wpc_of)

    # ERA5 frontal zone mask: |TFP| > threshold
    tfp_front = np.abs(tfp) > tfp_thresh

    # Build hybrid label:
    #   Priority: OF > CF > WF > SF (OF rarest, assign first)
    #   Each type requires: TFP detects front AND WPC agrees (within dilate cells)
    hybrid = np.zeros((n_time, n_lat, n_lon), dtype=np.int8)
    hybrid[tfp_front & wpc_sf_d] = LABEL["SF"]   # stationary (lowest priority)
    hybrid[tfp_front & wpc_wf_d] = LABEL["WF"]
    hybrid[tfp_front & wpc_cf_d] = LABEL["CF"]
    hybrid[tfp_front & wpc_of_d] = LABEL["OF"]   # occluded (highest priority)

    # Statistics
    total = hybrid.size
    for lname, lval in LABEL.items():
        pct = (hybrid == lval).sum() / total * 100
        print(f"    {lname}: {(hybrid==lval).sum():,} pixels  ({pct:.2f}%)")

    # Agreement metrics
    tfp_any   = (tfp_label > 0).sum()
    hybrid_any = (hybrid > 0).sum()
    both      = ((tfp_label > 0) & (hybrid > 0)).sum()
    print(f"    TFP-only fronts discarded: {tfp_any - both:,} "
          f"({(tfp_any-both)/max(tfp_any,1)*100:.1f}% of TFP pixels filtered)")
    print(f"    OF pixels gained: {(hybrid==LABEL['OF']).sum():,} (new class)")

    # Build output dataset (keep all ERA5 input variables, replace label)
    ds_out = xr.Dataset(
        {
            "t850":        train_sel["t850"],
            "u850":        train_sel["u850"],
            "v850":        train_sel["v850"],
            "tfp_850":     train_sel["tfp_850"],
            "front_label": (["time","lat","lon"], hybrid),
            # Keep original TFP label for comparison
            "tfp_label":   (["time","lat","lon"], tfp_label.astype(np.int8)),
        },
        coords={"time": train_sel.time, "lat": train_sel.lat, "lon": train_sel.lon},
    )
    ds_out.attrs["description"] = (
        "Hybrid front labels: ERA5 TFP position intersected with WPC image-extracted "
        "front type. front_label: 0=BG 1=CF 2=WF 3=SF 4=OF. "
        "tfp_label: original TFP-only labels (0=BG 1=CF 2=WF 3=SF) for comparison."
    )
    ds_out.attrs["tfp_threshold"]  = str(tfp_thresh)
    ds_out.attrs["wpc_dilation"]   = f"{dilate} grid cells (~{dilate*28} km)"
    ds_out.attrs["label_encoding"] = "0=BG 1=CF 2=WF 3=SF 4=OF"
    ds_out.attrs["method"] = (
        "Intersection: TFP detects front zone (|TFP|>thresh) AND dilated WPC label "
        "present within radius → assign WPC type. Discards TFP-only pixels (over-detection) "
        "and WPC-only pixels (calibration artifacts)."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4}
                for v in ["front_label", "tfp_label"]}
    if out_path.exists():
        out_path.unlink()   # prevent xarray append-mode conflict on existing file
    ds_out.to_netcdf(out_path, encoding=encoding)
    size_mb = out_path.stat().st_size / 1e6
    print(f"    Saved: {out_path.name}  ({size_mb:.1f} MB)")
    train.close(); wpc.close()
    return out_path


def verify_hybrid(year: int, n_samples: int = 4):
    """Quick visual sanity check for hybrid labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    out_path = OUT_DIR / f"hybrid_{year}.nc"
    if not out_path.exists():
        print(f"No hybrid labels for {year}, run without --verify first")
        return

    ds = xr.open_dataset(out_path)
    LAT = ds.lat.values; LON = ds.lon.values

    proj = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(25,25))
    ext  = [-135, -55, 18, 62]
    lcolor = {1:"#1565C0", 2:"#C62828", 3:"#00897B", 4:"#7B1FA2"}
    lname  = {1:"CF", 2:"WF", 3:"SF", 4:"OF"}

    # Pick 4 timesteps with interesting frontal activity
    hybrid = ds["front_label"].values
    tfp_lb = ds["tfp_label"].values
    activity = [(hybrid[t]>0).sum() for t in range(len(ds.time))]
    idxs = sorted(range(len(activity)), key=lambda i: -activity[i])[:n_samples]

    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4*n_samples),
                              subplot_kw={"projection": proj})
    fig.patch.set_facecolor("#f8f9fa")

    for row, ti in enumerate(idxs):
        t_str = str(ds.time.values[ti])[:16]
        for col, (label_arr, title) in enumerate([
            (tfp_lb[ti],    f"TFP label  {t_str}"),
            (hybrid[ti],    f"Hybrid label  {t_str}")
        ]):
            ax = axes[row, col]
            ax.set_extent(ext, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
            ax.add_feature(cfeature.OCEAN, facecolor="#daeef5")
            ax.add_feature(cfeature.COASTLINE, lw=0.8, edgecolor="#333")
            ax.add_feature(cfeature.BORDERS,   lw=0.4, edgecolor="#888")
            ax.add_feature(cfeature.STATES,    lw=0.3, edgecolor="#aaa")
            ax.set_title(title, fontsize=10, fontweight="bold")
            for lv, lc in lcolor.items():
                ri, ci = np.where(label_arr == lv)
                if len(ri):
                    ax.scatter(LON[ci], LAT[ri], s=2, c=lc, alpha=0.8,
                               transform=ccrs.PlateCarree(), zorder=2, label=lname[lv])
            if row == 0 and col == 1:
                ax.legend(loc="lower left", fontsize=8, markerscale=4)

    fig.suptitle(f"Hybrid Labels Verification — {year}\n"
                 "Left: TFP-only  |  Right: Hybrid (TFP position + WPC type)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = Path(f"/Users/hayoungbong/Analysis/Front/figures/hybrid_verify_{year}.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Verification plot: {fig_path}")
    ds.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, required=True,
                   help="Years to process (must have ERA5 training NC and WPC labels NC)")
    p.add_argument("--tfp-thresh", type=float, default=TFP_THRESH_DEFAULT,
                   help=f"TFP threshold for frontal zone detection (default {TFP_THRESH_DEFAULT})")
    p.add_argument("--dilate", type=int, default=DILATE_CELLS,
                   help=f"WPC dilation radius in grid cells (default {DILATE_CELLS})")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--verify", type=int, metavar="YEAR",
                   help="Generate verification plot for YEAR after building")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building hybrid labels: {args.years}")
    print(f"  TFP threshold: {args.tfp_thresh}  |  WPC dilation: {args.dilate} cells")
    print()

    for year in args.years:
        try:
            build_hybrid_one_year(year,
                                  tfp_thresh=args.tfp_thresh,
                                  dilate=args.dilate,
                                  overwrite=args.overwrite)
        except FileNotFoundError as e:
            print(f"  {year}: SKIP — {e}")
        print()

    if args.verify:
        verify_hybrid(args.verify)


if __name__ == "__main__":
    main()
