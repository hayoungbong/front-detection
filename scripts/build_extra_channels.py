"""
Build Extra Channel Files from ERA5 regional
=============================================
Extracts 8 variables from ERA5 regional monthly files (5-85N, 180W-10E),
crops to training domain (15-70N, 170-50W), and saves annual NetCDF files
used as 12-channel input to train_unet_v4.py.

Output variables (per year, training domain 221x481):
  z500  : geopotential at 500 hPa         [m²/s²]
  q850  : specific humidity at 850 hPa    [kg/kg]
  w850  : vertical velocity at 850 hPa    [Pa/s]
  msl   : mean sea level pressure         [Pa]
  t925  : temperature at 925 hPa          [K]
  t2m   : 2-metre temperature             [K]
  u10   : 10-metre U wind component       [m/s]
  v10   : 10-metre V wind component       [m/s]

Output: /Volumes/SSD_Hayoung/fronts/training/extra_channels_YYYY.nc

Usage:
  python build_extra_channels.py --years 2019 2020 2021 2022 2023 2024 2025
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ERA5_PL  = Path("/Volumes/SSD_Hayoung/ERA5/pressure_level")
ERA5_SFC = Path("/Volumes/SSD_Hayoung/ERA5/single_level")
OUT_DIR  = Path("/Volumes/SSD_Hayoung/fronts/training")

# ── Training domain (matches era5_YYYY_training.nc: 221×481) ───────────────
# Regional ERA5 lat: 85→5°N descending (N first); lon: -180→10°E ascending
# Training data lat is S→N ascending (15→70°N), so we crop then flip.
LAT_N, LAT_S = 70.0, 15.0
LON_W, LON_E = -170.0, -50.0


def crop_domain(da, lat_name="latitude", lon_name="longitude"):
    """Crop DataArray to training domain and flip lat to S→N."""
    da = da.sel(
        {lat_name: slice(LAT_N, LAT_S),   # N→S order, so N first
         lon_name: slice(LON_W, LON_E)}
    )
    return da.isel({lat_name: slice(None, None, -1)})


def build_extra_one_year(year: int, overwrite: bool = False) -> Path:
    out_path = OUT_DIR / f"extra_channels_{year}.nc"
    if out_path.exists() and not overwrite:
        print(f"  {year}: already exists, skipping (--overwrite to force)")
        return out_path

    print(f"  {year}: loading ERA5 regional monthly files...")

    pl_files  = sorted(ERA5_PL.glob(f"era5_PL_{year}??.nc"))
    sfc_files = sorted(ERA5_SFC.glob(f"era5_SFC_{year}??.nc"))

    if not pl_files:
        raise FileNotFoundError(f"No PL files found for {year}")
    if not sfc_files:
        raise FileNotFoundError(f"No SFC files found for {year}")

    print(f"    PL months: {len(pl_files)}  SFC months: {len(sfc_files)}")

    # Read month-by-month to avoid large simultaneous I/O on external SSD
    z500_list, q850_list, w850_list, msl_list = [], [], [], []
    t925_list, t2m_list, u10_list, v10_list   = [], [], [], []

    # Match PL and SFC files by YYYYMM
    pl_map  = {f.name[8:14]: f for f in pl_files}   # era5_PL_YYYYMM.nc
    sfc_map = {f.name[9:15]: f for f in sfc_files}  # era5_SFC_YYYYMM.nc
    months  = sorted(set(pl_map) & set(sfc_map))

    for ym in months:
        print(f"    {ym}...", end=" ", flush=True)
        pl  = xr.open_dataset(pl_map[ym]).rename({"valid_time": "time"})
        sfc = xr.open_dataset(sfc_map[ym]).rename({"valid_time": "time"})

        def pl_var(v, lev):
            return crop_domain(
                pl[v].sel(pressure_level=lev).drop_vars("pressure_level")
            ).rename({"latitude": "lat", "longitude": "lon"}).load()

        def sfc_var(v):
            return crop_domain(sfc[v]).rename({"latitude": "lat", "longitude": "lon"}).load()

        z500_list.append(pl_var("z", 500.0))
        q850_list.append(pl_var("q", 850.0))
        w850_list.append(pl_var("w", 850.0))
        t925_list.append(pl_var("t", 925.0))
        msl_list.append(sfc_var("msl"))
        t2m_list.append(sfc_var("t2m"))
        u10_list.append(sfc_var("u10"))
        v10_list.append(sfc_var("v10"))

        pl.close(); sfc.close()
        print("ok")

    z500 = xr.concat(z500_list, dim="time")
    q850 = xr.concat(q850_list, dim="time")
    w850 = xr.concat(w850_list, dim="time")
    t925 = xr.concat(t925_list, dim="time")
    msl  = xr.concat(msl_list,  dim="time")
    t2m  = xr.concat(t2m_list,  dim="time")
    u10  = xr.concat(u10_list,  dim="time")
    v10  = xr.concat(v10_list,  dim="time")

    common_times = np.intersect1d(z500.time.values, msl.time.values)
    print(f"    Common timesteps: {len(common_times)}")

    z500 = z500.sel(time=common_times)
    q850 = q850.sel(time=common_times)
    w850 = w850.sel(time=common_times)
    t925 = t925.sel(time=common_times)
    msl  = msl.sel(time=common_times)
    t2m  = t2m.sel(time=common_times)
    u10  = u10.sel(time=common_times)
    v10  = v10.sel(time=common_times)

    # Build output dataset
    ds_out = xr.Dataset(
        {
            "z500": z500.astype("float32"),
            "q850": q850.astype("float32"),
            "w850": w850.astype("float32"),
            "msl":  msl.astype("float32"),
            "t925": t925.astype("float32"),
            "t2m":  t2m.astype("float32"),
            "u10":  u10.astype("float32"),
            "v10":  v10.astype("float32"),
        }
    )
    ds_out.attrs["description"] = (
        "Extra ERA5 regional channels for 12-channel U-Net Run 5. "
        "Training domain: 15-70N, 170-50W (221x481). 0.25 deg, 6-hourly. "
        "z500 [m2/s2], q850 [kg/kg], w850 [Pa/s], msl [Pa], "
        "t925 [K], t2m [K], u10 [m/s], v10 [m/s]."
    )
    ds_out.attrs["source"] = "ERA5 regional (Copernicus CDS, /Volumes/SSD_Hayoung/ERA5)"
    ds_out.attrs["year"] = str(year)

    print(f"    z500: {float(z500.mean()):.1f} m2/s2  (500hPa geopotential)")
    print(f"    q850: {float(q850.mean())*1000:.3f} g/kg  (850hPa specific humidity)")
    print(f"    w850: {float(w850.mean()):.4f} Pa/s  (850hPa vertical velocity)")
    print(f"    msl:  {float(msl.mean())/100:.1f} hPa  (mean sea level pressure)")
    print(f"    t925: {float(t925.mean())-273.15:.1f} °C  (925hPa temperature)")
    print(f"    t2m:  {float(t2m.mean())-273.15:.1f} °C  (2m temperature)")
    print(f"    u10:  {float(u10.mean()):.2f} m/s  (10m U wind)")
    print(f"    v10:  {float(v10.mean()):.2f} m/s  (10m V wind)")

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    ds_out.to_netcdf(out_path, encoding=encoding)
    size_mb = out_path.stat().st_size / 1e6
    print(f"    Saved: {out_path.name}  ({size_mb:.1f} MB)")

    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    print(f"Building extra channels: {args.years}")
    for year in args.years:
        try:
            build_extra_one_year(year, overwrite=args.overwrite)
        except FileNotFoundError as e:
            print(f"  {year}: SKIP — {e}")
        print()


if __name__ == "__main__":
    main()
