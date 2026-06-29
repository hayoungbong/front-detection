"""
U-Net training dataset generation from ERA5.

Source:
  Local ERA5 files (/Volumes/SSD_Hayoung/ERA5/pressure_level/)
  ARCO ERA5 fallback for years without local files.

Processing:
  850hPa T smoothing → TFP → temperature advection →
  CF/WF/SF label assignment → annual NetCDF output

Output:
  /Volumes/SSD_Hayoung/fronts/training/era5_{year}_training.nc
    Variables: t850, u850, v850, tfp_850, front_label
    front_label: 0=background, 1=CF, 2=WF, 3=SF

Usage:
    python build_training_data.py 2020 2022
    python build_training_data.py 2020 2022 --test   # January only per year
"""

import sys, warnings, argparse
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from datetime import datetime

ERA5_DIR  = Path('/Users/hayoungbong/Analysis/Front/data/era5')
PL_DIR    = ERA5_DIR / 'pressure_level'
OUT_DIR   = Path('/Users/hayoungbong/Analysis/Front/data/training')

TARGET_KM   = 400
GRAD_THRESH = 0.006    # minimum |∇T| for front detection (K/km)
TADV_THRESH = 0.5e-5   # temperature advection threshold for CF/WF separation (K/s)

# Output domain (matched to WPC North America coverage)
LAT_MIN, LAT_MAX = 15.0, 70.0
LON_MIN, LON_MAX = -170.0, -50.0

BG, CF, WF, SF = 0, 1, 2, 3


# ── Physics functions ──────────────────────────────────────────────────────────
def adaptive_smooth(T, lats, target_km=TARGET_KM):
    T = T.copy().astype(np.float32)
    dlat = abs(float(lats[1]) - float(lats[0]))
    sigma_ns = target_km / (dlat * 111.0)
    T = gaussian_filter1d(T, sigma=sigma_ns, axis=0, mode='nearest')
    for i, lat in enumerate(lats):
        cos_lat = max(np.cos(np.deg2rad(abs(float(lat)))), 0.09)
        sigma_ew = min(target_km / (dlat * 111.0 * cos_lat), 120)
        T[i] = gaussian_filter1d(T[i], sigma=sigma_ew, mode='nearest')
    return T


def compute_tfp_and_grads(T, lats, lons):
    """Return TFP, |∇T|, ∂T/∂x, ∂T/∂y (all in K/km)."""
    R = 6371.0
    lons2d, lats2d = np.meshgrid(lons, lats)
    lat_r = np.deg2rad(lats2d)
    lon_r = np.deg2rad(lons2d)
    dy = R * np.diff(lat_r, axis=0, append=lat_r[-1:])
    dx = R * np.cos(lat_r) * np.diff(lon_r, axis=1, append=lon_r[:, -1:])
    dx = np.where(np.abs(dx) < 5.0, np.sign(dx + 1e-9) * 5.0, dx)
    dy = np.where(np.abs(dy) < 1e-3, 1e-3, dy)
    dTdx = np.diff(T, axis=1, append=T[:, -1:]) / dx
    dTdy = np.diff(T, axis=0, append=T[-1:])    / dy
    mag  = np.sqrt(dTdx**2 + dTdy**2)
    mag  = np.where(mag < 1e-10, 1e-10, mag)
    ux, uy = dTdx / mag, dTdy / mag
    dmx = np.diff(mag, axis=1, append=mag[:, -1:]) / dx
    dmy = np.diff(mag, axis=0, append=mag[-1:])    / dy
    tfp = -(dmx * ux + dmy * uy) * 1e4
    return tfp, mag, dTdx, dTdy


def make_front_label(TFP, u, v, dTdx, dTdy, grad_mag):
    """Assign CF/WF/SF labels via TFP zero-crossing + temperature advection (int8).
    Returns (label, tadv) — tadv saved separately as a continuous regression target."""
    sign = np.sign(TFP)
    hcross = np.zeros_like(TFP, dtype=bool)
    vcross = np.zeros_like(TFP, dtype=bool)
    hcross[:, :-1] = (sign[:, :-1] * sign[:, 1:]) < 0
    vcross[:-1, :] = (sign[:-1, :] * sign[1:, :]) < 0
    front_mask = (hcross | vcross) & (grad_mag > GRAD_THRESH)

    # u,v in m/s; dTdx/dTdy in K/km → divide by 1000 to get K/s
    tadv = -(u * dTdx / 1000.0 + v * dTdy / 1000.0)

    label = np.zeros(TFP.shape, dtype=np.int8)
    label[front_mask & (tadv < -TADV_THRESH)] = CF
    label[front_mask & (tadv >  TADV_THRESH)] = WF
    label[front_mask & (np.abs(tadv) <= TADV_THRESH)] = SF
    return label, tadv


def mask_edges(arr, n=4):
    out = arr.copy()
    out[:n, :] = 0; out[-n:, :] = 0
    out[:, :n] = 0; out[:, -n:] = 0
    return out


# ── Data sources ───────────────────────────────────────────────────────────────
def open_local(year: int, month: int):
    path = PL_DIR / f'era5_PL_{year}{month:02d}.nc'
    if not path.exists():
        return None
    return xr.open_dataset(path)


_arco_ds = None
def open_arco():
    global _arco_ds
    if _arco_ds is None:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token='anon')
        _arco_ds = xr.open_zarr(
            fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'),
            chunks={}, consolidated=True)
    return _arco_ds


# ── Per-month processing ───────────────────────────────────────────────────────
def process_month(year: int, month: int) -> tuple | None:
    """
    Process one month. Uses local files if available, falls back to ARCO.
    Returns (times, lats, lons, t850, u850, v850, tfp, labels) or None on failure.
    """
    ds_local = open_local(year, month)

    if ds_local is not None:
        lons_raw = ds_local.longitude.values if 'longitude' in ds_local.coords else ds_local.lon.values
        if lons_raw.max() > 180:
            ds_local = ds_local.assign_coords(
                longitude=((ds_local.longitude + 180) % 360 - 180)
            ).sortby('longitude')

        lat_dim  = 'latitude'  if 'latitude'  in ds_local.coords else 'lat'
        lon_dim  = 'longitude' if 'longitude' in ds_local.coords else 'lon'
        lev_dim  = 'pressure_level' if 'pressure_level' in ds_local.coords else 'level'
        time_dim = 'valid_time' if 'valid_time' in ds_local.coords else 'time'

        sub = ds_local.sel({
            lat_dim: slice(LAT_MAX, LAT_MIN),
            lon_dim: slice(LON_MIN, LON_MAX),
            lev_dim: 850,
        })
        sub = sub.isel({time_dim: sub[time_dim].dt.hour.isin([0, 6, 12, 18]).values})

        times = sub[time_dim].values
        lats  = sub[lat_dim].values
        lons  = sub[lon_dim].values

        import time as _time
        t0 = _time.time()
        t_raw = sub['t'].values.astype(np.float32) - 273.15
        u_arr = sub['u'].values.astype(np.float32)
        v_arr = sub['v'].values.astype(np.float32)
        print(f'local load: {_time.time()-t0:.1f}s', end=' ', flush=True)
        ds_local.close()

    else:
        print(f'    (ARCO fallback: {year}-{month:02d})', flush=True)
        ds = open_arco()
        lat_s = slice(LAT_MAX, LAT_MIN)
        lon_s_360 = slice(LON_MIN + 360, LON_MAX + 360)

        import pandas as pd
        month_times = pd.date_range(f'{year}-{month:02d}-01', periods=1, freq='MS')
        sts = month_times[0]
        ets = (sts + pd.DateOffset(months=1))
        time_s = slice(sts, ets - pd.Timedelta(hours=1))

        sub = ds[['temperature', 'u_component_of_wind', 'v_component_of_wind']].sel(
            latitude=lat_s, longitude=lon_s_360, level=850, time=time_s)
        sub = sub.isel(time=sub.time.dt.hour.isin([0, 6, 12, 18]))

        import time as _time
        t0 = _time.time()
        loaded = sub.compute()
        print(f'ARCO load: {_time.time()-t0:.1f}s', end=' ', flush=True)

        times = loaded.time.values
        lats  = loaded.latitude.values
        lons  = loaded.longitude.values - 360

        t_raw = loaded['temperature'].values.astype(np.float32) - 273.15
        u_arr = loaded['u_component_of_wind'].values.astype(np.float32)
        v_arr = loaded['v_component_of_wind'].values.astype(np.float32)

    n = len(times)
    t850_out     = np.empty((n,) + t_raw.shape[1:], dtype=np.float32)
    tfp_out      = np.empty_like(t850_out)
    tadv_out     = np.empty_like(t850_out)
    grad_mag_out = np.empty_like(t850_out)
    u850_out  = u_arr.astype(np.float32)
    v850_out  = v_arr.astype(np.float32)
    label_out = np.empty((n,) + t_raw.shape[1:], dtype=np.int8)

    for i in range(n):
        T_s = adaptive_smooth(t_raw[i], lats)
        tfp, gm, dTdx, dTdy = compute_tfp_and_grads(T_s, lats, lons)
        tfp_m = mask_edges(tfp)
        label, tadv = make_front_label(tfp_m, u_arr[i], v_arr[i], dTdx, dTdy, gm)
        t850_out[i]     = T_s + 273.15
        tfp_out[i]      = tfp
        label_out[i]    = label
        tadv_out[i]     = tadv
        grad_mag_out[i] = gm

    return times, lats, lons, t850_out, u850_out, v850_out, tfp_out, label_out, tadv_out, grad_mag_out


# ── Per-year processing ────────────────────────────────────────────────────────
def process_year(year: int, months: list[int]):
    out_file = OUT_DIR / f'era5_{year}_training.nc'
    if out_file.exists():
        print(f'{year}: already exists, skipping ({out_file.stat().st_size/1e9:.2f} GB)')
        return

    all_times = []; all_t850 = []; all_u850 = []; all_v850 = []
    all_tfp   = []; all_labels = []; all_tadv = []; all_grad_mag = []
    lats_ref = lons_ref = None

    for month in months:
        print(f'  {year}-{month:02d} processing...', end=' ', flush=True)
        result = process_month(year, month)
        if result is None:
            print('failed, skipping')
            continue
        times, lats, lons, t850, u850, v850, tfp, labels, tadv, grad_mag = result
        lats_ref = lats; lons_ref = lons
        all_times.append(times); all_t850.append(t850)
        all_u850.append(u850);   all_v850.append(v850)
        all_tfp.append(tfp);     all_labels.append(labels)
        all_tadv.append(tadv);   all_grad_mag.append(grad_mag)
        total = labels.size
        cf_pct = 100*(labels==CF).sum()/total
        wf_pct = 100*(labels==WF).sum()/total
        sf_pct = 100*(labels==SF).sum()/total
        print(f'{len(times)} steps  CF:{cf_pct:.2f}% WF:{wf_pct:.2f}% SF:{sf_pct:.2f}%')

    if not all_times:
        print(f'{year}: no data processed')
        return

    print(f'{year}: saving...', end=' ', flush=True)
    times_all    = np.concatenate(all_times)
    t850_all     = np.concatenate(all_t850).astype(np.float32)
    u850_all     = np.concatenate(all_u850).astype(np.float32)
    v850_all     = np.concatenate(all_v850).astype(np.float32)
    tfp_all      = np.concatenate(all_tfp).astype(np.float32)
    labels_all   = np.concatenate(all_labels).astype(np.int8)
    tadv_all     = np.concatenate(all_tadv).astype(np.float32)
    grad_mag_all = np.concatenate(all_grad_mag).astype(np.float32)

    ds_out = xr.Dataset(
        {
            't850':        (['time','lat','lon'], t850_all,
                            {'units':'K', 'long_name':'850hPa Temperature (smoothed)'}),
            'u850':        (['time','lat','lon'], u850_all,
                            {'units':'m/s', 'long_name':'850hPa U-wind'}),
            'v850':        (['time','lat','lon'], v850_all,
                            {'units':'m/s', 'long_name':'850hPa V-wind'}),
            'tfp_850':     (['time','lat','lon'], tfp_all,
                            {'units':'K/(100km)^2', 'long_name':'Thermal Front Parameter'}),
            'front_label': (['time','lat','lon'], labels_all,
                            {'units':'', 'long_name':'Front Type Label (classification target)',
                             'flag_values':'0 1 2 3',
                             'flag_meanings':'background CF WF SF'}),
            'tadv_850':    (['time','lat','lon'], tadv_all,
                            {'units':'K/s', 'long_name':'850hPa Temperature Advection -v·∇T (regression target)'}),
            'grad_mag_850':(['time','lat','lon'], grad_mag_all,
                            {'units':'K/km', 'long_name':'850hPa Temperature Gradient Magnitude |∇T| (regression target)'}),
        },
        coords={'time': times_all, 'lat': lats_ref, 'lon': lons_ref}
    )
    ds_out.attrs['description'] = (
        f'ERA5 850hPa U-Net training data ({year}). '
        'Input: t850, u850, v850. '
        'Classification target: front_label (0=BG,1=CF,2=WF,3=SF). '
        'Regression targets: tfp_850, tadv_850, grad_mag_850.'
    )
    encoding = {v: {'dtype':'float32','zlib':True,'complevel':4}
                for v in ['t850','u850','v850','tfp_850','tadv_850','grad_mag_850']}
    encoding['front_label'] = {'dtype':'int8','zlib':True,'complevel':4}

    ds_out.to_netcdf(out_file, encoding=encoding)
    size_gb = out_file.stat().st_size / 1e9
    print(f'done → {out_file.name}  ({size_gb:.2f} GB)')

    total = labels_all.size
    print(f'  Label distribution: BG={100*(labels_all==BG).sum()/total:.2f}%  '
          f'CF={100*(labels_all==CF).sum()/total:.2f}%  '
          f'WF={100*(labels_all==WF).sum()/total:.2f}%  '
          f'SF={100*(labels_all==SF).sum()/total:.2f}%')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year_start', type=int, nargs='?', default=2020)
    parser.add_argument('year_end',   type=int, nargs='?', default=2022)
    parser.add_argument('--test', action='store_true', help='January only per year')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    months = [1] if args.test else list(range(1, 13))

    print(f'=== ERA5 training data generation: {args.year_start}–{args.year_end} ===')
    print(f'Source: local {PL_DIR}  (ARCO fallback if missing)')
    print(f'Output: {OUT_DIR}')
    print(f'Months: {months}')
    print()

    for year in range(args.year_start, args.year_end + 1):
        local_files = list(PL_DIR.glob(f'era5_PL_{year}*.nc'))
        n_local = len([f for f in local_files if not f.name.endswith('.tmp')])
        if n_local == 0 and year <= 2022:
            print(f'{year}: no local files, skipping')
            continue
        print(f'\n=== {year} (local: {n_local} months) ===')
        process_year(year, months)

    print('\nAll done.')


if __name__ == '__main__':
    main()
