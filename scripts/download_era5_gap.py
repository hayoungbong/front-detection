"""
ERA5 regional download for NASA Discover — fills gap before /css/era5/ coverage.

Domain  : 85°N–5°N, 180°W–10°E  (North America + North Atlantic)
          Same as Mac download_era5_regional.py

/css/era5/ actual coverage (verified 2026-06-29):
  pressure-level : Y2018/M11–M12 only, then 2019–present  (M01–M10 2018 missing!)
  single-level   : Y2007/M01–present  (may be partial)

Default fills the gap safely:
  PL  1940–2018  (covers 2018 M01–M10 missing from CSS)
  SFC 1940–2007  (covers full 2007 in case CSS is partial)

Storage estimate (regional, North America + North Atlantic):
  PL  per year  ~27 GB   → 1940–2018 (79 yr) ~2.1 TB
  SFC per year  ~ 4 GB   → 1940–2007 (68 yr) ~0.3 TB

Prerequisites:
  pip install --user cdsapi
  ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <your-CDS-API-key>   # https://cds.climate.copernicus.eu/profile

Usage (from login node — login nodes have internet):
  screen -S era5gap
  module load python/GEOSpyD/24.11.3-0/3.12
  python download_era5_gap.py                          # PL 1940-2017, SFC 1940-2006
  python download_era5_gap.py --pl  2000 2017          # PL only, 2000-2017
  python download_era5_gap.py --pl  2000 2017 --no-sfc # PL only
  python download_era5_gap.py --workers 4              # fewer parallel requests
"""

import argparse
import datetime
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Output paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path('/home/hbong/nobackup/DATA/ERA5/download/front')
PL_DIR   = BASE_DIR / 'pressure_level'
SFC_DIR  = BASE_DIR / 'single_level'
LOG_FILE = BASE_DIR / 'download_gap.log'

# Safe defaults (CSS PL only has 2018 M11-M12; SFC 2007 may be partial)
CSS_PL_FIRST_YEAR  = 2019   # download through 2018 to fill M01-M10 gap
CSS_SFC_FIRST_YEAR = 2008   # download through 2007 to be safe

# ── ERA5 request parameters (identical to Mac download_era5_regional.py) ───────
AREA   = [85, -180, 5, 10]   # [N, W, S, E] — North America + North Atlantic
HOURS  = ['00:00', '06:00', '12:00', '18:00']
LEVELS = ['500', '700', '850', '900', '925', '950', '1000']

PL_VARS = [
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'geopotential',
    'specific_humidity',
    'vertical_velocity',
]
SFC_VARS = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    'surface_pressure',
]

# ── Logging ────────────────────────────────────────────────────────────────────
for d in (PL_DIR, SFC_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

_tls = threading.local()

def get_client():
    if not hasattr(_tls, 'client'):
        import cdsapi
        _tls.client = cdsapi.Client(quiet=True)
    return _tls.client


# ── Download one month ─────────────────────────────────────────────────────────
def _download_one(kind: str, year: int, month: int) -> tuple:
    label = f'{kind} {year}-{month:02d}'
    if kind == 'PL':
        path    = PL_DIR  / f'era5_PL_{year}{month:02d}.nc'
        tmp     = PL_DIR  / f'era5_PL_{year}{month:02d}.nc.tmp'
        dataset = 'reanalysis-era5-pressure-levels'
        request = {
            'product_type':    ['reanalysis'],
            'variable':         PL_VARS,
            'pressure_level':   LEVELS,
            'year':  [str(year)],
            'month': [f'{month:02d}'],
            'day':   [f'{d:02d}' for d in range(1, 32)],
            'time':  HOURS,
            'area':  AREA,
            'data_format':     'netcdf',
            'download_format': 'unarchived',
        }
    else:
        path    = SFC_DIR / f'era5_SFC_{year}{month:02d}.nc'
        tmp     = SFC_DIR / f'era5_SFC_{year}{month:02d}.nc.tmp'
        dataset = 'reanalysis-era5-single-levels'
        request = {
            'product_type': ['reanalysis'],
            'variable':      SFC_VARS,
            'year':  [str(year)],
            'month': [f'{month:02d}'],
            'day':   [f'{d:02d}' for d in range(1, 32)],
            'time':  HOURS,
            'area':  AREA,
            'data_format':     'netcdf',
            'download_format': 'unarchived',
        }

    if path.exists():
        return label, True, f'skip ({path.stat().st_size/1e9:.2f} GB)'
    if tmp.exists():
        tmp.unlink()
        log.warning('%s: stale .tmp removed, retrying', label)

    t0 = time.time()
    try:
        get_client().retrieve(dataset, request).download(str(tmp))
        tmp.rename(path)
        elapsed = (time.time() - t0) / 60
        size_gb = path.stat().st_size / 1e9
        return label, True, f'{size_gb:.2f} GB  {elapsed:.1f} min'
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return label, False, str(e)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_download(pl_range, sfc_range, workers: int):
    today = (datetime.date.today().year, datetime.date.today().month)

    tasks = []
    if pl_range:
        start, end = pl_range
        for year in range(end, start - 1, -1):
            for month in range(12, 0, -1):
                if (year, month) <= today:
                    tasks.append(('PL', year, month))

    if sfc_range:
        start, end = sfc_range
        for year in range(end, start - 1, -1):
            for month in range(12, 0, -1):
                if (year, month) <= today:
                    tasks.append(('SFC', year, month))

    if not tasks:
        log.info('Nothing to download.')
        return

    n_pl  = sum(1 for t in tasks if t[0] == 'PL')
    n_sfc = sum(1 for t in tasks if t[0] == 'SFC')
    est_gb = n_pl * 2.25 + n_sfc * 0.33   # regional estimates
    free_gb = shutil.disk_usage(str(BASE_DIR)).free / 1e9

    log.info('=== ERA5 regional gap download ===')
    log.info('Domain   : 85N–5N, 180W–10E (North America + North Atlantic)')
    log.info('Output   : %s', BASE_DIR)
    log.info('Tasks    : %d total  (PL=%d  SFC=%d)', len(tasks), n_pl, n_sfc)
    log.info('Est size : ~%.0f GB', est_gb)
    log.info('Disk free: %.0f GB', free_gb)
    if free_gb < est_gb:
        log.warning('WARNING: estimated download (%.0f GB) may exceed free disk (%.0f GB)',
                    est_gb, free_gb)

    t0 = time.time()
    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, *t): t for t in tasks}
        for fut in as_completed(futures):
            label, success, msg = fut.result()
            if success:
                if 'skip' in msg:
                    log.info('  skip  %s  %s', label, msg)
                    skip += 1
                else:
                    log.info('  done  %s  %s', label, msg)
                    ok += 1
            else:
                log.error('  FAIL  %s  %s', label, msg)
                err += 1

    elapsed = (time.time() - t0) / 60
    log.info('=== Done: new=%d  skip=%d  error=%d  %.1f min ===', ok, skip, err, elapsed)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pl',  nargs=2, type=int, metavar=('START', 'END'),
                   default=[1940, CSS_PL_FIRST_YEAR - 1],
                   help=f'PL year range (default 1940–{CSS_PL_FIRST_YEAR-1})')
    p.add_argument('--sfc', nargs=2, type=int, metavar=('START', 'END'),
                   default=[1940, CSS_SFC_FIRST_YEAR - 1],
                   help=f'SFC year range (default 1940–{CSS_SFC_FIRST_YEAR-1})')
    p.add_argument('--no-pl',  action='store_true', help='skip pressure-level download')
    p.add_argument('--no-sfc', action='store_true', help='skip single-level download')
    p.add_argument('--workers', type=int, default=4,
                   help='parallel CDS requests (default 4; CDS queues anyway)')
    args = p.parse_args()

    pl_range  = None if args.no_pl  else args.pl
    sfc_range = None if args.no_sfc else args.sfc
    run_download(pl_range, sfc_range, args.workers)


if __name__ == '__main__':
    main()
