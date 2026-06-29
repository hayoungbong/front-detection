"""
ERA5 regional download: 6-hourly, North America + surrounding oceans.

Domain: 5°N–85°N, 180°W–10°E  (generous ETC coverage)
  - Captures full Pacific/Atlantic ETC genesis and decay regions
  - Polar lows up to 85°N, subtropical fronts down to 5°N

Variables:
  pressure-level (500, 700, 850, 900, 925, 950, 1000 hPa): T, u, v, Z, q, omega
  single-level: 2mT, 2m dewpoint, 10m u/v, MSL, surface pressure

Download strategy:
  - Newest year first (2026 → year_start), month 12 → 1
  - Resume: skips existing .nc files, deletes and retries .tmp files
  - Safe to interrupt and restart at any time

Usage:
    python download_era5_regional.py              # 2019-2026
    python download_era5_regional.py 2019 2022   # specific year range
    python download_era5_regional.py 2020 2020   # single year
"""

import sys, time, logging, threading, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cdsapi

OUT_DIR  = Path('/Users/hayoungbong/Analysis/Front/data/era5')
PL_DIR   = OUT_DIR / 'pressure_level'
SFC_DIR  = OUT_DIR / 'single_level'
LOG_FILE = str(OUT_DIR / 'download_regional.log')

PL_DIR.mkdir(parents=True, exist_ok=True)
SFC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# 5°N–85°N, 180°W–10°E — generous ETC + polar low coverage
AREA   = [85, -180, 5, 10]   # [N, W, S, E]
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

# CDS API requires one client instance per thread
_tls = threading.local()

def get_client():
    if not hasattr(_tls, 'client'):
        _tls.client = cdsapi.Client(quiet=True)
    return _tls.client


def _download_one(kind: str, year: int, month: int) -> tuple:
    """Download a single month. Returns (label, success, message)."""
    label = f'{kind} {year}-{month:02d}'
    if kind == 'PL':
        path = PL_DIR / f'era5_PL_{year}{month:02d}.nc'
        tmp  = PL_DIR / f'era5_PL_{year}{month:02d}.nc.tmp'
        dataset = 'reanalysis-era5-pressure-levels'
        request = {
            'product_type': 'reanalysis',
            'variable': PL_VARS,
            'pressure_level': LEVELS,
            'year': str(year), 'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': HOURS, 'area': AREA,
            'data_format': 'netcdf', 'download_format': 'unarchived',
        }
    else:
        path = SFC_DIR / f'era5_SFC_{year}{month:02d}.nc'
        tmp  = SFC_DIR / f'era5_SFC_{year}{month:02d}.nc.tmp'
        dataset = 'reanalysis-era5-single-levels'
        request = {
            'product_type': 'reanalysis',
            'variable': SFC_VARS,
            'year': str(year), 'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': HOURS, 'area': AREA,
            'data_format': 'netcdf', 'download_format': 'unarchived',
        }

    if path.exists():
        return label, True, f'skip (already exists: {path.stat().st_size/1e9:.2f} GB)'

    if tmp.exists():
        tmp.unlink()
        log.warning('%s: stale .tmp found — deleted, retrying', label)

    t0 = time.time()
    try:
        get_client().retrieve(dataset, request, str(tmp))
        tmp.rename(path)
        elapsed = (time.time() - t0) / 60
        size_gb = path.stat().st_size / 1e9
        return label, True, f'{size_gb:.2f} GB  {elapsed:.1f} min'
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return label, False, str(e)


def download_all(year_start: int, year_end: int, max_workers: int = 6):
    """Submit all tasks at once — newest year first."""
    import datetime, collections
    current_ym = (datetime.date.today().year, datetime.date.today().month)

    tasks = []
    for year in range(year_end, year_start - 1, -1):
        for month in range(12, 0, -1):
            if (year, month) > current_ym:
                continue
            tasks.append(('PL',  year, month))
            tasks.append(('SFC', year, month))

    free_gb = shutil.disk_usage(str(OUT_DIR)).free / 1e9
    log.info('Disk free: %.0f GB', free_gb)
    log.info('=== Regional download %d→%d  (%d tasks, %d workers) ===',
             year_end, year_start, len(tasks), max_workers)

    t0 = time.time()
    year_counts = collections.defaultdict(lambda: {'ok': 0, 'skip': 0, 'err': 0})
    total_ok = total_skip = total_err = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, *t): t for t in tasks}
        for fut in as_completed(futures):
            label, success, msg = fut.result()
            year = int(label.split()[-1].split('-')[0])
            if success:
                if 'skip' in msg:
                    log.info('  ✓ skip  %s', label)
                    year_counts[year]['skip'] += 1
                    total_skip += 1
                else:
                    log.info('  ✓ done  %s  %s', label, msg)
                    year_counts[year]['ok'] += 1
                    total_ok += 1
            else:
                log.error('  ✗ fail  %s  %s', label, msg)
                year_counts[year]['err'] += 1
                total_err += 1

    elapsed = (time.time() - t0) / 60
    log.info('\n=== Summary ===')
    for year in sorted(year_counts, reverse=True):
        c = year_counts[year]
        log.info('  %d: new=%d  skip=%d  err=%d', year, c['ok'], c['skip'], c['err'])
    log.info('Total: new=%d  skip=%d  err=%d  %.1f min', total_ok, total_skip, total_err, elapsed)


def main():
    year_start = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    year_end   = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    workers    = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    download_all(year_start, year_end, max_workers=workers)


if __name__ == '__main__':
    main()
