"""
3-panel front comparison: TFP Physics | U-Net Prediction | WPC Analyst

Fetches ERA5 data from the ARCO public Zarr archive (no local ERA5 needed),
computes TFP-based physics labels, runs U-Net inference, and overlays the
WPC coded-surface analyst front lines — all in one Lambert Conformal figure.

Panels
------
  Left   : TFP zero-crossing + temperature-advection labels (physics baseline)
  Centre : U-Net pixel-level prediction (trained on TFP labels 2020-2021)
  Right  : WPC analyst hand-drawn fronts from coded_sfc bulletins

WPC data availability (local):
  2024-04-02 ~ 2024-04-07
  2026-06-12 ~ 2026-06-21

Key observation: TFP and U-Net tend to over-detect relative to WPC analyst,
which only draws meteorologically significant fronts. Switching the training
target to WPC labels is the logical next step for improving fidelity.

Usage
-----
    python compare_unet_wpc.py                        # all available WPC times
    python compare_unet_wpc.py --time 2024-04-03T00   # single timestep
    python compare_unet_wpc.py --period 2024           # 2024 period only
    python compare_unet_wpc.py --period 2026           # 2026 period only
"""

import sys, warnings, argparse
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import gcsfs
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from datetime import datetime

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.insert(0, str(Path(__file__).parent))
from parse_coded_sfc import load_period

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = Path('/Volumes/SSD_Hayoung/fronts/models')
CODED_SFC_ROOT = Path('/Volumes/SSD_Hayoung/fronts/coded_sfc')
OUT_DIR        = Path('/Users/hayoungbong/Analysis/Front/figures/compare_unet_wpc')

EXTENT      = [-135, -50, 22, 63]          # display domain [W E S N]
GRAD_THRESH = 0.006
TADV_THRESH = 0.5e-5
TARGET_KM   = 400
BUF         = 8                            # buffer degrees for TFP edge suppression

# Training domain (must match training data)
LAT_MIN, LAT_MAX = 15.0, 70.0
LON_MIN, LON_MAX = -170.0, -50.0          # western longitudes

FRONT_COLORS = {'CF': '#1565C0', 'WF': '#C62828', 'SF': '#2E7D32',
                'OF': '#6A1B9A', 'TROF': '#E65100'}


# ── U-Net (must match train_unet.py) ─────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,   base)
        self.enc2 = ConvBlock(base,    base*2)
        self.enc3 = ConvBlock(base*2,  base*4)
        self.enc4 = ConvBlock(base*4,  base*8)
        self.bot  = ConvBlock(base*8,  base*16)
        self.pool = nn.MaxPool2d(2)
        self.up4  = nn.ConvTranspose2d(base*16, base*8,  2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8)
        self.up3  = nn.ConvTranspose2d(base*8,  base*4,  2, stride=2)
        self.dec3 = ConvBlock(base*8,  base*4)
        self.up2  = nn.ConvTranspose2d(base*4,  base*2,  2, stride=2)
        self.dec2 = ConvBlock(base*4,  base*2)
        self.up1  = nn.ConvTranspose2d(base*2,  base,    2, stride=2)
        self.dec1 = ConvBlock(base*2,  base)
        self.head = nn.Conv2d(base, num_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def load_model():
    ckpts = sorted(MODEL_DIR.glob('unet_e30_b8_best.pt'))
    if not ckpts:
        ckpts = sorted(MODEL_DIR.glob('unet_*_best.pt'))
    ckpt_path = ckpts[-1]
    print(f'Model: {ckpt_path.name}')
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['norm_stats'], device


# ── Physics ───────────────────────────────────────────────────────────────────
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


def compute_tfp_grads(T, lats, lons):
    R = 6371.0
    lons2d, lats2d = np.meshgrid(lons, lats)
    lat_r, lon_r = np.deg2rad(lats2d), np.deg2rad(lons2d)
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


def tfp_labels(TFP, u, v, dTdx, dTdy, gm):
    sign = np.sign(TFP)
    hc = np.zeros_like(TFP, dtype=bool); vc = np.zeros_like(TFP, dtype=bool)
    hc[:, :-1] = (sign[:, :-1] * sign[:, 1:]) < 0
    vc[:-1, :] = (sign[:-1, :] * sign[1:, :]) < 0
    mask = (hc | vc) & (gm > GRAD_THRESH)
    tadv = -(u * dTdx / 1000.0 + v * dTdy / 1000.0)
    label = np.zeros(TFP.shape, dtype=np.int8)
    label[mask & (tadv < -TADV_THRESH)] = 1   # CF
    label[mask & (tadv >  TADV_THRESH)] = 2   # WF
    label[mask & (np.abs(tadv) <= TADV_THRESH)] = 3  # SF
    return label


# ── ERA5 fetch ────────────────────────────────────────────────────────────────
def fetch_era5(ds, time_str):
    dt = datetime.strptime(time_str, '%Y-%m-%dT%H')
    # Full training domain + buffer for TFP
    lat_s = slice(LAT_MAX + BUF, LAT_MIN - BUF)
    lon_360_min = (LON_MIN + 360) % 360
    lon_360_max = (LON_MAX + 360) % 360
    lon_s = slice(lon_360_min - BUF, lon_360_max + BUF)

    T_K = ds['temperature'].sel(time=dt, level=850, latitude=lat_s, longitude=lon_s
                                ).values.squeeze().astype(np.float32)
    u   = ds['u_component_of_wind'].sel(time=dt, level=850, latitude=lat_s, longitude=lon_s
                                        ).values.squeeze().astype(np.float32)
    v   = ds['v_component_of_wind'].sel(time=dt, level=850, latitude=lat_s, longitude=lon_s
                                        ).values.squeeze().astype(np.float32)

    lats = ds['latitude'].sel(latitude=lat_s).values
    lons = ds['longitude'].sel(longitude=lon_s).values
    return T_K - 273.15, u, v, lats, lons


# ── U-Net inference ───────────────────────────────────────────────────────────
def run_unet(model, norm, device, T_s, u, v, tfp, lats, lons):
    """Run U-Net on the full training domain, return prediction on training domain."""
    channels = []
    for arr, key in zip([T_s + 273.15, u, v, tfp],
                        ['t850', 'u850', 'v850', 'tfp_850']):
        mu, sigma = norm[key]
        channels.append((arr.astype(np.float32) - mu) / sigma)

    x = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(device)
    H, W = x.shape[-2:]
    m = 16
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    if pH or pW:
        x = F.pad(x, (0, pW, 0, pH), mode='reflect')

    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(dim=1).squeeze().cpu().numpy()[:H, :W]
    return pred


# ── Plot ──────────────────────────────────────────────────────────────────────
LABEL_COLORS = {0: '#d4d4d4', 1: '#1565C0', 2: '#C62828', 3: '#2E7D32'}

def scatter_labels(ax, label_map, lons2d, lats2d):
    for cls, color in [(1,'#1565C0'), (2,'#C62828'), (3,'#2E7D32')]:
        ys, xs = np.where(label_map == cls)
        if len(ys):
            ax.scatter(lons2d[ys, xs], lats2d[ys, xs],
                       s=4, color=color, alpha=0.85,
                       transform=ccrs.PlateCarree(), zorder=3)


def make_figure(time_str, T_s, lats, lons, phys_label, unet_pred, wpc_fronts):
    lons2d, lats2d = np.meshgrid(lons, lats)
    proj = ccrs.LambertConformal(central_longitude=-97, central_latitude=38)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8),
                             subplot_kw={'projection': proj})
    fig.suptitle(
        f'850hPa Front Detection  |  {time_str.replace("T"," ")}Z\n'
        f'Left: TFP Physics  |  Center: U-Net (trained 2020-2021)  |  Right: WPC Analyst',
        fontsize=12, fontweight='bold')

    for ax in axes:
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE,  linewidth=0.7)
        ax.add_feature(cfeature.BORDERS,    linewidth=0.4, linestyle=':')
        ax.add_feature(cfeature.STATES,     linewidth=0.25, edgecolor='#999')
        ax.add_feature(cfeature.LAND,       facecolor='#f5f5f5', zorder=0)
        ax.add_feature(cfeature.OCEAN,      facecolor='#dceef8', zorder=0)
        ax.gridlines(linewidth=0.3, color='gray', alpha=0.4)
        ax.pcolormesh(lons2d, lats2d, T_s, cmap='RdBu_r', vmin=-30, vmax=30,
                      transform=ccrs.PlateCarree(), shading='auto', zorder=1, alpha=0.6)

    legend_pts = [
        Line2D([0],[0], color='#1565C0', marker='o', ms=5, ls='none', label='CF'),
        Line2D([0],[0], color='#C62828', marker='o', ms=5, ls='none', label='WF'),
        Line2D([0],[0], color='#2E7D32', marker='o', ms=5, ls='none', label='SF'),
    ]

    # Panel 1: TFP physics
    axes[0].set_title('TFP Physics', fontsize=11, fontweight='bold')
    scatter_labels(axes[0], phys_label, lons2d, lats2d)
    axes[0].legend(handles=legend_pts, loc='lower left', fontsize=8)

    # Panel 2: U-Net
    axes[1].set_title('U-Net Prediction', fontsize=11, fontweight='bold')
    scatter_labels(axes[1], unet_pred, lons2d, lats2d)
    axes[1].legend(handles=legend_pts, loc='lower left', fontsize=8)

    # Panel 3: WPC analyst
    axes[2].set_title('WPC Analyst', fontsize=11, fontweight='bold')
    drawn = set()
    for front in wpc_fronts:
        ft = front['type']
        if ft not in FRONT_COLORS: continue
        ls = '--' if ft == 'SF' else '-'
        axes[2].plot(front['lons'], front['lats'],
                     color=FRONT_COLORS[ft], lw=2.2, linestyle=ls,
                     transform=ccrs.PlateCarree(), zorder=3)
        drawn.add(ft)
    wpc_legend = [
        Line2D([0],[0], color=FRONT_COLORS[t], lw=2,
               ls='--' if t=='SF' else '-', label=t)
        for t in ['CF','WF','SF','OF','TROF'] if t in drawn
    ]
    if wpc_legend:
        axes[2].legend(handles=wpc_legend, loc='lower left', fontsize=8)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time',   type=str, default=None,
                        help='single timestep YYYY-MM-DDTHH')
    parser.add_argument('--period', choices=['2024','2026','all'], default='all')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading model...')
    model, norm, device = load_model()

    print('Loading WPC coded_sfc...')
    all_wpc = load_period(CODED_SFC_ROOT)
    all_wpc = [w for w in all_wpc if w['valid_time'] and w['valid_time'].hour % 6 == 0]

    if args.time:
        dt_target = datetime.strptime(args.time, '%Y-%m-%dT%H')
        all_wpc = [w for w in all_wpc if w['valid_time'] == dt_target]
    elif args.period == '2024':
        all_wpc = [w for w in all_wpc if w['valid_time'].year == 2024]
    elif args.period == '2026':
        all_wpc = [w for w in all_wpc if w['valid_time'].year == 2026]

    all_wpc = sorted(all_wpc, key=lambda x: x['valid_time'])
    print(f'Timesteps to process: {len(all_wpc)}')

    print('Connecting to ARCO ERA5...')
    fs = gcsfs.GCSFileSystem(token='anon')
    ds = xr.open_zarr(
        fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'),
        chunks=None, consolidated=True)

    for wpc in all_wpc:
        dt = wpc['valid_time']
        time_str = dt.strftime('%Y-%m-%dT%H')
        print(f'  {time_str} ...', end=' ', flush=True)

        try:
            T_C, u, v, lats_buf, lons_buf = fetch_era5(ds, time_str)
        except Exception as e:
            print(f'ERA5 error: {e}')
            continue

        # Training domain indices within buffered domain
        lons_w = lons_buf - 360  # convert 0-360 → western
        i_lat = np.where((lats_buf >= LAT_MIN) & (lats_buf <= LAT_MAX))[0]
        i_lon = np.where((lons_w   >= LON_MIN) & (lons_w   <= LON_MAX))[0]

        T_s_buf = adaptive_smooth(T_C, lats_buf)
        tfp_buf, gm_buf, dTdx_buf, dTdy_buf = compute_tfp_grads(
            T_s_buf, lats_buf, lons_buf)

        # Crop to training domain
        T_s   = T_s_buf[np.ix_(i_lat, i_lon)]
        u_c   = u[np.ix_(i_lat, i_lon)]
        v_c   = v[np.ix_(i_lat, i_lon)]
        tfp_c = tfp_buf[np.ix_(i_lat, i_lon)]
        gm_c  = gm_buf[np.ix_(i_lat, i_lon)]
        dTdx_c = dTdx_buf[np.ix_(i_lat, i_lon)]
        dTdy_c = dTdy_buf[np.ix_(i_lat, i_lon)]
        lats  = lats_buf[i_lat]
        lons  = lons_w[i_lon]

        phys  = tfp_labels(tfp_c, u_c, v_c, dTdx_c, dTdy_c, gm_c)
        unet  = run_unet(model, norm, device, T_s, u_c, v_c, tfp_c, lats, lons)

        fig = make_figure(time_str, T_s, lats, lons, phys, unet, wpc['fronts'])
        out = OUT_DIR / f'unet_vs_wpc_{dt.strftime("%Y%m%d%H")}.png'
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'saved → {out.name}')

    print(f'\nDone. Figures in {OUT_DIR}')


if __name__ == '__main__':
    main()
