"""
U-Net Front Detection — Run 7 (12→11-channel REGRESSION, ERA5-only)
====================================================================
Instead of classifying discrete front types, this run predicts the continuous
ERA5-derived frontal diagnostic fields directly. No WPC labels, no arbitrary
threshold — threshold-free and climate-change robust (see REPORT §4.3, §7.2).

Inputs (11 channels, tfp_850 dropped to avoid target leakage):
  t850/u850/v850 (from era5_YYYY_training.nc)
  z500/q850/w850/msl/t925/t2m/u10/v10 (from extra_channels_YYYY.nc)

Targets (continuous, from era5_YYYY_training.nc — whichever are present):
  tfp_850       Thermal Front Parameter   (front location)
  tadv_850      temperature advection     (CF/WF sign emerges naturally)
  grad_mag_850  |∇T| at 850 hPa           (frontal intensity)

Each input and target is standardized (per-channel mean/std over the training
years). The model predicts standardized targets; metrics are reported as
normalized RMSE and Pearson correlation (scale-free) per target.

Usage:
  python train_unet_regression.py --train 2019 2024 --val 2025 2025 --epochs 30 --batch 32 \
      --data-root /discover/nobackup/projects/giss/paleofun/hbong/front/data
  python train_unet_regression.py --resume ...
"""

import os, argparse, time, csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path

_logfile = None
def log(msg):
    print(msg, flush=True)
    if _logfile:
        _logfile.write(msg + '\n'); _logfile.flush()

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths (overridable via --data-root or individual --*-dir flags) ────────
_DEFAULT_ROOT = Path('/Volumes/SSD_Hayoung/fronts')
EXTRA_DIR  = _DEFAULT_ROOT / 'training'    # holds era5_YYYY_training.nc + extra_channels_YYYY.nc
MODEL_DIR  = _DEFAULT_ROOT / 'models'

# ── Config ─────────────────────────────────────────────────────────────────
# 11 inputs: tfp_850 deliberately dropped (it is a regression target).
BASE_VARS    = ['t850', 'u850', 'v850']                                     # from era5_training
EXTRA_VARS   = ['z500', 'q850', 'w850', 'msl', 't925', 't2m', 'u10', 'v10']  # from extra_channels
INPUT_VARS   = BASE_VARS + EXTRA_VARS
IN_CH        = len(INPUT_VARS)

# Candidate regression targets; only those present in era5_training are used.
TARGET_VARS  = ['tfp_850', 'tadv_850', 'grad_mag_850']


# ── Device ─────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Resolve which targets actually exist across all training years ──────────
def resolve_targets(years: list) -> list:
    present = None
    for year in years:
        tp = EXTRA_DIR / f'era5_{year}_training.nc'
        if not tp.exists():
            continue
        ds = xr.open_dataset(tp)
        have = {v for v in TARGET_VARS if v in ds}
        ds.close()
        present = have if present is None else (present & have)
    targets = [v for v in TARGET_VARS if present and v in present]
    if not targets:
        raise RuntimeError(f'No regression targets {TARGET_VARS} found in era5_training files.')
    return targets


# ── Normalization (inputs + targets) ────────────────────────────────────────
def compute_norm_stats(years: list, targets: list) -> dict:
    vars_all = INPUT_VARS + targets
    sums    = {v: 0.0 for v in vars_all}
    sq_sums = {v: 0.0 for v in vars_all}
    count   = {v: 0   for v in vars_all}

    for year in years:
        tp = EXTRA_DIR / f'era5_{year}_training.nc'
        ep = EXTRA_DIR / f'extra_channels_{year}.nc'
        if not tp.exists() or not ep.exists():
            continue
        b = xr.open_dataset(tp); e = xr.open_dataset(ep)
        common = np.intersect1d(b.time.values, e.time.values)
        b = b.sel(time=common); e = e.sel(time=common)
        for v in vars_all:
            ds = e if v in EXTRA_VARS else b
            vals = ds[v].values.astype(np.float64)
            sums[v]    += vals.sum()
            sq_sums[v] += (vals ** 2).sum()
            count[v]   += vals.size
        b.close(); e.close()

    norm = {}
    for v in vars_all:
        mu  = sums[v] / count[v]
        std = float(np.sqrt(max(sq_sums[v] / count[v] - mu ** 2, 1e-12)))
        norm[v] = (float(mu), std)
    return norm


# ── Dataset ────────────────────────────────────────────────────────────────
class RegFrontDataset(Dataset):
    """Returns (float32 [11, H, W] inputs, float32 [n_targets, H, W] targets)."""

    def __init__(self, years: list, norm: dict, targets: list, pad_to: int = 16):
        self.norm = norm
        self.targets = targets
        self.data_x = []
        self.data_y = []

        for year in years:
            tp = EXTRA_DIR / f'era5_{year}_training.nc'
            ep = EXTRA_DIR / f'extra_channels_{year}.nc'
            if not tp.exists():
                print(f'  [skip] era5_{year}_training.nc not found'); continue
            if not ep.exists():
                print(f'  [skip] extra_channels_{year}.nc not found'); continue

            print(f'  {year} loading...', end=' ', flush=True)
            t0 = time.time()
            b = xr.open_dataset(tp); e = xr.open_dataset(ep)
            common = np.intersect1d(b.time.values, e.time.values)
            b = b.sel(time=common); e = e.sel(time=common)

            xch = []
            for v in INPUT_VARS:
                ds = e if v in EXTRA_VARS else b
                arr = ds[v].values.astype(np.float32)
                mu, sigma = norm[v]
                xch.append((arr - mu) / sigma)
            ych = []
            for v in targets:
                arr = b[v].values.astype(np.float32)
                mu, sigma = norm[v]
                ych.append((arr - mu) / sigma)

            x_all = np.stack(xch, axis=1)   # [T, 11, H, W]
            y_all = np.stack(ych, axis=1)   # [T, n_targets, H, W]
            b.close(); e.close()

            T, C, H, W = x_all.shape
            pH = (pad_to - H % pad_to) % pad_to
            pW = (pad_to - W % pad_to) % pad_to
            if pH or pW:
                x_all = np.pad(x_all, ((0,0),(0,0),(0,pH),(0,pW)), mode='reflect')
                y_all = np.pad(y_all, ((0,0),(0,0),(0,pH),(0,pW)), mode='reflect')

            for i in range(T):
                self.data_x.append(x_all[i])
                self.data_y.append(y_all[i])
            print(f'{T} steps  {(time.time()-t0):.0f}s', flush=True)

        print(f'  Total: {len(self.data_x):,} samples ({years})', flush=True)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data_x[idx].astype(np.float32)),
                torch.from_numpy(self.data_y[idx].astype(np.float32)))


# ── U-Net (regression head: no softmax) ──────────────────────────────────────
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
    def __init__(self, in_ch, n_out, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,    base)
        self.enc2 = ConvBlock(base,     base*2)
        self.enc3 = ConvBlock(base*2,   base*4)
        self.enc4 = ConvBlock(base*4,   base*8)
        self.bot  = ConvBlock(base*8,   base*16)
        self.pool = nn.MaxPool2d(2)
        self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16,  base*8)
        self.up3  = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8,   base*4)
        self.up2  = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4,   base*2)
        self.up1  = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
        self.dec1 = ConvBlock(base*2,   base)
        self.head = nn.Conv2d(base, n_out, 1)   # continuous outputs, no activation

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ── Metrics: normalized RMSE and Pearson r per target ───────────────────────
def compute_metrics(preds: np.ndarray, targs: np.ndarray, targets: list) -> dict:
    # preds, targs: [N, n_targets, H, W] in normalized units
    out = {}
    for c, name in enumerate(targets):
        p = preds[:, c].ravel().astype(np.float64)
        t = targs[:, c].ravel().astype(np.float64)
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        pm, tm = p.mean(), t.mean()
        cov = ((p - pm) * (t - tm)).mean()
        denom = p.std() * t.std() + 1e-12
        r = float(cov / denom)
        out[name] = {'rmse': rmse, 'r': r}
    return out


# ── Training ───────────────────────────────────────────────────────────────
def train(args):
    global _logfile
    device = get_device()
    print(f'Device: {device}')

    train_years = list(range(args.train[0], args.train[1] + 1))
    val_years   = list(range(args.val[0],   args.val[1]   + 1))
    print(f'Train: {train_years}  Val: {val_years}')

    targets = resolve_targets(train_years)
    n_out = len(targets)
    print(f'Regression targets ({n_out}): {targets}')

    run_tag     = args.run_name if args.run_name else f'unet_reg_{train_years[0]}-{train_years[-1]}_e{args.epochs}_b{args.batch}'
    log_path    = MODEL_DIR / f'{run_tag}.log'
    csv_path    = MODEL_DIR / f'{run_tag}_metrics.csv'
    resume_ckpt = MODEL_DIR / f'{run_tag}_resume.pt'

    _logfile = open(log_path, 'a')
    print(f'Log: {log_path}')

    print('Computing normalization stats...')
    norm = compute_norm_stats(train_years, targets)
    for k in INPUT_VARS + targets:
        mu, std = norm[k]
        tag = 'TARGET' if k in targets else 'input'
        print(f'  [{tag}] {k}: mean={mu:.4g}  std={std:.4g}')

    print('Loading datasets...')
    train_ds = RegFrontDataset(train_years, norm, targets)
    val_ds   = RegFrontDataset(val_years,   norm, targets)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(in_ch=IN_CH, n_out=n_out).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'U-Net parameters: {n_params/1e6:.1f}M  ({IN_CH} channels → {n_out} regression targets)')

    criterion = nn.SmoothL1Loss()   # Huber: robust to the rare large frontal values
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_r = -1.0
    start_epoch = 1

    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'val_loss'] +
                [f'rmse_{t}' for t in targets] +
                [f'r_{t}' for t in targets] +
                ['r_mean', 'sec'])

    if args.resume and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_r  = ckpt['best_val_r']
        print(f'Resumed from epoch {ckpt["epoch"]}  (best r={best_val_r:.3f})')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if out.shape[-2:] != yb.shape[-2:]:
                out = out[..., :yb.shape[-2], :yb.shape[-1]]
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                if out.shape[-2:] != yb.shape[-2:]:
                    out = out[..., :yb.shape[-2], :yb.shape[-1]]
                val_loss += criterion(out, yb).item()
                all_p.append(out.cpu().numpy())
                all_t.append(yb.cpu().numpy())
        val_loss /= len(val_loader)

        metrics = compute_metrics(np.concatenate(all_p), np.concatenate(all_t), targets)
        r_mean = float(np.mean([metrics[t]['r'] for t in targets]))
        elapsed = time.time() - t0

        r_str = '  '.join(f'{t}:r={metrics[t]["r"]:.3f}' for t in targets)
        log(f'Epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}→{val_loss:.4f}  '
            f'{r_str}  r_mean:{r_mean:.3f}  {elapsed:.0f}s')

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, f'{train_loss:.6f}', f'{val_loss:.6f}'] +
                [f'{metrics[t]["rmse"]:.4f}' for t in targets] +
                [f'{metrics[t]["r"]:.4f}' for t in targets] +
                [f'{r_mean:.4f}', f'{elapsed:.0f}'])

        if r_mean > best_val_r:
            best_val_r = r_mean
            best_ckpt = MODEL_DIR / f'{run_tag}_best.pt'
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'norm_stats': norm, 'targets': targets, 'val_r': best_val_r,
                        'metrics': metrics, 'in_ch': IN_CH, 'n_out': n_out}, best_ckpt)
            log(f'  → saved best: {best_ckpt.name}  (r={best_val_r:.3f})')

        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_val_r': best_val_r, 'norm_stats': norm,
                    'targets': targets, 'in_ch': IN_CH, 'n_out': n_out}, resume_ckpt)

    log(f'\nTraining complete. Best val r (mean): {best_val_r:.3f}')
    if _logfile: _logfile.close()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    global EXTRA_DIR, MODEL_DIR

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train',     nargs=2, type=int, default=[2019, 2024],
                   metavar=('START', 'END'))
    p.add_argument('--val',       nargs=2, type=int, default=[2025, 2025],
                   metavar=('START', 'END'))
    p.add_argument('--epochs',    type=int,   default=30)
    p.add_argument('--batch',     type=int,   default=32)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--resume',    action='store_true')
    p.add_argument('--data-root', type=str,   default=None,
                   help='Override data dirs: <root>/training, <root>/models')
    p.add_argument('--extra-dir', type=str,   default=None)
    p.add_argument('--model-dir', type=str,   default=None)
    p.add_argument('--run-name',  type=str,   default=None,
                   help='Prefix for saved model files (e.g. run8_reg → run8_reg_best.pt)')
    args = p.parse_args()

    if args.data_root:
        root = Path(args.data_root)
        EXTRA_DIR = root / 'training'
        MODEL_DIR = root / 'models'
    if args.extra_dir: EXTRA_DIR = Path(args.extra_dir)
    if args.model_dir: MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == '__main__':
    main()
