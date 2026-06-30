"""
U-Net weather front classification at 850 hPa over North America.

Architecture
------------
Standard 4-level encoder-decoder U-Net with skip connections.
  Encoder: 64 → 128 → 256 → 512 → 1024 channels (bottleneck)
  Decoder: reverse with ConvTranspose2d upsampling
  Head   : 1×1 Conv → 4 class logits (pixel-wise)

Input channels (4): t850, u850, v850, tfp_850
Output classes (4): 0=Background, 1=Cold Front, 2=Warm Front, 3=Stationary Front

Training details
----------------
  Loss      : Focal Loss (γ=2) with inverse-frequency class weights
  Optimizer : AdamW (lr=1e-4, weight_decay=1e-4)
  Scheduler : CosineAnnealingLR → lr×0.01 by final epoch
  Labels    : TFP zero-crossing + temperature advection sign
  Domain    : 15–70°N, 170°W–50°W (North America), 0.25° grid
  Achieved  : mean F1=0.675 (CF:0.764 WF:0.731 SF:0.531) at epoch 30

Usage
-----
    python train_unet_classifier_legacy.py                              # train (2020–2021 / val 2022)
    python train_unet_classifier_legacy.py --epochs 50 --batch 8
    python train_unet_classifier_legacy.py --train 2000 2019 --val 2020
    python train_unet_classifier_legacy.py --resume                     # continue from *_resume.pt
    python train_unet_classifier_legacy.py --predict 2022-06-15T00      # single-timestep inference
"""

import os, argparse, time, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path

_logfile = None
def log(msg: str):
    print(msg, flush=True)
    if _logfile:
        _logfile.write(msg + '\n')
        _logfile.flush()

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths (overridable via --data-dir / --model-dir / --hybrid-dir) ───────────
DATA_DIR   = Path('/Users/hayoungbong/Analysis/Front/data/training')
MODEL_DIR  = Path('/Volumes/SSD_Hayoung/fronts/models')
HYBRID_DIR = Path('/Volumes/SSD_Hayoung/fronts/hybrid_labels')

# Overridden at runtime by --no-tfp / --hybrid
CLASS_NAMES = ['BG', 'CF', 'WF', 'SF']
N_CLASSES   = 4
CHANNELS    = ['t850', 'u850', 'v850', 'tfp_850']
IN_CH       = 4
USE_HYBRID  = False   # if True, load from hybrid_YYYY.nc (WPC labels)

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Normalization statistics (computed from training data) ─────────────────────
NORM_STATS = {
    # channel: (mean, std) — defaults; overridden by compute_norm_stats() at runtime
    't850':    (280.0, 20.0),
    'u850':    (0.0,   15.0),
    'v850':    (0.0,   10.0),
    'tfp_850': (0.0,    2.0),
}


def _data_path(year: int) -> Path:
    if USE_HYBRID:
        return HYBRID_DIR / f'hybrid_{year}.nc'
    return DATA_DIR / f'era5_{year}_training.nc'


def compute_norm_stats(years: list[int]) -> dict:
    """Compute per-channel mean and std over all training years."""
    n = len(CHANNELS)
    sums    = np.zeros(n, dtype=np.float64)
    sq_sums = np.zeros(n, dtype=np.float64)
    count   = 0

    for year in years:
        path = _data_path(year)
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        for i, var in enumerate(CHANNELS):
            vals = ds[var].values.astype(np.float64)
            sums[i]    += vals.sum()
            sq_sums[i] += (vals ** 2).sum()
            if i == 0:
                count += vals.size
        ds.close()

    means = sums   / count
    stds  = np.sqrt(sq_sums / count - means ** 2)
    return {v: (float(means[i]), float(stds[i])) for i, v in enumerate(CHANNELS)}


# ── Dataset ────────────────────────────────────────────────────────────────────
class FrontDataset(Dataset):
    """
    6-hourly ERA5 snapshots loaded entirely into memory (eliminates per-epoch disk I/O).
    input:  float32 [4, H, W]  (normalized)
    label:  int64   [H, W]     (0-3)
    """

    def __init__(self, years: list[int], norm_stats: dict, pad_to_multiple: int = 16):
        self.norm = norm_stats
        self.pad  = pad_to_multiple
        self.data_x = []
        self.data_y = []

        for year in years:
            path = _data_path(year)
            if not path.exists():
                print(f'  [skip] {path.name} not found')
                continue
            print(f'  {year} loading...', end=' ', flush=True)
            t0 = time.time()
            ds = xr.open_dataset(path)
            channels = []
            for var in CHANNELS:
                arr = ds[var].values.astype(np.float32)
                mu, sigma = norm_stats[var]
                channels.append((arr - mu) / sigma)
            x_all = np.stack(channels, axis=1)   # [T, C, H, W]
            y_all = ds['front_label'].values.copy()  # [T, H, W]
            if USE_HYBRID:
                # Remap label 4 (OF) → 3 for contiguous class indices
                y_all[y_all == 4] = 3
            ds.close()
            _, _, H, W = x_all.shape
            m = pad_to_multiple
            pH = (m - H % m) % m
            pW = (m - W % m) % m
            if pH or pW:
                x_all = np.pad(x_all, ((0,0),(0,0),(0,pH),(0,pW)), mode='reflect')
                y_all = np.pad(y_all, ((0,0),(0,pH),(0,pW)), mode='constant')
            for i in range(len(x_all)):
                self.data_x.append(x_all[i])
                self.data_y.append(y_all[i])
            print(f'{len(x_all)} steps  {(time.time()-t0):.0f}s', flush=True)

        print(f'  Total samples: {len(self.data_x):,}  ({years})', flush=True)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx].astype(np.float32)
        y = self.data_y[idx].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


# ── U-Net ──────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    4-level U-Net: encoder 64→128→256→512→1024 (bottleneck) → decoder (reverse).
    """
    def __init__(self, in_ch=IN_CH, num_classes=N_CLASSES, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,   base)
        self.enc2 = ConvBlock(base,    base*2)
        self.enc3 = ConvBlock(base*2,  base*4)
        self.enc4 = ConvBlock(base*4,  base*8)
        self.bot  = ConvBlock(base*8,  base*16)

        self.pool = nn.MaxPool2d(2)

        self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8)
        self.up3  = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8,  base*4)
        self.up2  = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4,  base*2)
        self.up1  = nn.ConvTranspose2d(base*2,  base,   2, stride=2)
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


# ── Loss function ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for severe class imbalance (BG ~99%).
    gamma=2: down-weights easy samples (BG), focuses training on hard samples (CF/WF/SF).
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B,H,W]
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def class_weights(years: list[int], device) -> torch.Tensor:
    """Compute inverse-frequency class weights from training label distribution."""
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for year in years:
        path = _data_path(year)
        if not path.exists():
            continue
        ds = xr.open_dataset(path)
        labels = ds['front_label'].values
        for c in range(N_CLASSES):
            counts[c] += (labels == c).sum()
        ds.close()

    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-8)   # sqrt dampens extreme values
    w /= w.mean()
    print(f'  Class weights: {dict(zip(CLASS_NAMES, w.round(2)))}')
    return torch.tensor(w, dtype=torch.float32, device=device)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Per-class F1 and IoU."""
    results = {}
    for c in range(N_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        iou  = tp / (tp + fp + fn + 1e-8)
        results[CLASS_NAMES[c]] = {'f1': float(f1), 'iou': float(iou)}
    return results


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = get_device()
    print(f'Device: {device}')

    train_years = list(range(args.train[0], args.train[1] + 1))
    val_years   = list(range(args.val[0],   args.val[1]   + 1))
    print(f'Train: {train_years}  Val: {val_years}')

    print('Computing normalization statistics...')
    norm = compute_norm_stats(train_years)
    for k, (mu, std) in norm.items():
        print(f'  {k}: mean={mu:.2f}  std={std:.2f}')

    print('Loading data...')
    train_ds = FrontDataset(train_years, norm)
    val_ds   = FrontDataset(val_years,   norm)

    # num_workers=0: data is already in memory, forking would add overhead
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # model
    model = UNet(in_ch=IN_CH, num_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params/1e6:.1f}M  in_ch={IN_CH}  n_classes={N_CLASSES}', flush=True)

    w = class_weights(train_years, device)
    criterion = FocalLoss(gamma=2.0, weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_f1 = 0.0
    start_epoch = 1
    suffix = ''
    if USE_HYBRID:
        suffix += '_hybrid'
    if 'tfp_850' not in CHANNELS:
        suffix += '_notfp'
    run_tag = f'unet_{train_years[0]}-{train_years[-1]}_e{args.epochs}_b{args.batch}{suffix}'
    resume_ckpt = MODEL_DIR / f'{run_tag}_resume.pt'
    csv_path    = MODEL_DIR / f'{run_tag}_metrics.csv'

    import csv
    front_classes = [c for c in CLASS_NAMES if c != 'BG']
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'val_loss']
                + [f'f1_{c.lower()}' for c in front_classes]
                + ['f1_mean', 'sec']
            )

    if args.resume and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_f1  = ckpt['best_val_f1']
        print(f'Resumed from epoch {ckpt["epoch"]}  (best F1={best_val_f1:.3f})', flush=True)
    elif args.resume:
        print(f'No resume checkpoint found at {resume_ckpt}, starting from scratch.', flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = logits[..., :y.shape[-2], :y.shape[-1]]
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = logits[..., :y.shape[-2], :y.shape[-1]]
                val_loss += criterion(logits, y).item()
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        val_loss /= len(val_loader)
        metrics   = compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels))
        mean_front_f1 = np.mean([metrics[c]['f1'] for c in front_classes])
        elapsed = time.time() - t0

        f1_str = '  '.join(f'{c}:{metrics[c]["f1"]:.3f}' for c in front_classes)
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'loss {train_loss:.4f}→{val_loss:.4f}  '
              f'F1 {f1_str}  mean:{mean_front_f1:.3f}  {elapsed:.0f}s', flush=True)

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, f'{train_loss:.6f}', f'{val_loss:.6f}']
                + [f'{metrics[c]["f1"]:.4f}' for c in front_classes]
                + [f'{mean_front_f1:.4f}', f'{elapsed:.0f}']
            )

        if mean_front_f1 > best_val_f1:
            best_val_f1 = mean_front_f1
            ckpt = MODEL_DIR / f'{run_tag}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'norm_stats': norm,
                'val_f1': best_val_f1,
                'metrics': metrics,
            }, ckpt)
            print(f'  → checkpoint saved: {ckpt.name}  (F1={best_val_f1:.3f})', flush=True)

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'norm_stats': norm,
        }, resume_ckpt)

    print(f'\nTraining complete. Best val F1 (CF/WF/SF mean): {best_val_f1:.3f}', flush=True)
    print(f'Model: {MODEL_DIR}/{run_tag}_best.pt', flush=True)


# ── Inference / visualisation ─────────────────────────────────────────────────
def predict(args):
    """Single-timestep inference and 2-panel (physics label vs U-Net) plot."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    device = get_device()

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpts = sorted(MODEL_DIR.glob('unet_*_best.pt'), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            print('No checkpoint found. Run training first.')
            return
        ckpt_path = ckpts[-1]
    print(f'Loading model: {ckpt_path.name}')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm = ckpt['norm_stats']

    model = UNet().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    target_date = np.datetime64(args.predict)
    year = int(str(target_date)[:4])
    path = DATA_DIR / f'era5_{year}_training.nc'
    if not path.exists():
        print(f'{path} not found')
        return

    ds = xr.open_dataset(path)
    times = ds['time'].values
    idx = np.argmin(np.abs(times - target_date))
    actual_time = times[idx]
    print(f'Timestep: {actual_time}')

    channels = []
    for var in ['t850', 'u850', 'v850', 'tfp_850']:
        arr = ds[var].values[idx].astype(np.float32)
        mu, sigma = norm[var]
        channels.append((arr - mu) / sigma)

    lats = ds['lat'].values
    lons = ds['lon'].values
    true_label = ds['front_label'].values[idx]
    ds.close()

    x = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(device)
    m = 16
    H, W = x.shape[-2:]
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    if pH or pW:
        x = F.pad(x, (0, pW, 0, pH), mode='reflect')

    with torch.no_grad():
        logits = model(x)
    pred = logits.argmax(dim=1).squeeze().cpu().numpy()[:H, :W]

    cmap = mcolors.ListedColormap(['#d4d4d4', '#2196F3', '#F44336', '#9C27B0'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_c = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(axes, [true_label, pred],
                               ['Physics Label (TFP)', 'U-Net Prediction']):
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm_c,
                           shading='nearest')
        ax.set_title(f'{title}\n{actual_time}', fontsize=12)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    plt.colorbar(im, ax=axes, ticks=[0,1,2,3],
                 label='0=BG  1=CF  2=WF  3=SF')
    out = Path('/Users/hayoungbong/Analysis/Front/figures') / f'unet_pred_{str(actual_time)[:13].replace("T","_")}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',   nargs=2, type=int, default=[2020, 2021],
                        metavar=('START', 'END'))
    parser.add_argument('--val',     nargs=2, type=int, default=[2022, 2022],
                        metavar=('START', 'END'))
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--batch',   type=int, default=4)
    parser.add_argument('--lr',      type=float, default=1e-4)
    parser.add_argument('--predict', type=str, default=None,
                        help='target datetime for inference (YYYY-MM-DDTHH)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='explicit checkpoint path for --predict (default: most recent best.pt)')
    parser.add_argument('--no-tfp', action='store_true',
                        help='drop tfp_850 from input channels (3-ch ablation)')
    parser.add_argument('--hybrid', action='store_true',
                        help='use hybrid WPC labels from hybrid_YYYY.nc (5-class)')
    parser.add_argument('--hybrid-dir', type=str, default=None,
                        help='directory containing hybrid_YYYY.nc files')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from unet_*_resume.pt checkpoint')
    parser.add_argument('--data-dir',  type=str, default=None,
                        help='directory containing era5_YYYY_training.nc files')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='directory to save checkpoints and metrics')
    args = parser.parse_args()

    global DATA_DIR, MODEL_DIR, HYBRID_DIR, CHANNELS, IN_CH, N_CLASSES, CLASS_NAMES, USE_HYBRID
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.model_dir:
        MODEL_DIR = Path(args.model_dir)
    if args.hybrid_dir:
        HYBRID_DIR = Path(args.hybrid_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Apply --no-tfp: 3-channel model (drop tfp_850)
    if args.no_tfp:
        CHANNELS = ['t850', 'u850', 'v850']
        IN_CH    = 3

    # Apply --hybrid: 4-class WPC labels (BG/CF/WF/OF — SF not extractable from WPC GIFs)
    # hybrid_YYYY.nc front_label values: 0=BG, 1=CF, 2=WF, 4=OF
    # We remap 4→3 so label indices are contiguous: 0=BG, 1=CF, 2=WF, 3=OF
    if args.hybrid:
        USE_HYBRID  = True
        N_CLASSES   = 4
        CLASS_NAMES = ['BG', 'CF', 'WF', 'OF']

    print(f'Channels ({IN_CH}): {CHANNELS}')
    print(f'Classes  ({N_CLASSES}): {CLASS_NAMES}')
    print(f'Hybrid labels: {USE_HYBRID}')

    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
