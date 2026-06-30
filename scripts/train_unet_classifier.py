"""
U-Net Front Detection — Run 4 (12-channel, 5-class hybrid labels)
=================================================================
Extends Run 3 with 4 additional input channels:
  - 12 input channels: t850/u850/v850/tfp_850 (from hybrid NC) +
                       z500/q850/w850/msl/t925/t2m/u10/v10 (from extra_channels NC)
  - 5 output classes: 0=BG 1=CF 2=WF 3=SF 4=OF
  - Hybrid labels (ERA5 TFP position ∩ WPC expert type)

Data layout:
  /Volumes/SSD_Hayoung/fronts/hybrid_labels/hybrid_YYYY.nc
      → variables: t850, u850, v850, tfp_850, front_label (5-class), tfp_label
  /Volumes/SSD_Hayoung/fronts/training/extra_channels_YYYY.nc
      → variables: z500, q850, w850, msl, t925, t2m, u10, v10

Usage:
  python train_unet_classifier.py                              # default: train 2022 2023 2024, val 2025
  python train_unet_classifier.py --train 2022 2023 2024 --val 2025 --epochs 30
  python train_unet_classifier.py --predict 2026-01-15T00      # inference
  python train_unet_classifier.py --resume
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ── Paths (overridable via --data-root or individual --*-dir flags) ────────
_DEFAULT_ROOT = Path('/Volumes/SSD_Hayoung/fronts')
HYBRID_DIR = _DEFAULT_ROOT / 'hybrid_labels'
EXTRA_DIR  = _DEFAULT_ROOT / 'training'
MODEL_DIR  = _DEFAULT_ROOT / 'models'

# ── Config ─────────────────────────────────────────────────────────────────
IN_CH       = 12
N_CLASSES   = 5
CLASS_NAMES = ['BG', 'CF', 'WF', 'SF', 'OF']

# Label source: hybrid (5-class, ERA5×WPC) [default] or TFP (4-class, ERA5 only).
# Run 5: hybrid labels.  Run 6: TFP labels (same 12 channels, isolates label effect).
USE_TFP = False

BASE_VARS  = ['t850', 'u850', 'v850', 'tfp_850']               # from hybrid NC
EXTRA_VARS = ['z500', 'q850', 'w850', 'msl', 't925', 't2m', 'u10', 'v10']  # from extra_channels NC
ALL_VARS   = BASE_VARS + EXTRA_VARS

# ── Distributed setup ──────────────────────────────────────────────────────
def setup_ddp():
    """Init process group if launched with torchrun; otherwise single-GPU."""
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ── Device ─────────────────────────────────────────────────────────────────
def get_device(local_rank=0):
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')


# ── Normalization ──────────────────────────────────────────────────────────
def compute_norm_stats(years: list) -> dict:
    """Compute per-channel mean/std over all training years."""
    sums    = np.zeros(IN_CH, dtype=np.float64)
    sq_sums = np.zeros(IN_CH, dtype=np.float64)
    count   = 0

    for year in years:
        hp = HYBRID_DIR / f'hybrid_{year}.nc'
        ep = EXTRA_DIR  / f'extra_channels_{year}.nc'
        if not hp.exists() or not ep.exists():
            continue
        h = xr.open_dataset(hp)
        e = xr.open_dataset(ep)
        # Discover: hybrid has labels only; ERA5 vars are in era5_YYYY_training.nc
        tp = EXTRA_DIR / f'era5_{year}_training.nc'
        b = xr.open_dataset(tp) if ('t850' not in h and tp.exists()) else h

        # Align times
        common = np.intersect1d(h.time.values, e.time.values)
        common = np.intersect1d(common, b.time.values)
        h = h.sel(time=common); e = e.sel(time=common); b = b.sel(time=common)

        for i, var in enumerate(ALL_VARS):
            ds = b if var in BASE_VARS else e
            vals = ds[var].values.astype(np.float64)
            sums[i]    += vals.sum()
            sq_sums[i] += (vals ** 2).sum()
            if i == 0:
                count += vals.size
        h.close(); e.close()
        if b is not h: b.close()

    means = sums / count
    stds  = np.sqrt(np.maximum(sq_sums / count - means ** 2, 1e-12))
    return {v: (float(means[i]), float(stds[i])) for i, v in enumerate(ALL_VARS)}


# ── Dataset ────────────────────────────────────────────────────────────────
class HybridFrontDataset(Dataset):
    """
    Loads hybrid_YYYY.nc (base channels + 5-class label) and
    extra_channels_YYYY.nc (z500/q850/w850/msl) into memory.
    Returns (float32 [8, H, W], int64 [H, W]).
    """

    def __init__(self, years: list, norm_stats: dict, pad_to: int = 16):
        self.norm = norm_stats
        self.data_x = []
        self.data_y = []

        for year in years:
            hp = HYBRID_DIR / f'hybrid_{year}.nc'
            ep = EXTRA_DIR  / f'extra_channels_{year}.nc'
            if not hp.exists():
                print(f'  [skip] hybrid_{year}.nc not found'); continue
            if not ep.exists():
                print(f'  [skip] extra_channels_{year}.nc not found'); continue

            print(f'  {year} loading...', end=' ', flush=True)
            t0 = time.time()

            h = xr.open_dataset(hp)
            e = xr.open_dataset(ep)
            # Discover: hybrid has labels only; ERA5 vars are in era5_YYYY_training.nc.
            # TFP mode (Run 6): always need era5_training for the 4-class TFP labels.
            tp = EXTRA_DIR / f'era5_{year}_training.nc'
            b = xr.open_dataset(tp) if (('t850' not in h or USE_TFP) and tp.exists()) else h

            # Align timestamps (inner)
            common = np.intersect1d(h.time.values, e.time.values)
            common = np.intersect1d(common, b.time.values)
            h = h.sel(time=common)
            e = e.sel(time=common)
            b = b.sel(time=common)

            channels = []
            for var in ALL_VARS:
                ds   = b if var in BASE_VARS else e
                arr  = ds[var].values.astype(np.float32)
                mu, sigma = norm_stats[var]
                channels.append((arr - mu) / sigma)

            x_all = np.stack(channels, axis=1)        # [T, 12, H, W]
            # Run 6 (TFP): labels from era5_training (0-3).  Run 5 (hybrid): from hybrid (0-4).
            y_src = b if (USE_TFP and b is not h) else h
            y_all = y_src['front_label'].values.astype(np.int8)  # [T, H, W]
            h.close(); e.close()
            if b is not h: b.close()

            # Pad to multiple of pad_to for U-Net pooling
            T, C, H, W = x_all.shape
            pH = (pad_to - H % pad_to) % pad_to
            pW = (pad_to - W % pad_to) % pad_to
            if pH or pW:
                x_all = np.pad(x_all, ((0,0),(0,0),(0,pH),(0,pW)), mode='reflect')
                y_all = np.pad(y_all, ((0,0),(0,pH),(0,pW)), mode='constant')

            for i in range(T):
                self.data_x.append(x_all[i])
                self.data_y.append(y_all[i])

            print(f'{T} steps  {(time.time()-t0):.0f}s', flush=True)

        print(f'  Total: {len(self.data_x):,} samples ({years})', flush=True)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.data_x[idx].astype(np.float32)),
                torch.from_numpy(self.data_y[idx].astype(np.int64)))


# ── U-Net ──────────────────────────────────────────────────────────────────
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
    def __init__(self, in_ch=IN_CH, num_classes=N_CLASSES, base=64):
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
        self.head = nn.Conv2d(base, num_classes, 1)

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


# ── Loss ───────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma; self.weight = weight

    def forward(self, logits, targets):
        log_p  = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = log_pt.exp()
        focal  = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def class_weights(years: list, device) -> torch.Tensor:
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for year in years:
        # TFP mode (Run 6): labels in era5_training.  Hybrid mode (Run 5): in hybrid NC.
        src = (EXTRA_DIR / f'era5_{year}_training.nc') if USE_TFP \
              else (HYBRID_DIR / f'hybrid_{year}.nc')
        if not src.exists(): continue
        ds = xr.open_dataset(src)
        labels = ds['front_label'].values
        for c in range(N_CLASSES):
            counts[c] += (labels == c).sum()
        ds.close()

    w = np.ones(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        if counts[c] > 0:
            w[c] = 1.0 / np.sqrt(counts[c] / counts.sum() + 1e-8)
        else:
            w[c] = 0.0   # no training signal → zero weight (SF=0 pixels)
    if w.sum() > 0:
        w /= w[w > 0].mean()
    print(f'  Class weights: {dict(zip(CLASS_NAMES, w.round(2)))}')
    return torch.tensor(w, dtype=torch.float32, device=device)


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    results = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        results[name] = {'f1': float(f1), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    return results


# ── Training ───────────────────────────────────────────────────────────────
def train(args):
    global _logfile  # noqa: must declare global to modify module-level _logfile

    local_rank, world_size = setup_ddp()
    is_ddp  = world_size > 1
    is_main = (local_rank == 0)   # only rank 0 logs / saves

    device = get_device(local_rank)
    if is_main:
        print(f'Device: {device}  |  world_size={world_size}  |  AMP={args.amp}  |  compile={args.compile}')

    train_years = list(range(args.train[0], args.train[1] + 1))
    val_years   = list(range(args.val[0],   args.val[1]   + 1))
    if is_main:
        print(f'Train: {train_years}  Val: {val_years}')

    label_tag   = 'tfp' if USE_TFP else 'hybrid'   # Run 6 = tfp, Run 5 = hybrid
    detail_tag  = f'{label_tag}_{IN_CH}ch_{N_CLASSES}cls_{train_years[0]}-{train_years[-1]}'
    run_tag     = f'{args.run_name}_{detail_tag}' if args.run_name else f'unet_{detail_tag}_e{args.epochs}_b{args.batch}'
    log_path    = MODEL_DIR / f'{run_tag}.log'
    csv_path    = MODEL_DIR / f'{run_tag}_metrics.csv'
    resume_ckpt = MODEL_DIR / f'{run_tag}_resume.pt'

    if is_main:
        _logfile = open(log_path, 'a')
        print(f'Log: {log_path}')

    if is_main:
        print('Computing normalization stats...')
    norm = compute_norm_stats(train_years)
    if is_main:
        for k, (mu, std) in norm.items():
            print(f'  {k}: mean={mu:.4g}  std={std:.4g}')

    if is_main:
        print('Loading datasets...')
    train_ds = HybridFrontDataset(train_years, norm)
    val_ds   = HybridFrontDataset(val_years,   norm)

    pin = (device.type == 'cuda')
    nw  = args.workers
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=local_rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  sampler=train_sampler,
                                  num_workers=nw, pin_memory=pin,
                                  persistent_workers=(nw > 0))
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=nw, pin_memory=pin,
                                  persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=nw, pin_memory=pin,
                            persistent_workers=(nw > 0))

    model = UNet(in_ch=IN_CH, num_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f'U-Net parameters: {n_params/1e6:.1f}M  ({IN_CH} channels → {N_CLASSES} classes)')

    if args.compile and hasattr(torch, 'compile'):
        if is_main:
            print('Compiling model with torch.compile ...')
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    w = class_weights(train_years, device)
    criterion = FocalLoss(gamma=2.0, weight=w)
    raw_params = model.module.parameters() if is_ddp else model.parameters()
    optimizer = torch.optim.AdamW(raw_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Mixed precision: bf16 on CUDA (no GradScaler needed)
    use_amp  = args.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    best_val_f1 = 0.0
    start_epoch = 1

    # Header for CSV
    front_classes = [c for c in CLASS_NAMES if c != 'BG']  # CF WF SF OF
    if is_main and not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'val_loss'] +
                [f'f1_{c.lower()}' for c in front_classes] +
                ['f1_mean', 'sec']
            )

    if args.resume and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        raw_model = model.module if is_ddp else model
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt['best_val_f1']
        if is_main:
            print(f'Resumed from epoch {ckpt["epoch"]}  (best F1={best_val_f1:.3f})')

    for epoch in range(start_epoch, args.epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for batch_i, (xb, yb) in enumerate(train_loader):
            if args.smoke and batch_i >= 10:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                 dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = logits[..., :yb.shape[-2], :yb.shape[-1]]
                loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # Validation (only rank 0 in DDP to avoid duplicating val data)
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                     dtype=amp_dtype, enabled=use_amp):
                    logits = model(xb)
                    if logits.shape[-2:] != yb.shape[-2:]:
                        logits = logits[..., :yb.shape[-2], :yb.shape[-1]]
                val_loss += criterion(logits, yb).item()
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        val_loss /= len(val_loader)

        if is_main:
            metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels))
            f1s = [metrics[c]['f1'] for c in front_classes if metrics[c]['tp'] + metrics[c]['fn'] > 0]
            mean_f1 = float(np.mean(f1s)) if f1s else 0.0
            elapsed = time.time() - t0

            f1_str = '  '.join(f'{c}:{metrics[c]["f1"]:.3f}' for c in front_classes)
            msg = (f'Epoch {epoch:3d}/{args.epochs}  '
                   f'loss {train_loss:.4f}→{val_loss:.4f}  '
                   f'F1 {f1_str}  mean:{mean_f1:.3f}  {elapsed:.0f}s')
            log(msg)

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [epoch, f'{train_loss:.6f}', f'{val_loss:.6f}'] +
                    [f'{metrics[c]["f1"]:.4f}' for c in front_classes] +
                    [f'{mean_f1:.4f}', f'{elapsed:.0f}']
                )

            raw_model = model.module if is_ddp else model
            if mean_f1 > best_val_f1:
                best_val_f1 = mean_f1
                best_ckpt = MODEL_DIR / f'{run_tag}_best.pt'
                torch.save({'epoch': epoch, 'model_state': raw_model.state_dict(),
                            'norm_stats': norm, 'val_f1': best_val_f1,
                            'metrics': metrics, 'in_ch': IN_CH, 'n_classes': N_CLASSES}, best_ckpt)
                log(f'  → saved best: {best_ckpt.name}  (F1={best_val_f1:.3f})')

            torch.save({'epoch': epoch, 'model_state': raw_model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_val_f1': best_val_f1, 'norm_stats': norm,
                        'in_ch': IN_CH, 'n_classes': N_CLASSES}, resume_ckpt)

    if is_main:
        log(f'\nTraining complete. Best val F1: {best_val_f1:.3f}')
        if _logfile: _logfile.close()
    cleanup_ddp()


# ── Inference ──────────────────────────────────────────────────────────────
def predict(args):
    """Run 12-channel inference on a single timestep and save comparison figure."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    device = get_device()
    ckpts  = sorted(MODEL_DIR.glob('unet_v4_*_best.pt'))
    if not ckpts:
        print('No v4 checkpoint found. Run training first.'); return
    ckpt_path = ckpts[-1]
    print(f'Loading: {ckpt_path.name}')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm = ckpt['norm_stats']

    model = UNet(in_ch=ckpt.get('in_ch', IN_CH),
                 num_classes=ckpt.get('n_classes', N_CLASSES)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    target = np.datetime64(args.predict)
    year   = int(str(target)[:4])
    hp = HYBRID_DIR / f'hybrid_{year}.nc'
    ep = EXTRA_DIR  / f'extra_channels_{year}.nc'

    # Fall back to WPC labels for 2026 (no hybrid)
    wpc_path = Path('/Volumes/SSD_Hayoung/fronts/wpc_labels') / f'wpc_labels_{year}.nc'

    if hp.exists() and ep.exists():
        h = xr.open_dataset(hp); e = xr.open_dataset(ep)
        common = np.intersect1d(h.time.values, e.time.values)
        h = h.sel(time=common); e = e.sel(time=common)
        idx = int(np.argmin(np.abs(h.time.values - target)))
        actual = h.time.values[idx]
        channels = []
        for var in ALL_VARS:
            ds = h if var in BASE_VARS else e
            arr = ds[var].values[idx].astype(np.float32)
            mu, sigma = norm[var]
            channels.append((arr - mu) / sigma)
        lats, lons = h.lat.values, h.lon.values
        true_label = h['front_label'].values[idx]
        label_src  = 'Hybrid label'
        h.close(); e.close()
    else:
        print(f'hybrid_{year}.nc or extra_channels_{year}.nc not found.')
        print(f'Attempting with ERA5_global data for {year}...')
        # For 2026: we only have extra_channels (from ERA5_global)
        # Use zeros for base channels (not ideal, but allows code test)
        # TODO: build full 2026 training data
        return

    x = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(device)
    H, W = x.shape[-2:]
    m = 16
    pH = (m - H % m) % m; pW = (m - W % m) % m
    if pH or pW:
        x = F.pad(x, (0, pW, 0, pH), mode='reflect')
    with torch.no_grad():
        pred = model(x).argmax(1).squeeze().cpu().numpy()[:H, :W]

    # Plot
    proj = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(25,25))
    ext  = [-130, -60, 20, 60]
    cmap = mcolors.ListedColormap(['#d4d4d4','#1565C0','#C62828','#00897B','#7B1FA2'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm_c = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             subplot_kw={'projection': proj})
    lons2d, lats2d = np.meshgrid(lons, lats)
    for ax, data, title in zip(axes,
        [true_label, pred],
        [f'{label_src}\n{actual}', f'U-Net v4 Prediction\n{actual}']):
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,      facecolor='#f0ede8')
        ax.add_feature(cfeature.OCEAN,     facecolor='#c8e6f5')
        ax.add_feature(cfeature.COASTLINE, lw=0.7, edgecolor='#333')
        ax.add_feature(cfeature.BORDERS,   lw=0.4, edgecolor='#666')
        ax.add_feature(cfeature.STATES,    lw=0.25, edgecolor='#999')
        im = ax.pcolormesh(lons2d, lats2d, data, cmap=cmap, norm=norm_c,
                           transform=ccrs.PlateCarree(), shading='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')

    cb = plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.03,
                      fraction=0.04, shrink=0.6, ticks=[0,1,2,3,4])
    cb.set_ticklabels(['BG','CF','WF','SF','OF'])
    fig.suptitle(f'U-Net v4 (12-ch, 5-class hybrid)  |  {actual}',
                 fontsize=12, fontweight='bold')

    tag = str(actual)[:13].replace('T','-').replace(':','')
    out = Path(f'/Users/hayoungbong/Analysis/Front/figures/predict_v4_{tag}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    global HYBRID_DIR, EXTRA_DIR, MODEL_DIR  # allow CLI overrides
    global USE_TFP, N_CLASSES, CLASS_NAMES   # label-source switch (Run 6)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train',      nargs=2, type=int, default=[2022, 2024],
                   metavar=('START', 'END'))
    p.add_argument('--val',        nargs=2, type=int, default=[2025, 2025],
                   metavar=('START', 'END'))
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--batch',      type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--resume',     action='store_true')
    p.add_argument('--tfp-labels', action='store_true',
                   help='Run 6: train on 4-class TFP labels (from era5_YYYY_training.nc) '
                        'with the same 12 channels, instead of 5-class hybrid labels')
    p.add_argument('--predict',    type=str,   default=None,
                   help='Run inference: e.g. 2026-01-15T00')
    p.add_argument('--data-root',  type=str,   default=None,
                   help='Override all data dirs: <root>/hybrid_labels, <root>/training, <root>/models')
    p.add_argument('--hybrid-dir', type=str,   default=None)
    p.add_argument('--extra-dir',  type=str,   default=None)
    p.add_argument('--model-dir',  type=str,   default=None)
    # ── Optimization flags ──
    p.add_argument('--amp',     action='store_true', default=True,
                   help='bf16 mixed precision on CUDA (default: on)')
    p.add_argument('--no-amp',  dest='amp', action='store_false',
                   help='Disable mixed precision')
    p.add_argument('--workers', type=int, default=4,
                   help='DataLoader num_workers (default: 4; use 0 on MPS)')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the model (~15-30%% speedup on A100/H100)')
    # ── Smoke-test flag ──
    p.add_argument('--run-name',  type=str, default=None,
                   help='Optional suffix appended to model filename (e.g. run8, run8b)')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: 1 year, 2 epochs, 10 batches — verifies the full pipeline fast')
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch  = max(args.batch, 8)

    # Run 6: switch to 4-class TFP labels (BG/CF/WF/SF), same 12 input channels
    if args.tfp_labels:
        USE_TFP     = True
        N_CLASSES   = 4
        CLASS_NAMES = ['BG', 'CF', 'WF', 'SF']
        print('Label source: TFP 4-class (Run 6) — 12 channels, no OF')
    else:
        print('Label source: hybrid 5-class (Run 5) — 12 channels, with OF')

    if args.data_root:
        root = Path(args.data_root)
        HYBRID_DIR = root / 'hybrid_labels'
        EXTRA_DIR  = root / 'training'
        MODEL_DIR  = root / 'models'
    if args.hybrid_dir: HYBRID_DIR = Path(args.hybrid_dir)
    if args.extra_dir:  EXTRA_DIR  = Path(args.extra_dir)
    if args.model_dir:  MODEL_DIR  = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
