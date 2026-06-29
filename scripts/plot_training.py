"""
Plot U-Net training metrics from CSV.

Usage:
    python plot_training.py                          # latest CSV in MODEL_DIR
    python plot_training.py unet_e30_b8_metrics.csv  # specific file
"""

import sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

MODEL_DIR = Path('/Volumes/SSD_Hayoung/fronts/models')
FIG_DIR   = Path('/Users/hayoungbong/Analysis/Front/figures')


def load_csv(path: Path) -> dict:
    import csv
    rows = list(csv.DictReader(open(path)))
    keys = ['epoch', 'train_loss', 'val_loss', 'f1_cf', 'f1_wf', 'f1_sf', 'f1_mean', 'sec']
    return {k: [float(r[k]) for r in rows] for k in keys}


def load_log(path: Path) -> dict:
    """Fallback: parse raw .log file (same format as /tmp/unet_train.log)."""
    pat = r'Epoch\s+(\d+)/\d+\s+loss\s+([\d.]+)→([\d.]+)\s+F1 CF:([\d.]+) WF:([\d.]+) SF:([\d.]+)\s+mean:([\d.]+)\s+(\d+)s'
    d = {k: [] for k in ['epoch','train_loss','val_loss','f1_cf','f1_wf','f1_sf','f1_mean','sec']}
    for m in re.finditer(pat, path.read_text()):
        vals = [int(m.group(1)), float(m.group(2)), float(m.group(3)),
                float(m.group(4)), float(m.group(5)), float(m.group(6)),
                float(m.group(7)), int(m.group(8))]
        for k, v in zip(d, vals):
            d[k].append(v)
    return d


def plot(d: dict, tag: str):
    e = d['epoch']
    best_e = e[d['f1_mean'].index(max(d['f1_mean']))]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(e, d['train_loss'], 'o-', color='#e67e22', lw=2, label='Train')
    ax1.plot(e, d['val_loss'],   's-', color='#8e44ad', lw=2, label='Val')
    ax1.set_title('Loss (Focal)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Focal Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    # F1
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(e, d['f1_cf'],   'o-', color='#e74c3c', lw=2, label='CF')
    ax2.plot(e, d['f1_wf'],   's-', color='#3498db', lw=2, label='WF')
    ax2.plot(e, d['f1_sf'],   '^-', color='#27ae60', lw=2, label='SF')
    ax2.plot(e, d['f1_mean'], 'D--', color='#2c3e50', lw=2, label='Mean')
    ax2.axvline(best_e, color='gray', ls=':', alpha=0.7)
    ax2.annotate(f'Best {max(d["f1_mean"]):.3f}\n(ep {best_e})',
                 xy=(best_e, max(d['f1_mean'])),
                 xytext=(best_e + 0.5, max(d['f1_mean']) - 0.06),
                 fontsize=8, color='gray')
    ax2.set_title('F1 by Class (validation)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1')
    ax2.set_ylim(0, 0.9); ax2.legend(); ax2.grid(alpha=0.3)

    # Train-Val gap
    ax3 = fig.add_subplot(gs[1, 0])
    gap = [t - v for t, v in zip(d['train_loss'], d['val_loss'])]
    colors = ['#e74c3c' if g > 0 else '#3498db' for g in gap]
    ax3.bar(e, gap, color=colors, alpha=0.7)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Train − Val Loss Gap\n(red: train>val / blue: val>train)',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Δ Loss')
    ax3.grid(alpha=0.3, axis='y')

    # Time per epoch
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(e, [s / 60 for s in d['sec']], 'o-', color='#7f8c8d', lw=1.5)
    ax4.set_title('Time per Epoch', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('Minutes')
    ax4.set_ylim(0, 20); ax4.grid(alpha=0.3)

    fig.suptitle(
        f'U-Net Training Summary  ({tag})  |  epoch 1–{e[-1]}',
        fontsize=13, fontweight='bold', y=1.01)

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f'training_summary_{tag}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        if not src.is_absolute():
            src = MODEL_DIR / src
    else:
        csvs = sorted(MODEL_DIR.glob('*_metrics.csv'))
        logs = [Path('/tmp/unet_train.log')]
        src  = csvs[-1] if csvs else (logs[0] if logs[0].exists() else None)
        if src is None:
            print('No metrics file found.')
            sys.exit(1)

    print(f'Source: {src}')
    tag = src.stem.replace('_metrics', '') if src.suffix == '.csv' else 'from_log'
    d   = load_csv(src) if src.suffix == '.csv' else load_log(src)
    print(f'Epochs loaded: {len(d["epoch"])}')
    plot(d, tag)
