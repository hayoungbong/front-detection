# Pipeline Details

Step-by-step execution guide. See README.md for project overview.

---

## Step 0 — Explore ARCO ERA5 (optional)

```bash
python scripts/check_arco.py
```

Lists available variables and coordinates in the ARCO public ERA5 Zarr archive.

---

## Step 1 — Download ERA5 (CDS API)

```bash
python scripts/download_era5_batch.py              # 2000–2026, global
python scripts/download_era5_batch.py 2000 2019   # specific year range
```

- **Output:** `/Volumes/SSD_Hayoung/ERA5/{pressure_level,single_level}/`
- **Variables (PL):** T, u, v, Z, q, ω  @  500/700/850/900/950/1000 hPa
- **Variables (SFC):** 2mT, Td2m, 10m u/v, MSLP, Ps
- **Resolution:** 0.25°, 6-hourly, global
- **Resume-safe:** stale `.tmp` files deleted and retried automatically

---

## Step 2 — Build TFP Training Data

```bash
python scripts/build_training_data.py --years 2019 2020 2021 2022 2023 2024 2025
```

Computes TFP, smooths temperature (adaptive Gaussian ~400 km), and assigns
CF/WF/SF labels based on TFP zero-crossings and temperature advection sign.

**Output:** `/Volumes/SSD_Hayoung/fronts/training/era5_YYYY_training.nc`
**Variables:** `t850`, `u850`, `v850`, `tfp_850`, `front_label` (0=BG 1=CF 2=WF 3=SF)

### TFP interpretation

```
       [Cold air]     FRONT LINE     [Warm air]
TFP :    < 0       →     = 0     ←    > 0
```

Labels: CF if T-advection < 0 at TFP zero-crossing, WF if > 0, SF if near zero.

---

## Step 3 — [NEW] Extract WPC Front Labels from Image Archive

```bash
# Full 2007–2026 batch (downloads & caches GIFs, ~17h first run)
bash scripts/run_wpc_batch.sh

# Re-extract from cache with updated calibration params (~30 min, no download)
bash scripts/run_wpc_reextract.sh

# Single date for testing
python scripts/extract_wpc_fronts.py --date 2020-06-15 --hours 12 --verify
```

Extracts CF/WF/OF/SF front masks from WPC `namfntsfc` GIF archive (2007–present).

### Pipeline within this step:
```
WPC GIF image
    │
    ├─ Color detection (fixed RGB palette per front type)
    │   CF=(0,0,255)/(0,178,238)  WF=(255,0,0)  OF=(145,44,238)
    │   Isobars are DARK red (139,0,0) → separable from bright-red WF.
    │   (No magenta in the palette: WPC has no stationary-front colour.)
    │
    ├─ Clean: drop sub-extent components (H/L letters & pressure numbers are
    │   small; fronts are long), then skeletonize to a 1-px centerline.
    │
    ├─ Stationary fronts (SF): recovered from the red+blue corridor (the
    │   alternating warm/cold symbols), and removed from CF/WF to undo the
    │   double-count.
    │
    └─ Full-affine projection → ERA5 0.25° grid
        Includes a real ~10° image rotation (b,d ≠ 0); LCC lon0=-100°W.
        Coastline-ICP calibrated, stable across 2007–2026.
```

**Output:** `/Volumes/SSD_Hayoung/fronts/wpc_labels/wpc_labels_YYYY.nc`
**Coverage:** 2007–2026, 6-hourly, 28,449 timesteps, ~60 MB total
**Cache:** `/Volumes/SSD_Hayoung/fronts/wpc_gif_cache/` (~6.7 GB, ~28,000 GIFs)

### Calibration notes (overhauled 2026-06)

WPC images use a Lambert Conformal Conic (LCC) projection, but the pixel↔grid
mapping needs a **full 2D affine** — the image axes are rotated ~10° relative
to the LCC axes (consistent across 2007–2026). The earlier **diagonal** affine
ignored this rotation and placed fronts ~100–360 km off.

Current calibration (`calibrate_from_coastline`): ICP of the GIF's black
coastline/border pixels against Natural Earth geography, fitting a full affine.
Validated against analyst bulletins (n=64): CF recall 364→216 km after the fix.

> Earlier broken states now fixed: (1) the cleaning filter rejected curved
> fronts and returned **0 px**; (2) the diagonal projection rotation error;
> (3) SF was unextractable and always empty. See `REPORT.md §6.5`.

---

## Step 4 — [NEW] Build Hybrid Labels

```bash
# Mac (reads from /Volumes/SSD_Hayoung/)
python scripts/build_hybrid_labels.py --years 2019 2020 2021 2022 2023 2024 2025
python scripts/build_hybrid_labels.py --years 2019 2024 --tfp-thresh 0.12 --dilate 2
python scripts/build_hybrid_labels.py --years 2024 --verify 2024  # with QC plot

# NASA Discover (self-contained: downloads WPC GIFs, projects, intersects TFP)
python scripts/build_hybrid_discover.py --years 2019 2020 2021 2022 2023 2024 2025
```

> **Discover coordinate note (critical):** Discover training files store lat **ascending**
> (15→70N, from `sortby('lat')` in `build_training_data_discover.py`), while the WPC
> extraction grid is lat **descending** (70→15N). Intersecting by raw array index
> silently N-S flips the masks — OF (near TFP threshold) drops from ~104 K to ~26 K.
> Always align by coordinate: `wpc = wpc.reindex(lat=tr["lat"].values, lon=tr["lon"].values, fill_value=0)`.

**Novel contribution:** Intersects ERA5 TFP (accurate position) with WPC labels
(accurate front type) to produce superior 5-class training targets.

### Intersection logic

```
For each grid cell (i, j) at each timestep:

  if |TFP(i,j)| > threshold:          # ERA5 confirms frontal gradient
      AND WPC_CF within ±2 cells:   → label = CF (1)
      AND WPC_WF within ±2 cells:   → label = WF (2)
      AND WPC_SF within ±2 cells:   → label = SF (3)
      AND WPC_OF within ±2 cells:   → label = OF (4)  ← NEW CLASS
      else:                          → label = BG (0)  [TFP over-detection removed]

  else:                               → label = BG (0)  [no ERA5 gradient]
```

The ±2 cell (±50 km) dilation of WPC labels compensates for the residual
projection offset of the image extraction (now ~100–200 km after the
full-affine fix; see Step 3). With SF recovery, the `WPC_SF → SF` branch is
finally non-empty, so hybrid labels carry all five classes.

### Why this is novel

No prior study has combined:
- ERA5 thermodynamic field positions (exact grid alignment)
- WPC analyst type classification (expert CF/WF/OF judgment)
- 20-year image archive (2007–2026)
- Skeletonized centerlines (removes thickness bias)

Most ML front detection work uses either TFP auto-labels (no type distinction,
over-detection) or coded bulletins (limited archive, text-only coordinates).
This hybrid approach uses both sources for their respective strengths.

**Output:** `/Volumes/SSD_Hayoung/fronts/hybrid_labels/hybrid_YYYY.nc`
**Variables:** `t850`, `u850`, `v850`, `tfp_850`, `front_label` (0–4), `tfp_label` (0–3)
**Label encoding:** 0=BG 1=CF 2=WF 3=SF 4=OF

---

## Step 5 — Build Extra Channels

```bash
# Mac (reads from /Volumes/SSD_Hayoung/ERA5/)
python scripts/build_extra_channels.py --years 2019 2025 --overwrite

# NASA Discover (reads from /css/era5/)
python scripts/build_extra_channels_discover.py --years 2019 2025
```

Extracts 8 additional atmospheric variables per year for Run 5 12-channel input.

**Output:** `extra_channels_YYYY.nc`
**Variables:** `z500`, `q850`, `w850`, `msl`, `t925`, `t2m`, `u10`, `v10`

---

## Step 6 — Train U-Net

```bash
# Run 4: 4-channel TFP labels
python scripts/train_unet.py --epochs 30 --batch 8 \
    --train 2019 2020 2021 2022 2023 2024 --val 2025

# Run 5a: 12-channel, hybrid labels (completed ep28, F1=0.693 — OF=0 due to label bugs)
# Run 5b: 12-channel, corrected hybrid labels (active on Discover A100, OF F1=0.172 at ep3)
python scripts/train_unet_v4.py \
    --train 2019 2024 --val 2025 2025 \
    --epochs 30 --batch 32 \
    --data-root /path/to/fronts/data

python scripts/train_unet_v4.py --resume   # continue from checkpoint
```

> **Discover note:** On Discover, SLURM runs the root-level `train_unet_v4.py` (not
> `scripts/`). After editing `scripts/train_unet_v4.py`, always `cp scripts/train_unet_v4.py train_unet_v4.py`.
> Discover hybrid files contain only `front_label` (not t850/u850/etc.); the training
> script reads ERA5 vars from `era5_YYYY_training.nc` automatically when t850 is absent.

Checkpoints: `models/`

---

## Step 7 — Plot Training Summary

```bash
python scripts/plot_training.py
```

Generates training progress figures (Loss, F1 by class, Run comparison).

---

## Step 8 — Inference / Prediction

```bash
python scripts/train_unet_v4.py --predict 2025-06-15T00
```

---

## Step 10 — Compare with WPC Analyst

```bash
python scripts/compare_unet_wpc.py --time 2024-04-03T00
python scripts/compare_unet_wpc.py --period 2026
```

---

## Step 11 — TFP Visualisation (standalone)

```bash
python scripts/batch_tfp_arco.py 2024 2024
python scripts/plot_tfp_global.py 2024-09-26T18 0.2
```

---

## File Structure

```
/Volumes/SSD_Hayoung/
├── ERA5/
│   ├── pressure_level/   era5_PL_YYYYMM.nc
│   └── single_level/     era5_SFC_YYYYMM.nc
└── fronts/
    ├── training/         era5_YYYY_training.nc      (TFP-based labels, 4 class)
    ├── extra_channels/   extra_channels_YYYY.nc     (z500/q850/w850/msl/t925/t2m/u10/v10)
    ├── hybrid_labels/    hybrid_YYYY.nc             (Hybrid labels, 5 class + OF)
    ├── wpc_labels/       wpc_labels_YYYY.nc         (WPC image extraction)
    ├── wpc_gif_cache/    namfntsfc*.gif             (28,442 cached images)
    ├── coded_sfc/        codsus*_hr                 (WPC text bulletins, 2-week)
    ├── models/           unet_*.pt, *_metrics.csv
    └── tfp_labels/       tfp_YYYY.nc

/Users/hayoungbong/Analysis/Front/
├── README.md             Project overview + novel contribution description
├── PIPELINE.md           This file
├── scripts/
│   ├── extract_wpc_fronts.py     WPC image extraction + skeletonization
│   ├── build_hybrid_labels.py    Hybrid label builder (TFP ∩ WPC)
│   ├── build_training_data.py    ERA5 TFP training data
│   ├── train_unet.py             U-Net training + inference
│   └── ...
└── figures/
    ├── wpc_quality_analysis.png  20-year WPC label quality report
    ├── calibration_comparison.png  Old vs new calibration vs truth
    ├── hybrid_verify_YYYY.png    Hybrid label QC plots
    └── training_v2_progress.png  Training curves
```
