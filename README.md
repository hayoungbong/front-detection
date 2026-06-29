# Weather Front Detection with Deep Learning

Automated detection and classification of synoptic-scale weather fronts
(Cold / Warm / Stationary / Occluded) over North America and surrounding
oceans using ERA5 reanalysis and a U-Net deep learning model.

---

## Project Summary

| Item | Detail |
|------|--------|
| **Domain** | 5–85°N, 180°W–10°E (ETC full lifecycle coverage) |
| **Resolution** | 0.25°, 6-hourly (00/06/12/18 UTC) |
| **Input data** | ERA5 reanalysis (CDS API), 1970–2026 |
| **Labels** | TFP-based (Runs 1–2) → Hybrid ERA5×WPC 5-class (Runs 3–5) |
| **Model** | U-Net (Focal Loss γ=2, AdamW, CosineAnnealingLR), 4→12 channels |
| **Best result** | Run 2: Mean F1 = **0.768** · Run 4 in progress (ep12+, best 0.642) · Run 5 running on A100 |
| **Compute** | Mac M-series (MPS) · NASA Discover A100 GPU |

---

## Training Run History

| Run | Script | Input channels | Label strategy | Train data | Epochs | Mean F1 | Status |
|-----|--------|---------------|----------------|------------|--------|---------|--------|
| **Run 1** | `train_unet.py` | 4 (t850/u850/v850/tfp) | TFP classification | 2020–2021 | 30 | 0.675 | ✅ Complete |
| **Run 2** | `train_unet.py` | 4 (t850/u850/v850/tfp) | TFP classification | 2019–2021 | 30 | **0.768** | ✅ Complete |
| **Run 3** | `train_unet_v3.py` | 8 (+z500/q850/w850/msl) | Hybrid ERA5×WPC (5-class+OF) | 2024 only | 10 | 0.127* | ✅ Smoke test |
| **Run 4** | `train_unet.py` | 4 (t850/u850/v850/tfp) | TFP classification | 2019–2024 | 30 | 0.642† | 🔄 In progress (Discover CPU ep12+) |
| **Run 5** | `train_unet_v4.py` | **12** (base 4 + z500/q850/w850/msl/t925/t2m/u10/v10) | **Hybrid ERA5×WPC, 5-class (BG/CF/WF/SF/OF)** | 2019–2024 | 30 | — | 🔄 **Running (Discover A100 GPU)** |

*Run 3 smoke test: 2024 only, 10 epochs, not converged.
†Run 4 best so far at epoch 10 (CF=0.751, WF=0.697, SF=0.476), still training.

**WPC label-pipeline overhaul (2026-06):** the WPC ground-truth extraction was fixed —
front extraction (had silently returned 0 px), a ~10° projection rotation (fronts were
100–360 km off), and the **stationary-front class** (recovered from the alternating
red/blue symbols, previously always empty). 20 years of labels regenerated; SF populated
for the first time, unblocking the 5-class **Run 5**. See `REPORT.md` §6.5 / `PIPELINE.md` Step 3.

### Run 4 — Current Results (Epoch 10/30, NASA Discover CPU)

Run 4 establishes the TFP 4-channel ceiling with 6 years of training data.
Directly comparable to Run 2 (same model, same label strategy, 2× training data).

| Epoch | CF | WF | SF | Mean |
|-------|----|----|-----|------|
| 1 | 0.617 | 0.549 | 0.301 | 0.489 |
| 5 | 0.653 | 0.602 | 0.383 | 0.546 |
| 10 | **0.751** | **0.697** | **0.476** | **0.642** |

*Training ongoing. Expected to surpass Run 2 (0.768) by epoch 20–25.*

### Run 2 — Final Metrics (Epoch 30, train 2019–2021 / val 2022)

| Class | F1 | Notes |
|-------|----|-------|
| Cold Front (CF) | 0.837 | Strong signal from temperature gradient |
| Warm Front (WF) | 0.822 | Consistent with CF |
| Stationary Front (SF) | 0.645 | Challenging — low temperature advection |
| **Mean** | **0.768** | Best checkpoint at epoch 30 |

### Run 3 — Smoke Test (Epoch 10, 2024 data only)

| Class | F1 | Notes |
|-------|----|-------|
| Cold Front (CF) | 0.165 | Still improving |
| Warm Front (WF) | 0.153 | Still improving |
| Stationary Front (SF) | 0.000 | Insufficient data |
| Occluded Front (OF) | 0.063 | **First successful OF detection** |
| **Mean** | **0.127** | Not converged — 1 year only |

---

## Run 5: 12-Channel Hybrid (Training)

Run 5 is the first full hybrid training run — combining ERA5 thermodynamics
(position accuracy) with WPC analyst labels (expert type judgment) and
expanded 12-channel atmospheric input.

### 12-Channel Input

| Channel | Variable | Source | Physical role |
|---------|----------|--------|---------------|
| 1 | `t850` | ERA5 PL | 850 hPa temperature |
| 2 | `u850` | ERA5 PL | 850 hPa zonal wind |
| 3 | `v850` | ERA5 PL | 850 hPa meridional wind |
| 4 | `tfp_850` | Derived | Thermal Front Parameter |
| 5 | `z500` | ERA5 PL | 500 hPa geopotential (trough/ridge) |
| 6 | `q850` | ERA5 PL | 850 hPa specific humidity (moisture transport) |
| 7 | `w850` | ERA5 PL | 850 hPa vertical velocity (ascent/descent) |
| 8 | `msl` | ERA5 SFC | Mean sea level pressure (synoptic pattern) |
| 9 | `t925` | ERA5 PL | 925 hPa temperature (boundary layer) |
| 10 | `t2m` | ERA5 SFC | 2 m temperature |
| 11 | `u10` | ERA5 SFC | 10 m zonal wind |
| 12 | `v10` | ERA5 SFC | 10 m meridional wind |

### 5 Output Classes

| Label | Class | Description |
|-------|-------|-------------|
| 0 | BG | Background (no front) |
| 1 | CF | Cold Front |
| 2 | WF | Warm Front |
| 3 | SF | Stationary Front |
| 4 | OF | Occluded Front |

SF (Stationary Front) is now fully populated for the first time, after
the WPC label pipeline overhaul that recovered SF from the alternating
red/blue symbol pattern (previously always 0 px).

### Auto-Chain
Run 5 starts automatically after Run 4 via `run5_chain.sh`:
1. Wait for Run 4 to finish (process-based polling)
2. Build `extra_channels_YYYY.nc` for 2019–2025 from ERA5 regional (no new downloads)
3. Rebuild hybrid labels 2022–2025 with corrected WPC (SF populated)
4. Launch `train_unet_v4.py --train 2019 2024 --val 2025 2025 --epochs 30`

---

## Run 4: What's New and Why

Run 4 rebuilds the training pipeline from scratch with a new ERA5 regional
download and 7-variable training data format.

### Key Improvements Over Run 2

| | Run 2 | Run 4 |
|--|-------|-------|
| Training data source | Old 5-variable files | **New 7-variable pipeline** |
| ERA5 download | Original regional (no 925 hPa) | **New download with 925 hPa** |
| Training years | 2019–2021 | **2019–2024** (6 years) |
| Training samples | ~4,380 | **~8,768** (2× increase) |
| Validation | Internal years | **2025 (held-out year)** |
| Data leakage | Not checked | **Verified clean split** |
| Regression targets | None | **tadv_850, grad_mag_850 available** |

### New 7-Variable Training Format

Training files now include 7 variables (vs. 5 in Runs 1–2):

| Variable | Role |
|----------|------|
| `t850` | Input: 850 hPa temperature |
| `u850` | Input: 850 hPa zonal wind |
| `v850` | Input: 850 hPa meridional wind |
| `tfp_850` | Input: Thermal Front Parameter |
| `front_label` | Classification target (BG/CF/WF/SF) |
| `tadv_850` | Regression target: temperature advection |
| `grad_mag_850` | Regression target: \|∇T\| magnitude |

The regression targets (`tadv_850`, `grad_mag_850`) are not used by `train_unet.py`
but are available for future regression-based training.

### Regression Targets: Why They Matter

Continuous physical fields as regression targets are climate-change-robust:
a fixed TFP classification threshold becomes biased as Arctic Amplification
weakens meridional temperature gradients over decades. Continuous regression
does not suffer from this threshold drift.

| Variable | Physical meaning | Advantage |
|----------|-----------------|-----------|
| `tadv_850` | Temperature advection (-**v**·∇T) | Sign gives CF/WF naturally, magnitude gives intensity |
| `grad_mag_850` | \|∇T\| at 850 hPa | Frontal intensity without discretization |

---

## Compute Infrastructure

Training runs in parallel across three platforms:

| Platform | Device | Batch | Role |
|----------|--------|-------|------|
| MacBook Pro (M-series) | Apple MPS | 8 | Development, Run 4 |
| NASA Discover HPC | NVIDIA A100-SXM4-40GB | 32 | Run 5 GPU training |

**Discover:** SLURM `gpu_a100` partition, PyTorch 2.6.0 + CUDA 12.4, ERA5 at `/css/era5/`.

---

## Pipeline

```
ERA5 Regional Download (CDS API)
  5–85°N, 180°W–10°E, 1970–2026
  PL: T/u/v/Z/q/ω @ 500/700/850/900/925/950/1000 hPa
  SFC: 2mT, Td, 10m wind, MSLP, Ps
         │
         ▼
build_training_data.py
  ├─ Crop to training domain (15–70°N, 170–50°W)
  ├─ Compute TFP, |∇T|, temperature advection
  ├─ Generate classification labels (CF/WF/SF/BG)
  └─ Save regression targets (tfp_850, tadv_850, grad_mag_850)
         │
         ├─────────────────────────────────┐
         ▼                                 ▼
  WPC GIF Archive                   Extra Channels
  extract_wpc_fronts.py             build_extra_channels.py
  CF/WF/OF/SF extraction            z500, q850, w850, msl,
  Skeletonize → LCC → ERA5 grid     t925, t2m, u10, v10
         │                                 │
         ▼                                 │
  build_hybrid_labels.py                   │
  TFP position ∩ WPC type                  │
  5-class: BG/CF/WF/SF/OF                  │
         │                                 │
         └──────────────┬──────────────────┘
                        ▼
               U-Net Training
         train_unet_v4.py (Run 5)
         12-channel input, 5-class output
         Focal Loss γ=2, AdamW, CosineAnnealingLR
                        │
                        ▼
              Evaluation vs WPC analyst
              compare_unet_wpc.py
```

---

## Label Strategies

### TFP Classification (Runs 1–2)
Objective labels derived entirely from ERA5 thermodynamics.
CF/WF/SF determined by TFP zero-crossing location + temperature advection sign.
No human judgment involved — fully reproducible, globally applicable.
**Limitation**: arbitrary threshold (0.12 K/(100km)²) and systematic
over-detection vs. human-drawn fronts.

### Hybrid ERA5×WPC (Runs 3–4)
Intersects ERA5 TFP (precise position) with WPC analyst images (expert type):
```
Label(i,j) = WPC_type   if |TFP| > threshold AND WPC within ±50 km
           = Background  otherwise
```
**Benefit**: eliminates TFP over-detection; adds Occluded Front class (impossible with TFP alone).
**Trade-off**: inherits WPC analyst subjectivity and limited to 2007–present archive.

### Regression Targets (Run 4+)
Predict continuous physical fields instead of discrete categories.
No arbitrary thresholds — climate-change robust, physically consistent.
CF/WF distinction emerges naturally from temperature advection sign.

---

## Scripts

### Core Pipeline

| Script | Step | Description |
|--------|------|-------------|
| `download_era5_regional.py` | 1 | ERA5 regional download (5–85°N), 1970–2026, 925 hPa included, resume-safe |
| `download_era5_batch.py` | 1 | ERA5 global download for separate server |
| `build_training_data.py` | 2 | TFP computation, classification labels + regression targets |
| `extract_wpc_fronts.py` | 3 | WPC GIF → color extraction → skeletonize → LCC → ERA5 grid |
| `calibrate_wpc_projection.py` | 3 | LCC projection calibration using WPC coded bulletins |
| `build_hybrid_labels.py` | 4 | ERA5 TFP ∩ WPC labels → 5-class hybrid NetCDF |
| `build_extra_channels.py` | 5 | Extract z500/q850/w850/msl/t925/t2m/u10/v10 per year |
| `train_unet.py` | 6 | Run 1/2: 4-channel, 4-class TFP labels |
| `train_unet_v3.py` | 6 | Run 3: 8-channel, 5-class hybrid labels |
| `train_unet_v4.py` | 6 | Run 4: 12-channel, 5-class hybrid + regression option |
| `compare_unet_wpc.py` | 7 | 3-panel evaluation: TFP / U-Net / WPC analyst |
| `analyze_run2.py` | 7 | Run 2 comprehensive analysis and WPC comparison |

### Analysis & Utilities

| Script | Description |
|--------|-------------|
| `parse_coded_sfc.py` | Parse WPC coded surface bulletins (exact lat/lon front coordinates) |
| `verify_wpc_projection.py` | Verify WPC LCC projection quality |
| `diagnose_wpc_projection.py` | Diagnose projection calibration errors |
| `add_925hpa_to_existing.py` | Interpolate 925 hPa into legacy ERA5 files (deprecated — use regional download) |

---

## Data Structure

```
/Volumes/SSD_Hayoung/ERA5/               ← symlinked from data/era5/
├── pressure_level/   era5_PL_YYYYMM.nc  T,u,v,Z,q,ω @ 7 levels (incl. 925 hPa)
├── single_level/     era5_SFC_YYYYMM.nc 2mT,Td,10m wind,MSLP,Ps
├── pressure_level_v1/                   Legacy files (850/900/950/1000 only)
└── single_level_v1/                     Legacy files

/Volumes/SSD_Hayoung/fronts/
├── training/         era5_YYYY_training.nc   t850/u850/v850/tfp + labels + regression targets
├── hybrid_labels/    hybrid_YYYY.nc           5-class hybrid labels (2019–2025)
├── wpc_labels/       wpc_labels_YYYY.nc       WPC extracted fronts on ERA5 grid
├── wpc_gif_cache/    namfntsfc*.gif            ~28,000 images cached
├── extra_channels/   extra_channels_YYYY.nc   z500/q850/w850/msl/t925/t2m/u10/v10
└── models/
    ├── unet_2019-2024_e30_b8_best.pt          Run 2 best checkpoint
    ├── unet_2019-2024_e30_b8_metrics.csv      Run 2 per-epoch metrics
    ├── unet_v3_hybrid_2024-2024_e10_b8_best.pt  Run 3 smoke test
    └── unet_v3_hybrid_2024-2024_e10_b8_metrics.csv
```

---

## Model Architecture

```
Input [C, 221, 481]    C = 4 / 8 / 12 channels

Encoder
  ConvBlock(C → 64)    [64, 221, 481]
  MaxPool + CB(→128)   [128, 111, 241]
  MaxPool + CB(→256)   [256,  56, 121]
  MaxPool + CB(→512)   [512,  28,  61]
  MaxPool + CB(→1024)  [1024, 14,  31]  ← bottleneck

Decoder (ConvTranspose2d + skip connection)
  1024+512 → 512
   512+256 → 256
   256+128 → 128
   128+ 64 →  64

Head: 1×1 Conv → N classes (4 or 5) or regression outputs
Parameters: ~31M
```

ConvBlock = Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → ReLU

---

## Setup

```bash
conda activate geospy_env
pip install xarray zarr gcsfs scipy cartopy matplotlib cdsapi
pip install torch torchvision opencv-python-headless scikit-image

# CDS API key required at ~/.cdsapirc
```

---

## References

- Renard & Clarke (1965): Thermal Front Parameter
- Berry et al. (2011): Global front climatology — *Geophys. Res. Lett.*
- Hewson (1998): Objective fronts by TFP — *Meteorological Applications*
- Biard & Kunkel (2019): Automated front detection with ML
- Ronneberger et al. (2015): U-Net — *MICCAI 2015*
- Lin et al. (2017): Focal Loss — *ICCV 2017*
- Hersbach et al. (2020): ERA5 global reanalysis — *QJRMS*
