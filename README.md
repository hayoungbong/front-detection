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
| **Best result** | Run 2: F1=**0.768** (4ch TFP) · Run 5b: F1=**0.255** (12ch hybrid corrected, OF=0.242, complete) |
| **Compute** | Mac M-series (MPS) · NASA Discover A100 GPU |

---

## Training Run History

| Run | Channels | Label strategy | Training data | Best F1 | Status |
|-----|---------|---------------|--------------|---------|--------|
| 1 | 4 | TFP classification | 2020–2021 | 0.675 | ✅ Complete |
| 2 | 4 | TFP classification | 2019–2021 | **0.768** | ✅ Complete |
| 3 | 8 | Hybrid WPC (smoke test)† | 2024 only | 0.127 | ✅ Complete |
| 4 | 4 | TFP classification | 2019–2024 | 0.642* | 🔄 A100 queued |
| 5a | 12 | Hybrid ERA5×WPC (OF bug)‡ | 2019–2024 | 0.693 | ✅ Complete |
| **5b** | **12** | **Hybrid corrected, 5-class** | **2019–2024** | **0.255** | ✅ Complete |
| 6 | 12 | TFP labels (12ch baseline) | 2019–2024 | — | 🔄 A100 queued |
| 7 | 11 | ERA5 regression (continuous) | 2019–2024 | — | 🔄 A100 queued |

†Run 3: 1 year, 10 epochs; not converged — first OF detection (F1=0.063).  
‡Run 5a: OF silently zero due to N-S coordinate flip + priority overwrite in `build_hybrid_discover.py`.  
*Run 4 CPU partial (ep10 best); resubmitted on A100 for full training.

### Run 5b — Final Results (2026-06-29, A100)

First complete hybrid run with corrected 5-class labels.
Label counts (2019): CF=524 K · WF=215 K · SF=352 K · OF=104 K (Berry et al. 2011 consistent).

| Ep | CF | WF | SF | OF | Mean |
|----|----|----|----|----|------|
| 1 | 0.181 | 0.092 | 0.223 | 0.081 | 0.144 |
| 5 | 0.266 | 0.140 | 0.262 | 0.165 | 0.208 |
| 10 | 0.275 | 0.160 | 0.277 | 0.190 | 0.225 |
| 20 | 0.302 | 0.173 | 0.278 | 0.221 | 0.243 |
| **30 (final)** | **0.315** | **0.175** | **0.289** | **0.242** | **0.255** |

OF detected from ep1 — Run 5a had OF=0 for all 28 epochs.  
Plateau ep25–30 (F1 0.252–0.255); training-set expansion (2007–2018 hybrid labels) is the next step.

**WPC label-pipeline overhaul (2026-06):** fixed front extraction (0-px bug), ~10° projection
rotation (100–360 km offset), and SF class (alternating red/blue, previously undetected).
20 years of labels regenerated; SF populated for the first time. See `REPORT.md` §6.5.

---

## Run 5: 12-Channel Hybrid

Run 5 combines ERA5 thermodynamics (position accuracy) with WPC analyst labels (expert type
judgment) over 12 input channels. Run 5a (ep28, F1=0.693) had OF=0 throughout due to two bugs
in `build_hybrid_discover.py`: (1) OF priority overwrite by CF/WF/SF, (2) N-S coordinate flip
(Discover lat ascending vs. WPC descending). Fixed via `wpc.reindex(lat=..., lon=...)` with OF
assigned last. Run 5b is the corrected run (ep30, F1=0.255, OF=0.242).

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
| MacBook Pro (M-series) | Apple MPS | 8 | Development, Runs 1–3 |
| NASA Discover PRISM | NVIDIA A100-SXM4-40GB | 32 | Runs 4–7 (A100 GPU) |

**Discover:** SLURM `gpu_a100` partition, PyTorch 2.6.0 + CUDA 12.4, ERA5 at `/css/era5/`.
The A100-SXM4-40GB (40 GB HBM2) is a top-tier research GPU — the HPC standard before H100,
still widely used for scientific ML. For this U-Net scale, the bottleneck is NetCDF I/O,
not GPU compute; once training begins, the A100 handles batch 32 with ample headroom.

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
