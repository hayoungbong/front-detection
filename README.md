# Weather Front Detection with Deep Learning

Automated detection and classification of synoptic-scale weather fronts
(Cold / Warm / Stationary / Occluded) over North America using ERA5 reanalysis
and a U-Net deep learning model.

---

## Project Summary

| Item | Detail |
|------|--------|
| **Domain** | 15–70°N, 170–50°W (North American frontal zone) |
| **Resolution** | 0.25°, 6-hourly (00/06/12/18 UTC) |
| **Input data** | ERA5 reanalysis (CDS API), 2019–2025 |
| **Labels** | TFP-based (Runs 1–2, 4, 6) · Hybrid ERA5×WPC 5-class (Runs 3, 5) |
| **Models** | U-Net classification (4–12 ch) · U-Net regression (11 ch → 3 targets) |
| **Best F1** | Run 4: **0.718** (4-ch) · Run 5b: **0.255** incl. OF |
| **Best r** | Run 7: **r=0.993** (physical regression) |
| **Compute** | Mac M-series MPS (inference) · NASA Discover A100 (training) |

---

## Training Run History

| Run | Ch | Labels | Train | Best F1 / r | Status |
|-----|----|--------|-------|-------------|--------|
| 1 | 4 | TFP | 2020–21 | F1=0.675 | ✅ |
| 2 | 4 | TFP | 2019–21 | F1=**0.768** | ✅ |
| 3 | 8 | Hybrid WPC† | 2024 only | F1=0.127 | ✅ |
| **4** | **4** | **TFP** | **2019–24** | **F1=0.718** | **✅ Best classifier** |
| 5a | 12 | Hybrid (OF bug‡) | 2019–24 | OF=0 | ✅ ablation |
| **5b** | **12** | **Hybrid corrected** | **2019–24** | **F1=0.255, OF=0.242** | **✅ OF-capable** |
| 6 | 12 | TFP (12ch ablation) | 2019–24 | F1=0.688 | ✅ |
| **7** | **11** | **Regression** | **2019–24** | **r=0.993** | **✅ Best predictor** |

†Run 3: smoke test, 10 epochs, first OF detection.  
‡Run 5a: OF=0 throughout due to coordinate mismatch in label builder.

---

## Run Progression and Key Findings

```
Run 1 (4ch, 2yr)  →  Run 2 (4ch, 3yr)  →  Run 4 (4ch, 6yr)  ← BEST CLASSIFIER
                                                  ↓ hybrid labels
                                            Run 5b (12ch, 5-class)  ← OF CAPABLE

Run 6 (12ch TFP ablation): 12-ch < 4-ch by 0.030 F1
→ Finding: extra channels hurt when labels are TFP-derived (redundant with t850)

Run 7 (11ch → tfp/tadv/∇T regression): r = 0.993
→ Finding: U-Net backbone can learn physical diagnostics near-perfectly
```

---

## Scientific Applications

### 1. Hurricane ET Analysis

6-hourly front detection through TC lifetime → timeseries of CF/WF fraction.

- **Ida 2021**: WF ingestion post-landfall → ET signal detected
- **Ian 2022**: CF interaction through FL panhandle track
- **Helene 2024**: WF fraction peaks **exactly** at ET onset (9/27); model independently
  confirms IBTrACS ET classification. Post-ET frontal gap connects to NJ historic drought.

### 2. Wildfire Risk

Fire Weather Index (FWI) + CF probability maps. Cold front passage = fire-spread
precursor (post-frontal dry + gusty). Case study: CA 2021-08-29.

### 3. Front Climatology (2019–2025)

- Annual and seasonal CF/WF/SF frequency maps
- Zonal mean frequency by latitude
- Linear trend per grid cell (2019–2025)
- Fire hotspot (FIRMS) overlay

---

## Repository Structure

```
front-detection/
├── scripts/
│   ├── train_unet_v4.py          # Classifier (Runs 4–6)
│   ├── train_unet_reg.py         # Regressor (Run 7)
│   ├── build_training_data.py    # ERA5 → training.nc
│   ├── build_extra_channels.py   # Extra 8 channels
│   ├── build_hybrid_discover.py  # Hybrid ERA5×WPC labels
│   ├── download_era5_*.py        # CDS download helpers
│   └── submit_run{4,5,6,7}_gpu.sh  # Discover A100 SLURM scripts
├── README.md
├── REPORT.md
└── PIPELINE.md
```

---

## Quick Start (inference)

```bash
conda activate geospy_env
export KMP_DUPLICATE_LIB_OK=TRUE

# Load Run 4 (classification)
import torch
from scripts.train_unet_v4 import UNet
ckpt  = torch.load("run4_best.pt", weights_only=False)
model = UNet(in_ch=4, num_classes=4)
model.load_state_dict(ckpt["model_state"])

# Load Run 7 (regression)
from scripts.train_unet_reg import UNet as UNetReg
ckpt  = torch.load("run7_best.pt", weights_only=False)
model = UNetReg(in_ch=ckpt["in_ch"], n_out=ckpt["n_out"])
model.load_state_dict(ckpt["model_state"])

# Always pad to multiple of 16
import numpy as np
H, W = x.shape[-2:]
pH = (16 - H%16)%16; pW = (16 - W%16)%16
x  = np.pad(x, ((0,0),(0,0),(0,pH),(0,pW)), mode="reflect")
pred = model(torch.tensor(x[None]))[..., :H, :W]
```

---

## Data

| Source | Variable | Temporal coverage |
|--------|----------|-------------------|
| ERA5 pressure levels | t/u/v/z/q/w @ 850/500/925 hPa | 2019–2025 |
| ERA5 single level | msl/t2m/u10/v10/tp | 2019–2025 |
| FIRMS MODIS | Fire hotspots | 2000–2025 |
| IBTrACS | TC best tracks (NA/EP/WP) | 1842–2025 |
| WPC frontal analysis | GIF archives | 2007–2024 |
