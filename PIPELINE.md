# Analysis Pipeline

End-to-end workflow, from raw ERA5 reanalysis to trained front models and their
scientific applications. Paths and cluster-specific details are omitted; this
describes the *logical* stages.

---

## Stage 0 — Data acquisition

| Data | Source | Used for |
|------|--------|----------|
| ERA5 pressure & single levels | Copernicus CDS API | Model input fields |
| WPC surface analysis | NOAA/WPC archive | Expert front types (Hybrid labels) |
| IBTrACS | NOAA best-track archive | Tropical-cyclone application |
| FIRMS (MODIS/VIIRS) | NASA FIRMS | Wildfire application |

```bash
python3 scripts/download_era5_batch.py --type pressure_level --start 2010 --end 2025
python3 scripts/download_era5_batch.py --type single_level   --start 2010 --end 2025
python3 scripts/download_ibtracs.py
python3 scripts/download_firms.py --start 2010 --end 2025
```

---

## Stage 1 — Pre-processing & label construction

```bash
# Core training fields: t850/u850/v850, TFP, and TFP-based front_label
python3 scripts/build_training_data.py   --years 2010 ... 2025

# Extra input channels: z500/q850/w850/msl/t925/t2m/u10/v10
python3 scripts/build_extra_channels.py  --years 2010 ... 2025

# Hybrid labels: ERA5/TFP position  ∩  WPC expert type (CF/WF/SF/OF)
python3 scripts/build_hybrid_labels.py   --years 2010 ... 2025
```

The **Hybrid label** step is the heart of the pipeline: it keeps the ERA5/TFP
front *position* only where the WPC analysis confirms a front *type* nearby,
suppressing human-drawn noise while gaining the type information (including
occluded fronts) that TFP alone cannot provide.

---

## Stage 2 — Model training

```bash
# Classifier — discrete front types from ERA5 + Hybrid labels
python3 scripts/train_unet_classifier.py \
    --train 2010 2024 --val 2025 2025 --epochs 30 --run-name run8

# Regressor — continuous ERA5 frontal fields, no WPC labels
python3 scripts/train_unet_regression.py \
    --train 2010 2024 --val 2025 2025 --epochs 30 --run-name run8_reg
```

Each run saves a `.pt` checkpoint (the trained network weights). Training scales
from a single GPU to multi-GPU distributed training for the longer runs.

---

## Stage 3 — Inference & applications

A trained `.pt` model can be applied to any ERA5 time step.

```bash
# Tropical-cyclone / front interaction (extratropical transition)
python3 scripts/plot_hurricane_front.py --tc HELENE2024 --model run4

# Wildfire risk: cold-front passage + fire-weather index
python3 scripts/plot_wildfire_risk.py --mode casestudy --date 2021-08-29

# Multi-year front climatology and trends
python3 scripts/front_climatology.py --years 2010 ... 2025
```

---

## Core scripts

| Script | Role |
|--------|------|
| `build_training_data.py` | Build ERA5 training fields + TFP labels |
| `build_extra_channels.py` | Build the additional input channels |
| `build_hybrid_labels.py` | **Fuse ERA5/TFP positions with WPC types** |
| `train_unet_classifier.py` | Train the front-type classifier |
| `train_unet_regression.py` | Train the WPC-independent regressor |
| `front_climatology.py` | Apply a model across many years → climatology |
| `plot_hurricane_front.py` | TC / front interaction case studies |
| `plot_wildfire_risk.py` | Front / wildfire-weather analysis |

---

## Environment

Python 3.10+ with PyTorch, xarray, and the scientific Python stack. ERA5 access
requires a (free) Copernicus CDS account; FIRMS access requires a (free) NASA
FIRMS map key.
