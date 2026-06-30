# Weather Front Detection with Deep Learning

## Overview

A PyTorch U-Net learns to detect and classify synoptic-scale weather fronts
(Cold / Warm / Stationary / Occluded) over North America from ERA5 reanalysis.
It is trained on **Hybrid labels** that fuse objective ERA5/TFP front *positions*
with expert WPC front *types* — giving an automatic, reproducible front product
that can be generated for any period and linked to hurricanes, drought, and
wildfire.

TFP (Thermal Front Parameter) is a scalar field derived purely from ERA5
temperature gradients that marks where air-mass boundaries are sharpening.

---

## Method

### The idea

Weather fronts are one of the most important features on a surface analysis
map, but the authoritative operational product still depends on **human
forecaster judgment** — analysts manually place each boundary, guided by model
output and observations. This project asks: *can a neural network learn to
reproduce that analysis automatically, from a limited set of physical fields?*

If it can, then fronts become a **quantitative, reproducible layer** that can be
produced for any time step — past, present, or future — and linked to the
high-impact phenomena fronts are tied to: hurricane transitions, drought onset,
wildfire spread, and long-term climate trends.

### ERA5 — the input

ERA5 is a
global atmospheric reanalysis (0.25°, hourly) that gives a physically consistent
"best estimate" of the atmosphere. We feed the model a stack of ERA5 fields
(temperature, wind, geopotential, humidity, etc. at selected pressure levels).

### TFP — an objective front locator

The **Thermal Front Parameter (TFP)** is a diagnostic computed *purely* from the
ERA5 temperature field. It measures where horizontal temperature gradients are
tightening, which is exactly where fronts form. TFP is fully **dynamical and
objective** — no human input — but it has one limitation: it can locate a front,
yet it **cannot tell what *type* of front** it is.

### U-Net — the model

U-Net is an encoder–decoder convolutional
network with skip connections, originally built for image segmentation. Front
detection *is* a segmentation problem — every grid cell is classified as
background or a front type — which makes U-Net a natural fit.

### Why "Hybrid" labels?

To teach the model front *types*, we need labels that include type information.
Two sources, two trade-offs:

| Source | Strength | Weakness |
|--------|----------|----------|
| **ERA5 / TFP** | Grid-accurate position; fully dynamical and explainable | Cannot classify front type |
| **WPC analysis** | Expert classification (CF/WF/SF/OF) | Human-drawn → subjective, positional error → adds noise that can lower ML performance |

The **Hybrid label** combines the best of both: it takes the **position from
ERA5/TFP** (clean, dynamically grounded) and the **type from WPC** (expert
judgment), keeping only places where the two agree. This suppresses the human
noise in WPC while gaining the type information ERA5 alone cannot provide — and
it is the only route to detecting **occluded fronts (OF)**, which TFP cannot see.

### Strategy

| Phase | Period (example) | Purpose |
|-------|------------------|---------|
| **Training**    | 2010–2024 | Learn fronts from ERA5 + Hybrid labels |
| **Validation**  | 2025      | Measure skill on unseen years |
| **Application** | 2026, historical (1940+), extreme events | Apply the trained model where no hand-drawn fronts exist |

---

## Results

### Front types

| Code | Type | What it is |
|------|------|------------|
| **CF** | Cold Front | Cold air advancing, undercutting warmer air ahead |
| **WF** | Warm Front | Warm air advancing and rising over retreating cold air |
| **SF** | Stationary Front | Boundary between air masses with little movement |
| **OF** | Occluded Front | A cold front overtaking a warm front, lifting the warm air aloft |

### How we measure skill

We report the **F1 score** — the harmonic mean of precision and recall (0 = worst,
1 = perfect). F1 is well suited here because fronts are *rare* compared with
background, so plain accuracy would be misleading. Each model trains for a set
number of **epochs** (full passes over the training data; typically 30).

### Progression of runs

The project grew through a series of runs, each answering one question. Compute
also evolved: the early runs ran on a **Mac (CPU, then Apple-silicon GPU)**, and
later runs moved to **NASA Discover A100 GPUs** (up to 4×A100 with distributed
training) — cutting per-epoch time from hours to minutes.

| Run | What it added | Why |
|-----|---------------|-----|
| **Run 2** | 4-channel TFP classifier, 3 training years | Establish a clean baseline that fronts are learnable |
| **Run 4** | Extended to 6 training years | More data → best classical classifier (F1 ≈ 0.72) |
| **Run 5** | 12-channel **Hybrid** labels | First model to detect **occluded fronts (OF)** |
| **Run 7** | **Regression** (ERA5-only, no WPC) | Remove human-label noise entirely — predict continuous frontal fields instead of discrete classes |
| **Run 8** | Hybrid + 15 training years, 4×A100 | Test whether more data restores type-classification skill at scale |

**Why a regression run (Run 7)?** Discrete labels inherit the subjectivity and
positional error of the human WPC analysis. Regression sidesteps this: instead
of predicting a front *class*, the model predicts the **continuous ERA5
diagnostic fields** (front location, temperature advection, gradient strength).
This is threshold-free, fully reproducible, and robust for climate-scale
application. It reached a near-perfect correlation (r ≈ 0.99) with the physical
target fields.

---

## Current Direction

**1. Improving the models** *(`.pt` = the trained PyTorch model — the millions of
learned network weights, plus optimizer state and epoch number)*
We are exploring which input **channels** help versus add noise, and how best to
use the type information extracted from WPC without degrading performance.

**2. Applications** *(in collaboration with domain experts)*

| Area | Question |
|------|----------|
| **Tropical / extratropical cyclones (TC/ETC)** | How do fronts interact with storms during transition? |
| **Drought** | Can front frequency anomalies signal dry-spell onset? |
| **Wildfire** | Does cold-front passage precede fire-spread conditions? |
| **Climatology** | How are front frequency and position shifting over decades? |

Each of these will be developed with advice from specialists in the respective
fields.

---

## Code

The project centers on two training scripts:

| Script | Purpose |
|--------|---------|
| `train_unet_classifier.py` | U-Net **classifier** — predicts discrete front types (CF/WF/SF/OF) from ERA5 + Hybrid labels |
| `train_unet_regression.py` | U-Net **regressor** — predicts continuous ERA5 frontal fields, no WPC labels |

Both share the same U-Net backbone and differ only in their output head and
labels. A trained model is saved as a `.pt` checkpoint and can be loaded for
inference on any ERA5 time step.

---

## Data

| Source | Role |
|--------|------|
| **ERA5** reanalysis | Model input (temperature, wind, geopotential, humidity at selected levels) |
| **WPC** surface analysis | Expert front types for Hybrid labels |
| **IBTrACS** | Tropical-cyclone tracks (TC/ETC application) |
| **FIRMS** | Satellite fire detections (wildfire application) |

---

## Origin & What's New

This work began from aaTman/fronts (a fork of ai2es/fronts), which provides a
TensorFlow front-detection toolkit. Our project
re-implements and substantially extends it:

- **PyTorch re-implementation** with mixed-precision and multi-GPU (4×A100) training.
- **Hybrid labels** — fusing objective ERA5/TFP (Thermal Front Parameter) front
  *positions* with expert WPC *types* (extracted directly from analysis archives),
  enabling **occluded-front detection** that TFP alone cannot achieve.
- **Regression branch** — a threshold-free, WPC-independent alternative that
  predicts continuous frontal fields (new; not in the original).
- **Downstream applications** — hurricane transition, drought, wildfire, and
  multi-decadal front climatology.
