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
| **ERA5 / TFP** | Grid-accurate position; objective and physically interpretable | Cannot classify front type |
| **WPC analysis** | Forecaster-classified types (CF/WF/SF/OF) | Analyst judgment → subjectivity and positional uncertainty → noise that can degrade ML performance |

The **Hybrid label** combines the best of both: it takes the **position from
ERA5/TFP** (clean, dynamically grounded) and the **type from WPC** (expert
judgment), keeping only places where the two agree. This suppresses the human
noise in WPC while gaining the type information ERA5 alone cannot provide — and
it is the only route to detecting **occluded fronts (OF)**, which TFP cannot see.

### Strategy

Periods are adjustable depending on data availability and the target application.

| Phase | Period | Purpose |
|-------|--------|---------|
| **Training**    | 2010–2024 | Learn fronts from ERA5 + Hybrid labels |
| **Validation**  | 2025      | Measure skill on unseen years |
| **Application** | 2026 onward; historical (1940+); extreme events | Apply the trained model wherever automated front analysis is needed |

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

**Classification runs** report the **F1 score** — the harmonic mean of precision
and recall (0 = worst, 1 = perfect). F1 is well suited here because fronts are
*rare* compared with background, so plain accuracy would be misleading.

**The regression run (Run 7)** predicts continuous physical values rather than
discrete labels, so F1 does not apply. Instead we report **Pearson correlation
(r)**, which measures how closely the predicted field tracks the actual ERA5
field at every grid point. An r ≈ 0.99 means the model reproduces the spatial
pattern of the physical field almost exactly.

Each model trains for a set number of **epochs** (full passes over the training
data; typically 30).

### Progression of runs

The project grew through a series of runs, each answering one question. Compute
also evolved: the early runs ran on a **Mac (CPU, then Apple-silicon GPU)**, and
later runs moved to **NASA Discover A100 GPUs** (up to 4×A100 with distributed
training) — cutting per-epoch time from hours to minutes.

| Run | What it added | Why |
|-----|---------------|-----|
| **Run 2** | 4-channel TFP classifier, 3 training years | Establish a clean baseline that fronts are learnable |
| **Run 4** | Extended to 6 training years | More data → best classical classifier (F1 ≈ 0.72) |
| **Run 5** | 12-channel **Hybrid** labels | First to detect **occluded fronts (OF)**; F1 lower than Run 4 because Hybrid labels are harder to learn |
| **Run 7** | **ERA5-only, no WPC labels** | Remove analyst-judgment noise — predict continuous physical fields instead of discrete classes (see below) |
| **Run 8** | Hybrid + 15 training years [4×A100] | Extension of Run 5; tests whether more data closes the F1 gap introduced by Hybrid labels |

**Why predict continuous fields instead of classes (Run 7)?**
Discrete front labels inherit the subjectivity and positional uncertainty of the
operational analysis. An alternative is to predict the underlying **continuous
ERA5 physical fields** — the Thermal Front Parameter, temperature advection, and
temperature-gradient magnitude — directly. This is called *regression* in
machine learning (continuous-value prediction, as opposed to *classification*
into discrete categories). Because the targets are physical quantities, the
natural metric is Pearson correlation (r) rather than F1. Run 7 reached r ≈ 0.99,
meaning the model reproduces the spatial pattern of each physical field
near-perfectly — with no dependence on analyst judgment.

---

## Current Direction

**1. Improving the models** *(`.pt` = the trained PyTorch model — millions of
learned weights, plus optimizer state and epoch number)*
Two open questions drive the next runs:
- **Channel selection**: adding more ERA5 input channels did not always help —
  Run 6 (12 channels) underperformed Run 4 (4 channels). We are testing which
  channels are genuinely informative and which hurt the model.
- **Whether and how to incorporate WPC front types**: using analyst-classified
  labels (Hybrid) enabled occluded-front detection but lowered F1 for the other
  front types. The trade-off between richer type information and model
  performance is still being resolved.

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
- **Regression branch** — predicts continuous ERA5 physical fields directly,
  making the output more tightly connected to atmospheric dynamics and independent
  of analyst judgment. Because the targets (TFP, temperature advection, gradient
  magnitude) are computed at 850 hPa rather than the surface, results may differ
  from a true surface-front analysis; however, the 850 hPa fields are closely
  coupled to moisture transport and frontal dynamics, making them a valuable
  indicator for large-scale atmospheric applications. Not present in the original
  codebase.
- **Downstream applications** — hurricane transition, drought, wildfire, and
  multi-decadal front climatology. The original codebase focuses solely on model
  training and detection; these scientific applications are new to this project.
