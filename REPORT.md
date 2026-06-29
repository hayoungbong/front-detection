# Deep Learning-Based Atmospheric Front Detection
## Methodological Comparison: Classification vs. Regression Approaches with ERA5

---

## 1. Introduction

Atmospheric fronts — boundaries between air masses of contrasting temperature
and humidity — are the primary drivers of mid-latitude weather. Despite their
importance, fronts are not objectively defined physical objects. They represent
zones of enhanced horizontal temperature gradient, not discrete lines. The
lines drawn by operational forecasters (e.g., NOAA/WPC) are analytical
conventions designed for communication, not direct observations of nature.

This report documents the development of a U-Net-based front detection system
using ERA5 reanalysis data, traces the evolution across four training runs,
and motivates a methodological shift from discrete classification toward
physically grounded continuous regression.

---

## 2. Data

### 2.1 ERA5 Reanalysis

- **Source**: ECMWF ERA5 via Copernicus Climate Data Store (CDS API)
- **Period**: 1970–2026 (ERA5 back extension pre-1979)
- **Domain**: 5°N–85°N, 180°W–10°E
  - Captures full ETC lifecycle: Pacific genesis (~180°W), Atlantic decay (~20°W), polar lows (85°N)
- **Resolution**: 0.25°, 6-hourly (00/06/12/18 UTC)
- **Pressure-level variables** (500/700/850/900/925/950/1000 hPa): T, u, v, Z, q, ω
- **Surface variables**: 2m temperature, 2m dewpoint, 10m winds, MSLP, surface pressure

### 2.2 Domain Design

The training/inference domain (15–70°N, 170–50°W) matches the WPC North America
surface analysis coverage. The download domain is intentionally broader to provide
spatial context for extratropical cyclones:

| Boundary | Training domain | Download domain | Rationale |
|----------|----------------|-----------------|-----------|
| North | 70°N | **85°N** | Polar lows, Arctic fronts |
| South | 15°N | **5°N** | Subtropical fronts |
| West | 170°W | **180°W** | Pacific ETC genesis |
| East | 50°W | **10°E** | Full North Atlantic storm track |

### 2.3 WPC Front Archive

NOAA/WPC publishes 6-hourly surface analysis GIF images (namfntsfc) from 2007–present.
Front types extracted: Cold Front (CF), Warm Front (WF), Stationary Front (SF),
Occluded Front (OF). Color detection → morphological skeletonization → Lambert
Conformal Conic projection → 0.25° ERA5 grid.

---

## 3. Physical Background

### 3.1 Thermal Front Parameter (TFP)

The TFP (Renard & Clarke 1965) identifies locations where the horizontal
temperature gradient is locally maximum — the objective front location.

$$\text{TFP} = \nabla|\nabla T| \cdot \left(\frac{-\nabla T}{|\nabla T|}\right)$$

**Computation procedure:**

1. Compute horizontal temperature gradient at each grid point (finite differences):
$$\nabla T = \left(\frac{\partial T}{\partial x},\ \frac{\partial T}{\partial y}\right),
\quad |\nabla T| = \sqrt{\left(\frac{\partial T}{\partial x}\right)^2 + \left(\frac{\partial T}{\partial y}\right)^2}$$

2. Apply the nabla operator to |∇T| → find where its gradient is zero

3. TFP = 0 marks the crest of the |∇T| ridge = objective front location

The sign of TFP indicates which side is warm (TFP > 0) vs. cold (TFP < 0).

### 3.2 Temperature Advection

$$\text{tadv} = -\mathbf{v} \cdot \nabla T = -\left(u\frac{\partial T}{\partial x} + v\frac{\partial T}{\partial y}\right)$$

- tadv > 0 (warm advection) → physically a **Warm Front**
- tadv < 0 (cold advection) → physically a **Cold Front**
- tadv ≈ 0 (no advection) → physically a **Stationary Front**

This sign emerges naturally from the ERA5 wind and temperature fields —
no analyst judgment required.

### 3.3 Temperature Gradient Magnitude

$$|\nabla T| \quad [\text{K/km}]$$

The continuous measure of frontal intensity. Unlike discrete labels,
this quantity tracks changes in frontal strength over time without
imposing an arbitrary threshold.

---

## 4. Label Strategies

### 4.1 TFP Classification (Runs 1–2)

TFP zero-crossings combined with temperature advection sign produce discrete labels:

```
|TFP| > threshold AND |∇T| > GRAD_THRESH:
    tadv < -δ  →  CF (1)
    tadv > +δ  →  WF (2)
    |tadv| ≤ δ →  SF (3)
otherwise      →  BG (0)
```

Fixed thresholds: TFP_THRESH = 0.12 K/(100km)², TADV_THRESH = 0.5×10⁻⁵ K/s.

**Limitations:**
- Thresholds lack physical justification — set empirically
- Information loss through discretization of continuous gradients
- **Climate change bias**: Arctic Amplification reduces the pole-to-equator
  temperature gradient over decades. A fixed threshold will systematically
  under-detect fronts in a warmer climate, making it impossible to distinguish
  real frontal frequency changes from threshold artifacts

### 4.2 Hybrid ERA5×WPC Classification (Run 3)

Intersects ERA5 TFP (precise grid-native position) with WPC analyst images
(expert front-type classification):

$$\text{label}(i,j) = \begin{cases} X \in \{\text{CF,WF,SF,OF}\} & \text{if } |\text{TFP}(i,j)| > \theta \text{ AND WPC}_X \text{ within } \pm 50\text{ km} \\ \text{BG} & \text{otherwise} \end{cases}$$

| Aspect | TFP alone | WPC image alone | **Hybrid** |
|--------|-----------|-----------------|-----------|
| Position accuracy | ✅ ERA5 grid | ❌ ~30 km offset | ✅ ERA5 grid |
| Type classification | ❌ threshold-based | ✅ expert judgment | ✅ expert judgment |
| Over-detection | ❌ systematic | ✅ analyst filtered | ✅ doubly filtered |
| Occluded Front | ❌ impossible | ✅ detected | ✅ **detected** |
| Archive coverage | ✅ 1970–2026 | ✅ 2007–present | ✅ 2007–present |

**Trade-off**: the model absorbs WPC analyst subjectivity — inconsistency between
analysts, systematic omission of weak fronts, software rendering changes (~2013)
causing year-to-year label thickness variation.

### 4.3 ERA5 Regression Targets (Run 4+)

Instead of discrete categories, predict continuous physical fields directly
from ERA5:

| Target variable | Physical meaning | Advantage |
|----------------|-----------------|-----------|
| `tfp_850` | TFP field | Continuous front location |
| `tadv_850` | Temperature advection (-**v**·∇T) | CF/WF sign emerges naturally |
| `grad_mag_850` | \|∇T\| magnitude | Frontal intensity without thresholds |

No arbitrary thresholds. No WPC dependency. Tracks climate-change-driven
changes in frontal intensity distributions continuously. Researchers can apply
any threshold post-hoc for their specific application.

---

## 5. Model Architecture

### 5.1 U-Net

Fully convolutional encoder-decoder with skip connections.
Input and output share the same spatial resolution (grid-to-grid mapping).

```
Input  [C, 221, 481]    C = 4 / 8 / 12 depending on run

Encoder                                     Decoder
ConvBlock(C → 64)    [64,  221, 481]  ←→  64+128  → 64
MaxPool + CB(→128)   [128, 111, 241]  ←→  128+256 → 128
MaxPool + CB(→256)   [256,  56, 121]  ←→  256+512 → 256
MaxPool + CB(→512)   [512,  28,  61]  ←→  512+1024→ 512
MaxPool + CB(→1024)  [1024, 14,  31]  ←  bottleneck

Head: 1×1 Conv → N classes (softmax) or continuous targets
Parameters: ~31M
```

ConvBlock = Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU

### 5.2 Training Setup

- **Loss**: Focal Loss (γ=2) with inverse-frequency class weights
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR (T_max=30, η_min=lr×0.01)
- **Batch size**: 8
- **Hardware**: Apple M3 Pro (MPS, 18 GB unified memory)

### 5.3 Class Imbalance

Frontal pixels constitute only ~1.2% of the training domain.

| Class | Pixel fraction | Weight |
|-------|---------------|--------|
| BG | 98.85% | 0.07 |
| CF | 0.57% | 0.87 |
| WF | 0.48% | 0.96 |
| SF | 0.10% | 2.11 |
| OF | <0.05% | ~4.0 |

---

## 6. Training Run Results

### 6.1 Run 1 — Baseline (Complete)

**Configuration:** 4-channel input, TFP classification labels, 2020–2021 training,
2022 validation, 30 epochs.

| Class | F1 |
|-------|-----|
| Cold Front (CF) | 0.764 |
| Warm Front (WF) | 0.731 |
| Stationary Front (SF) | 0.531 |
| **Mean** | **0.675** |

Spatial connectivity poor at epoch 1 (smoke test: Mean F1=0.449), improved
substantially with full 30-epoch training. Established that U-Net can learn
front positions from ERA5 thermodynamic fields.

### 6.2 Run 2 — Extended Dataset (Complete)

**Configuration:** Same 4-channel model, TFP classification labels,
2019–2024 training (3× more data), 2025 validation, 30 epochs.

Per-epoch progression:

| Epoch | Mean F1 | Notes |
|-------|---------|-------|
| 1 | 0.451 | |
| 10 | 0.621 | |
| 16 | 0.689 | Rapid improvement (CosineAnnealingLR late phase) |
| 30 | **0.768** | Best checkpoint, still improving |

**Final metrics (Epoch 30 — best):**

| Class | F1 |
|-------|-----|
| Cold Front (CF) | **0.837** |
| Warm Front (WF) | **0.822** |
| Stationary Front (SF) | **0.645** |
| **Mean** | **0.768** |

**Key observations:**
- No overfitting: validation loss consistently below training loss
  (2025 ERA5 patterns appear cleaner than 2019–2024 training period)
- SF remains lowest: temperature advection ≈ 0 is inherently ambiguous
- Run 2 U-Net over-detects by ×2.8 vs. WPC (TFP baseline: ×2.1)
  — expected, since Run 2 learns TFP labels which are themselves more
  generous than analyst-drawn fronts

**Comparison with WPC analyst (49 samples, 2025):**

Pixel-level F1 vs. WPC is low (~0.007) due to width mismatch: model
predicts 2–3 pixel bands while WPC labels are skeletonized to 1 pixel.
Qualitative comparison shows correct front positions and spatial structure.

### 6.3 Run 3 — Hybrid Labels, Smoke Test Only

**Configuration:** 8-channel input (Run 2 channels + z500/q850/w850/msl),
Hybrid ERA5×WPC labels (5-class: BG/CF/WF/SF/OF), 2024 training only,
2025 validation, 10 epochs.

| Epoch | CF | WF | SF | OF | Mean |
|-------|----|----|----|----|------|
| 1 | 0.104 | 0.090 | 0.000 | 0.000 | 0.065 |
| 5 | 0.148 | 0.116 | 0.000 | 0.054 | 0.106 |
| 10 | **0.165** | **0.153** | **0.000** | **0.063** | **0.127** |

**Status: smoke test only — 1 year of data, 10 epochs, not converged.**

**Key achievement:** First successful detection of Occluded Fronts (OF) —
a class impossible to generate from TFP alone, emerging from hybrid labels.

**Why it stopped here:** Full Run 3 requires hybrid labels and extra channels
for 2019–2025 (7 years), and 925 hPa was not available in the legacy ERA5
files. New regional ERA5 download (including 925 hPa from the start) is
currently in progress (1970–2026).

### 6.4 Run 4 — In Progress

**Configuration:** Same 4-channel model as Run 2 (`train_unet.py`),
TFP classification labels rebuilt with new data pipeline,
2019–2024 training (6 years), 2025 validation, 30 epochs, batch 8.

Run 4 is a clean rebuild of Run 2 with a new, higher-quality data pipeline:

| Aspect | Run 2 | Run 4 |
|--------|-------|-------|
| Training data source | Legacy 5-variable files | **New 7-variable pipeline** |
| ERA5 download | Original (no 925 hPa) | **New regional download with 925 hPa** |
| Training years | 2019–2021 | **2019–2024 (6 years)** |
| Training samples | ~4,380 | **~8,768** |
| Data leakage check | Not verified | **Confirmed clean (val=2025)** |
| Regression targets | Not available | **tadv_850, grad_mag_850 in files** |

**Smoke test results (5 epochs, train 2021–2024 / val 2025):**

| Epoch | CF | WF | SF | Mean |
|-------|----|----|----|----|
| 1 | 0.558 | 0.489 | 0.265 | 0.437 |
| 3 | 0.636 | 0.577 | 0.351 | 0.521 |
| 5 | **0.681** | **0.657** | **0.426** | **0.588** |

All metrics trending upward at epoch 5 — full 30-epoch run is currently
in progress (train 2019–2024, val 2025).

**Expected outcome:** Based on the smoke test trajectory and larger dataset
(2019–2024 vs. smoke test's 2021–2024), Run 4 is projected to meet or
exceed Run 2's best of F1=0.768.

### 6.5 WPC Label-Pipeline Overhaul (enables Run 5)

The WPC analyst fronts are the ground truth for the hybrid labels and the human
comparison. While preparing the 5-class hybrid run we found the GIF-extraction
pipeline (`extract_wpc_fronts.py`) was silently broken in three ways, and fixed
each (validated against analyst bulletins, n=64):

| Problem | Cause | Fix | Effect |
|---------|-------|-----|--------|
| Extraction returned **0 fronts** | bbox aspect-ratio filter rejected curved (hooked) fronts | curvature-invariant extent filter | fronts recovered |
| Fronts offset **100–360 km** | diagonal affine ignored a real ~10° image rotation | full 6-param affine, coastline-ICP calibrated (stable 2007–2026) | CF recall 364→216 km |
| **SF class always empty** | WPC has no stationary colour (alternating red/blue) | detect red+blue corridor; also removes CF/WF double-count | SF recovered (445/301 km); WF precision 1244→1063 |
| H/L letters & numbers leaked | same colour as fronts | drop sub-extent components | cleaner masks |

Rejected after validation (net-negative): polynomial / per-image projection
(distorts the data-sparse interior, hurts WF) and template H/L removal (clips
real warm-front curves). The simpler fixes above were strictly better.

**Outcome:** 20 years of labels (2007–2026) regenerated; SF populated for the
first time (2024: ~266 cells/map). This unblocks **Run 5** (5-class hybrid) and
fixes the `f1_sf=0` of Run 3. The earlier image-extracted WPC comparison (§6.2)
is preliminary and will be re-scored on the corrected reference. SF is physically
consistent with ERA5 (TFP front with near-zero temperature advection), so WPC
type and ERA5 dynamics agree — location stays ERA5-driven, WPC supplies the type.

---

## 7. Discussion

### 7.1 What Is a Front?

A meteorological front is not a line — it is a continuous gradient zone
spanning tens to hundreds of kilometers. The discrete lines drawn by WPC
analysts are operational simplifications, not direct measurements of nature.
Any labeling scheme that produces discrete categories imposes an arbitrary
boundary on what is physically a continuous field.

### 7.2 Climate Change Implications

Classification with fixed thresholds is problematic for long-period climate
studies. Arctic Amplification preferentially warms high latitudes, reducing
the meridional temperature gradient. If TFP_THRESH remains fixed at 0.12
K/(100km)², fewer grid cells will exceed the threshold in a warmer climate —
even if fronts are equally frequent or intense. This threshold artifact would
be misinterpreted as a real reduction in frontal activity.

Continuous regression targets (tfp, tadv, grad_mag) track the actual
distribution of frontal intensity over time, making climate trend analysis
independent of arbitrary thresholds.

### 7.3 Role of the U-Net

Although TFP and temperature advection can be computed directly from ERA5
using finite differences, the U-Net provides capabilities beyond local
gradient calculations:

| Capability | Finite differences | U-Net |
|-----------|--------------------|-------|
| Spatial context | Neighboring 1–2 grid cells | Hundreds of km pattern |
| Upper-level coupling | None | Learns 500 hPa trough → surface front relationship |
| Applicable to observations | ERA5 required | Any data with same variables |
| Inference speed | Fast | Faster after training |

### 7.4 On WPC Labels

Using WPC labels — even as part of a hybrid approach — means the model
learns to replicate analyst behavior rather than detect the physical phenomenon.
WPC labels embody:
- Inter-analyst subjectivity
- Systematic omission of weak fronts
- Software rendering changes (~2013 thickness variation)
- Coverage limited to North America post-2007

For pure physical research, a regression approach targeting ERA5-derived
continuous fields is more internally consistent and globally applicable.

---

## 8. Planned Next Steps

### Near-term
1. Complete ERA5 regional download (1970–2026) — currently in progress
2. Build training data for all years with regression targets
3. Build extra channels including t925 for 1970–2026
4. Full Run 3 (2019–2026, hybrid labels, 30+ epochs)
5. Full Run 4 (1970–2026, 12 channels, hybrid + regression, 30+ epochs)

### Medium-term
6. Distance-based evaluation metric: detection rate within N km of WPC front
   (replaces pixel F1, accounts for front width mismatch)
7. Climate analysis: frontal frequency and intensity trends 1970–2026
8. Frontogenesis function as additional regression target

### Long-term
9. Global domain extension beyond North America
10. Multi-reanalysis comparison (ERA5 / MERRA-2 / JRA-55) for uncertainty quantification
11. Real-time pipeline: automated download → inference → visualization

---

## References

- Renard, R.J. & Clarke, L.C. (1965). Experiments in numerical objective frontal analysis. *Mon. Wea. Rev.*, 93, 547–556.
- Hewson, T.D. (1998). Objective fronts by the thermal front parameter. *Meteorological Applications*, 5(1), 51–65.
- Berry, G. et al. (2011). A global climatology of atmospheric fronts. *Geophys. Res. Lett.*, 38, L04809.
- Biard, J.C. & Kunkel, K.E. (2019). Automated detection of weather fronts using a deep learning neural network. *Advances in Statistical Climatology*, 3, 103–117.
- Ronneberger, O. et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. *ICCV 2017*.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999–2049.
- Catto, J.L. & Pfahl, S. (2013). The importance of fronts for extreme precipitation. *J. Geophys. Res.*, 118, 10791–10801.
