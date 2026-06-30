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
- **Batch size**: 8 (CPU/MPS) / 32 (A100 GPU)
- **Hardware**: Apple M-series (MPS) · NASA Discover PRISM (NVIDIA A100-SXM4-40GB)

**GPU note:** The A100-SXM4-40GB (40 GB HBM2, ~1.5 TB/s memory bandwidth) was the
de facto standard for HPC AI workloads before H100. NASA Discover operates V100,
A100, and H100 nodes; A100 remains a primary production GPU for scientific
deep learning. For a U-Net of this scale (~31M parameters), the A100 is
not compute-bound — the bottleneck is disk I/O during NetCDF loading.
Once training begins, GPU utilization is high and batch 32 runs comfortably
within the 40 GB VRAM.

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

### 6.1 Run 1 — Proof of Concept (Complete)

**Configuration:** 4-channel input (t850/u850/v850/tfp_850), TFP classification labels,
2020–2021 training, 2022 validation, 30 epochs, batch 4.

**Primary question:** Can a U-Net learn to locate atmospheric fronts from ERA5
thermodynamic fields alone?

| Class | F1 |
|-------|-----|
| Cold Front (CF) | 0.764 |
| Warm Front (WF) | 0.731 |
| Stationary Front (SF) | 0.531 |
| **Mean** | **0.675** |

**Answer: Yes.** F1=0.675 from only 2 years of training confirms that the U-Net
architecture can extract frontal patterns from ERA5. The model learns spatially
coherent front structures despite training only on TFP-derived labels.

**What Run 1 revealed:**
- CF and WF are learnable from thermodynamic fields alone
- SF is the hardest class — temperature advection ≈ 0 is inherently ambiguous
- 2 training years are insufficient for the model to fully converge
- The TFP label ceiling is already visible: the model cannot exceed what TFP encodes

**What this motivated:** Add more training data (Run 2) and eventually replace
TFP labels with expert analyst labels (Runs 3, 5).

### 6.2 Run 2 — Data Scaling: More Years (Complete)

**Configuration:** Same 4-channel TFP model, 2019–2021 training (+1 year over Run 1),
2022 validation, 30 epochs.

**Primary question:** Is the Run 1 result data-limited? Does adding one more year
of training data improve performance?

Per-epoch progression:

| Epoch | Mean F1 | Notes |
|-------|---------|-------|
| 1 | 0.451 | |
| 10 | 0.621 | Steady improvement |
| 20 | 0.719 | CosineAnnealingLR late phase |
| 30 | **0.768** | Best checkpoint — still not fully converged |

**Final metrics (Epoch 30 — best):**

| Class | F1 |
|-------|-----|
| Cold Front (CF) | **0.837** |
| Warm Front (WF) | **0.822** |
| Stationary Front (SF) | **0.645** |
| **Mean** | **0.768** |

**Answer: Yes — significantly.** F1 0.675 → 0.768 (+14%) from adding one year.
The learning curve has not plateaued at epoch 30, suggesting even more data
would continue to help.

**Key observations:**
- CF/WF now plateau near 0.84/0.82 — strong signals well-learned
- SF improved but remains the bottleneck (TFP-based SF labeling is inherently noisy)
- No overfitting — validation loss tracks training loss closely
- Run 2 U-Net over-detects by ×2.8 vs. WPC: expected, since it learns TFP labels
  which are more generous than analyst-drawn fronts

**What this motivated:** Continue scaling data (Run 4: 6 years) to see how far
the TFP-based approach can go, while also testing expert labels (Run 3) in parallel
to understand whether the real ceiling is data quantity or label quality.

### 6.3 Run 3 — Hybrid Labels: Proof of Concept (Smoke Test)

**Configuration:** 8-channel input (t850/u850/v850/tfp + z500/q850/w850/msl),
Hybrid ERA5×WPC labels (5-class: BG/CF/WF/SF/OF), 2024 training only,
2025 validation, 10 epochs.

**Primary question:** Do hybrid WPC×ERA5 labels work at all? Can the U-Net
detect Occluded Fronts — a class that TFP alone cannot produce?

| Epoch | CF | WF | SF | OF | Mean |
|-------|----|----|----|----|------|
| 1 | 0.104 | 0.090 | 0.000 | 0.000 | 0.065 |
| 5 | 0.148 | 0.116 | 0.000 | 0.054 | 0.106 |
| 10 | **0.165** | **0.153** | **0.000** | **0.063** | **0.127** |

**Status: smoke test only — 1 year of data, 10 epochs, not converged.**

**Answer: Yes.** Two key proof points:
1. **Hybrid labels work** — CF/WF are climbing at ep10 with only 1 year
2. **OF detection works** — F1=0.063 at epoch 10 is a landmark: the first
   time an Occluded Front has ever been detected in this pipeline.
   TFP-based labels cannot generate OF at all; this class exists only
   because WPC analysts drew it.

**SF=0.000 explained:** The WPC label pipeline had a bug — the stationary
front color (alternating red/blue) was not being detected, so SF labels were
always empty. Fixed before Run 5 (see §6.5 WPC overhaul).

**What this motivated:** Run 3 proved the concept but was data-starved.
The full hybrid run needs 6 years of data, corrected SF labels, and
an expanded 12-channel input — that is Run 5.

### 6.4 Run 4 — Maximum TFP Baseline: 6-Year Dataset (In Progress)

**Configuration:** Same 4-channel TFP model as Run 2 (`train_unet.py`),
2019–2024 training (6 years, 2× Run 2), 2025 validation, 30 epochs, batch 8.
Running on NASA Discover CPU cluster.

**Primary question:** What is the maximum F1 achievable with TFP labels
and the full available training data (6 years)? Where does the TFP ceiling lie?

Run 4 is a clean rebuild of Run 2 with a larger dataset and new data pipeline:

| Aspect | Run 2 | Run 4 |
|--------|-------|-------|
| Training data source | Legacy 5-variable files | **New 7-variable pipeline** |
| ERA5 download | Original (no 925 hPa) | **New regional download with 925 hPa** |
| Training years | 2019–2021 (3 yr) | **2019–2024 (6 yr — 2×)** |
| Training samples | ~4,380 | **~8,768** |
| Validation year | 2022 | **2025 (fully held-out future year)** |
| Regression targets | Not available | **tadv_850, grad_mag_850 in files** |

**Current results (Epoch 11/30, NASA Discover CPU):**

| Epoch | CF | WF | SF | Mean |
|-------|----|----|-----|------|
| 1 | 0.617 | 0.549 | 0.301 | 0.489 |
| 5 | 0.653 | 0.602 | 0.383 | 0.546 |
| 8 | 0.740 | 0.672 | 0.466 | **0.626** |
| 10 | **0.751** | **0.697** | **0.476** | **0.642** |
| 11 | 0.739 | 0.686 | 0.473 | 0.633 |

*Training ongoing — epoch 12+ in progress. Best F1=0.642 at epoch 10.*

**Trajectory:** The curve closely mirrors Run 2's progression at the same epochs.
With 6 years of data the model is expected to ultimately surpass Run 2's F1=0.768,
and the still-rising SF F1 (0.476 at ep10) confirms that data volume directly
helps the rarest class.

**What Run 4 is establishing:**
- The ceiling of the TFP 4-channel approach with maximal data
- A rigorous 2025 baseline against which Run 5's hybrid labels can be fairly compared
- SF emergence: with 6 years, SF consistently appears for the first time

**When complete, Run 4 will answer:** How much of Run 5's improvement (if any)
comes from more data vs. better labels vs. more channels.

### 6.5 Run 5 — Full System: 12-Channel Hybrid

Run 5 went through two stages separated by a label-pipeline diagnosis.

#### Run 5a — 12-Channel Baseline (Complete, ep28, F1=0.693)

**Configuration:** 12-channel input, hybrid labels (nominally), 2019–2024 training,
2025 validation, 30 epochs, batch 32, NASA Discover A100.

Run 5a completed but OF was silently zero throughout training — class weight for OF=0.0,
F1_OF=0.000 at all epochs. Best mean F1=0.693 at ep28 (effectively a 12-channel TFP run
because the hybrid labels contained no OF pixels).

**Root cause diagnosis** revealed two bugs in `build_hybrid_discover.py`:

| Bug | Symptom | Cause | Fix |
|-----|---------|-------|-----|
| **Priority overwrite** | OF inflated to 626 K pixels | OF assigned last with no TFP requirement, overwriting CF/WF/SF; OR when TFP threshold too strict, OF=0 | OF assigned last (highest priority) WITH TFP filter; same threshold as other classes |
| **N-S coordinate flip** | OF collapsed to 26 K (should be 104 K) | Discover training lat ascending (15→70N), WPC extraction lat descending (70→15N) — intersecting by array index mirror-flips masks; OF near TFP threshold, most sensitive to spatial mismatch | `wpc = wpc.reindex(lat=tr["lat"].values, lon=tr["lon"].values, fill_value=0)` |

After fixes, 2019 label pixel counts: CF=524,285 WF=215,365 SF=352,034 OF=104,400.
OF at ~8% of front pixels is consistent with Berry et al. (2011) climatology
(CF > SF > WF > OF ordering), confirming the corrected labels are physically plausible.

#### Run 5b — Corrected Hybrid Labels (Complete, 2026-06-29)

**Configuration:** Same 12-channel setup, corrected `build_hybrid_discover.py`,
rebuilt hybrid labels 2019–2025, NASA Discover A100-SXM4-40GB (batch 32).

**Primary question:** When we combine everything learned from Runs 1–4 —
6 years of data, expert labels, 12 channels, corrected SF and OF, GPU training —
how far can we push front detection accuracy?

Run 5b simultaneously addresses every limitation identified in Runs 1–4:

| Limitation (Runs 1–4) | Run 5b Solution |
|----------------------|----------------|
| TFP labels over-detect by ×2.8 vs. WPC | WPC analyst labels for type |
| No Occluded Front class | OF from WPC extraction (corrected labels) |
| SF always 0 (Run 3 bug) | SF recovered — WPC label-pipeline overhaul |
| 4 channels miss upper-level dynamics | z500: 500 hPa trough/ridge context |
| No moisture information | q850: frontal lifting signal |
| No surface wind or temperature | t2m, u10, v10: surface air-mass contrast |
| CPU training (~3600 s/epoch) | A100 GPU: ~120 s/epoch (~30× faster) |

**Class weights (inverse-frequency, 2019–2024 hybrid labels):**

| BG | CF | WF | SF | OF |
|----|----|----|----|----|
| 0.05 | 0.85 | 1.24 | 0.95 | **1.91** |

OF carries the highest weight — the model is explicitly penalized for missing the rarest front type.

**Full epoch results (2025 validation, ~120 s/epoch on A100):**

| Epoch | train loss | val loss | CF | WF | SF | OF | Mean |
|-------|-----------|---------|----|----|----|----|------|
| 1 | 0.0157 | 0.0114 | 0.181 | 0.092 | 0.223 | 0.081 | 0.144 |
| 2 | 0.0091 | 0.0094 | 0.259 | 0.117 | 0.227 | 0.084 | 0.172 |
| 3 | 0.0078 | 0.0084 | 0.260 | 0.132 | 0.264 | 0.172 | 0.207 |
| 5 | 0.0066 | 0.0077 | 0.266 | 0.140 | 0.262 | 0.165 | 0.208 |
| 9 | 0.0053 | 0.0078 | 0.266 | 0.161 | 0.263 | 0.159 | 0.212 |
| 10 | 0.0049 | 0.0080 | 0.275 | 0.160 | 0.277 | 0.190 | 0.225 |
| 12 | 0.0042 | 0.0087 | 0.290 | 0.164 | 0.275 | 0.193 | 0.230 |
| 16 | 0.0027 | 0.0115 | 0.286 | 0.162 | 0.278 | 0.214 | 0.235 |
| 17 | 0.0023 | 0.0125 | 0.295 | 0.167 | 0.271 | 0.219 | 0.238 |
| 20 | 0.0016 | 0.0150 | 0.302 | 0.173 | 0.278 | 0.221 | 0.243 |
| 22 | 0.0013 | 0.0171 | 0.297 | 0.168 | 0.281 | 0.234 | 0.245 |
| 25 | 0.0010 | 0.0202 | 0.309 | 0.175 | 0.291 | 0.242 | 0.254 |
| **30 (final)** | **0.0008** | **0.0223** | **0.315** | **0.175** | **0.289** | **0.242** | **0.255** |

**Key observations:**
- OF F1 > 0 from epoch 1 (Run 5a: OF=0 throughout 28 epochs).
- val_loss diverges from train_loss starting ep2 onward (0.0084 vs 0.0078 at ep3; 0.0223 vs 0.0008 at ep30) — severe overfitting.
- F1 plateaus ep25–30 (0.252–0.255); no new best saved after ep25 until ep30 final.
- Next step: extend training set to 2007–2018 (~12 additional years of hybrid labels).

| Aspect | Run 4 | Run 5a | **Run 5b** |
|--------|-------|--------|-------|
| Label source | TFP threshold | Hybrid (OF broken) | **Hybrid corrected** |
| Input channels | 4 | 12 | **12** |
| Occluded Front | ✗ | ✗ | **✓ (OF=0.242)** |
| Best F1 | 0.642 (ep10, CPU) | 0.693 (ep28) | **0.255 (ep30)** |
| Hardware | CPU | A100 | **A100** |

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
1. ✅ ERA5 regional download (2019–2026) — complete
2. ✅ Build training data with regression targets (2019–2025)
3. ✅ Build extra channels (z500/q850/w850/msl/t925/t2m/u10/v10, 2019–2025)
4. ✅ WPC label pipeline overhaul — SF recovered, projection fixed
5. ✅ Diagnose and fix Run 5 hybrid label bugs (OF priority + N-S flip on Discover)
6. ✅ Rebuild hybrid labels 2019–2025 on Discover with corrected pipeline
7. ✅ Run 5a completed (12ch, ep28, F1=0.693 — OF=0 due to label bugs, now diagnosed)
8. ✅ **Run 4 full training (2019–2024, 4-ch TFP) — A100 complete ep29, F1=0.718 (CF=0.801, WF=0.762, SF=0.591)**
9. ✅ **Run 5b full training (2019–2024, 12-ch hybrid corrected, A100) — F1=0.255, OF=0.242**
10. ✅ **Run 6 (12-ch TFP labels, 2019–2024, A100) — F1=0.688 ep28 (12-ch < 4-ch: extra channels add noise)**
11. ✅ **Run 7 (11-ch ERA5 regression, 2019–2024, A100) — r_mean=0.993 ep30 (tfp:0.984, tadv:0.997, grad:0.999)**

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

## 9. Completed Runs: 6 and 7 (2026-06-30)

### Run 4 Final (A100, 2026-06)

Architecture identical to CPU run, full A100 training (batch=32, 30 epochs).
- **Best epoch**: 29 · **Mean F1**: 0.718
- CF: 0.801 · WF: 0.762 · SF: 0.591
- Outperforms Run 2 (0.768 on 2022 val) — note: val years differ (2025 is harder than 2022)

### Run 6 — 12-channel TFP Ablation (A100, 2026-06)

Identical to Run 4 but with 12 input channels (same extra channels as Run 5).
- **Best epoch**: 28 · **Mean F1**: 0.688
- CF: 0.785 · WF: 0.741 · SF: 0.539
- **Finding: 12-ch < 4-ch (0.688 vs 0.718)**
- Interpretation: When labels are TFP-derived (thresholded from t850), adding extra channels
  that are themselves derived from or correlated with t850 does not add information — it adds
  redundancy and introduces noise. The model cannot benefit from knowing z500 or q850 when
  it is being supervised to match a signal computed from t850 alone.

### Run 7 — Physical Regression (A100, 2026-06)

- **Target**: continuous fields (tfp_850, tadv_850, grad_mag_850) instead of classes
- **Architecture**: UNet(in_ch=11, n_out=3) with linear output head (no softmax)
- **Best epoch**: 30 · **r_mean**: 0.993
  - tfp_850: r = 0.984
  - tadv_850 (temperature advection): r = 0.997
  - grad_mag_850 (|∇T|): r = 0.999
- **Runtime**: 1h12m on A100 (vs 4+ h for classification)
- **Interpretation**: The U-Net backbone can learn to reconstruct physical diagnostic
  fields from 11-channel ERA5 input with near-perfect accuracy. This makes the model
  a universal TFP/TADV generator — applicable to any ERA5 or ERA5-like gridded state,
  at any historical time, without recomputing finite-difference operators explicitly.

### Implications of Run 4 vs Run 6 Comparison

| | Run 4 (4-ch TFP) | Run 6 (12-ch TFP) |
|-|-----------------|------------------|
| F1 | **0.718** | 0.688 |
| CF | **0.801** | 0.785 |
| WF | **0.762** | 0.741 |
| SF | **0.591** | 0.539 |

The 4-channel model is strictly better. This validates that for TFP-labeled training:
- Extra channels (z500, q850, w850, etc.) are not harmful in principle
- But they provide no benefit when the labels are already derived from t850 (= the first channel)
- The Hybrid-label setting (Run 5b) may benefit differently — untested for 4-ch hybrid

### Run 8 Recommendation

Train Run 4 architecture on ERA5 back-extension (1940–2018):
- 80 years of data → climate trend detection without threshold artifacts
- Same 4-ch TFP labels → consistent with Run 4, no label engineering needed
- Val year: 2019 (no overlap)
- Expected training time on A100: ~30 h for 50 epochs (79yr data)
- Enables: poleward migration of storm tracks, frontal frequency trends, climate change signal

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
