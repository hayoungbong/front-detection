# Research Report — Deep Learning for Weather Front Detection

**Author:** H. Bong

---

## 1. Motivation

Weather fronts — boundaries between air masses of different temperature and
humidity — drive a disproportionate share of high-impact weather:

- **Precipitation** from frontal lifting
- **Extreme winds** in post-frontal cold advection
- **Wildfire spread** behind dry, gusty cold fronts
- **Tropical-cyclone transition** when a storm merges with a frontal system
- **Drought onset** when frontal moisture supply stops

Fronts are still drawn **by hand** by human analysts. This project automates the
task with a U-Net trained on ERA5 reanalysis, so that fronts become a
quantitative, reproducible layer available for any period.

---

## 2. Model

All runs use a standard **U-Net** (encoder–bottleneck–decoder with skip
connections), a segmentation architecture that classifies every grid cell.

- 4 down/up levels, ~31M parameters
- **Focal loss** (γ=2) to handle the strong class imbalance (background dominates)
- AdamW optimizer with cosine learning-rate decay, ~30 epochs
- The regressor reuses the same backbone with a linear output head

---

## 3. Labels

| Label | How it is made | Trade-off |
|-------|----------------|-----------|
| **TFP** | Thermal Front Parameter computed from ERA5 temperature | Objective and grid-accurate, but cannot classify front type |
| **WPC** | Front types read from expert surface analyses | Correct types, but human-drawn → positional noise |
| **Hybrid** | ERA5/TFP *position* ∩ WPC *type* | Clean positions + expert types; enables occluded-front detection |

---

## 4. Results

The project advanced through a sequence of runs, each answering one question.
Compute evolved alongside — from a laptop (CPU, then Apple-silicon GPU) to
multi-GPU A100 training — steadily shrinking per-epoch time.

| Run | Setup | Outcome |
|-----|-------|---------|
| Baseline | 4-channel TFP classifier, few years | Fronts are learnable; clean reference |
| **Best classifier** | 4-channel TFP, 6 years | F1 ≈ **0.72** (CF 0.80, WF 0.76, SF 0.59) |
| **Hybrid** | 12-channel Hybrid labels | First model to detect **occluded fronts** |
| **Regression** | ERA5-only, continuous targets | r ≈ **0.99** with physical frontal fields |
| **Scaling** | Hybrid, 15 years, multi-GPU | Tests whether more data restores type-classification skill |

### Key findings

- **More data helps** — extending the training period consistently raised skill.
- **Extra channels can hurt** — when labels are TFP-derived, additional input
  channels are redundant and slightly *reduce* skill; channel selection matters.
- **Hybrid is the only route to occluded fronts**, but human-label noise makes
  the other classes harder; balancing this is an open problem.
- **Regression sidesteps label noise entirely** — predicting continuous ERA5
  diagnostics is threshold-free and reproducible, and reaches near-perfect
  correlation with the target fields.

---

## 5. Applications

### 5.1 Tropical-cyclone / front interaction
Applying the classifier through a storm's lifetime tracks how fronts wrap into
the circulation during extratropical transition. In the Helene (2024) case, the
modeled warm-front fraction peaks right at the transition onset, after which a
blocking ridge cut off frontal moisture — consistent with the subsequent
Northeast-US dry spell.

### 5.2 Wildfire
Cold-front passage brings dry, gusty post-frontal air. Overlaying modeled
cold-front probability with a fire-weather index highlights post-frontal
fire-spread windows.

### 5.3 Front climatology
Running the model across many years yields seasonal frequency maps, zonal-mean
distributions, and multi-decadal trends in front position and frequency.

---

## 6. Current direction

1. **Better models** — explore which input channels help, and how to use the
   WPC-derived type information without importing its noise.
2. **Applications with domain experts** — tropical/extratropical cyclones,
   drought, wildfire, and long-term climatology, each developed with specialist
   input.
3. **Historical extension** — applying trained models to the full ERA5 record
   (1940+) for climate-scale front trends.
