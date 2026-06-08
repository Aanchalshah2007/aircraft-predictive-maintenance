# Aircraft Engine Predictive Maintenance

Predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines from multivariate sensor data, enabling proactive maintenance before failure. Built on NASA's C-MAPSS dataset, the project compares a gradient-boosting baseline against a deep learning sequence model.

---

## Problem

Unplanned aircraft engine failures are costly and dangerous. If we can estimate how many operational cycles an engine has left before it degrades to failure, airlines can schedule maintenance proactively — reducing downtime and improving safety. This project frames that as a supervised regression problem: given a window of sensor readings, predict the Remaining Useful Life (RUL) in cycles.

---

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD001 subset**

- 100 engines run to failure under a single operating condition
- 21 sensor measurements + 3 operational settings per cycle
- Engine lifetimes range from 128 to 362 cycles
- Source: NASA C-MAPSS on Kaggle

---

## Approach

**1. Data exploration**
Analysed sensor behaviour across engine lifetimes and identified that 6 sensors had near-zero variance (no usable signal), leaving 15 informative sensors.

**2. Target engineering**
Computed RUL as `max_cycle − current_cycle` per engine, then applied **piecewise clipping at 125 cycles** (standard in C-MAPSS literature) so the model focuses on the meaningful pre-failure window rather than wasting capacity on early life where no degradation signal exists.

**3. Leakage-safe validation**
Used **engine-level train/validation splitting** (80/20) so no engine appears in both sets. This mirrors real deployment — predicting RUL for engines never seen during training. A naive random row-split would leak future cycles of the same engine into validation, inflating metrics.

**4. Feature engineering**
Created **75 features** including rolling means and standard deviations over 5- and 10-cycle windows (grouped by engine to avoid cross-engine bleed), then applied `StandardScaler` fit on training data only.

**5. Modeling**
- **XGBoost** baseline, tuned across 5 hyperparameter configurations
- **2-layer LSTM** (64 → 32 units, dropout 0.2) on 30-cycle sequences, with EarlyStopping

---

## Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| XGBoost (baseline) | 20.16 | 14.18 | 0.7642 |
| XGBoost (tuned) | 20.03 | 14.00 | 0.7673 |
| **LSTM** | **17.40** | **13.72** | **0.8270** |

The LSTM reduced RMSE by **13%** over the tuned XGBoost and raised R² from 0.77 to 0.83. Notably, XGBoost tuning gave only a 0.67% improvement — signalling it had hit an architectural ceiling. The LSTM broke through by modelling temporal dependencies in the sensor sequences directly, rather than relying on hand-engineered rolling features.

---

## Project Structure
