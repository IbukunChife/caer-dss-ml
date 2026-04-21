# Technical Methodology / Metodologia Técnica

## Dataset

- **Source:** CAER LIMS (Laboratory Information Management System)
- **Period:** January 2014 – February 2026
- **Total records:** 6,148
- **Sampling points:** Raw water (n=54), treated water (n=1,500), distribution network (n=4,434)
- **ML subset:** 10 samples with sulfate records (2016–2017)

## Features

| Feature | Description |
|---------|-------------|
| `ph` | Water pH |
| `turbidez` | Turbidity (NTU) |
| `cor` | Apparent color (uH) |
| `mes` | Collection month (1–12) |
| `periodo_chuvoso` | Binary flag: rainy season (Apr–Sep = 1) |

**Target (y):** `sulfato` — sulfate residual used as proxy for coagulant dosage (mg/L)

## Preprocessing

- Filter: city=BOA VISTA, source=RIO BRANCO, system=SÃO PEDRO, year≥2014
- Drop features with zero variance after imputation (temperatura, alcalinidade_bicarbonatos)
- StandardScaler normalization for Ridge Regression

## Validation

**Leave-One-Out Cross-Validation (LOOCV)** — standard protocol for n < 30 samples.

Each iteration: train on n-1 samples, predict on the held-out sample.
Metrics: MAE, RMSE, R².

## Models

### Ridge Regression
- Linear baseline with L2 regularization
- `alpha = 1.0`

### Random Forest
- `n_estimators = 500`
- `random_state = 42`

### XGBoost
- `n_estimators = 100`
- `max_depth = 2`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `reg_alpha = 1.0`, `reg_lambda = 1.0`

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) — based on cooperative game theory.
Applied to Random Forest and XGBoost to quantify each variable's contribution.

**Global importance (Random Forest):**

| Variable | Mean SHAP |
|----------|-----------|
| Cor Aparente | 0.391 |
| Turbidez | 0.284 |
| Período Chuvoso | 0.198 |
| pH | 0.087 |
| Mês | 0.040 |
