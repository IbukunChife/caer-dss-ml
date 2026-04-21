# CAER-DSS-ML — Decision Support System for Coagulant Dosing at CAER/RR

> **PT:** Sistema de Suporte à Decisão (DSS) baseado em Machine Learning para dosagem de coagulantes na ETA São Pedro da CAER (Companhia de Águas e Esgotos de Roraima), com diagnóstico quantitativo da lacuna de digitalização operacional.
>
> **EN:** Machine Learning-based Decision Support System (DSS) for coagulant dosing at the São Pedro Water Treatment Plant (CAER/RR, Brazil), with quantitative diagnosis of the operational digitalization gap.

---

## About / Sobre

This project is the technical output of a graduate monograph (Lato Sensu Specialization in Applied Computing for Industry 4.0 — UFRR) authored by **Ibukun Chife Didier Adjitche**, under supervision of **Prof. Leandro Nelinho Balico**.

**Central finding:** The CAER LIMS contains only quality parameters (turbidity, color, pH), not operational coagulant dosage records. This absence — the *operational digitalization gap* — is the main barrier to full ML deployment. This DSS provides the complete architectural framework ready for activation when operational data becomes available.

---

## Repository Structure / Estrutura

```
caer-dss-ml/
├── src/
│   ├── 01_monitoramento.py      # Module 1 — Historical monitoring & EDA
│   ├── 02_diagnostico.py        # Module 2 — Digitalization gap diagnosis
│   ├── 03_ml_pipeline.py        # Module 3 — ML pipeline (Ridge + RF + XGBoost + SHAP)
│   └── 04_app_dss_mvp.py        # Module 4 — Streamlit DSS app (MVP)
│
├── data/
│   └── README.md                # Data instructions (raw data not included)
│
├── models/
│   └── README.md                # Exported model description
│
├── results/
│   └── figures/                 # Generated plots and visualizations
│
├── docs/
│   └── methodology.md           # Technical methodology summary
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Modules / Módulos

### Module 1 — Historical Monitoring (`01_monitoramento.py`)
Processes 6,148 historical records from CAER's LIMS (2014–2026):
- Monthly time series of turbidity, color, pH, and residual chlorine
- Amazonian seasonality analysis (rainy: Apr–Sep vs dry: Oct–Mar)
- Compliance with Brazilian standard Portaria GM/MS nº 888/2021
- Statistical comparison: raw water × treated water × distribution network

### Module 2 — Digitalization Gap Diagnosis (`02_diagnostico.py`)
Produces the central scientific finding of the study:
- Analysis of 47 raw/treated sample pairs collected on the same date
- Pearson correlation r ≈ 0 → statistical proof of operational dosage absence
- Visual evidence of the Industry 4.0 gap at CAER

### Module 3 — ML Pipeline (`03_ml_pipeline.py`)
Predictive framework validated by Leave-One-Out Cross-Validation (LOOCV):
- **Ridge Regression** — linear interpretable baseline
- **Random Forest** — best performance: MAE = 1.99 mg/L, RMSE = 3.00 mg/L
- **XGBoost** — MAE = 2.23 mg/L, RMSE = 3.18 mg/L
- **SHAP** explanability: Apparent Color (0.391) > Turbidity (0.284) > Rainy Season (0.198)
- Exports trained models as `.pkl` for use in the MVP app

### Module 4 — DSS MVP App (`04_app_dss_mvp.py`)
Interactive Streamlit application:
- Manual sliders for pH, Turbidity, Apparent Color, Month
- Real-time prediction from all 3 models
- Individual SHAP waterfall chart per prediction

---

## Quick Start / Como Usar

### Requirements
- Python 3.9+
- See `requirements.txt`

### Installation

```bash
git clone https://github.com/IbukunChife/caer-dss-ml.git
cd caer-dss-ml
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Running the DSS App

```bash
streamlit run src/04_app_dss_mvp.py
```

> **Note:** The app uses pre-trained models from `models/`. No data file needed to run the MVP.

### Running the Full ML Pipeline

```bash
# Place your data file at data/dados-analise-agua.csv
python src/03_ml_pipeline.py
```

### Running the Analysis Modules

```bash
python src/01_monitoramento.py   # Historical monitoring
python src/02_diagnostico.py     # Digitalization gap diagnosis
```

---

## Data / Dados

Raw data is **not included** in this repository due to institutional confidentiality (CAER/RR LIMS data).

To use your own data, place a CSV file with `;` separator at `data/dados-analise-agua.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `nome_cidade` | City name (filter: `BOA VISTA`) |
| `nome_manancial` | Water source (filter: `RIO BRANCO`) |
| `nome_sistema_abastacimento` | System name (filter: `SÃO PEDRO`) |
| `data_coleta` | Collection date |
| `ph` | pH |
| `turbidez` | Turbidity (NTU) |
| `cor` | Apparent color (uH) |
| `sulfato` | Sulfate — proxy for coagulant dosage (mg/L) |

See `data/README.md` for full details.

---

## Results / Resultados

| Model | R² (LOOCV) | MAE (mg/L) | RMSE (mg/L) |
|-------|-----------|------------|-------------|
| Ridge Regression | -22.77 | 6.56 | 11.98 |
| Random Forest | -0.49 | **1.99** | **3.00** |
| XGBoost | -0.67 | 2.23 | 3.18 |

> Negative R² values confirm the digitalization gap diagnosis — not model failure. With n ≥ 100 real dosage samples, positive R² is expected.

Generated figures are available in `results/figures/`.

---

## Citation / Citação

```
ADJITCHE, Ibukun Chife Didier. Machine Learning para Dosagem de Coagulantes em ETA:
Diagnóstico de Digitalização na CAER/RR. Monografia (Especialização em Computação
Aplicada na Indústria 4.0) — Universidade Federal de Roraima, Boa Vista, 2026.
```

---

## License / Licença

MIT License — see [LICENSE](LICENSE).

---

## Author / Autor

**Ibukun Chife Didier Adjitche**
- Email: adjitchedidier1@gmail.com
- Advisor / Orientador: Prof. Leandro Nelinho Balico — UFRR
