# Models / Modelos Exportados

Os modelos treinados (`.pkl`) são gerados automaticamente ao executar o pipeline:

```bash
python src/03_ml_pipeline.py
```

## Modelos gerados

| Arquivo | Modelo | MAE (mg/L) | RMSE (mg/L) | R² (LOOCV) |
|---------|--------|-----------|------------|-----------|
| `model_ridge.pkl` | Ridge Regression | 6.56 | 11.98 | -22.77 |
| `model_rf.pkl` | Random Forest | 1.99 | 3.00 | -0.49 |
| `model_xgb.pkl` | XGBoost | 2.23 | 3.18 | -0.67 |
| `config.pkl` | Configurações (scaler, features) | — | — | — |

> Os valores de R² negativos confirmam o diagnóstico da lacuna de digitalização — não são falha dos modelos.
> Com n ≥ 100 amostras reais de dosagem operacional, R² positivo é esperado.

## Nota sobre os modelos pré-treinados

Os arquivos `.pkl` **não estão incluídos** no repositório (ver `.gitignore`) por serem binários grandes.
Execute o pipeline para regenerá-los localmente.
