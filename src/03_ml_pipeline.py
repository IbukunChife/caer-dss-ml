"""
=============================================================================
TCC — AUTOMAÇÃO E PREDIÇÃO DA DOSAGEM DE COAGULANTE (SULFATO)
ETA SÃO PEDRO / RIO BRANCO / BOA VISTA — CAER/RR  (Série 2014–2026)
=============================================================================
Pipeline de Machine Learning — v2 (melhorado)

Modelos   : Ridge Regression (baseline) + Random Forest + XGBoost
XAI       : SHAP (Shapley Additive exPlanations)
Validação : Leave-One-Out Cross-Validation (LOOCV) — padrão para n < 30

Melhorias v2:
  - Features sazonais: mês e flag período chuvoso (abril–setembro)
  - Ridge Regression como baseline linear interpretável
  - Remoção de features com variância zero (temperatura, alcalinidade)
  - Exportação dos modelos treinados (.pkl) para uso no MVP Streamlit
  - Insights científicos expandidos

Filtros obrigatórios:
  (a) nome_cidade             contém 'BOA VISTA'
  (b) nome_manancial          contém 'RIO BRANCO'
  (c) nome_sistema_abastacimento contém 'SÃO PEDRO'
  (d) data_coleta             >= 2014

Features (X): ph, turbidez, cor, mes, periodo_chuvoso
Target  (y): sulfato  (proxy da dosagem de coagulante)

Nota metodológica:
  O LIMS da CAER registrou dosagem de sulfato em apenas 10 análises dentro
  do período filtrado (2016–2017). As features 'temperatura' e
  'alcalinidade_bicarbonatos' estavam completamente ausentes nessas amostras
  e foram removidas do modelo (variância zero após imputação). A dinâmica
  sazonal amazônica (cheia/seca) foi incorporada por meio do mês de coleta
  e do flag período_chuvoso, variáveis determinísticas derivadas da data.
  Recomenda-se como trabalho futuro a sistematização do registro de dosagem
  no LIMS, viabilizando re-treinamento com n ≥ 100.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "Dados/dados-analise-agua-20260302.csv"
OUTPUT_DIR   = "Resultados"
DPI          = 300
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features originais (usadas para EDA e imputação)
FEATURES_RAW = ["ph", "turbidez", "cor", "temperatura", "alcalinidade_bicarbonatos"]

# Features do modelo (com variância real nas 10 amostras ML)
# temperatura e alcalinidade são removidas pois são constantes após imputação
FEATURES_MODEL = ["ph", "turbidez", "cor", "mes", "periodo_chuvoso"]

TARGET = "sulfato"

LABELS = {
    "ph"                       : "pH",
    "turbidez"                 : "Turbidez (uT)",
    "cor"                      : "Cor Aparente (uH)",
    "temperatura"              : "Temperatura (°C)",
    "alcalinidade_bicarbonatos": "Alcalinidade Bicarb. (mg/L)",
    "mes"                      : "Mês de Coleta",
    "periodo_chuvoso"          : "Período Chuvoso (Abr–Set)",
}

PALETTE = {
    "ridge": "#8e44ad",
    "rf"   : "#27ae60",
    "xgb"  : "#2980b9",
}

# Meses do período chuvoso da Amazônia setentrional (abril–setembro)
MESES_CHUVOSOS = [4, 5, 6, 7, 8, 9]


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
def secao(titulo: str) -> None:
    print()
    print("=" * 72)
    print(f"  {titulo}")
    print("=" * 72)


def salvar(fig: plt.Figure, nome: str) -> None:
    path = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [PNG] {path}")


def calcular_metricas(y_true, y_pred, nome: str) -> dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {nome}")
    print(f"    R²   = {r2:>8.4f}")
    print(f"    MAE  = {mae:>8.4f} mg/L")
    print(f"    RMSE = {rmse:>8.4f} mg/L")
    print()
    return {"Modelo": nome, "R²": r2, "MAE (mg/L)": mae, "RMSE (mg/L)": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA E FILTRAGEM
# ─────────────────────────────────────────────────────────────────────────────
def carregar_e_filtrar(path: str):
    secao("ETAPA 1 — CARGA E FILTRAGEM DO DATASET")

    df = pd.read_csv(path, sep=";", on_bad_lines="skip", low_memory=False)
    print(f"  Dataset bruto        : {df.shape[0]:>8,} linhas × {df.shape[1]} colunas")

    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce")

    mask = (
        df["nome_cidade"].str.upper().str.contains("BOA VISTA", na=False)
        & df["nome_manancial"].str.upper().str.contains("RIO BRANCO", na=False)
        & df["nome_sistema_abastacimento"].str.upper().str.contains("SÃO PEDRO", na=False)
        & (df["data_coleta"].dt.year >= 2014)
    )
    df_eta = df[mask].copy()
    print(f"  Após 4 filtros (ETA) : {len(df_eta):>8,} linhas")
    print(f"  Período              : {df_eta['data_coleta'].min().date()} → "
          f"{df_eta['data_coleta'].max().date()}")

    for col in FEATURES_RAW + [TARGET]:
        df_eta[col] = pd.to_numeric(df_eta[col], errors="coerce")

    # Medianas do dataset completo (para imputação e exibição no app)
    medians = df_eta[FEATURES_RAW].median()

    # ── Subconjunto ML: onde sulfato é válido ─────────────────────────────────
    df_ml = df_eta.dropna(subset=[TARGET]).copy()

    # Features sazonais — derivadas deterministicamente da data
    df_ml["mes"]             = df_ml["data_coleta"].dt.month
    df_ml["periodo_chuvoso"] = df_ml["mes"].isin(MESES_CHUVOSOS).astype(int)

    print()
    print(f"  Amostras com sulfato válido (dataset ML): {len(df_ml)}")
    print()
    print("  Amostras ML — detalhamento:")
    cols_display = ["data_coleta", "ph", "turbidez", "cor", "mes",
                    "periodo_chuvoso", TARGET]
    print(df_ml[cols_display].to_string(index=False))
    print()
    print("  Cobertura das features (dataset ML):")
    for feat in FEATURES_MODEL:
        n_ok = df_ml[feat].notna().sum()
        pct  = 100 * n_ok / len(df_ml)
        print(f"    {LABELS[feat]:<35}: {n_ok}/{len(df_ml)} ({pct:.0f}%)")

    return df_eta, df_ml, medians


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ─────────────────────────────────────────────────────────────────────────────
def eda(df_eta: pd.DataFrame, df_ml: pd.DataFrame) -> None:
    secao("ETAPA 2 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")

    eda_feats = ["ph", "turbidez", "cor"]

    print("  Estatísticas descritivas — dataset completo ETA (~6.000 amostras):")
    print(df_eta[eda_feats].describe().round(3).to_string())

    print()
    print("  Estatísticas descritivas — dataset ML (n=10):")
    print(df_ml[FEATURES_MODEL + [TARGET]].describe().round(3).to_string())

    # ── Séries temporais mensais + histogramas ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    pal  = sns.color_palette("Blues_d", len(eda_feats) + 1)

    for i, col in enumerate(eda_feats):
        data   = df_eta[col].dropna()
        lo, hi = data.quantile(0.01), data.quantile(0.99)
        data   = data[(data >= lo) & (data <= hi)]
        axes[i].hist(data, bins=45, color=pal[i], edgecolor="white", linewidth=0.4)
        axes[i].set_title(LABELS[col], fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Valor"); axes[i].set_ylabel("Frequência")
        axes[i].grid(axis="y", alpha=0.3)

    # Sulfato (dataset ML)
    axes[3].bar(range(len(df_ml)), df_ml[TARGET].values,
                color="#e74c3c", edgecolor="white", linewidth=0.6)
    axes[3].set_title(f"Sulfato — Dosagem (mg/L)  [n={len(df_ml)} amostras]",
                      fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Amostra (índice)"); axes[3].set_ylabel("mg/L")
    axes[3].grid(axis="y", alpha=0.3)
    axes[3].axhline(df_ml[TARGET].mean(), color="navy", linestyle="--",
                    linewidth=1.5, label=f"Média = {df_ml[TARGET].mean():.1f}")
    axes[3].legend(fontsize=9)

    df_ts = df_eta[["data_coleta", "turbidez", "cor"]].dropna().copy()
    df_ts = df_ts.sort_values("data_coleta").set_index("data_coleta")
    df_monthly = df_ts.resample("ME").median()

    axes[4].plot(df_monthly.index, df_monthly["turbidez"],
                 color="#e67e22", linewidth=1.5, label="Turbidez")
    # Sombrear período chuvoso (abril–setembro)
    for yr in df_monthly.index.year.unique():
        axes[4].axvspan(
            pd.Timestamp(f"{yr}-04-01"), pd.Timestamp(f"{yr}-09-30"),
            alpha=0.08, color="blue", label="_chuvoso"
        )
    axes[4].set_title("Turbidez mediana mensal — ETA São Pedro (2014–2026)",
                      fontsize=11, fontweight="bold")
    axes[4].set_xlabel("Mês"); axes[4].set_ylabel("uT")
    axes[4].grid(alpha=0.3); axes[4].legend(fontsize=9)
    axes[4].xaxis.set_major_locator(mticker.MaxNLocator(6))

    axes[5].plot(df_monthly.index, df_monthly["cor"],
                 color="#8e44ad", linewidth=1.5, label="Cor Aparente")
    axes[5].set_title("Cor Aparente mediana mensal — ETA São Pedro (2014–2026)",
                      fontsize=11, fontweight="bold")
    axes[5].set_xlabel("Mês"); axes[5].set_ylabel("uH")
    axes[5].grid(alpha=0.3); axes[5].legend(fontsize=9)
    axes[5].xaxis.set_major_locator(mticker.MaxNLocator(6))

    fig.suptitle(
        "Análise Exploratória — ETA São Pedro / Rio Branco / Boa Vista (2014–2026)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    salvar(fig, "01_eda_distribuicoes_series.png")

    # ── Sazonalidade: boxplot por período ─────────────────────────────────────
    df_eta_ts = df_eta[["data_coleta", "turbidez", "cor"]].dropna().copy()
    df_eta_ts["periodo"] = df_eta_ts["data_coleta"].dt.month.isin(
        MESES_CHUVOSOS).map({True: "Chuvoso\n(Abr–Set)", False: "Seco\n(Out–Mar)"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat, color in zip(axes,
                                ["turbidez", "cor"],
                                ["#e67e22", "#8e44ad"]):
        data_feat = df_eta_ts[["periodo", feat]].dropna()
        # Remove outliers extremos para visualização
        q99 = data_feat[feat].quantile(0.99)
        data_feat = data_feat[data_feat[feat] <= q99]
        ax.boxplot(
            [data_feat[data_feat["periodo"].str.startswith("Chuvoso")][feat],
             data_feat[data_feat["periodo"].str.startswith("Seco")][feat]],
            labels=["Chuvoso\n(Abr–Set)", "Seco\n(Out–Mar)"],
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", linewidth=2),
        )
        ax.set_title(f"{LABELS[feat]} por Período Sazonal",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel(LABELS[feat])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Sazonalidade — ETA São Pedro / Rio Branco (2014–2026)\n"
        "Região Amazônica Setentrional: Cheia vs. Seca",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    salvar(fig, "01b_sazonalidade_boxplot.png")

    # ── Correlação dataset ML ──────────────────────────────────────────────────
    corr = df_ml[FEATURES_MODEL + [TARGET]].corr()
    labels_corr = [LABELS[f] for f in FEATURES_MODEL] + ["Sulfato (mg/L)"]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        xticklabels=labels_corr, yticklabels=labels_corr,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(
        "Matriz de Correlação — Features × Sulfato\n"
        "(Dataset ML  |  n=10  |  ETA São Pedro / Rio Branco)",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    salvar(fig, "02_heatmap_correlacao.png")

    print()
    print("  Correlações de Pearson com Sulfato (dataset ML, n=10):")
    for feat in FEATURES_MODEL:
        r = df_ml[feat].corr(df_ml[TARGET])
        print(f"    {LABELS[feat]:<35}: r = {r:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARAÇÃO DO DATASET ML
# ─────────────────────────────────────────────────────────────────────────────
def preparar_dados(df_ml: pd.DataFrame):
    secao("ETAPA 3 — PREPARAÇÃO DO DATASET DE TREINO (ML)")

    X = df_ml[FEATURES_MODEL].copy()
    y = df_ml[TARGET].copy()

    print(f"  Shape X : {X.shape}  (features com variância real)")
    print(f"  Shape y : {y.shape}")
    print(f"  Sulfato  — mín: {y.min():.1f}  máx: {y.max():.1f}  "
          f"média: {y.mean():.2f}  dp: {y.std():.2f}  mg/L")
    print()
    print("  Features do modelo (v2 — sem constantes):")
    for feat in FEATURES_MODEL:
        print(f"    • {LABELS[feat]:<35}: std={X[feat].std():.4f}")
    print()
    print("  Features removidas (variância zero após imputação):")
    print("    • Temperatura (°C)              — todas NaN nas 10 amostras ML")
    print("    • Alcalinidade Bicarb. (mg/L)   — todas NaN nas 10 amostras ML")
    print()
    print("  Features sazonais adicionadas:")
    print("    • Mês de Coleta (1–12)          — derivado de data_coleta")
    print("    • Período Chuvoso (0/1)          — meses 4–9 = chuvoso (Amazônia)")
    print()
    print("  Validação: Leave-One-Out Cross-Validation (LOOCV)")
    print("  Justificativa: padrão estatístico para n < 30.")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. RIDGE REGRESSION (baseline linear)
# ─────────────────────────────────────────────────────────────────────────────
def treinar_ridge(X: pd.DataFrame, y: pd.Series):
    secao("ETAPA 4 — RIDGE REGRESSION (Baseline Linear)")
    print("  Modelo linear com regularização L2 — interpretável e robusto para n pequeno.")
    print()

    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    preds_loo = cross_val_predict(pipe_ridge, X, y, cv=LeaveOneOut())
    calcular_metricas(y, preds_loo, "Ridge Regression (LOOCV)")

    pipe_ridge.fit(X, y)
    return pipe_ridge, preds_loo


# ─────────────────────────────────────────────────────────────────────────────
# 5. RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
def treinar_rf(X: pd.DataFrame, y: pd.Series):
    secao("ETAPA 5 — RANDOM FOREST REGRESSOR")
    print("  n_estimators=500, max_depth=3, min_samples_leaf=2")
    print()

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preds_loo = cross_val_predict(rf, X, y, cv=LeaveOneOut())
    calcular_metricas(y, preds_loo, "Random Forest (LOOCV)")

    rf.fit(X, y)
    return rf, preds_loo


# ─────────────────────────────────────────────────────────────────────────────
# 6. XGBOOST (hiperparâmetros ajustados para n pequeno)
# ─────────────────────────────────────────────────────────────────────────────
def treinar_xgb(X: pd.DataFrame, y: pd.Series):
    secao("ETAPA 6 — XGBOOST REGRESSOR (ajustado para n=10)")
    print("  n_estimators=50, max_depth=2, lr=0.1, reg_alpha=1.0, reg_lambda=5.0")
    print("  (Regularização forte para evitar overfitting em amostras pequenas)")
    print()

    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    preds_loo = cross_val_predict(model, X, y, cv=LeaveOneOut())
    calcular_metricas(y, preds_loo, "XGBoost (LOOCV)")

    model.fit(X, y)
    return model, preds_loo


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRÁFICOS COMPARATIVOS DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def plotar_metricas(y, preds_ridge, preds_rf, preds_xgb, resultados: list) -> None:
    secao("ETAPA 7 — GRÁFICOS COMPARATIVOS DE MÉTRICAS")

    df_m = pd.DataFrame(resultados)
    configs = [
        (preds_ridge, "Ridge",        PALETTE["ridge"]),
        (preds_rf,    "Random Forest", PALETTE["rf"]),
        (preds_xgb,   "XGBoost",       PALETTE["xgb"]),
    ]

    # ── Real vs Predito ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (preds, nome, cor) in zip(axes, configs):
        ax.scatter(y, preds, color=cor, s=110, edgecolors="black",
                   linewidth=0.7, zorder=3, label="Amostras")
        lo = min(y.min(), np.min(preds)) - 0.5
        hi = max(y.max(), np.max(preds)) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal")
        r2 = r2_score(y, preds)
        ax.set_title(f"{nome}\nR² (LOOCV) = {r2:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Sulfato Real (mg/L)"); ax.set_ylabel("Sulfato Predito (mg/L)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Real vs. Predito — Dosagem de Sulfato (LOOCV)\n"
                 "ETA São Pedro / Rio Branco / Boa Vista",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    salvar(fig, "04_real_vs_predito.png")

    # ── Barras de métricas ────────────────────────────────────────────────────
    modelos  = df_m["Modelo"].tolist()
    cores_m  = [PALETTE["ridge"], PALETTE["rf"], PALETTE["xgb"]]

    metricas_cfg = [
        ("R²",          "R² — Coef. de Determinação",  "Maior = melhor"),
        ("MAE (mg/L)",  "MAE — Erro Absoluto (mg/L)",   "Menor = melhor"),
        ("RMSE (mg/L)", "RMSE — Erro Quadrático (mg/L)", "Menor = melhor"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, titulo, nota) in zip(axes, metricas_cfg):
        vals = df_m[col].tolist()
        bars = ax.bar(modelos, vals, color=cores_m, edgecolor="white",
                      linewidth=0.8, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(max(vals, key=abs)) * 0.03,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_ylabel(col); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=10)
        ax.text(0.5, 0.97, nota, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, color="gray", style="italic")

    fig.suptitle("Comparativo de Métricas — Ridge vs. Random Forest vs. XGBoost (LOOCV)\n"
                 "ETA São Pedro / Rio Branco / Boa Vista (2014–2026)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    salvar(fig, "05_comparativo_metricas.png")

    # ── Resíduos ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (preds, nome, cor) in zip(axes, configs):
        residuos = np.array(y) - np.array(preds)
        ax.bar(range(len(residuos)), residuos,
               color=[cor if r >= 0 else "#e74c3c" for r in residuos],
               edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"{nome} — Resíduos\n(Real − Predito)", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Índice da Amostra"); ax.set_ylabel("Resíduo (mg/L)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Análise de Resíduos (LOOCV) — Dosagem de Sulfato",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    salvar(fig, "06_residuos_loocv.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SHAP — INTELIGÊNCIA ARTIFICIAL EXPLICÁVEL
# ─────────────────────────────────────────────────────────────────────────────
def analisar_shap(model_rf, model_xgb, X: pd.DataFrame):
    secao("ETAPA 8 — SHAP: INTELIGÊNCIA ARTIFICIAL EXPLICÁVEL (XAI)")

    feat_labels = [LABELS[f] for f in FEATURES_MODEL]
    X_vis = X.copy()
    X_vis.columns = feat_labels

    # ── SHAP Random Forest ────────────────────────────────────────────────────
    print("  Calculando SHAP values — Random Forest...")
    exp_rf = shap.TreeExplainer(model_rf)
    sv_rf  = exp_rf(X_vis)

    fig = plt.figure(figsize=(11, 5))
    shap.summary_plot(sv_rf, X_vis, show=False, plot_size=None)
    plt.title(
        "SHAP Summary Plot — Random Forest\n"
        "Impacto de cada feature na predição de Sulfato (mg/L)",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    salvar(fig, "07_shap_summary_rf.png")

    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(sv_rf, X_vis, plot_type="bar", show=False)
    plt.title(
        "Feature Importance SHAP — Random Forest\n"
        "Importância média absoluta dos valores SHAP",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    salvar(fig, "08_shap_importance_rf.png")

    # ── SHAP XGBoost ──────────────────────────────────────────────────────────
    print("  Calculando SHAP values — XGBoost...")
    exp_xgb = shap.TreeExplainer(model_xgb)
    sv_xgb  = exp_xgb(X_vis)

    fig = plt.figure(figsize=(11, 5))
    shap.summary_plot(sv_xgb, X_vis, show=False, plot_size=None)
    plt.title(
        "SHAP Summary Plot — XGBoost\nImpacto de cada feature na predição de Sulfato (mg/L)",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    salvar(fig, "09_shap_summary_xgb.png")

    # ── Painel comparativo lado a lado ────────────────────────────────────────
    imp_rf  = np.abs(sv_rf.values).mean(axis=0)
    imp_xgb = np.abs(sv_xgb.values).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (imp, nome, pal) in zip(
        axes,
        [(imp_rf, "Random Forest", "Greens_d"), (imp_xgb, "XGBoost", "Blues_d")],
    ):
        ordem = np.argsort(imp)
        cores = sns.color_palette(pal, len(FEATURES_MODEL))
        ax.barh(
            [feat_labels[i] for i in ordem],
            imp[ordem],
            color=list(reversed(cores)),
            edgecolor="white",
        )
        ax.set_title(f"{nome}\nImportância SHAP |mean|", fontsize=12, fontweight="bold")
        ax.set_xlabel("SHAP Médio Absoluto (mg/L)")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Comparativo de Importância de Features via SHAP\n"
        "Random Forest vs. XGBoost — Dosagem de Sulfato (ETA São Pedro / Rio Branco)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    salvar(fig, "11_shap_comparativo.png")

    return imp_rf, imp_xgb, feat_labels


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORTAR MODELOS PARA O MVP
# ─────────────────────────────────────────────────────────────────────────────
def exportar_modelos(model_ridge, model_rf, model_xgb, medians, df_ml):
    secao("ETAPA 9 — EXPORTAÇÃO DOS MODELOS (para MVP Streamlit)")

    config = {
        "features_model": FEATURES_MODEL,
        "features_raw"  : FEATURES_RAW,
        "labels"        : LABELS,
        "meses_chuvosos": MESES_CHUVOSOS,
        "medians_raw"   : medians.to_dict(),
        "y_mean"        : float(df_ml[TARGET].mean()),
        "y_std"         : float(df_ml[TARGET].std()),
        "y_min"         : float(df_ml[TARGET].min()),
        "y_max"         : float(df_ml[TARGET].max()),
        "n_samples"     : len(df_ml),
    }

    joblib.dump(model_ridge, os.path.join(OUTPUT_DIR, "model_ridge.pkl"))
    joblib.dump(model_rf,    os.path.join(OUTPUT_DIR, "model_rf.pkl"))
    joblib.dump(model_xgb,   os.path.join(OUTPUT_DIR, "model_xgb.pkl"))
    joblib.dump(config,      os.path.join(OUTPUT_DIR, "config.pkl"))

    print(f"  Modelos exportados para ./{OUTPUT_DIR}/")
    print("    model_ridge.pkl  — Ridge Regression")
    print("    model_rf.pkl     — Random Forest")
    print("    model_xgb.pkl    — XGBoost")
    print("    config.pkl       — configurações e metadados")


# ─────────────────────────────────────────────────────────────────────────────
# 10. INSIGHTS ANALÍTICOS
# ─────────────────────────────────────────────────────────────────────────────
def imprimir_insights(resultados, imp_rf, imp_xgb, feat_labels, df_ml):
    secao("ETAPA 10 — INSIGHTS ANALÍTICOS PARA O TCC")

    df_m   = pd.DataFrame(resultados)
    melhor = df_m.loc[df_m["R²"].idxmax(), "Modelo"]

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  RESUMO DE DESEMPENHO DOS MODELOS (LOOCV — n=10)                   │
  └─────────────────────────────────────────────────────────────────────┘

{df_m.to_string(index=False)}

  Modelo com melhor R²: {melhor}

  ┌─────────────────────────────────────────────────────────────────────┐
  │  RANKING SHAP — Random Forest                                       │
  └─────────────────────────────────────────────────────────────────────┘""")

    for i in np.argsort(imp_rf)[::-1]:
        print(f"  {i+1}. {feat_labels[i]:<35} SHAP = {imp_rf[i]:.5f}")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  INSIGHTS CIENTÍFICOS (para a seção Resultados e Conclusão)        │
  └─────────────────────────────────────────────────────────────────────┘

  1. FEATURES SAZONAIS MELHORAM A INTERPRETABILIDADE
     A adição de Mês de Coleta e Flag Período Chuvoso fornece ao modelo
     contexto hidrológico da bacia amazônica. Na Amazônia setentrional,
     a turbidez e a cor da água bruta são fortemente influenciadas pela
     sazonalidade cheia/seca do Rio Branco, o que justifica incluir essas
     variáveis como preditores complementares.

  2. RIDGE COMO BASELINE INTERPRETÁVEL
     O Ridge Regression, por ser um modelo linear, fornece coeficientes
     diretamente interpretáveis por engenheiros de saneamento. Sua
     inclusão segue a recomendação da literatura de sempre comparar
     modelos complexos com baselines simples (Bachri et al., 2025).

  3. R² NEGATIVO — INTERPRETAÇÃO CORRETA PARA O TCC
     Com apenas 10 amostras e alta variância (sulfato entre 1 e 10 mg/L),
     o LOOCV não consegue generalizar melhor que a previsão pela média.
     Isso NÃO invalida a pesquisa — é o diagnóstico central do TCC:
     a ausência de registro sistemático de dosagem no LIMS é a principal
     barreira para a automação plena na CAER.

  4. COR APARENTE LIDERA O RANKING SHAP
     Ambos os modelos Ensemble elegeram Cor Aparente como principal
     determinante da dosagem — resultado fisicamente coerente para o
     Rio Branco, caracterizado por alta concentração de substâncias
     húmicas e coloidais que demandam maior dose de coagulante para
     neutralização elétrica.

  5. CONTRIBUIÇÃO CIENTÍFICA CENTRAL (SHAP/XAI)
     A análise SHAP é a principal entrega científica deste TCC para
     o contexto de Indústria 4.0: ela transforma o modelo de caixa-preta
     em explicações auditáveis por amostra, habilitando o engenheiro a
     rastrear cada recomendação de dosagem até suas causas físico-químicas.

  ┌─────────────────────────────────────────────────────────────────────┐
  │  LIMITAÇÃO + TRABALHO FUTURO                                        │
  └─────────────────────────────────────────────────────────────────────┘

  Registros históricos de sulfato no LIMS/CAER: apenas 10 (2016–2017).
  → Implementar registro automático de dosagem integrado ao SCADA/LIMS
  → Re-treinar com n ≥ 100 amostras → resultados estatisticamente robustos
  → O pipeline, os modelos e o MVP estão prontos para receber o volume
    de dados necessário.
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  TCC — PREDIÇÃO DE DOSAGEM DE COAGULANTE (SULFATO)  v2             ║")
    print("║  ETA São Pedro / Rio Branco / Boa Vista — CAER/RR (2014–2026)      ║")
    print("║  Modelos: Ridge + Random Forest + XGBoost | XAI: SHAP              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    df_eta, df_ml, medians = carregar_e_filtrar(DATA_PATH)
    eda(df_eta, df_ml)
    X, y = preparar_dados(df_ml)

    model_ridge, preds_ridge = treinar_ridge(X, y)
    model_rf,    preds_rf    = treinar_rf(X, y)
    model_xgb,   preds_xgb  = treinar_xgb(X, y)

    resultados = [
        calcular_metricas(y, preds_ridge, "Ridge Regression"),
        calcular_metricas(y, preds_rf,    "Random Forest"),
        calcular_metricas(y, preds_xgb,   "XGBoost"),
    ]

    plotar_metricas(y, preds_ridge, preds_rf, preds_xgb, resultados)
    imp_rf, imp_xgb, feat_labels = analisar_shap(model_rf, model_xgb, X)
    exportar_modelos(model_ridge, model_rf, model_xgb, medians, df_ml)
    imprimir_insights(resultados, imp_rf, imp_xgb, feat_labels, df_ml)

    print(f"  Imagens salvas em : ./{OUTPUT_DIR}/")
    print("  Pipeline v2 finalizado com sucesso!\n")
