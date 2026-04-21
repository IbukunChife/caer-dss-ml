"""
=============================================================================
DSS — MÓDULO 1: MONITORAMENTO HISTÓRICO DA QUALIDADE DA ÁGUA
ETA São Pedro / Rio Branco / Boa Vista — CAER/RR (2014–2026)
=============================================================================
Analisa os 6.148 registros do LIMS-CAER:
  - Séries temporais de turbidez, cor, pH e cloro residual
  - Sazonalidade amazônica (período chuvoso abr–set vs seco out–mar)
  - Conformidade com a Portaria GM/MS nº 888/2021
  - Comparação estatística água bruta × tratada × rede

Gera figuras em: dss/Figuras/
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "Dados", "dados-analise-agua-20260302.csv")
FIG_DIR    = os.path.join(BASE_DIR, "Figuras")
DPI        = 300
os.makedirs(FIG_DIR, exist_ok=True)

MESES_CHUVOSOS = [4, 5, 6, 7, 8, 9]

# Limites VMP — Portaria GM/MS nº 888/2021
VMP = {
    "turbidez":           5.0,   # uT
    "cor":               15.0,   # uH
    "ph_min":             6.0,
    "ph_max":             9.5,
    "cloro_residual_livre": 0.2, # mg/L (mínimo)
}

CORES = {
    "bruta":   "#e67e22",
    "tratada": "#27ae60",
    "rede":    "#2980b9",
    "chuvoso": "#3498db",
    "seco":    "#e74c3c",
}


def salvar(fig, nome):
    path = os.path.join(FIG_DIR, nome)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {path}")


def secao(titulo):
    print()
    print("=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA E FILTRAGEM
# ─────────────────────────────────────────────────────────────────────────────
def carregar():
    secao("ETAPA 1 — CARGA E FILTRAGEM")

    df = pd.read_csv(DATA_PATH, sep=";", on_bad_lines="skip", low_memory=False)
    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce")

    mask = (
        df["nome_cidade"].str.upper().str.contains("BOA VISTA", na=False)
        & df["nome_manancial"].str.upper().str.contains("RIO BRANCO", na=False)
        & df["nome_sistema_abastacimento"].str.upper().str.contains("SÃO PEDRO", na=False)
        & (df["data_coleta"].dt.year >= 2014)
    )
    eta = df[mask].copy()

    for col in ["ph", "turbidez", "cor", "cloro_residual_livre", "condutividade"]:
        eta[col] = pd.to_numeric(eta[col], errors="coerce")

    eta["mes"]      = eta["data_coleta"].dt.month
    eta["ano"]      = eta["data_coleta"].dt.year
    eta["periodo"]  = eta["mes"].isin(MESES_CHUVOSOS).map(
        {True: "Chuvoso\n(Abr–Set)", False: "Seco\n(Out–Mar)"}
    )

    bruta   = eta[eta["nome_ponto_amostragem"].str.upper().str.contains("NATURA", na=False)].copy()
    tratada = eta[eta["nome_ponto_amostragem"].str.upper().str.contains("SAÍDA DO TRATAMENTO", na=False)].copy()
    rede    = eta[eta["nome_ponto_amostragem"].str.upper().str.contains("REDE", na=False)].copy()

    print(f"  Total ETA São Pedro   : {len(eta):>6,} registros (2014–2026)")
    print(f"  Água in natura (bruta): {len(bruta):>6,} registros")
    print(f"  Saída do tratamento   : {len(tratada):>6,} registros")
    print(f"  Rede de distribuição  : {len(rede):>6,} registros")

    return eta, bruta, tratada, rede


# ─────────────────────────────────────────────────────────────────────────────
# 2. SÉRIE TEMPORAL MENSAL — ÁGUA TRATADA
# ─────────────────────────────────────────────────────────────────────────────
def fig_serie_temporal(tratada):
    secao("ETAPA 2 — SÉRIE TEMPORAL MENSAL (ÁGUA TRATADA)")

    df_ts = tratada[["data_coleta", "turbidez", "cor", "ph", "cloro_residual_livre"]].copy()
    df_ts = df_ts.sort_values("data_coleta").set_index("data_coleta")
    monthly = df_ts.resample("ME").agg(["median", "std"])

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    params = [
        ("turbidez",            "Turbidez (uT)",           "#27ae60", VMP["turbidez"],     "VMP = 5 uT"),
        ("cor",                 "Cor Aparente (uH)",        "#8e44ad", VMP["cor"],           "VMP = 15 uH"),
        ("ph",                  "pH",                       "#2980b9", None,                 None),
        ("cloro_residual_livre","Cloro Residual Livre (mg/L)","#e67e22", VMP["cloro_residual_livre"], "Mínimo = 0.2 mg/L"),
    ]

    for ax, (col, label, cor, vmp_val, vmp_label) in zip(axes, params):
        med = monthly[col]["median"]
        std = monthly[col]["std"]

        ax.plot(med.index, med.values, color=cor, linewidth=1.8, label="Mediana mensal")
        ax.fill_between(med.index,
                         (med - std).clip(lower=0), med + std,
                         alpha=0.15, color=cor, label="±1 DP")

        # Sombrear períodos chuvosos
        for yr in med.index.year.unique():
            ax.axvspan(
                pd.Timestamp(f"{yr}-04-01"),
                pd.Timestamp(f"{yr}-09-30"),
                alpha=0.07, color="#3498db"
            )

        if vmp_val:
            ax.axhline(vmp_val, color="red", linestyle="--", linewidth=1.2,
                       label=vmp_label, zorder=5)

        if col == "ph":
            ax.axhline(VMP["ph_min"], color="red", linestyle="--", linewidth=1.2, label=f"VMP min = {VMP['ph_min']}")
            ax.axhline(VMP["ph_max"], color="darkred", linestyle=":", linewidth=1.2, label=f"VMP max = {VMP['ph_max']}")

        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8))

    # Legenda de sombreamento
    patch_chuva = mpatches.Patch(color="#3498db", alpha=0.15, label="Período chuvoso (Abr–Set)")
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + [patch_chuva],
                   labels=axes[0].get_legend_handles_labels()[1] + ["Período chuvoso (Abr–Set)"],
                   fontsize=8, loc="upper right")

    fig.suptitle(
        "Série Temporal da Qualidade da Água Tratada — ETA São Pedro / Rio Branco\n"
        "CAER · Boa Vista–RR · 2014–2026 · n = 1.460 amostras",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    salvar(fig, "M01_serie_temporal_tratada.png")
    print("  Período chuvoso destacado em azul claro. VMP em vermelho tracejado.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAZONALIDADE — BRUTA vs TRATADA
# ─────────────────────────────────────────────────────────────────────────────
def fig_sazonalidade(bruta, tratada):
    secao("ETAPA 3 — SAZONALIDADE: ÁGUA BRUTA vs TRATADA")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, col, label, unit in zip(
        axes,
        ["turbidez", "cor"],
        ["Turbidez", "Cor Aparente"],
        ["uT", "uH"],
    ):
        dados = []
        labels_box = []
        cores_box  = []

        for ponto, df_p, cor_p in [
            ("Bruta\nChuvoso", bruta[bruta["periodo"].str.startswith("Chuvoso")][col].dropna(), "#f39c12"),
            ("Bruta\nSeco",    bruta[bruta["periodo"].str.startswith("Seco")][col].dropna(),    "#e67e22"),
            ("Tratada\nChuvoso", tratada[tratada["periodo"].str.startswith("Chuvoso")][col].dropna(), "#2ecc71"),
            ("Tratada\nSeco",    tratada[tratada["periodo"].str.startswith("Seco")][col].dropna(),    "#27ae60"),
        ]:
            # Remove outliers extremos (p99) para visualização
            q99 = df_p.quantile(0.99) if len(df_p) > 0 else 1
            dados.append(df_p[df_p <= q99].values)
            labels_box.append(ponto)
            cores_box.append(cor_p)

        bp = ax.boxplot(
            dados,
            labels=labels_box,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        for patch, cor_p in zip(bp["boxes"], cores_box):
            patch.set_facecolor(cor_p)
            patch.set_alpha(0.7)

        ax.set_title(f"{label} ({unit}) — Sazonalidade\nBruta vs Tratada",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(axis="y", alpha=0.3)

        # Linha VMP
        vmp_val = VMP.get(col)
        if vmp_val:
            ax.axhline(vmp_val, color="red", linestyle="--", linewidth=1.5,
                       label=f"VMP = {vmp_val} {unit}")
            ax.legend(fontsize=9)

    fig.suptitle(
        "Sazonalidade Amazônica — Qualidade da Água\n"
        "ETA São Pedro / Rio Branco · CAER · Boa Vista–RR (2014–2026)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    salvar(fig, "M02_sazonalidade_bruta_vs_tratada.png")

    # Estatísticas textuais
    for col, label in [("turbidez", "Turbidez"), ("cor", "Cor")]:
        print(f"\n  {label} — Mediana por período:")
        print(f"    Bruta  Chuvoso : {bruta[bruta['periodo'].str.startswith('Chuvoso')][col].median():.1f}")
        print(f"    Bruta  Seco    : {bruta[bruta['periodo'].str.startswith('Seco')][col].median():.1f}")
        print(f"    Tratada Chuvoso: {tratada[tratada['periodo'].str.startswith('Chuvoso')][col].median():.1f}")
        print(f"    Tratada Seco   : {tratada[tratada['periodo'].str.startswith('Seco')][col].median():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFORMIDADE VMP — PORTARIA 888/2021
# ─────────────────────────────────────────────────────────────────────────────
def fig_conformidade(tratada, rede):
    secao("ETAPA 4 — ANÁLISE DE CONFORMIDADE (VMP Portaria 888/2021)")

    resultados = {}

    for ponto_nome, df_p in [("Saída Tratamento", tratada), ("Rede Distribuição", rede)]:
        n_turb  = df_p["turbidez"].notna().sum()
        n_cor   = df_p["cor"].notna().sum()
        n_ph    = df_p["ph"].notna().sum()

        conf_turb = (df_p["turbidez"] <= VMP["turbidez"]).sum() / n_turb * 100 if n_turb else 0
        conf_cor  = (df_p["cor"] <= VMP["cor"]).sum() / n_cor * 100 if n_cor else 0
        conf_ph   = (
            (df_p["ph"] >= VMP["ph_min"]) & (df_p["ph"] <= VMP["ph_max"])
        ).sum() / n_ph * 100 if n_ph else 0

        resultados[ponto_nome] = {
            "Turbidez ≤ 5 uT": conf_turb,
            "Cor ≤ 15 uH":     conf_cor,
            "pH 6,0–9,5":      conf_ph,
        }
        print(f"\n  {ponto_nome}:")
        print(f"    Turbidez ≤ 5 uT : {conf_turb:.1f}%")
        print(f"    Cor ≤ 15 uH     : {conf_cor:.1f}%")
        print(f"    pH 6,0–9,5      : {conf_ph:.1f}%")

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cores_bar = ["#27ae60", "#2980b9", "#8e44ad"]

    for ax, (ponto_nome, dados) in zip(axes, resultados.items()):
        parametros = list(dados.keys())
        valores    = list(dados.values())

        bars = ax.barh(parametros, valores,
                       color=cores_bar, edgecolor="white", linewidth=0.8, height=0.5)

        for bar, val in zip(bars, valores):
            ax.text(val - 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="right",
                    fontsize=12, fontweight="bold", color="white")

        ax.axvline(100, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
        ax.axvline(95, color="orange", linewidth=1.0, linestyle=":", alpha=0.7, label="Referência 95%")
        ax.set_xlim(50, 102)
        ax.set_xlabel("Conformidade (%)", fontsize=11)
        ax.set_title(f"{ponto_nome}\nConformidade com VMP (Portaria 888/2021)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Análise de Conformidade da Água — ETA São Pedro / Rio Branco\n"
        "CAER · Boa Vista–RR · 2014–2026",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    salvar(fig, "M03_conformidade_vmp.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EFICIÊNCIA DO TRATAMENTO (bruta → tratada)
# ─────────────────────────────────────────────────────────────────────────────
def fig_eficiencia(bruta, tratada):
    secao("ETAPA 5 — EFICIÊNCIA DO TRATAMENTO")

    stats_b = bruta[["turbidez", "cor"]].describe()
    stats_t = tratada[["turbidez", "cor"]].describe()

    for col, label, unit in [("turbidez", "Turbidez", "uT"), ("cor", "Cor", "uH")]:
        med_b = stats_b.loc["50%", col]
        med_t = stats_t.loc["50%", col]
        red   = (1 - med_t / med_b) * 100 if med_b > 0 else 0
        print(f"\n  {label}:")
        print(f"    Bruta   mediana: {med_b:.2f} {unit}")
        print(f"    Tratada mediana: {med_t:.2f} {unit}")
        print(f"    Redução         : {red:.1f}%")

    # Figura: distribuição bruta vs tratada
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, label, unit in zip(
        axes,
        ["turbidez", "cor"],
        ["Turbidez", "Cor Aparente"],
        ["uT", "uH"],
    ):
        dados_b = bruta[col].dropna()
        dados_t = tratada[col].dropna()

        # Clip p99 para visualização
        hi = max(dados_b.quantile(0.99), dados_t.quantile(0.99))
        dados_b = dados_b[dados_b <= hi]
        dados_t = dados_t[dados_t <= hi]

        bins = np.linspace(0, hi, 40)
        ax.hist(dados_b.values, bins=bins, alpha=0.65, color=CORES["bruta"],
                label=f"Bruta  (n={len(bruta):,})", edgecolor="white", linewidth=0.4)
        ax.hist(dados_t.values, bins=bins, alpha=0.65, color=CORES["tratada"],
                label=f"Tratada (n={len(tratada):,})", edgecolor="white", linewidth=0.4)

        vmp_val = VMP.get(col)
        if vmp_val:
            ax.axvline(vmp_val, color="red", linestyle="--", linewidth=1.5,
                       label=f"VMP = {vmp_val} {unit}")

        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("Frequência")
        ax.set_title(f"Distribuição — {label}\nÁgua Bruta vs Tratada", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Eficiência do Tratamento — ETA São Pedro / Rio Branco\n"
        "CAER · Boa Vista–RR · 2014–2026",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    salvar(fig, "M04_eficiencia_tratamento.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. VARIAÇÃO ANUAL — BOXPLOT POR ANO
# ─────────────────────────────────────────────────────────────────────────────
def fig_anual(tratada):
    secao("ETAPA 6 — VARIAÇÃO ANUAL DA QUALIDADE (ÁGUA TRATADA)")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax, col, label, unit, cor in zip(
        axes,
        ["turbidez", "cor"],
        ["Turbidez", "Cor Aparente"],
        ["uT", "uH"],
        [CORES["tratada"], "#8e44ad"],
    ):
        anos    = sorted(tratada["ano"].dropna().unique().astype(int))
        dados   = [tratada[tratada["ano"] == a][col].dropna().values for a in anos]
        dados   = [d for d in dados if len(d) > 0]
        anos_ok = [a for a, d in zip(anos, dados) if len(d) > 0]

        bp = ax.boxplot(
            dados,
            labels=anos_ok,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=2, alpha=0.3, color=cor),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(cor)
            patch.set_alpha(0.6)

        vmp_val = VMP.get(col)
        if vmp_val:
            ax.axhline(vmp_val, color="red", linestyle="--", linewidth=1.5,
                       label=f"VMP = {vmp_val} {unit}", zorder=5)

        ax.set_ylabel(f"{label} ({unit})", fontsize=11)
        ax.set_title(f"{label} — Variação Anual · Saída do Tratamento",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Ano", fontsize=11)
    fig.suptitle(
        "Tendência Anual da Qualidade da Água Tratada — ETA São Pedro\n"
        "CAER · Boa Vista–RR · 2014–2026",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    salvar(fig, "M05_variacao_anual.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DSS — MÓDULO 1: MONITORAMENTO HISTÓRICO DA QUALIDADE DA ÁGUA")
    print("  ETA São Pedro / Rio Branco / Boa Vista — CAER/RR (2014–2026)")
    print("=" * 70)

    eta, bruta, tratada, rede = carregar()
    fig_serie_temporal(tratada)
    fig_sazonalidade(bruta, tratada)
    fig_conformidade(tratada, rede)
    fig_eficiencia(bruta, tratada)
    fig_anual(tratada)

    print()
    print("=" * 70)
    print("  MÓDULO 1 CONCLUÍDO — 5 figuras geradas em dss/Figuras/")
    print("=" * 70)
