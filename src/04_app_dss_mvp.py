"""
=============================================================================
DSS — Sistema de Suporte à Decisão para Dosagem de Coagulantes
ETA São Pedro / CAER — Boa Vista, Roraima

TCC: Machine Learning para Dosagem de Coagulantes em ETA:
     Diagnóstico de Digitalização na CAER/RR
Autor     : Ibukun Chife Didier Adjitche
Orientador: Prof. Leandro Nelinho Balico
UFRR — Especialização em Computação Aplicada na Indústria 4.0
=============================================================================
Executar (da raiz do projeto):
    .venv/bin/streamlit run dss/04_app_dss.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────────────────────────────────────
DSS_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(DSS_DIR)
DATA_PATH  = os.path.join(ROOT_DIR, "Dados", "dados-analise-agua-20260302.csv")
FIG_DIR    = os.path.join(DSS_DIR, "Figuras")
MOD_DIR    = os.path.join(ROOT_DIR, "Resultados")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
MESES_CHUVOSOS  = [4, 5, 6, 7, 8, 9]
FEATURES_MODEL  = ["ph", "turbidez", "cor", "mes", "periodo_chuvoso"]
FEATURES_RAW    = ["ph", "turbidez", "cor", "temperatura", "alcalinidade_bicarbonatos"]
TARGET          = "sulfato"
RANDOM_STATE    = 42

LABELS = {
    "ph"              : "pH",
    "turbidez"        : "Turbidez (uT)",
    "cor"             : "Cor Aparente (uH)",
    "mes"             : "Mês de Coleta",
    "periodo_chuvoso" : "Período Chuvoso",
}

VMP = {"turbidez": 5.0, "cor": 15.0, "ph_min": 6.0, "ph_max": 9.5}

CORES = {
    "primary"  : "#1a6fa8",
    "success"  : "#27ae60",
    "warning"  : "#e67e22",
    "danger"   : "#e74c3c",
    "purple"   : "#8e44ad",
    "dark"     : "#2c3e50",
}

MESES_NOMES = {
    1:"Jan", 2:"Fev", 3:"Mar", 4:"Abr", 5:"Mai", 6:"Jun",
    7:"Jul", 8:"Ago", 9:"Set", 10:"Out", 11:"Nov", 12:"Dez"
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSS CAER — Sistema de Suporte à Decisão",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .header-box {
        background: linear-gradient(135deg, #1a6fa8 0%, #0d4f7a 100%);
        color: white; padding: 1.8rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; text-align: center;
    }
    .header-box h1 { color: white; font-size: 1.7rem; margin: 0; }
    .header-box p  { color: #b3d9f5; margin: 0.4rem 0 0; font-size: 0.9rem; }

    .kpi-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-top: 4px solid #1a6fa8; height: 100%;
    }
    .kpi-card .kpi-val  { font-size: 2rem; font-weight: 800; color: #1a6fa8; }
    .kpi-card .kpi-lab  { font-size: 0.8rem; color: #555; margin-top: 0.2rem; }
    .kpi-card.green  { border-top-color: #27ae60; }
    .kpi-card.green .kpi-val  { color: #27ae60; }
    .kpi-card.red    { border-top-color: #e74c3c; }
    .kpi-card.red .kpi-val    { color: #e74c3c; }
    .kpi-card.orange { border-top-color: #e67e22; }
    .kpi-card.orange .kpi-val { color: #e67e22; }

    .pred-box {
        background: linear-gradient(135deg, #1a6fa8, #27ae60);
        color: white; border-radius: 14px; padding: 2rem;
        text-align: center; margin: 1rem 0;
    }
    .pred-box .dose { font-size: 3.5rem; font-weight: 900; }
    .pred-box .unit { font-size: 1.1rem; opacity: 0.9; }

    .alert   { background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:0.8rem; margin:0.5rem 0; }
    .info    { background:#d1ecf1; border:1px solid #17a2b8; border-radius:8px; padding:0.8rem; margin:0.5rem 0; }
    .success { background:#d4edda; border:1px solid #28a745; border-radius:8px; padding:0.8rem; margin:0.5rem 0; }
    .danger  { background:#f8d7da; border:1px solid #dc3545; border-radius:8px; padding:0.8rem; margin:0.5rem 0; }

    .modulo-title {
        font-size: 1.25rem; font-weight: 700; color: #1a6fa8;
        border-left: 5px solid #1a6fa8; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }
    div[data-testid="stSidebar"] { background: #f0f4f8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE — DADOS E MODELOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando dados históricos da CAER...")
def carregar_dados():
    df = pd.read_csv(DATA_PATH, sep=";", on_bad_lines="skip", low_memory=False)
    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce")

    mask = (
        df["nome_cidade"].str.upper().str.contains("BOA VISTA", na=False)
        & df["nome_manancial"].str.upper().str.contains("RIO BRANCO", na=False)
        & df["nome_sistema_abastacimento"].str.upper().str.contains("SÃO PEDRO", na=False)
        & (df["data_coleta"].dt.year >= 2014)
    )
    eta = df[mask].copy()
    for col in FEATURES_RAW + [TARGET, "cloro_residual_livre", "condutividade"]:
        eta[col] = pd.to_numeric(eta[col], errors="coerce")

    eta["mes"]     = eta["data_coleta"].dt.month
    eta["ano"]     = eta["data_coleta"].dt.year
    eta["periodo"] = eta["mes"].isin(MESES_CHUVOSOS).map(
        {True: "Chuvoso (Abr–Set)", False: "Seco (Out–Mar)"}
    )

    bruta   = eta[eta["nome_ponto_amostragem"].str.upper().str.contains("NATURA",  na=False)].copy()
    tratada = eta[eta["nome_ponto_amostragem"].str.upper().str.contains("SAÍDA DO TRATAMENTO", na=False)].copy()
    rede    = eta[eta["nome_ponto_amostragem"].str.upper().str.contains("REDE",    na=False)].copy()

    df_ml = eta.dropna(subset=[TARGET]).copy()
    df_ml["mes"]             = df_ml["data_coleta"].dt.month
    df_ml["periodo_chuvoso"] = df_ml["mes"].isin(MESES_CHUVOSOS).astype(int)

    return eta, bruta, tratada, rede, df_ml


@st.cache_resource(show_spinner="Carregando modelos de Machine Learning...")
def carregar_modelos():
    try:
        rf     = joblib.load(os.path.join(MOD_DIR, "model_rf.pkl"))
        xgb_m  = joblib.load(os.path.join(MOD_DIR, "model_xgb.pkl"))
        ridge  = joblib.load(os.path.join(MOD_DIR, "model_ridge.pkl"))
        config = joblib.load(os.path.join(MOD_DIR, "config.pkl"))
        return rf, xgb_m, ridge, config
    except Exception:
        return None, None, None, {}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 💧 DSS CAER")
        st.markdown("**ETA São Pedro · Rio Branco**")
        st.markdown("---")

        pagina = st.radio(
            "Navegação",
            [
                "🏠  Início",
                "📊  Módulo 1 — Monitoramento",
                "🔍  Módulo 2 — Diagnóstico",
                "🤖  Módulo 3 — Predição + XAI",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<small>**TCC · UFRR · 2026**<br>"
            "Ibukun Chife Didier Adjitche<br>"
            "Orientador: Prof. Leandro Balico<br>"
            "Especialização em Computação<br>"
            "Aplicada na Indústria 4.0</small>",
            unsafe_allow_html=True,
        )
    return pagina


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: INÍCIO
# ─────────────────────────────────────────────────────────────────────────────
def page_inicio():
    st.markdown("""
    <div class="header-box">
        <h1>💧 Sistema de Suporte à Decisão — Dosagem de Coagulantes</h1>
        <p>ETA São Pedro · CAER · Boa Vista – Roraima · 2014–2026</p>
    </div>
    """, unsafe_allow_html=True)

    eta, bruta, tratada, rede, df_ml = carregar_dados()

    # KPIs principais
    n_turb_ok  = (tratada["turbidez"] <= VMP["turbidez"]).sum()
    n_turb_tot = tratada["turbidez"].notna().sum()
    conf_turb  = 100 * n_turb_ok / n_turb_tot if n_turb_tot else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-val">{len(eta):,}</div>
            <div class="kpi-lab">Registros históricos<br>LIMS-CAER (2014–2026)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card green">
            <div class="kpi-val">{conf_turb:.1f}%</div>
            <div class="kpi-lab">Conformidade turbidez<br>Portaria GM/MS 888/2021</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card red">
            <div class="kpi-val">0</div>
            <div class="kpi-lab">Registros digitais<br>de dosagem operacional</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card orange">
            <div class="kpi-val">{len(df_ml)}</div>
            <div class="kpi-lab">Amostras com sulfato<br>registrado no LIMS</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Descrição dos módulos
    st.markdown('<div class="modulo-title">Arquitetura do Sistema DSS</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        **📊 Módulo 1 — Monitoramento**

        Analisa os **6.148 registros históricos** do LIMS-CAER, produzindo:
        - Série temporal de qualidade (2014–2026)
        - Sazonalidade amazônica (cheia vs. seca)
        - Conformidade com a Portaria 888/2021
        - Eficiência do tratamento (bruta → tratada)
        """)
    with m2:
        st.markdown("""
        **🔍 Módulo 2 — Diagnóstico**

        Diagnóstico quantitativo da **lacuna de digitalização**:
        - Análise dos 47 pares bruta × tratada
        - Correlação r ≈ 0: evidência do controle manual ativo
        - Mapeamento do que o LIMS registra vs. o que falta
        - Tabela-resumo do diagnóstico operacional
        """)
    with m3:
        st.markdown("""
        **🤖 Módulo 3 — Predição + XAI**

        Framework preditivo com **ML + SHAP**:
        - Ridge Regression (baseline) · Random Forest · XGBoost
        - Validação LOOCV (n = 10 amostras)
        - Explicabilidade SHAP por predição
        - Análise de impacto econômico potencial
        """)

    st.markdown("---")
    st.markdown("""
    <div class="info">
    ℹ️ <strong>Sobre o sistema:</strong> O DSS está plenamente funcional nos Módulos 1 e 2.
    O Módulo 3 opera com os dados disponíveis (n = 10 amostras de sulfato registradas no LIMS).
    Quando a CAER digitalizar os registros de dosagem operacional, o pipeline será re-treinado e
    o sistema atingirá capacidade preditiva completa.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: MÓDULO 1 — MONITORAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def page_monitoramento():
    st.markdown('<div class="modulo-title">📊 Módulo 1 — Monitoramento Histórico da Qualidade da Água</div>', unsafe_allow_html=True)
    st.markdown("**ETA São Pedro · Rio Branco · Boa Vista–RR · 2014–2026 · n = 6.148 registros**")

    eta, bruta, tratada, rede, _ = carregar_dados()

    # KPIs do módulo
    med_turb_b = bruta["turbidez"].median()
    med_turb_t = tratada["turbidez"].median()
    red_turb   = (1 - med_turb_t / med_turb_b) * 100
    conf_turb  = 100 * (tratada["turbidez"] <= VMP["turbidez"]).sum() / tratada["turbidez"].notna().sum()
    conf_cor   = 100 * (tratada["cor"] <= VMP["cor"]).sum() / tratada["cor"].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{med_turb_b:.1f} uT", "Turbidez bruta (mediana)", "orange"),
        (c2, f"{med_turb_t:.2f} uT", "Turbidez tratada (mediana)", "green"),
        (c3, f"{red_turb:.1f}%",     "Redução de turbidez", "green"),
        (c4, f"{conf_turb:.1f}%",    "Conformidade turbidez (VMP 5 uT)", "green"),
    ]
    for col, val, lab, cor in cards:
        with col:
            st.markdown(f"""<div class="kpi-card {cor}">
                <div class="kpi-val">{val}</div>
                <div class="kpi-lab">{lab}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Série Temporal", "🌧️ Sazonalidade", "✅ Conformidade", "⚗️ Eficiência"])

    # ── TAB 1: Série temporal ────────────────────────────────────────────────
    with tab1:
        st.markdown("##### Turbidez e Cor Aparente — Saída do Tratamento (mediana mensal)")

        df_ts = tratada[["data_coleta", "turbidez", "cor"]].copy()
        df_ts = df_ts.sort_values("data_coleta").set_index("data_coleta")
        monthly = df_ts.resample("ME").median().reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Turbidez (uT)", "Cor Aparente (uH)"),
                            vertical_spacing=0.1)

        for yr in monthly["data_coleta"].dt.year.unique():
            for row in [1, 2]:
                fig.add_vrect(
                    x0=f"{yr}-04-01", x1=f"{yr}-09-30",
                    fillcolor="lightblue", opacity=0.15, line_width=0,
                    row=row, col=1
                )

        fig.add_trace(go.Scatter(x=monthly["data_coleta"], y=monthly["turbidez"],
                                 mode="lines", name="Turbidez", line=dict(color="#27ae60", width=1.8)),
                      row=1, col=1)
        fig.add_hline(y=VMP["turbidez"], line_dash="dash", line_color="red",
                      annotation_text="VMP = 5 uT", row=1, col=1)

        fig.add_trace(go.Scatter(x=monthly["data_coleta"], y=monthly["cor"],
                                 mode="lines", name="Cor", line=dict(color="#8e44ad", width=1.8)),
                      row=2, col=1)
        fig.add_hline(y=VMP["cor"], line_dash="dash", line_color="red",
                      annotation_text="VMP = 15 uH", row=2, col=1)

        fig.update_layout(height=480, showlegend=False,
                          title_text="Qualidade da Água Tratada — ETA São Pedro (sombreado = período chuvoso)")
        st.plotly_chart(fig, use_container_width=True)

        fpath = os.path.join(FIG_DIR, "M01_serie_temporal_tratada.png")
        if os.path.exists(fpath):
            with st.expander("Ver figura completa (pH + Cloro Residual incluídos)"):
                st.image(fpath, use_container_width=True)

    # ── TAB 2: Sazonalidade ──────────────────────────────────────────────────
    with tab2:
        st.markdown("##### Variação Sazonal — Chuvoso (Abr–Set) vs. Seco (Out–Mar)")

        col_a, col_b = st.columns(2)

        for col_st, feat, label, unit, color_c, color_s in [
            (col_a, "turbidez", "Turbidez", "uT", "#3498db", "#e74c3c"),
            (col_b, "cor",      "Cor Aparente", "uH", "#2980b9", "#c0392b"),
        ]:
            with col_st:
                dados_b_c = bruta[bruta["periodo"].str.startswith("Chuvoso")][feat].dropna()
                dados_b_s = bruta[bruta["periodo"].str.startswith("Seco")][feat].dropna()
                dados_t_c = tratada[tratada["periodo"].str.startswith("Chuvoso")][feat].dropna()
                dados_t_s = tratada[tratada["periodo"].str.startswith("Seco")][feat].dropna()

                fig = go.Figure()
                for nome, dados, cor in [
                    ("Bruta · Chuvoso", dados_b_c, color_c),
                    ("Bruta · Seco",    dados_b_s, color_s),
                    ("Tratada · Chuvoso", dados_t_c, "#2ecc71"),
                    ("Tratada · Seco",    dados_t_s, "#27ae60"),
                ]:
                    q99 = dados.quantile(0.99)
                    fig.add_trace(go.Box(y=dados[dados <= q99], name=nome,
                                        marker_color=cor, boxpoints=False))

                vmp_val = VMP.get(feat)
                if vmp_val:
                    fig.add_hline(y=vmp_val, line_dash="dash", line_color="red",
                                  annotation_text=f"VMP = {vmp_val} {unit}")

                fig.update_layout(
                    title=f"{label} ({unit}) — Sazonalidade",
                    height=380, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info">
        🌧️ <strong>Sazonalidade amazônica:</strong> No período chuvoso, a turbidez mediana da água bruta
        aumenta ~12% (21 → 23,5 uT), com picos de até <strong>101 uT</strong>. A ETA mantém a água tratada
        dentro do VMP em ambos os períodos, evidenciando a eficiência do controle manual dos operadores.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: Conformidade ───────────────────────────────────────────────────
    with tab3:
        st.markdown("##### Conformidade com a Portaria GM/MS nº 888/2021")

        resultados = {}
        for nome_p, df_p in [("Saída do Tratamento", tratada), ("Rede de Distribuição", rede)]:
            n_turb = df_p["turbidez"].notna().sum()
            n_cor  = df_p["cor"].notna().sum()
            n_ph   = df_p["ph"].notna().sum()
            resultados[nome_p] = {
                "Turbidez ≤ 5 uT":    100 * (df_p["turbidez"] <= 5).sum() / n_turb  if n_turb else 0,
                "Cor ≤ 15 uH":         100 * (df_p["cor"]      <= 15).sum() / n_cor   if n_cor  else 0,
                "pH entre 6,0 e 9,5":  100 * ((df_p["ph"] >= 6) & (df_p["ph"] <= 9.5)).sum() / n_ph if n_ph else 0,
            }

        col_c, col_d = st.columns(2)
        for col_st, (ponto, dados) in zip([col_c, col_d], resultados.items()):
            with col_st:
                fig = go.Figure(go.Bar(
                    x=list(dados.values()),
                    y=list(dados.keys()),
                    orientation="h",
                    marker_color=["#27ae60" if v >= 95 else "#e74c3c" for v in dados.values()],
                    text=[f"{v:.1f}%" for v in dados.values()],
                    textposition="inside",
                    textfont=dict(color="white", size=13),
                ))
                fig.add_vline(x=95, line_dash="dot", line_color="orange",
                              annotation_text="95% ref.")
                fig.update_layout(
                    title=ponto, xaxis_range=[50, 102],
                    xaxis_title="Conformidade (%)", height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="success">
        ✅ <strong>Resultado:</strong> A ETA São Pedro opera com <strong>98,6% de conformidade</strong>
        para turbidez e <strong>99,0%</strong> para cor aparente na saída do tratamento — desempenho
        notável considerando a alta variabilidade sazonal da água bruta do Rio Branco.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 4: Eficiência ────────────────────────────────────────────────────
    with tab4:
        st.markdown("##### Eficiência do Tratamento — Distribuição Bruta vs. Tratada")

        col_e, col_f = st.columns(2)
        for col_st, feat, label, unit in [
            (col_e, "turbidez", "Turbidez", "uT"),
            (col_f, "cor",      "Cor Aparente", "uH"),
        ]:
            with col_st:
                dados_b = bruta[feat].dropna()
                dados_t = tratada[feat].dropna()
                hi = max(dados_b.quantile(0.99), dados_t.quantile(0.99))

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=dados_b[dados_b <= hi], nbinsx=30,
                    name=f"Bruta (n={len(bruta):,})",
                    marker_color="#e67e22", opacity=0.65
                ))
                fig.add_trace(go.Histogram(
                    x=dados_t[dados_t <= hi], nbinsx=30,
                    name=f"Tratada (n={len(tratada):,})",
                    marker_color="#27ae60", opacity=0.65
                ))
                vmp_val = VMP.get(feat)
                if vmp_val:
                    fig.add_vline(x=vmp_val, line_dash="dash", line_color="red",
                                  annotation_text=f"VMP = {vmp_val}")
                fig.update_layout(
                    barmode="overlay",
                    title=f"{label} ({unit})", height=340,
                    xaxis_title=f"{label} ({unit})", yaxis_title="Frequência",
                )
                st.plotly_chart(fig, use_container_width=True)

        med_b = bruta["turbidez"].median()
        med_t = tratada["turbidez"].median()
        red   = (1 - med_t / med_b) * 100
        st.metric("Redução mediana de turbidez (bruta → tratada)", f"{red:.1f}%",
                  f"{med_b:.1f} uT → {med_t:.2f} uT")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: MÓDULO 2 — DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────────────
def page_diagnostico():
    st.markdown('<div class="modulo-title">🔍 Módulo 2 — Diagnóstico da Lacuna de Digitalização</div>', unsafe_allow_html=True)
    st.markdown("**Evidência quantitativa: por que a CAER ainda não pode usar ML para predizer dosagem**")

    eta, bruta, tratada, rede, df_ml = carregar_dados()

    # Construir pares
    bruta_d   = bruta.copy()
    tratada_d = tratada.copy()
    bruta_d["data_dia"]   = bruta_d["data_coleta"].dt.date
    tratada_d["data_dia"] = tratada_d["data_coleta"].dt.date

    bruta_ag   = bruta_d.groupby("data_dia")[["turbidez", "cor", "ph"]].median().reset_index()
    tratada_ag = tratada_d.groupby("data_dia")[["turbidez", "cor", "ph"]].median().reset_index()
    pares      = pd.merge(bruta_ag, tratada_ag, on="data_dia", suffixes=("_bruta", "_tratada")).dropna()
    pares["data_dia"] = pd.to_datetime(pares["data_dia"])
    pares["periodo"]  = pares["data_dia"].dt.month.isin(MESES_CHUVOSOS).map(
        {True: "Chuvoso (Abr–Set)", False: "Seco (Out–Mar)"}
    )

    # KPIs do módulo
    r_turb, p_turb = stats.pearsonr(pares["turbidez_bruta"], pares["turbidez_tratada"])
    r_cor,  p_cor  = stats.pearsonr(pares["cor_bruta"],      pares["cor_tratada"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-val">{len(pares)}</div>
            <div class="kpi-lab">Pares bruta × tratada<br>(mesma data)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card orange">
            <div class="kpi-val">r = {r_turb:+.3f}</div>
            <div class="kpi-lab">Correlação turbidez<br>bruta × tratada (p={p_turb:.2f})</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card orange">
            <div class="kpi-val">r = {r_cor:+.3f}</div>
            <div class="kpi-lab">Correlação cor aparente<br>bruta × tratada (p={p_cor:.2f})</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card red">
            <div class="kpi-val">0</div>
            <div class="kpi-lab">Registros digitais de<br>dosagem operacional</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📉 Scatter Pares", "🗺️ Mapa da Lacuna", "📋 Tabela-Resumo"])

    # ── TAB 1: Scatter pares ─────────────────────────────────────────────────
    with tab1:
        st.markdown("##### Dispersão dos Pares Bruta × Tratada — A Evidência do r ≈ 0")

        col_a, col_b = st.columns(2)
        for col_st, xc, yc, label, unit in [
            (col_a, "turbidez_bruta", "turbidez_tratada", "Turbidez", "uT"),
            (col_b, "cor_bruta",      "cor_tratada",      "Cor Aparente", "uH"),
        ]:
            with col_st:
                r_val, p_val = stats.pearsonr(pares[xc], pares[yc])
                slope, intercept, _, _, _ = stats.linregress(pares[xc], pares[yc])
                x_line = np.linspace(pares[xc].min(), pares[xc].max(), 100)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pares[xc], y=pares[yc],
                    mode="markers",
                    marker=dict(
                        color=["#3498db" if "Chuvoso" in p else "#e74c3c" for p in pares["periodo"]],
                        size=9, line=dict(color="white", width=0.8)
                    ),
                    customdata=pares["periodo"],
                    hovertemplate=f"Bruta: %{{x:.1f}} {unit}<br>Tratada: %{{y:.2f}} {unit}<br>%{{customdata}}<extra></extra>",
                    name="Pares"
                ))
                fig.add_trace(go.Scatter(
                    x=x_line, y=slope * x_line + intercept,
                    mode="lines", line=dict(color="gray", dash="dash", width=1.5),
                    name=f"Regressão (r={r_val:+.3f})"
                ))
                lim = max(pares[xc].max(), pares[yc].max()) * 1.05
                fig.add_trace(go.Scatter(
                    x=[0, lim], y=[0, lim],
                    mode="lines", line=dict(color="black", dash="dot", width=1),
                    name="Identidade (r=1)", opacity=0.4
                ))
                fig.update_layout(
                    title=f"{label} ({unit})<br><b>r = {r_val:+.3f} · p = {p_val:.3f}</b>",
                    xaxis_title=f"{label} BRUTA ({unit})",
                    yaxis_title=f"{label} TRATADA ({unit})",
                    height=380,
                    legend=dict(orientation="h", yanchor="top", y=-0.15)
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="alert">
        ⚠️ <strong>Por que r ≈ 0 é evidência científica, não falha?</strong><br>
        Se os operadores dosassem coagulante em proporção fixa à qualidade da água bruta,
        esperaríamos r próximo de 1. A correlação nula prova que eles compensam ativamente
        as variações — ajustando a dosagem de forma a entregar sempre água de qualidade constante.
        O problema: esse ajuste é feito manualmente e <strong>não tem registro digital</strong>.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Mapa da lacuna ────────────────────────────────────────────────
    with tab2:
        st.markdown("##### O que o LIMS registra · O que está ausente · O que o DSS propõe")

        fpath = os.path.join(FIG_DIR, "D02_lacuna_digitalizacao.png")
        if os.path.exists(fpath):
            st.image(fpath, use_container_width=True)
        else:
            st.info("Execute `python dss/02_diagnostico.py` para gerar esta figura.")

        st.markdown("""
        <div class="danger">
        🚨 <strong>Lacuna identificada:</strong> A CAER possui 6.148 registros de qualidade de água
        (parâmetros físico-químicos), mas <strong>zero registros digitais da dosagem operacional</strong>
        de coagulante — a variável de controle central do processo. Esta é a barreira à Indústria 4.0.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: Tabela-resumo ──────────────────────────────────────────────────
    with tab3:
        st.markdown("##### Tabela-Resumo do Diagnóstico — ETA São Pedro / CAER")

        n_turb_ok = (tratada["turbidez"] <= VMP["turbidez"]).sum()
        n_turb_tt = tratada["turbidez"].notna().sum()
        n_cor_ok  = (tratada["cor"] <= VMP["cor"]).sum()
        n_cor_tt  = tratada["cor"].notna().sum()

        dados_tab = {
            "Indicador": [
                "Registros LIMS — ETA São Pedro (2014–2026)",
                "  → Água in natura (bruta)",
                "  → Saída do tratamento",
                "  → Rede de distribuição",
                "Pares bruta + tratada na mesma data",
                "Correlação turbidez bruta × tratada (r)",
                "Correlação cor bruta × tratada (r)",
                "Registros digitais de dosagem operacional",
                f"Conformidade turbidez (≤ {VMP['turbidez']} uT) — saída",
                f"Conformidade cor (≤ {VMP['cor']} uH) — saída",
            ],
            "Valor": [
                f"{len(eta):,}",
                f"{len(bruta):,}",
                f"{len(tratada):,}",
                f"{len(rede):,}",
                f"{len(pares)}",
                f"{r_turb:+.3f}",
                f"{r_cor:+.3f}",
                "0",
                f"{100*n_turb_ok/n_turb_tt:.1f}%",
                f"{100*n_cor_ok/n_cor_tt:.1f}%",
            ],
            "Status": [
                "✅ Disponível", "✅ Disponível", "✅ Disponível", "✅ Disponível",
                "✅ Disponível",
                "≈ 0 (esperado)", "≈ 0 (esperado)",
                "❌ AUSENTE",
                "✅ Excelente", "✅ Excelente",
            ],
        }
        df_tab = pd.DataFrame(dados_tab)

        def colorir(row):
            if "❌" in str(row["Status"]):
                return ["background-color: #fadbd8"] * 3
            elif "✅" in str(row["Status"]):
                return ["background-color: #d5f5e3"] * 3
            elif "≈ 0" in str(row["Status"]):
                return ["background-color: #fef9e7"] * 3
            return [""] * 3

        st.dataframe(
            df_tab.style.apply(colorir, axis=1),
            use_container_width=True, hide_index=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: MÓDULO 3 — PREDIÇÃO + XAI
# ─────────────────────────────────────────────────────────────────────────────
def page_predicao():
    st.markdown('<div class="modulo-title">🤖 Módulo 3 — Framework Preditivo com IA Explicável (SHAP)</div>', unsafe_allow_html=True)

    rf, xgb_m, ridge, config = carregar_modelos()
    eta, bruta, tratada, rede, df_ml = carregar_dados()

    if rf is None:
        st.error("Modelos não encontrados. Execute `python ml_pipeline.py` para treinar e exportar.")
        return

    tab1, tab2, tab3 = st.tabs(["🎯 Predição Interativa", "📊 Desempenho dos Modelos", "💰 Impacto Econômico"])

    # ── TAB 1: Predição interativa ────────────────────────────────────────────
    with tab1:
        st.markdown("##### Simule uma Predição — Insira os Parâmetros da Água Bruta")

        col_inp, col_res = st.columns([1, 1])

        with col_inp:
            mes_atual = datetime.date.today().month
            periodo_atual = 1 if mes_atual in MESES_CHUVOSOS else 0

            ph_val      = st.slider("pH",                  4.0,  9.5, 7.0,  0.1)
            turb_val    = st.slider("Turbidez (uT)",        0.1, 120.0, 20.0, 0.5)
            cor_val     = st.slider("Cor Aparente (uH)",    1.0, 160.0, 55.0, 1.0)
            mes_val     = st.selectbox("Mês de Coleta",
                                       options=list(range(1, 13)),
                                       format_func=lambda m: f"{m} — {MESES_NOMES[m]}",
                                       index=mes_atual - 1)
            chuv_val    = st.radio("Período",
                                   [0, 1],
                                   format_func=lambda v: "🌧️ Chuvoso (Abr–Set)" if v == 1 else "☀️ Seco (Out–Mar)",
                                   index=periodo_atual,
                                   horizontal=True)

            X_input = pd.DataFrame([[ph_val, turb_val, cor_val, mes_val, chuv_val]],
                                   columns=FEATURES_MODEL)

        with col_res:
            # Predições dos 3 modelos
            pred_rf    = float(rf.predict(X_input)[0])
            pred_xgb   = float(xgb_m.predict(X_input)[0])
            pred_ridge = float(ridge.predict(X_input)[0])

            y_min = config.get("y_min", 1.0)
            y_max = config.get("y_max", 10.0)
            pred_rf_clip    = np.clip(pred_rf,    y_min, y_max)
            pred_xgb_clip   = np.clip(pred_xgb,   y_min, y_max)
            pred_ridge_clip = np.clip(pred_ridge,  y_min, y_max)

            st.markdown(f"""
            <div class="pred-box">
                <div style="font-size:0.9rem; opacity:0.85; margin-bottom:0.3rem">
                    ⭐ Random Forest (melhor modelo)
                </div>
                <div class="dose">{pred_rf_clip:.2f}</div>
                <div class="unit">mg/L de sulfato de alumínio</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Comparativo entre modelos:**")
            df_preds = pd.DataFrame({
                "Modelo": ["Ridge Regression", "Random Forest", "XGBoost"],
                "Predição (mg/L)": [f"{pred_ridge_clip:.2f}", f"{pred_rf_clip:.2f}", f"{pred_xgb_clip:.2f}"],
            })
            st.dataframe(df_preds, use_container_width=True, hide_index=True)

            # SHAP para Random Forest
            st.markdown("**Explicação SHAP — Por que o modelo chegou neste valor:**")
            try:
                X_train   = df_ml[FEATURES_MODEL]
                explainer = shap.TreeExplainer(rf)
                sv        = explainer(X_input)
                vals      = sv.values[0]
                feats     = [LABELS.get(f, f) for f in FEATURES_MODEL]

                fig_shap = go.Figure(go.Bar(
                    x=vals,
                    y=feats,
                    orientation="h",
                    marker_color=["#e74c3c" if v > 0 else "#3498db" for v in vals],
                    text=[f"{v:+.3f}" for v in vals],
                    textposition="outside",
                ))
                fig_shap.add_vline(x=0, line_color="black", line_width=1)
                fig_shap.update_layout(
                    title=f"SHAP — contribuição de cada variável<br>(base = {config.get('y_mean', 7.6):.1f} mg/L)",
                    xaxis_title="Impacto na predição (mg/L)",
                    height=300,
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP não disponível nesta sessão: {e}")

            st.markdown("""
            <div class="alert">
            ⚠️ <b>Nota metodológica:</b> O modelo foi treinado com n = 10 amostras.
            Os valores são indicativos. A digitalização dos dados operacionais da CAER
            permitirá re-treinamento com n ≥ 100 e desempenho preditivo robusto.
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Desempenho dos modelos ─────────────────────────────────────────
    with tab2:
        n_ml = len(df_ml)
        st.markdown(f"##### Métricas de Desempenho — LOOCV · n = {n_ml} amostras (LIMS 2016–2017)")

        # Calcular métricas dinamicamente via LOOCV nos modelos carregados
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        X_ml = df_ml[FEATURES_MODEL].copy()
        y_ml = df_ml[TARGET].copy()
        loo  = LeaveOneOut()

        @st.cache_data(show_spinner="Calculando métricas LOOCV...")
        def calcular_metricas_loocv(n):
            preds_ridge = cross_val_predict(ridge,  X_ml, y_ml, cv=loo)
            preds_rf    = cross_val_predict(rf,     X_ml, y_ml, cv=loo)
            preds_xgb   = cross_val_predict(xgb_m, X_ml, y_ml, cv=loo)
            rows = []
            for nome, preds, ind in [
                ("Ridge Regression", preds_ridge, "Baseline linear"),
                ("Random Forest",    preds_rf,    "⭐ Melhor modelo"),
                ("XGBoost",          preds_xgb,   "Boa regularização"),
            ]:
                r2   = r2_score(y_ml, preds)
                mae  = mean_absolute_error(y_ml, preds)
                rmse = np.sqrt(mean_squared_error(y_ml, preds))
                rows.append({
                    "Modelo":      nome,
                    "R²":          f"{r2:.2f}",
                    "MAE (mg/L)":  f"{mae:.2f}",
                    "RMSE (mg/L)": f"{rmse:.2f}",
                    "Indicação":   ind,
                })
            return pd.DataFrame(rows)

        df_metricas = calcular_metricas_loocv(n_ml)
        st.dataframe(df_metricas, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info">
        ℹ️ <strong>Sobre o R² negativo:</strong> Um R² negativo indica que o modelo performa
        pior que simplesmente prever a média — resultado esperado com apenas n = 10 amostras
        e alta dispersão (dosagens de 1 a 10 mg/L). Este não é um sinal de falha do algoritmo,
        mas a confirmação estatística de que a amostra é insuficiente para generalização.
        Com n ≥ 100 amostras de dosagem operacional, o pipeline está pronto para re-treinamento.
        </div>
        """, unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fpath = os.path.join(ROOT_DIR, "Resultados", "05_comparativo_metricas.png")
            if os.path.exists(fpath):
                st.image(fpath, caption="Comparativo de métricas — Ridge vs RF vs XGBoost",
                         use_container_width=True)
        with col_d:
            fpath = os.path.join(ROOT_DIR, "Resultados", "04_real_vs_predito.png")
            if os.path.exists(fpath):
                st.image(fpath, caption="Real vs. Predito — LOOCV",
                         use_container_width=True)

        st.markdown("##### Análise SHAP — Importância Global das Variáveis")
        col_e, col_f = st.columns(2)
        with col_e:
            fpath = os.path.join(ROOT_DIR, "Resultados", "07_shap_summary_rf.png")
            if os.path.exists(fpath):
                st.image(fpath, caption="SHAP Summary — Random Forest", use_container_width=True)
        with col_f:
            fpath = os.path.join(ROOT_DIR, "Resultados", "08_shap_importance_rf.png")
            if os.path.exists(fpath):
                st.image(fpath, caption="SHAP Importance — Random Forest", use_container_width=True)

    # ── TAB 3: Impacto econômico ──────────────────────────────────────────────
    with tab3:
        st.markdown("##### Projeção de Impacto Econômico — Digitalização + Otimização de Dosagem")

        st.markdown("""
        Estimativa do ganho econômico potencial quando a CAER digitalizar os dados de dosagem
        e o DSS for re-treinado com n ≥ 100 amostras.
        """)

        col_p, col_q = st.columns([1, 1])
        with col_p:
            capacidade  = st.number_input("Capacidade da ETA (L/s)", 50, 1000, 150, 10)
            dose_atual  = st.number_input("Dosagem média atual estimada (mg/L)", 1.0, 30.0, 7.6, 0.5)
            preco_ton   = st.number_input("Preço do coagulante (R$/ton)", 500, 5000, 1500, 100)
            reducao_pct = st.slider("Redução estimada com otimização (%)", 5, 30, 12)

        with col_q:
            dias        = 365
            vol_dia     = capacidade * 86400 / 1000             # m³/dia
            dose_kg_dia = dose_atual * vol_dia / 1000           # kg/dia
            dose_ton_ano= dose_kg_dia * dias / 1000             # ton/ano
            custo_atual = dose_ton_ano * preco_ton              # R$/ano

            economia    = custo_atual * reducao_pct / 100
            dose_nova   = dose_atual * (1 - reducao_pct / 100)

            st.metric("Consumo atual estimado (ton/ano)", f"{dose_ton_ano:.1f} ton")
            st.metric("Custo atual estimado (R$/ano)",    f"R$ {custo_atual:,.0f}")
            st.metric("Economia anual com otimização",    f"R$ {economia:,.0f}",
                      delta=f"-{reducao_pct}% de coagulante")
            st.metric("Dosagem otimizada estimada",       f"{dose_nova:.1f} mg/L",
                      delta=f"-{dose_atual - dose_nova:.1f} mg/L")

        fig_eco = go.Figure(go.Bar(
            x=["Cenário Atual\n(Manual)", f"Cenário Otimizado\n(DSS, -{reducao_pct}%)"],
            y=[custo_atual, custo_atual - economia],
            marker_color=["#e74c3c", "#27ae60"],
            text=[f"R$ {custo_atual:,.0f}", f"R$ {custo_atual - economia:,.0f}"],
            textposition="outside",
        ))
        fig_eco.update_layout(
            title=f"Impacto Econômico Anual — ETA {capacidade} L/s",
            yaxis_title="Custo com coagulante (R$/ano)",
            height=340,
        )
        st.plotly_chart(fig_eco, use_container_width=True)

        st.markdown(f"""
        <div class="success">
        💰 <strong>Potencial de economia:</strong> Com redução de {reducao_pct}% no consumo de coagulante
        via dosagem otimizada pelo DSS, a ETA São Pedro poderia economizar aproximadamente
        <strong>R$ {economia:,.0f} por ano</strong> em insumos químicos — sem comprometer
        a qualidade da água entregue à população.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pagina = sidebar()

    if "Início" in pagina:
        page_inicio()
    elif "Módulo 1" in pagina:
        page_monitoramento()
    elif "Módulo 2" in pagina:
        page_diagnostico()
    elif "Módulo 3" in pagina:
        page_predicao()


if __name__ == "__main__":
    main()
