"""
Gráficos explicativos do projeto CAER-DSS-ML
Gera visualizações para o README e documentação
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Arquitetura do DSS — 3 módulos
# ─────────────────────────────────────────────────────────────────────────────
def plot_dss_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    colors = ['#2196F3', '#4CAF50', '#FF9800']
    modules = [
        ('Módulo 1\nMonitoramento\nHistórico', 'src/01_monitoramento.py\n6.148 registros LIMS\n2014–2026', 1.0),
        ('Módulo 2\nDiagnóstico da\nLacuna Digital', 'src/02_diagnostico.py\n47 pares bruta×tratada\nr ≈ 0 (Pearson)', 5.2),
        ('Módulo 3\nInferência\nPreditiva (XAI)', 'src/03_ml_pipeline.py\nRidge + RF + XGBoost\nSHAP + LOOCV', 9.4),
    ]

    for (title, desc, x), color in zip(modules, colors):
        box = FancyBboxPatch((x, 1.2), 3.6, 3.4,
                             boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.15,
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.8, 4.0, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        ax.text(x + 1.8, 2.6, desc, ha='center', va='center',
                fontsize=8.5, color='#333333', linespacing=1.6)

    # Setas entre módulos
    for x_start in [4.6, 8.8]:
        ax.annotate('', xy=(x_start + 0.6, 3.0), xytext=(x_start, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

    # Dados de entrada
    ax.text(2.8, 0.6, 'LIMS-CAER (CSV)', ha='center', fontsize=9,
            color='#555555', style='italic')
    ax.annotate('', xy=(2.8, 1.2), xytext=(2.8, 0.9),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

    # MVP Streamlit
    box_mvp = FancyBboxPatch((10.5, 0.3), 3.0, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#9C27B0', alpha=0.15,
                              edgecolor='#9C27B0', linewidth=1.5)
    ax.add_patch(box_mvp)
    ax.text(12.0, 0.7, 'MVP Streamlit\n(src/04_app_dss_mvp.py)',
            ha='center', va='center', fontsize=8, color='#9C27B0', fontweight='bold')
    ax.annotate('', xy=(11.2, 1.2), xytext=(11.2, 1.1),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))

    ax.set_title('Arquitetura do DSS — CAER/RR (ETA São Pedro)',
                 fontsize=13, fontweight='bold', pad=12, color='#222')

    path = os.path.join(OUT, 'arch_dss_modules.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Salvo: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lacuna de Digitalização — O que o LIMS tem vs. o que falta
# ─────────────────────────────────────────────────────────────────────────────
def plot_digitalization_gap():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    # Lado esquerdo — O que o LIMS registra
    box_l = FancyBboxPatch((0.3, 0.5), 4.8, 4.0,
                            boxstyle="round,pad=0.2",
                            facecolor='#4CAF50', alpha=0.12,
                            edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(box_l)
    ax.text(2.7, 4.1, 'O que o LIMS registra', ha='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    items_l = ['✔  pH', '✔  Turbidez (NTU)', '✔  Cor aparente (uH)',
               '✔  Cloro residual', '✔  Alcalinidade', '✔  6.148 registros (2014–2026)']
    for i, item in enumerate(items_l):
        ax.text(0.7, 3.5 - i * 0.5, item, fontsize=9.5, color='#1B5E20')

    # Lado direito — O que falta
    box_r = FancyBboxPatch((6.9, 0.5), 4.8, 4.0,
                            boxstyle="round,pad=0.2",
                            facecolor='#F44336', alpha=0.12,
                            edgecolor='#F44336', linewidth=2)
    ax.add_patch(box_r)
    ax.text(9.3, 4.1, 'O que NÃO existe digitalmente', ha='center',
            fontsize=11, fontweight='bold', color='#B71C1C')
    items_r = ['✘  Dosagem de sulfato de alumínio', '✘  Dosagem de PAC',
               '✘  Hora/turno do operador', '✘  Resultado do Jar Test',
               '✘  Volume tratado por batelada', '✘  Registros operacionais']
    for i, item in enumerate(items_r):
        ax.text(7.1, 3.5 - i * 0.5, item, fontsize=9.5, color='#B71C1C')

    # Seta central — lacuna
    ax.annotate('', xy=(6.7, 2.5), xytext=(5.3, 2.5),
                arrowprops=dict(arrowstyle='<->', color='#FF6F00', lw=2.5))
    ax.text(6.0, 2.9, 'LACUNA\nDIGITAL', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#FF6F00')

    ax.set_title('Diagnóstico da Lacuna de Digitalização — CAER/RR\n'
                 'Correlação Pearson r ≈ 0 entre água bruta e tratada (n=47 pares)',
                 fontsize=12, fontweight='bold', color='#222', pad=10)

    path = os.path.join(OUT, 'diagnostico_lacuna_digital.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Salvo: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pipeline ML — fluxo completo
# ─────────────────────────────────────────────────────────────────────────────
def plot_ml_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    steps = [
        ('Dados\nLIMS-CAER', '#607D8B', 'n=10\n(sulfato\nregistrado)'),
        ('Features\n(X)', '#2196F3', 'pH, turbidez,\ncor, mês,\nchuvoso'),
        ('LOOCV\nn-1 / 1', '#9C27B0', 'Leave-One-Out\nCross-Validation'),
        ('3 Modelos', '#FF9800', 'Ridge\nRF\nXGBoost'),
        ('SHAP\n(XAI)', '#4CAF50', 'Explicabilidade\npor variável'),
        ('MVP\nStreamlit', '#F44336', 'Predição\ninterativa'),
    ]

    xs = [0.4, 2.5, 4.6, 6.7, 8.8, 10.9]
    for (label, color, sub), x in zip(steps, xs):
        box = FancyBboxPatch((x, 1.2), 1.9, 2.2,
                             boxstyle="round,pad=0.12",
                             facecolor=color, alpha=0.18,
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.95, 2.8, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color)
        ax.text(x + 0.95, 1.8, sub, ha='center', va='center',
                fontsize=8, color='#333', linespacing=1.5)

    # Setas
    for x in [2.3, 4.4, 6.5, 8.6, 10.7]:
        ax.annotate('', xy=(x + 0.2, 2.3), xytext=(x, 2.3),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

    ax.set_title('Pipeline de Machine Learning — CAER-DSS-ML',
                 fontsize=12, fontweight='bold', color='#222', pad=10)

    path = os.path.join(OUT, 'ml_pipeline_fluxo.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Salvo: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resumo de métricas — comparativo visual
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_summary():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('#F8F9FA')

    models = ['Ridge', 'Random Forest', 'XGBoost']
    mae    = [6.56, 1.99, 2.23]
    rmse   = [11.98, 3.00, 3.18]
    colors = ['#607D8B', '#4CAF50', '#FF9800']

    for ax, values, ylabel, title in zip(
        axes,
        [mae, rmse],
        ['MAE (mg/L)', 'RMSE (mg/L)'],
        ['MAE — Erro Absoluto Médio', 'RMSE — Raiz do Erro Quadrático Médio']
    ):
        bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=1.2, width=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold', color='#222', pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_facecolor('#F8F9FA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{val:.2f}', ha='center', fontsize=9.5, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.25)

    fig.suptitle('Desempenho dos Modelos — LOOCV (n=10, ETA São Pedro / CAER/RR)',
                 fontsize=12, fontweight='bold', color='#222', y=1.02)
    plt.tight_layout()

    path = os.path.join(OUT, 'resumo_metricas.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Salvo: {path}')


if __name__ == '__main__':
    plot_dss_architecture()
    plot_digitalization_gap()
    plot_ml_pipeline()
    plot_metrics_summary()
    print('\nTodos os gráficos explicativos gerados.')
