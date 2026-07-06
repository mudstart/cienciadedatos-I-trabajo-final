"""Utilidades transversales: TAD-collection, métricas manuales y gráficos.

Se emplean tipos de datos abstractos del módulo `collections` (namedtuple,
Counter, defaultdict, OrderedDict) para cumplir el requisito de uso apropiado
de TAD-collection.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, namedtuple

import matplotlib

matplotlib.use("Agg")  # backend no interactivo para guardar figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# --- TAD: estructura inmutable para la matriz de confusión binaria ----------
MatrizConfusion = namedtuple("MatrizConfusion", ["TP", "FP", "TN", "FN"])
Metrica = namedtuple("Metrica", ["nombre", "valor"])


def confusion_binaria_manual(y_true, y_pred, positivo=1) -> MatrizConfusion:
    """Calcula TP, FP, TN, FN manualmente (sin sklearn)."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    conteo = defaultdict(int)
    for yt, yp in zip(y_true, y_pred):
        real_pos = yt == positivo
        pred_pos = yp == positivo
        if real_pos and pred_pos:
            conteo["TP"] += 1
        elif not real_pos and pred_pos:
            conteo["FP"] += 1
        elif not real_pos and not pred_pos:
            conteo["TN"] += 1
        else:
            conteo["FN"] += 1
    return MatrizConfusion(conteo["TP"], conteo["FP"], conteo["TN"], conteo["FN"])


def f1_manual(y_true, y_pred, positivo=1) -> dict:
    """f1_score calculado a mano a partir de TP/FP/TN/FN."""
    m = confusion_binaria_manual(y_true, y_pred, positivo)
    precision = m.TP / (m.TP + m.FP) if (m.TP + m.FP) else 0.0
    recall = m.TP / (m.TP + m.FN) if (m.TP + m.FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return OrderedDict(
        TP=m.TP, FP=m.FP, TN=m.TN, FN=m.FN,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
    )


def distribucion(serie) -> Counter:
    """Devuelve un Counter (TAD) con la distribución de una serie categórica."""
    return Counter(serie.dropna().astype(str))


# --- Gráficos ---------------------------------------------------------------

def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return ruta


def plot_histograma(serie, titulo, ruta):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(serie.dropna(), kde=True, ax=ax, color="#4C72B0")
    ax.set_title(titulo)
    return guardar(fig, ruta)


def plot_boxplot(df, x, y, titulo, ruta, orden=None, logy=False):
    """Boxplot por grupo.

    Parameters
    ----------
    orden : list, opcional
        Orden explícito de las categorías del eje x (p. ej. ordinal).
    logy : bool
        Si True usa escala simétrica-logarítmica en el eje y, adecuada para
        variables muy asimétricas con valores en cero (p. ej. clics del VLE).
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.boxplot(data=df, x=x, y=y, ax=ax, order=orden,
                showfliers=True, fliersize=2, color="#4C72B0")
    if logy:
        ax.set_yscale("symlog")
        ax.set_ylabel(f"{y} (escala log)")
    ax.set_title(titulo)
    plt.xticks(rotation=20, ha="right")
    return guardar(fig, ruta)


def plot_correlacion(df_num, titulo, ruta):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df_num.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, annot_kws={"size": 7})
    ax.set_title(titulo)
    return guardar(fig, ruta)


def plot_scatter(df, x, y, hue, titulo, ruta):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, palette="viridis")
    ax.set_title(titulo)
    return guardar(fig, ruta)


def plot_matriz_confusion(cm, etiquetas, titulo, ruta):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=etiquetas, yticklabels=etiquetas, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    return guardar(fig, ruta)


def plot_importancias(nombres, valores, titulo, ruta):
    orden = np.argsort(valores)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=np.array(valores)[orden], y=np.array(nombres)[orden],
                ax=ax, color="#55A868")
    ax.set_title(titulo)
    return guardar(fig, ruta)
