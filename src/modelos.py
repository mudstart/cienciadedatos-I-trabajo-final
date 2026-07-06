"""Modelos de Machine Learning encapsulados con POO.

Jerarquía:
    ModeloBase (abstracta)
      ├── ClasificadorOULAD   -> targets dicotómicos y ordinales (multiclase)
      ├── RegresorOULAD       -> targets de intervalo/razón (score)
      └── SegmentadorKMeans   -> aprendizaje no supervisado (tendencias)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


class ModeloBase(ABC):
    """Contrato común para todos los modelos (encapsula fit/predict/evaluar)."""

    def __init__(self, nombre, estimador):
        self.nombre = nombre
        self.estimador = estimador
        self.entrenado = False

    def fit(self, X, y):
        self.estimador.fit(X, y)
        self.entrenado = True
        return self

    def predict(self, X):
        return self.estimador.predict(X)

    @abstractmethod
    def evaluar(self, X_test, y_test) -> OrderedDict:
        ...

    def importancias(self, columnas):
        """Importancia de variables si el estimador la expone."""
        if hasattr(self.estimador, "feature_importances_"):
            return dict(zip(columnas, self.estimador.feature_importances_))
        if hasattr(self.estimador, "coef_"):
            coef = np.ravel(self.estimador.coef_)
            return dict(zip(columnas, np.abs(coef)))
        return {}


class ClasificadorOULAD(ModeloBase):
    """Clasificación supervisada (binaria u ordinal tratada como multiclase)."""

    ALGORITMOS = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    }

    def __init__(self, algoritmo):
        super().__init__(algoritmo, self.ALGORITMOS[algoritmo]())

    def evaluar(self, X_test, y_test) -> OrderedDict:
        y_pred = self.predict(X_test)
        res = OrderedDict()
        res["precision_macro"] = precision_score(
            y_test, y_pred, average="macro", zero_division=0)
        res["recall_macro"] = recall_score(
            y_test, y_pred, average="macro", zero_division=0)
        res["f1_macro"] = f1_score(
            y_test, y_pred, average="macro", zero_division=0)
        res["accuracy"] = accuracy_score(y_test, y_pred)
        res["roc_auc"] = self._roc_auc(X_test, y_test)
        return res

    def _roc_auc(self, X_test, y_test):
        clases = np.unique(y_test)
        if len(clases) != 2 or not hasattr(self.estimador, "predict_proba"):
            return np.nan
        proba = self.estimador.predict_proba(X_test)[:, 1]
        try:
            return roc_auc_score(y_test, proba)
        except ValueError:
            return np.nan

    def matriz_confusion(self, X_test, y_test):
        return confusion_matrix(y_test, self.predict(X_test))


class RegresorOULAD(ModeloBase):
    """Regresión supervisada para targets de intervalo/razón (p. ej. score)."""

    ALGORITMOS = {
        "linear_regression": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=42),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
    }

    def __init__(self, algoritmo):
        super().__init__(algoritmo, self.ALGORITMOS[algoritmo]())

    def evaluar(self, X_test, y_test) -> OrderedDict:
        y_pred = self.predict(X_test)
        res = OrderedDict()
        res["mse"] = mean_squared_error(y_test, y_pred)
        res["rmse"] = float(np.sqrt(res["mse"]))
        res["r2"] = r2_score(y_test, y_pred)
        return res


class SegmentadorKMeans(ModeloBase):
    """Aprendizaje no supervisado para detectar tendencias/segmentos."""

    def __init__(self, k=3):
        super().__init__(f"kmeans_k{k}",
                         KMeans(n_clusters=k, n_init=10, random_state=42))
        self.k = k

    def fit(self, X, y=None):
        self.estimador.fit(X)
        self.entrenado = True
        self.labels_ = self.estimador.labels_
        return self

    def evaluar(self, X_test, y_test=None) -> OrderedDict:
        labels = self.estimador.predict(X_test)
        res = OrderedDict(k=self.k)
        try:
            res["silhouette"] = silhouette_score(X_test, labels)
        except ValueError:
            res["silhouette"] = np.nan
        return res
