"""Pipeline OSEMN sobre el dataset OULAD (RDBMS) + experimento Kongo (PI2).

OSEMN = Obtain, Scrub, Explore, Model, iNterpret.

- Dataset principal (PI): OULAD leído de MySQL. Unidad = estudiante-módulo-
  presentación. Targets derivados de `final_result` (dicotómico y ordinal) y de
  la nota media `score_medio` (intervalo/razón).
- Experimento complementario (PI2): Excel anonimizado "Kongo", target binario,
  del que se reportan además msePI2 y r2PI2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from . import config, utils
    from .db import FuenteOULAD, leer_kongo
    from .modelos import ClasificadorOULAD, RegresorOULAD, SegmentadorKMeans
except ImportError:
    import config
    import utils
    from db import FuenteOULAD, leer_kongo
    from modelos import ClasificadorOULAD, RegresorOULAD, SegmentadorKMeans

# Variables predictoras (se excluyen las derivadas de la evaluación para evitar
# fuga de información hacia final_result, y date_unregistration que determina
# directamente el estado "Withdrawn").
FEAT_NUM = ["num_of_prev_attempts", "studied_credits", "date_registration",
            "imd_ord", "age_ord", "vle_total_clics", "vle_interacciones",
            "vle_sitios"]
FEAT_CAT = ["gender", "region", "highest_education", "disability"]


class PipelineOULAD:
    def __init__(self, engine=None, verbose: bool = True):
        self.fuente = FuenteOULAD(engine)
        self.verbose = verbose
        self.df = None
        self.reporte_eda = {}
        self.resultados = {}
        self.figuras = []

    def log(self, *a):
        if self.verbose:
            print(*a)

    # ------------------------------------------------------------------ O
    def obtain(self):
        self.log("[O] Leyendo OULAD desde MySQL...")
        self.si = self.fuente.student_info()
        self.reg = self.fuente.registration()
        self.vle = self.fuente.vle_agg()
        self.asc = self.fuente.assess_agg()
        for n, d in [("student_info", self.si), ("registration", self.reg),
                     ("vle_agg", self.vle), ("assess_agg", self.asc)]:
            self.log(f"    - {n}: {d.shape}")
        return self

    # ------------------------------------------------------------------ S
    def scrub(self):
        df = self.si.merge(self.reg, on=config.CLAVE, how="left")
        df = df.merge(self.vle, on=config.CLAVE, how="left")
        df = df.merge(self.asc, on=config.CLAVE, how="left")

        # Targets desde final_result
        df["final_result"] = df["final_result"].astype(str).str.strip()
        df["target_bin"] = df["final_result"].map(config.RESULTADO_APRUEBA)
        df["target_ord"] = df["final_result"].map(config.RESULTADO_ORD)
        df = df.dropna(subset=["target_bin", "target_ord"])
        df["target_bin"] = df["target_bin"].astype(int)
        df["target_ord"] = df["target_ord"].astype(int)
        df["score_medio"] = pd.to_numeric(df["score_medio"], errors="coerce")

        # Ordinales
        df["imd_ord"] = df["imd_band"].astype(str).str.strip().apply(
            lambda v: config.ORDEN_IMD.index(v) if v in config.ORDEN_IMD else -1)
        df["age_ord"] = df["age_band"].astype(str).str.strip().apply(
            lambda v: config.ORDEN_EDAD.index(v) if v in config.ORDEN_EDAD else -1)

        # Missing values
        na_antes = df[FEAT_NUM].isna().sum().to_dict()
        for c in ["vle_total_clics", "vle_interacciones", "vle_sitios"]:
            df[c] = df[c].fillna(0)
        df["date_registration"] = df["date_registration"].fillna(
            df["date_registration"].median())
        self.reporte_eda["na_antes"] = {k: int(v) for k, v in na_antes.items() if v}

        for c in FEAT_CAT:
            df[c] = df[c].astype(str).str.strip()

        self.df = df.reset_index(drop=True)
        self.log(f"[S] Tabla analítica final: {self.df.shape}")
        return self

    def _features(self, sub: pd.DataFrame) -> pd.DataFrame:
        num = [c for c in FEAT_NUM if c in sub.columns]
        X = sub[num].copy()
        X = pd.concat([X, pd.get_dummies(sub[FEAT_CAT], drop_first=True)], axis=1)
        return X.fillna(0)

    # ------------------------------------------------------------------ E
    def explore(self):
        df = self.df
        num = df[FEAT_NUM + ["target_bin", "target_ord", "score_medio"]]
        self.reporte_eda["describe"] = num.describe().T
        self.reporte_eda["kurtosis"] = num.kurtosis(numeric_only=True)
        self.reporte_eda["skew"] = num.skew(numeric_only=True)
        self.reporte_eda["dist_final_result"] = utils.distribucion(df["final_result"])

        F = config.FIGURES_DIR
        self.figuras = []
        sc = df["score_medio"].dropna()
        if len(sc):
            self.figuras.append(utils.plot_histograma(
                sc, "Distribución de la nota media (score)", F / "hist_score.png"))
        self.figuras.append(utils.plot_boxplot(
            df, "final_result", "vle_total_clics",
            "Clics del VLE por resultado final", F / "box_vle_result.png",
            orden=["Withdrawn", "Fail", "Pass", "Distinction"], logy=True))
        self.figuras.append(utils.plot_correlacion(
            num, "Matriz de correlación", F / "correlacion.png"))
        df_s = df.dropna(subset=["score_medio"])
        if len(df_s):
            self.figuras.append(utils.plot_scatter(
                df_s, "vle_total_clics", "score_medio", "target_bin",
                "Clics VLE vs nota (color = aprueba)", F / "scatter_vle_score.png"))
        self.log(f"[E] EDA completo. {len(self.figuras)} figuras.")
        return self

    # ------------------------------------------------------------------ M
    def model(self, test_size=0.25, random_state=42):
        X = self._features(self.df)
        columnas = list(X.columns)
        Xs = StandardScaler().fit_transform(X)

        self._clasificar(Xs, self.df["target_bin"].values, columnas, "binario",
                         ["no_aprueba", "aprueba"], test_size, random_state,
                         con_mse=True)
        self._clasificar(Xs, self.df["target_ord"].values, columnas, "ordinal",
                         ["Withdrawn", "Fail", "Pass", "Distinction"],
                         test_size, random_state)

        # Regresión sobre la nota media (intervalo/razón)
        sub = self.df.dropna(subset=["score_medio"]).reset_index(drop=True)
        Xr = StandardScaler().fit_transform(self._features(sub))
        self._regresar(Xr, sub["score_medio"].values, columnas, test_size,
                       random_state)

        self._kmeans(Xs)
        return self

    def _clasificar(self, X, y, columnas, etiqueta, clases, test_size,
                    random_state, con_mse=False):
        estr = y if (len(np.unique(y)) > 1 and np.bincount(y).min() >= 2) else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=estr)
        resumen, mejor, imp = {}, None, {}
        for alg in ClasificadorOULAD.ALGORITMOS:
            modelo = ClasificadorOULAD(alg).fit(X_tr, y_tr)
            metr = modelo.evaluar(X_te, y_te)
            y_pred = modelo.predict(X_te)
            pd.DataFrame({"y_test": y_te, "y_pred": y_pred}).to_csv(
                config.PREDICTIONS_DIR / f"pred_{etiqueta}_{alg}.csv", index=False)
            if etiqueta == "binario":
                metr["f1_manual"] = utils.f1_manual(y_te, y_pred, positivo=1)
                if con_mse:
                    metr["mse"] = mean_squared_error(y_te, y_pred)
                    metr["r2"] = r2_score(y_te, y_pred)
            utils.plot_matriz_confusion(
                modelo.matriz_confusion(X_te, y_te), clases,
                f"Matriz de confusión ({etiqueta} - {alg})",
                config.FIGURES_DIR / f"cm_{etiqueta}_{alg}.png")
            resumen[alg] = metr
            if mejor is None or metr["f1_macro"] > resumen[mejor]["f1_macro"]:
                mejor, imp = alg, modelo.importancias(columnas)
        if imp:
            utils.plot_importancias(
                list(imp.keys()), list(imp.values()),
                f"Importancia de variables ({etiqueta} - {mejor})",
                config.FIGURES_DIR / f"importancias_{etiqueta}.png")
        self.resultados[f"clasificacion_{etiqueta}"] = {
            "metricas": resumen, "mejor": mejor, "importancias": imp,
            "n_train": len(y_tr), "n_test": len(y_te)}
        self.log(f"[M] Clasificación {etiqueta}: mejor={mejor} "
                 f"f1_macro={resumen[mejor]['f1_macro']:.3f}")

    def _regresar(self, X, y, columnas, test_size, random_state):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        resumen, mejor, imp = {}, None, {}
        for alg in RegresorOULAD.ALGORITMOS:
            modelo = RegresorOULAD(alg).fit(X_tr, y_tr)
            metr = modelo.evaluar(X_te, y_te)
            y_pred = modelo.predict(X_te)
            pd.DataFrame({"y_test": y_te, "y_pred": y_pred}).to_csv(
                config.PREDICTIONS_DIR / f"pred_regresion_{alg}.csv", index=False)
            resumen[alg] = metr
            if mejor is None or metr["r2"] > resumen[mejor]["r2"]:
                mejor, imp = alg, modelo.importancias(columnas)
        if imp:
            utils.plot_importancias(
                list(imp.keys()), list(imp.values()),
                f"Importancia de variables (regresión - {mejor})",
                config.FIGURES_DIR / "importancias_regresion.png")
        self.resultados["regresion_score"] = {
            "metricas": resumen, "mejor": mejor, "importancias": imp,
            "n_train": len(y_tr), "n_test": len(y_te)}
        self.log(f"[M] Regresión score: mejor={mejor} r2={resumen[mejor]['r2']:.3f}")

    def _kmeans(self, X):
        resumen = {}
        for k in (2, 3, 4):
            resumen[k] = SegmentadorKMeans(k).fit(X).evaluar(X)
        self.resultados["no_supervisado"] = resumen
        self.log("[M] KMeans evaluado para k=2,3,4")

    # ------------------------------------------------ PI2 (experimento Kongo)
    def pi2(self, test_size=0.25, random_state=42):
        """Modelo binario sobre el experimento complementario 'Kongo'.

        Reporta métricas de clasificación más msePI2 y r2PI2 (binario).
        """
        k = leer_kongo()
        ad = k["assess_detail"].copy()
        vle = k["vle_clickstream"].copy()
        agg = (vle.groupby("guid_student_id")
               .agg(vle_total_clics=("sum_clics", "sum"),
                    vle_sesiones=("sum_clics", "count"),
                    vle_sitios=("guid_site_id", "nunique")).reset_index())
        df = ad.merge(agg, on="guid_student_id", how="left")
        df["status"] = df["status"].astype(str).str.strip().str.lower()
        df["y"] = (df["status"] == "finished").astype(int)
        # date_real_days se excluye (fuga). Features simples y robustas:
        feats = ["weight", "num_of_prev_attempts", "vle_total_clics",
                 "vle_sesiones", "vle_sitios"]
        feats = [c for c in feats if c in df.columns]
        X = df[feats].fillna(0)
        if "assessment_type" in df.columns:
            X = pd.concat([X, pd.get_dummies(
                df["assessment_type"].astype(str), prefix="atype",
                drop_first=True)], axis=1)
        Xs = StandardScaler().fit_transform(X.fillna(0))
        y = df["y"].values
        estr = y if np.bincount(y).min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xs, y, test_size=test_size, random_state=random_state, stratify=estr)
        resumen, mejor = {}, None
        for alg in ClasificadorOULAD.ALGORITMOS:
            modelo = ClasificadorOULAD(alg).fit(X_tr, y_tr)
            metr = modelo.evaluar(X_te, y_te)
            y_pred = modelo.predict(X_te)
            metr["f1_manual"] = utils.f1_manual(y_te, y_pred, positivo=1)
            metr["msePI2"] = mean_squared_error(y_te, y_pred)
            metr["r2PI2"] = r2_score(y_te, y_pred)
            pd.DataFrame({"y_research_test": y_te, "y_research_pred": y_pred}).to_csv(
                config.PREDICTIONS_DIR / f"pred_pi2_{alg}.csv", index=False)
            resumen[alg] = metr
            if mejor is None or metr["f1_macro"] > resumen[mejor]["f1_macro"]:
                mejor = alg
        self.resultados["pi2_kongo"] = {
            "metricas": resumen, "mejor": mejor,
            "n_train": len(y_tr), "n_test": len(y_te)}
        self.log(f"[PI2] Kongo: mejor={mejor} "
                 f"f1_macro={resumen[mejor]['f1_macro']:.3f} "
                 f"msePI2={resumen[mejor]['msePI2']:.3f}")
        return self

    # ------------------------------------------------------------------ N
    def interpret(self):
        filas = []
        for tarea, info in self.resultados.items():
            if tarea == "no_supervisado":
                for kk, m in info.items():
                    filas.append({"tarea": "kmeans", "modelo": f"k={kk}",
                                  "metrica": "silhouette",
                                  "valor": m.get("silhouette")})
                continue
            for alg, metr in info["metricas"].items():
                for nombre, valor in metr.items():
                    if nombre == "f1_manual":
                        continue
                    filas.append({"tarea": tarea, "modelo": alg,
                                  "metrica": nombre, "valor": valor})
        rep = pd.DataFrame(filas)
        rep.to_csv(config.OUTPUTS_DIR / "reporte_metricas_general.csv", index=False)
        self.reporte_metricas = rep
        self.log("[N] Reporte general guardado en outputs/")
        return self

    # ---- Exportación de resumen completo (EDA + resultados) -------------
    @staticmethod
    def _safe(o):
        import numpy as _np
        if isinstance(o, dict):
            return {k: PipelineOULAD._safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [PipelineOULAD._safe(v) for v in o]
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return None if _np.isnan(o) else float(o)
        return o

    def exportar_resumen(self, ruta=None):
        """Guarda un JSON con el EDA y todas las métricas (todo en un archivo)."""
        import json
        eda = {}
        if self.df is not None:
            eda["n_filas"] = int(len(self.df))
            eda["distribucion_final_result"] = {
                k: int(v) for k, v in
                dict(self.reporte_eda.get("dist_final_result", {})).items()}
            eda["missing_imputados"] = self.reporte_eda.get("na_antes", {})
            if "describe" in self.reporte_eda:
                eda["describe"] = self.reporte_eda["describe"].round(4).to_dict(
                    orient="index")
            eda["kurtosis"] = {k: round(float(v), 4) for k, v in
                               self.reporte_eda.get("kurtosis", {}).items()}
            eda["skew"] = {k: round(float(v), 4) for k, v in
                          self.reporte_eda.get("skew", {}).items()}
        salida = {"eda": self._safe(eda), "resultados": self._safe(self.resultados)}
        ruta = ruta or (config.OUTPUTS_DIR / "resumen_resultados.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(salida, f, indent=2, ensure_ascii=False)
        self.log(f"[N] Resumen completo (EDA + métricas) guardado en {ruta}")
        return salida

    def run(self):
        return (self.obtain().scrub().explore().model().pi2().interpret())
