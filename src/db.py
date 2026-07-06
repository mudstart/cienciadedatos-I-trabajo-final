"""Capa de acceso a datos (DAL) sobre el RDBMS OULAD (MySQL).

`FuenteOULAD` lee las tablas del esquema OULAD y empuja las agregaciones
pesadas (clics del VLE, notas de evaluación) al motor SQL mediante GROUP BY,
evitando traer millones de filas a memoria. El experimento complementario
"Kongo" (PI2) se lee del Excel con `leer_kongo`.
"""
from __future__ import annotations

import pandas as pd

try:
    from . import config
except ImportError:  # ejecución como script
    import config


def crear_engine(silencioso: bool = True):
    """Crea un engine de SQLAlchemy hacia MySQL. Devuelve None si falla."""
    try:
        from sqlalchemy import create_engine

        url = (
            f"mysql+pymysql://{config.DB_USER}:{config.DB_PASS}"
            f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        )
        eng = create_engine(url)
        eng.connect().close()
        return eng
    except Exception as e:
        if not silencioso:
            print("No se pudo conectar a MySQL:", e)
        return None


class FuenteOULAD:
    """Acceso al esquema OULAD. Requiere un engine válido (MySQL o compatible)."""

    def __init__(self, engine=None):
        self.engine = engine or crear_engine()
        if self.engine is None:
            raise RuntimeError(
                "No hay conexión a la base de datos OULAD. Verifica que MySQL "
                "esté corriendo y las credenciales en src/config.py.")

    # ---- Lecturas simples ----------------------------------------------
    def tabla(self, nombre: str) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM `{nombre}`", self.engine)

    def student_info(self) -> pd.DataFrame:
        return self.tabla("student_info")

    def registration(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT id_student, code_module, code_presentation, "
            "date_registration, date_unregistration FROM student_registration",
            self.engine)

    # ---- Agregaciones en SQL (eficientes) ------------------------------
    def vle_agg(self) -> pd.DataFrame:
        """Actividad del VLE agregada por estudiante-módulo-presentación."""
        return pd.read_sql(
            "SELECT id_student, code_module, code_presentation, "
            "SUM(sum_click) AS vle_total_clics, "
            "COUNT(*) AS vle_interacciones, "
            "COUNT(DISTINCT id_site) AS vle_sitios "
            "FROM student_vle "
            "GROUP BY id_student, code_module, code_presentation",
            self.engine)

    def assess_agg(self) -> pd.DataFrame:
        """Notas de evaluación agregadas por estudiante-módulo-presentación."""
        return pd.read_sql(
            "SELECT sa.id_student, a.code_module, a.code_presentation, "
            "AVG(sa.score) AS score_medio, "
            "COUNT(*) AS n_evaluaciones, "
            "SUM(sa.is_banked) AS n_banked "
            "FROM student_assessment sa "
            "JOIN assessments a ON a.id_assessment = sa.id_assessment "
            "GROUP BY sa.id_student, a.code_module, a.code_presentation",
            self.engine)


def leer_kongo() -> dict:
    """Lee del Excel las hojas del experimento complementario 'Kongo' (PI2)."""
    return {clave: pd.read_excel(config.EXCEL_PATH, sheet_name=hoja)
            for clave, hoja in config.HOJAS_KONGO.items()}


# Compatibilidad: engine global (puede ser None si no hay MySQL)
engine = crear_engine()
