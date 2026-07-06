"""Configuración central del proyecto OULAD.

Define rutas, credenciales del RDBMS y los metadatos del esquema OULAD real
(7 tablas) que vive en MySQL, así como el mapeo del experimento complementario
"Kongo" (PI2) almacenado en el Excel anonimizado.
"""
from pathlib import Path

# --- Rutas del proyecto -----------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
EXCEL_PATH = RAIZ / "AnonymisezData_oulad_context-Kongo-2024 (2).xlsx"
OUTPUTS_DIR = RAIZ / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SCHEMA_DIR = RAIZ / "sql"

for _d in (OUTPUTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, SCHEMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Credenciales RDBMS (MySQL) --------------------------------------------
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "oulad"
DB_USER = "root"
DB_PASS = "12345678"

# --- Esquema OULAD real (tablas en MySQL) ----------------------------------
TABLAS_OULAD = [
    "assessments", "courses", "student_assessment", "student_info",
    "student_registration", "student_vle", "vle",
]

# Hojas del experimento complementario "Kongo" (PI2) en el Excel
HOJAS_KONGO = {
    "assess_detail": "Assesss_detail",
    "vle_clickstream": "VLE_clickStream",
}

# --- Codificación de variables ordinales -----------------------------------
ORDEN_IMD = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
             "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
ORDEN_EDAD = ["0-35", "35-55", "55<="]

# --- Targets derivados de final_result -------------------------------------
# Ordinal: Withdrawn < Fail < Pass < Distinction
RESULTADO_ORD = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}
# Dicotómico: aprueba (Pass/Distinction) = 1 ; no aprueba (Fail/Withdrawn) = 0
RESULTADO_APRUEBA = {"Withdrawn": 0, "Fail": 0, "Pass": 1, "Distinction": 1}

# Clave natural de la unidad de análisis
CLAVE = ["id_student", "code_module", "code_presentation"]
