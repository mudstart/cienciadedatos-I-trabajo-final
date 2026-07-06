"""Valida que la base de datos MySQL contenga el esquema OULAD correcto.

Verifica, tabla por tabla: existencia, columnas esperadas, número de filas (>0)
y la sanidad de final_result (4 clases). Úsalo antes de 'python main.py'.

Uso:
    python validar_bd.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
from sqlalchemy import inspect  # noqa: E402
from db import crear_engine  # noqa: E402

OK, FAIL = "[ OK ]", "[FALLA]"

ESQUEMA = {
    "assessments": {"id_assessment", "code_module", "code_presentation",
                    "assessment_type", "date_due", "weight"},
    "courses": {"code_module", "code_presentation", "module_presentation_length"},
    "student_assessment": {"id_assessment", "id_student", "date_submitted",
                           "is_banked", "score"},
    "student_info": {"id_student", "code_module", "code_presentation", "gender",
                     "region", "highest_education", "imd_band", "age_band",
                     "num_of_prev_attempts", "studied_credits", "disability",
                     "final_result"},
    "student_registration": {"id_student", "code_module", "code_presentation",
                             "date_registration", "date_unregistration"},
    "student_vle": {"id_student", "id_site", "code_module", "code_presentation",
                    "date_interaction", "sum_click"},
    "vle": {"id_site", "code_module", "code_presentation", "activity_type",
            "week_from", "week_to"},
}


def validar(engine):
    insp = inspect(engine)
    tablas = set(insp.get_table_names())
    print("Tablas encontradas:", sorted(tablas), "\n")
    problemas = 0
    print(f"{'Tabla':22s} {'Existe':7s} {'Filas':9s} Columnas")
    print("-" * 70)
    for t, cols_esp in ESQUEMA.items():
        if t not in tablas:
            print(f"{t:22s} {'NO':7s} {'-':9s} {FAIL} no existe")
            problemas += 1
            continue
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM `{t}`", engine)["n"].iloc[0]
        cols = {c["name"] for c in insp.get_columns(t)}
        faltan = cols_esp - cols
        estado = OK if (n > 0 and not faltan) else FAIL
        if estado == FAIL:
            problemas += 1
        nota = "" if not faltan else f" faltan: {sorted(faltan)}"
        if n == 0:
            nota += " (tabla vacía)"
        print(f"{t:22s} {'SÍ':7s} {n:<9d} {estado}{nota}")

    print("\nSanidad de student_info.final_result:")
    if "student_info" in tablas:
        dist = pd.read_sql(
            "SELECT final_result, COUNT(*) n FROM student_info "
            "GROUP BY final_result ORDER BY n DESC", engine)
        for _, r in dist.iterrows():
            print(f"   {r['final_result']:12s} {r['n']}")
        esperadas = {"Pass", "Fail", "Withdrawn", "Distinction"}
        if esperadas.issubset(set(dist["final_result"])):
            print(f"   {OK} contiene las 4 clases esperadas.")
        else:
            print(f"   {FAIL} faltan clases: {esperadas - set(dist['final_result'])}")
            problemas += 1
    return problemas


def main():
    engine = crear_engine(silencioso=False)
    if engine is None:
        print(f"{FAIL} No se pudo conectar a MySQL. Revisa src/config.py.")
        sys.exit(1)
    print(f"{OK} Conexión establecida.\n")
    problemas = validar(engine)
    print("\n" + "=" * 70)
    if problemas == 0:
        print(f"{OK} ESQUEMA OULAD VÁLIDO. Puedes ejecutar 'python main.py'.")
    else:
        print(f"{FAIL} {problemas} problema(s). Revisa el esquema OULAD en MySQL.")
        sys.exit(1)


if __name__ == "__main__":
    main()
