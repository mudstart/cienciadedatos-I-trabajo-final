"""Inspecciona el esquema real de la base de datos MySQL.

Imprime, para cada tabla: número de filas, columnas con su tipo y 2 filas de
muestra. Sirve para adaptar el pipeline al esquema real (nombres de columnas).

Uso:
    python inspeccionar_bd.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
from sqlalchemy import inspect  # noqa: E402
from db import crear_engine  # noqa: E402

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)


def main():
    engine = crear_engine()
    if engine is None:
        print("ERROR: no se pudo conectar a MySQL. Revisa src/config.py.")
        sys.exit(1)

    insp = inspect(engine)
    tablas = insp.get_table_names()
    print(f"Base de datos con {len(tablas)} tablas: {tablas}\n")
    print("=" * 80)

    for t in tablas:
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM `{t}`", engine)["n"].iloc[0]
        cols = insp.get_columns(t)
        print(f"\n### TABLA: {t}   ({n} filas, {len(cols)} columnas)")
        print("Columnas (nombre : tipo):")
        for c in cols:
            print(f"   - {c['name']} : {c['type']}")
        try:
            muestra = pd.read_sql(f"SELECT * FROM `{t}` LIMIT 2", engine)
            print("Muestra (2 filas):")
            print(muestra.to_string(index=False))
        except Exception as e:
            print(f"   (no se pudo leer muestra: {e})")
        print("-" * 80)


if __name__ == "__main__":
    main()
