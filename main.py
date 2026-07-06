"""Punto de entrada: ejecuta el pipeline OSEMN completo sobre OULAD (MySQL) + PI2.

Uso:
    python main.py

Requiere conexión a MySQL con el esquema OULAD cargado (7 tablas). Genera en
outputs/: figuras, CSVs de predicción (y_test, y_pred / y_research_*) y el
reporte general de métricas. Guarda además outputs/resumen_resultados.json.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
from db import crear_engine  # noqa: E402
from pipeline import PipelineOULAD  # noqa: E402


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    return o


def main():
    engine = crear_engine(silencioso=False)
    if engine is None:
        print("\nERROR: no se pudo conectar a MySQL. Revisa src/config.py y que "
              "el esquema OULAD esté cargado. Para validar: python validar_bd.py")
        sys.exit(1)

    pipe = PipelineOULAD(engine=engine, verbose=True)
    pipe.run()

    print("\n===== RESUMEN DE MÉTRICAS =====")
    print(pipe.reporte_metricas.round(4).to_string(index=False))

    print("\n===== f1 MANUAL (clasificación binaria OULAD, mejor modelo) =====")
    info = pipe.resultados["clasificacion_binario"]
    for kk, v in info["metricas"][info["mejor"]]["f1_manual"].items():
        print(f"  {kk}: {v}")

    print("\n===== PI2 (Kongo) msePI2 / r2PI2 =====")
    p2 = pipe.resultados["pi2_kongo"]
    m = p2["metricas"][p2["mejor"]]
    print(f"  modelo: {p2['mejor']}  msePI2={m['msePI2']:.4f}  r2PI2={m['r2PI2']:.4f}")

    # Guardar resumen completo (EDA + métricas) en un solo archivo
    pipe.exportar_resumen()
    print("\nResumen completo guardado en outputs/resumen_resultados.json")


if __name__ == "__main__":
    main()
