# Proyecto Final — Machine Learning sobre OULAD

Guía de ejecución **desde cero**. El proyecto entrena modelos de ML sobre el
dataset **OULAD** (leído desde **MySQL**) y lo complementa con el experimento
anónimo **"Kongo"** (Excel) como conjunto de investigación (PI2).

---

## 1. Arquitectura de datos

- **Dataset principal (PI): OULAD en MySQL** — 7 tablas (`student_info`,
  `student_assessment`, `student_vle`, `vle`, `assessments`, `courses`,
  `student_registration`). Unidad de análisis: estudiante-módulo-presentación.
- **Experimento complementario (PI2): Excel "Kongo"** — target binario; se
  reportan además `msePI2` y `r2PI2`.

**Targets del modelo principal** (derivados de `final_result`):

- Dicotómico: aprueba (Pass/Distinction) vs no aprueba (Fail/Withdrawn).
- Ordinal: Withdrawn < Fail < Pass < Distinction.
- Intervalo/razón: nota media `score_medio`.

---

## 2. Requisitos

- Python 3.10+ y Git.
- **MySQL 8+** con el esquema OULAD cargado (las 7 tablas). Es obligatorio:
  el modelo principal lee de la base de datos.

---

## 3. Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd cienciadedatos-I-trabajo-final

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configurar y validar la base de datos

1. Ajusta las credenciales en `src/config.py` (`DB_HOST`, `DB_PORT`, `DB_NAME`,
   `DB_USER`, `DB_PASS`).

2. Si necesitas crear el esquema desde cero:

   ```bash
   gunzip -k sql/oulad_schema.sql.gz
   mysql -u root -p < sql/oulad_schema.sql
   # luego importa los CSV oficiales de OULAD a las tablas
   ```

3. **Inspecciona** el esquema real (nombres de tablas y columnas):

   ```bash
   python inspeccionar_bd.py
   ```

4. **Valida** que el esquema OULAD esté correcto:

   ```bash
   python validar_bd.py
   ```

   Debe reportar las 7 tablas con sus columnas y las 4 clases de
   `final_result`. Si todo está OK, continúa.

---

## 5. Ejecutar el pipeline completo

```bash
python main.py
```

Ejecuta las fases OSEMN sobre OULAD + el experimento PI2 (Kongo). Genera en
`outputs/`: figuras, CSVs de predicción y el reporte de métricas, y guarda
`outputs/resumen_resultados.json`.

---

## 6. Análisis interactivo (notebook)

```bash
jupyter lab
```

Abre `notebooks/01_eda.ipynb` y ejecuta las celdas en orden.

---

## 7. Salidas generadas (`outputs/`)

- `predictions/` — CSVs caso a caso:
  - `pred_binario_*.csv`, `pred_ordinal_*.csv`, `pred_regresion_*.csv` (OULAD).
  - `pred_pi2_*.csv` con columnas `y_research_test`, `y_research_pred` (Kongo).
- `reporte_metricas_general.csv` — todas las métricas de todos los modelos.
- `figures/` — histograma, boxplot, correlación, scatter, matrices de
  confusión e importancia de variables.
- `resumen_resultados.json` — resumen estructurado (incluye `msePI2`/`r2PI2`).

El artículo APA está en `informe/Articulo_OULAD_APA.docx`.

---

## 8. Estructura del proyecto

```
cienciadedatos-I-trabajo-final/
├── main.py                 # Ejecuta el pipeline OSEMN (OULAD + PI2)
├── inspeccionar_bd.py      # Vuelca el esquema real de la BD
├── validar_bd.py           # Valida el esquema OULAD
├── requirements.txt
├── README.md
├── AnonymisezData_...xlsx  # Experimento complementario Kongo (PI2)
├── src/
│   ├── config.py           # Rutas, credenciales, esquema y targets
│   ├── db.py               # DAL: lectura OULAD + agregaciones SQL + Kongo
│   ├── utils.py            # Métricas manuales (f1) y gráficos (TAD-collection)
│   ├── modelos.py          # Clases de modelos supervisados y KMeans
│   └── pipeline.py         # Clase PipelineOULAD (OSEMN + PI2)
├── notebooks/01_eda.ipynb
├── sql/oulad_schema.sql(.gz)
├── informe/Articulo_OULAD_APA.docx
└── outputs/                # (se genera al ejecutar)
```

---

## 9. Solución de problemas

| Problema | Solución |
|----------|----------|
| `No se pudo conectar a MySQL` | Verifica que el servidor esté corriendo y las credenciales en `src/config.py`. |
| `Table ... doesn't exist` | Corre `python inspeccionar_bd.py` y `python validar_bd.py` para ver el esquema real. Los nombres de tabla deben coincidir con los del esquema OULAD. |
| `ModuleNotFoundError` | Activa el entorno virtual e instala `requirements.txt`. |
| El modelo principal usa pocos datos | Confirma con `validar_bd.py` que `student_info` tenga las ~32k filas y las 4 clases. |

---

## 10. Reproducibilidad

Todos los modelos usan `random_state=42` y partición fija 75/25.
