# Credit Risk Review Prioritization

Proyecto reproducible para estimar `P(not.fully.paid = 1)` en préstamos históricos de LendingClub y priorizar casos para revisión humana. **No es un sistema automático de aprobación o rechazo de préstamos.** Una probabilidad alta indica mayor afinidad estimada con la clase positiva, no certeza de impago.

El conjunto contiene 9.578 observaciones de 2007–2010. La aplicación del modelo a datos actuales requiere validación externa, evaluación de equidad y controles de gobernanza antes de cualquier uso real.

## Flujo de modelado

- El notebook de EDA, `notebooks/credit_risk_analysis.ipynb`, se conserva intacto.
- La modelización y selección final están en `notebooks/credit_risk_modeling.ipynb`.
- Target: `not.fully.paid`; clase 0 = pagado completamente, clase 1 = no pagado completamente.
- Se comparan baseline, regresión logística, Random Forest y XGBoost; la evaluación operativa final se centra en regresión logística sin balanceo, regresión logística con RandomOverSampler, Random Forest con `class_weight` y XGBoost ajustado.
- La selección se realiza en validación por Recall@K y la jerarquía de métricas documentada; PR-AUC de validación cruzada sobre entrenamiento informa estabilidad. Test se reserva para evaluación final y no interviene en selección del modelo, capacidad o threshold.
- Los escenarios K = 500, 750, 1.000, 1.250 y 1.500 son capacidades hipotéticas, no medidas reales de un equipo de LendingClub.

La evaluación reproducible seleccionó **Regresión logística sin balanceo** con capacidad hipotética de 1.000 revisiones. En validación obtuvo Recall@1.000 = 0,7296, 224 TP, 776 FP, 83 FN y PR-AUC = 0,2995; la PR-AUC media de validación cruzada en entrenamiento fue 0,2816 (desviación estándar 0,0134). El test reservado confirmó Recall@1.000 = 0,7329 (225 TP, 775 FP, 82 FN) y PR-AUC = 0,3042. Se eligió con validación; el resultado de test solo se informa como evaluación final.

El threshold de producción es **0,13264508**, fijado en validación a partir del score del préstamo situado en el puesto 1.000. Aplicado sin cambios al test, generó 972 alertas (50,73 % de los préstamos), recall = 0,7134, 219 TP, 753 FP y 88 FN. Las demás configuraciones y capacidades están en `reports/`; la selección no se basó solo en accuracy, ROC-AUC o F2.

Recall@K del modelo seleccionado, calculado por ranking directo de probabilidades:

| K | TP | FP | FN | Recall@K | Precision@K | Cartera revisada |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 135 | 365 | 172 | 0,4397 | 0,2700 | 26,10 % |
| 750 | 183 | 567 | 124 | 0,5961 | 0,2440 | 39,14 % |
| 1.000 | 225 | 775 | 82 | 0,7329 | 0,2250 | 52,19 % |
| 1.250 | 257 | 993 | 50 | 0,8371 | 0,2056 | 65,24 % |
| 1.500 | 286 | 1.214 | 21 | 0,9316 | 0,1907 | 78,29 % |

El threshold por capacidad es el score del puesto K en validación; al aplicarlo sin cambios a test, los empates y el cambio de muestra pueden alterar el número de alertas:

| K de validación | Threshold | Alertas test | Cartera test revisada | Recall test |
|---:|---:|---:|---:|---:|
| 500 | 0,19476845 | 475 | 24,79 % | 0,4300 |
| 750 | 0,15858615 | 739 | 38,57 % | 0,5896 |
| 1.000 | 0,13264508 | 972 | 50,73 % | 0,7134 |
| 1.250 | 0,11138934 | 1.235 | 64,46 % | 0,8339 |
| 1.500 | 0,09008706 | 1.502 | 78,39 % | 0,9316 |

Thresholds derivados en validación para los objetivos de recall del modelo seleccionado y resultados al aplicarlos sin cambios al test:

| Recall objetivo | Threshold validación | Recall validación | Alertas test | Recall test | FP test | FN test |
|---:|---:|---:|---:|---:|---:|---:|
| 0,90 | 0,10031500 | 0,9023 | 1.388 | 0,8990 | 1.112 | 31 |
| 0,95 | 0,08552488 | 0,9511 | 1.539 | 0,9414 | 1.250 | 18 |
| 0,98 | 0,06962319 | 0,9805 | 1.708 | 0,9772 | 1.408 | 7 |

El número real de alertas con un threshold validado puede diferir ligeramente de K al cambiar de muestra. El endpoint `/prioritize` respeta K exactamente para el lote que recibe; `/predict` y `/predict-file` aplican el threshold del artefacto.

## Preparación del entorno y evaluación

En PowerShell desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"
```

Ejecutar el notebook completo y regenerar el artefacto:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/credit_risk_modeling.ipynb
```

Las salidas persistentes de la evaluación se escriben en `reports/`; el bundle de inferencia, en `artifacts/credit_risk_model_bundle.joblib`.

## API de priorización

La API carga el bundle generado por el notebook; no reentrena modelos ni imputadores durante las solicitudes.

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/health` | Estado del servicio, modelo, target y threshold |
| `POST` | `/predict` | Puntuar un préstamo |
| `POST` | `/predict-file` | Puntuar un objeto/lista JSON o un CSV y devolver los casos ordenados por probabilidad descendente |
| `POST` | `/prioritize` | Ordenar registros JSON o CSV y, opcionalmente, marcar los primeros K como `review` |
| `GET` | `/sample-json` | Obtener un ejemplo de entrada válido |
| `GET` | `/docs` | Documentación Swagger/OpenAPI |

Ejecutar localmente:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

La API no incorpora autenticación ni TLS. Mantén el servicio en loopback para desarrollo; antes de exponerlo en una red, añade autenticación, TLS, control de acceso, monitorización y los controles de privacidad/gobernanza requeridos.

Ejemplo PowerShell:

```powershell
$sample = Invoke-RestMethod http://127.0.0.1:8000/sample-json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body ($sample | ConvertTo-Json -Depth 5)
```

`/predict` devuelve `probability_not_fully_paid`, `probability_fully_paid`, `prediction`, `prediction_label` y el threshold guardado en el bundle. `prediction = 1` / `review` significa que la probabilidad supera el threshold operativo; `prediction = 0` / `no_review` significa que no lo supera. Ninguna salida supone una decisión crediticia definitiva.

`/predict-file` acepta un objeto o una lista JSON. Incluye un `rank` y devuelve los casos de mayor probabilidad primero. Ejemplo:

```json
[
  {"credit.policy": 1, "purpose": "debt_consolidation", "int.rate": 0.1189, "installment": 829.1, "log.annual.inc": 11.3504, "dti": 19.48, "fico": 737, "days.with.cr.line": 5639.9583, "revol.bal": 28854, "revol.util": 52.1, "inq.last.6mths": 0, "delinq.2yrs": 0, "pub.rec": 0}
]
```

Ambos endpoints también aceptan un CSV como `multipart/form-data`, adjuntado en el campo `file`. La primera fila debe contener los 13 nombres de feature del modelo (por ejemplo, `credit.policy`, `int.rate` y `days.with.cr.line`); cada fila posterior representa un préstamo. Las celdas vacías se tratan como valores ausentes. Ejemplo:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict-file `
  -F "file=@loans.csv"
```

`/prioritize` puede aplicar una capacidad máxima al lote recibido. Con CSV, envía `review_capacity` como otro campo multipart opcional:

```json
{
  "review_capacity": 1,
  "records": [
    {"credit.policy": 1, "purpose": "debt_consolidation", "int.rate": 0.1189, "installment": 829.1, "log.annual.inc": 11.3504, "dti": 19.48, "fico": 737, "days.with.cr.line": 5639.9583, "revol.bal": 28854, "revol.util": 52.1, "inq.last.6mths": 0, "delinq.2yrs": 0, "pub.rec": 0}
  ]
}
```

Con `review_capacity`, los K primeros quedan etiquetados `review` y el resto `no_review`; si se omite, se usa el threshold del bundle. Esto es priorización de trabajo para revisión humana, no aprobación/rechazo automático.

```powershell
curl.exe -X POST http://127.0.0.1:8000/prioritize `
  -F "file=@loans.csv" `
  -F "review_capacity=10"
```

## Datos ausentes y valores inválidos

- Deben enviarse las 13 features del modelo; una feature puede valer `null`.
- Los faltantes numéricos se imputan con la mediana aprendida por el `SimpleImputer` del pipeline durante entrenamiento.
- `purpose` ausente se imputa con la moda entrenada. No se calculan medias, medianas o categorías nuevas a partir de la solicitud.
- Los límites numéricos de la API son el mínimo y máximo observados en el dataset histórico utilizado; son límites empíricos de entrada, no reglas de negocio o límites regulatorios.
- `purpose` se limita a las categorías observadas al entrenar.
- Un valor fuera del rango, una categoría no observada, una variable extra o un target `not.fully.paid` enviado como entrada se rechaza con HTTP 422. La respuesta identifica el campo y el motivo.

## Docker

Después de ejecutar el notebook y generar `artifacts/credit_risk_model_bundle.joblib`:

```powershell
docker build -t credit-risk-review .
docker run --rm -p 127.0.0.1:8000:8000 credit-risk-review
```

La imagen instala versiones de inferencia fijadas en `deployment/requirements.txt` para mantener compatibilidad con el artefacto. No incorpora `.env`, los CSV ni los notebooks. Swagger queda disponible en `http://localhost:8000/docs`.

## Tests y equivalencia notebook/API

```powershell
pytest -q
python deployment/validate_production_model.py
```

Los tests cubren endpoints, ranking/capacidad, imputación, target no admitido y errores 422. El validador reconstruye el split de test determinista, compara las probabilidades servidas con el pipeline guardado mediante `numpy.testing.assert_allclose`, verifica predicciones binarias exactas y confirma el threshold y los nombres de features.

## Estructura

```text
app/main.py                              API FastAPI
artifacts/credit_risk_model_bundle.joblib Pipeline, imputador, features, threshold y metadata
data/loan_data.csv                       Dataset histórico
deployment/requirements.txt              Dependencias fijadas para inferencia/Docker
deployment/validate_production_model.py  Equivalencia notebook/API
notebooks/credit_risk_analysis.ipynb     EDA (sin modificaciones)
notebooks/credit_risk_modeling.ipynb     Modelado y evaluación operativa
reports/                                 Tablas, ranking y resultados reproducibles
src/credit_risk/evaluation.py            Métricas de ranking y umbrales
src/credit_risk/notebook_reporting.py     Evaluación final y exportación del bundle
tests/                                   Tests de evaluación y API
Dockerfile                               Contenedor de inferencia
```
