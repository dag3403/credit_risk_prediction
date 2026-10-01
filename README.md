# Credit Risk Review Prioritization

Proyecto de análisis y modelado de riesgo crediticio que estima la probabilidad de que un préstamo no se pague completamente y permite ordenar solicitudes para revisión humana. Incluye análisis exploratorio, comparación de modelos, evaluación con capacidad limitada y una API de inferencia con FastAPI.

> **Uso responsable:** este proyecto es demostrativo y utiliza datos históricos de préstamos de LendingClub de 2007–2010. Una puntuación alta no demuestra que un préstamo vaya a impagarse ni debe utilizarse por sí sola para aprobar, rechazar o fijar las condiciones de un crédito. El modelo no está validado para decisiones actuales; cualquier uso real requeriría validación externa, evaluación de equidad, controles de privacidad y supervisión adecuada.

## Contenido

- [Datos y metodología](#datos-y-metodología)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y reproducción](#instalación-y-reproducción)
- [API](#api)
- [Validación y tests](#validación-y-tests)
- [Limitaciones](#limitaciones)

## Datos y metodología

El dataset incluido en [`data/loan_data.csv`](data/loan_data.csv) contiene 9.578 préstamos concedidos entre 2007 y 2010. El objetivo es `not.fully.paid`: `1` indica que el préstamo no se pagó completamente y `0`, que se pagó por completo. El modelo utiliza 13 variables, entre ellas el propósito del préstamo, el tipo de interés, el historial crediticio y los indicadores de utilización del crédito.

El análisis exploratorio está en [`notebooks/credit_risk_analysis.ipynb`](notebooks/credit_risk_analysis.ipynb) y el proceso de modelado y evaluación, en [`notebooks/credit_risk_modeling.ipynb`](notebooks/credit_risk_modeling.ipynb). Se comparan modelos baseline, regresión logística, Random Forest y XGBoost. La selección se hace con datos de validación, priorizando Recall@K para capacidades hipotéticas de revisión; el conjunto de test reservado se utiliza solo para la evaluación final.

El artefacto de inferencia versionado es [`artifacts/credit_risk_model_bundle.joblib`](artifacts/credit_risk_model_bundle.joblib). Contiene el pipeline entrenado, el umbral operativo, los nombres de las variables y metadatos. La API lo carga para puntuar registros; no vuelve a entrenar el modelo durante las solicitudes.

## Resultados

La configuración seleccionada en validación es una **regresión logística sin balanceo**, con capacidad hipotética de 1.000 revisiones. El umbral operativo guardado en el artefacto es **0,13264508**, calculado en validación a partir de la puntuación del caso situado en el puesto 1.000.

| Evaluación | Recall@1.000 | PR-AUC | Resultado adicional |
|---|---:|---:|---|
| Validación | 0,7296 | 0,2995 | 224 positivos identificados de 307 |
| Test reservado | 0,7329 | 0,3042 | 225 positivos identificados de 307 |
| Test con el umbral operativo | — | — | 972 alertas; recall 0,7134 |

Recall@K ordena directamente por probabilidad y selecciona exactamente los primeros K registros. En el test, los resultados del ranking fueron:

| K | Recall@K | Precision@K | Casos positivos identificados |
|---:|---:|---:|---:|
| 500 | 0,4397 | 0,2700 | 135 |
| 750 | 0,5961 | 0,2440 | 183 |
| 1.000 | 0,7329 | 0,2250 | 225 |
| 1.250 | 0,8371 | 0,2056 | 257 |
| 1.500 | 0,9316 | 0,1907 | 286 |

La aplicación de un umbral fijado en validación a otra muestra no garantiza exactamente K alertas: la distribución de puntuaciones y los empates pueden cambiar el recuento. `/prioritize` permite aplicar una capacidad exacta al lote recibido. Los resultados y métricas complementarios están en [`reports/`](reports/).

## Estructura del repositorio

```text
app/                 API de inferencia FastAPI
artifacts/           Pipeline de producción serializado
data/                Dataset histórico utilizado
deployment/          Dependencias de inferencia y validador de equivalencia
notebooks/           Análisis exploratorio y modelado
reports/             Métricas, selección y resultados reproducibles
src/credit_risk/     Funciones de evaluación y generación de informes
tests/               Tests de evaluación y API
Dockerfile           Imagen del servicio de inferencia
requirements.txt     Dependencias para notebooks, entrenamiento y tests
```

## Instalación y reproducción

Se recomienda **Python 3.14**, que es la versión utilizada por la imagen Docker del proyecto.

Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias del proyecto:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para ejecutar los notebooks de forma interactiva:

```bash
jupyter notebook
```

Abrir `notebooks/credit_risk_modeling.ipynb` y ejecutar sus celdas para reproducir el entrenamiento y la evaluación. El notebook actualiza el artefacto del modelo y los informes de `reports/`. También puede ejecutarse de forma no interactiva:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/credit_risk_modeling.ipynb
```

El dataset y el artefacto utilizado por la API ya están incluidos en el repositorio; volver a ejecutar el notebook es necesario para reproducir o regenerar el modelo, no para iniciar el servicio.

## API

La API carga el bundle existente y ofrece estos endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y metadatos básicos del modelo |
| `GET` | `/sample-json` | Ejemplo de registro válido |
| `POST` | `/predict` | Puntúa un préstamo y aplica el umbral del artefacto |
| `POST` | `/predict-file` | Puntúa un registro, una lista JSON o un CSV y devuelve los resultados ordenados |
| `POST` | `/prioritize` | Ordena un lote y, opcionalmente, marca los primeros K para revisión |
| `GET` | `/docs` | Documentación interactiva Swagger/OpenAPI |

### Ejecución local

Instalar las dependencias de inferencia y arrancar el servicio:

```bash
python -m pip install -r deployment/requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

La documentación interactiva queda disponible en <http://127.0.0.1:8000/docs>. Para probar una predicción en PowerShell:

```powershell
$sample = Invoke-RestMethod http://127.0.0.1:8000/sample-json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body ($sample | ConvertTo-Json -Depth 5)
```

`/predict` devuelve las probabilidades de ambas clases, `prediction`, `prediction_label` y el umbral usado. `review` significa que la probabilidad supera el umbral configurado; `no_review`, que no lo supera. Estas etiquetas indican prioridad de revisión, no una decisión crediticia.

### Priorización por capacidad

`/prioritize` acepta un objeto JSON con una lista `records`. Puede incluir `review_capacity` para marcar como `review` los K casos mejor puntuados y como `no_review` los restantes. Si se omite, se conserva la clasificación por umbral del artefacto.

```json
{
  "review_capacity": 10,
  "records": [
    {
      "credit.policy": 1,
      "purpose": "debt_consolidation",
      "int.rate": 0.1189,
      "installment": 829.1,
      "log.annual.inc": 11.3504,
      "dti": 19.48,
      "fico": 737,
      "days.with.cr.line": 5639.9583,
      "revol.bal": 28854,
      "revol.util": 52.1,
      "inq.last.6mths": 0,
      "delinq.2yrs": 0,
      "pub.rec": 0
    }
  ]
}
```

`/predict-file` acepta un registro o una lista JSON y devuelve los casos ordenados por probabilidad descendente, con su posición (`rank`). Tanto `/predict-file` como `/prioritize` aceptan también un archivo CSV mediante `multipart/form-data`, en el campo `file`. El CSV debe tener encabezados para las 13 variables del modelo:

```bash
curl -X POST http://127.0.0.1:8000/prioritize \
  -F "file=@loans.csv" \
  -F "review_capacity=10"
```

Se admiten valores ausentes (`null` en JSON y celdas vacías en CSV); el pipeline utiliza las imputaciones aprendidas durante el entrenamiento. Los valores numéricos fuera de los rangos observados, categorías no conocidas, columnas adicionales o el target `not.fully.paid` como entrada se rechazan con HTTP 422.

### Docker

Construir y ejecutar el servicio desde la raíz del repositorio:

```bash
docker build -t credit-risk-review .
docker run --rm -p 127.0.0.1:8000:8000 credit-risk-review
```

La imagen utiliza las dependencias de inferencia fijadas en [`deployment/requirements.txt`](deployment/requirements.txt), incluye el artefacto versionado y publica el servicio únicamente en el loopback del host. No incluye el dataset ni los notebooks.

## Validación y tests

Ejecutar la suite de tests:

```bash
pytest -q
```

Verificar que las predicciones servidas por la API coinciden con el pipeline guardado en el conjunto de test:

```bash
python deployment/validate_production_model.py
```

El validador reconstruye la partición determinista de test y comprueba la equivalencia de probabilidades, predicciones, umbral y nombres de variables. Los detalles de la última validación se guardan en [`reports/production_equivalence.json`](reports/production_equivalence.json).

## Limitaciones

- Los datos describen préstamos históricos de 2007–2010 y no representan necesariamente las condiciones, población o políticas de crédito actuales.
- Las capacidades de revisión evaluadas son escenarios hipotéticos; no representan la capacidad real de un equipo.
- Las métricas agregadas no evalúan por sí solas equidad, impacto sobre grupos protegidos, deriva temporal ni adecuación regulatoria.
- La API no incorpora autenticación ni TLS. Mantenerla en loopback para pruebas locales; antes de exponerla a una red, añadir controles de acceso, transporte seguro, monitorización y las salvaguardas de privacidad y gobernanza pertinentes.
