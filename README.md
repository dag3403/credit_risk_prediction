# Credit Risk & Loan Repayment Prediction

Proyecto de análisis de riesgo crediticio con datos históricos de préstamos de LendingClub de 2007 a 2010. El objetivo es explorar las características de los préstamos y estudiar la predicción de si fueron pagados completamente. La variable objetivo es `not.fully.paid`.

**Estado actual:** Análisis exploratorio de datos (EDA) completado y documentado en `notebooks/credit_risk_analysis.ipynb`.

## Estructura del proyecto

```text
credit-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── loan_data.csv
├── notebooks/
│   ├── credit_risk_analysis.ipynb
│   └── loan_data.csv
├── src/credit_risk_modeling.ipynb
│   └── ...
└── models/
    └── ...
```

## Entorno virtual para ejecutar los notebooks

Se recomienda trabajar con un entorno virtual para que los notebooks queden aislados del resto del sistema:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"
```

Después, desde VS Code se puede seleccionar el kernel `Python (credit-risk)` para ejecutar `notebooks/credit_risk_analysis.ipynb` y `notebooks/credit_risk_modeling.ipynb`.

## Trabajo realizado

En el notebook `notebooks/credit_risk_analysis.ipynb` se llevó a cabo una exploración completa del conjunto de datos:

- Carga y validación del dataset `data/loan_data.csv`.
- Revisión de calidad de datos: ausencia de valores nulos y sin filas duplicadas.
- Análisis de la variable objetivo `not.fully.paid`.
- Descripción univariante de variables numéricas (tipo de interés, cuota, DTI, FICO, saldo rotativo, etc.).
- Visualización de distribuciones mediante histogramas y gráficos de barras.
- Evaluación de asociaciones por intervalos de FICO con la tasa observada de impago.
- Comparación de grupos pagados y no pagados para variables clave mediante pruebas no paramétricas.
- Identificación de variables con mayor potencial predictivo y preparación para etapas posteriores de preprocesamiento y modelado.

## Hallazgos clave

### Distribución de la variable objetivo

Se registraron 9.578 préstamos en el dataset. La proporción observada de préstamos no pagados completamente fue:

- `not.fully.paid = 0`: 8.045 casos (84,0%)
- `not.fully.paid = 1`: 1.533 casos (16,0%)

La tasa de impago observada en el conjunto es del 16,01%.

### Relación entre FICO y riesgo

La tasa de impago aumenta claramente a medida que baja la puntuación FICO:

| Intervalo FICO | Nº de préstamos | Tasa no pagado completamente |
|---|---:|---:|
| Hasta 660 | 489 | 30,879% |
| 661–700 | 3.732 | 19,373% |
| 701–740 | 3.127 | 15,286% |
| 741–780 | 1.709 | 8,777% |
| Más de 780 | 521 | 5,950% |

Estos resultados confirman que los prestatarios con menor puntuación FICO presentan un riesgo de impago significativamente mayor.

### Variables con mayor diferenciación

Se observaron diferencias relevantes entre préstamos pagados y no pagados en variables como:

- `fico`
- `int.rate`
- `dti`
- `revol.util`
- `inq.last.6mths`
- `delinq.2yrs`
- `pub.rec`

En particular, el riesgo de impago se asocia con tasas de interés más altas, mayor endeudamiento, peor utilización del crédito y peor historial crediticio.

## Tecnologías utilizadas

- Python
- Pandas y NumPy
- Matplotlib y Seaborn
- SciPy
- Jupyter Notebook

## Próximos pasos

1. Preprocesamiento de variables categóricas y numéricas.
2. Construcción de pipeline para modelado predictivo.
3. Entrenamiento y comparación de modelos de clasificación.
4. Evaluación con métricas adecuadas para datos desbalanceados.
5. Interpretabilidad del modelo y análisis de resultados.
6. Documentación final del proceso y recomendaciones de negocio.
