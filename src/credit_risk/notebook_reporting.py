"""Run the final notebook evaluation and export the production model bundle."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.credit_risk.evaluation import run_operational_evaluation


CAPACITIES = [500, 750, 1000, 1250, 1500]
TARGET_RECALLS = [0.90, 0.95, 0.98]
PRIMARY_MODELS = [
    "Regresión logística — sin balanceo",
    "Regresión logística — RandomOverSampler",
    "Random Forest — class_weight",
    "XGBoost ajustado",
]
REQUIRED_FEATURES = [
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _input_schema(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for feature in REQUIRED_FEATURES:
        column = data[feature]
        if feature == "purpose":
            schema[feature] = {
                "type": "categorical",
                "categories": sorted(column.dropna().astype(str).unique().tolist()),
                "missing_allowed": True,
            }
            continue
        schema[feature] = {
            "type": "integer" if pd.api.types.is_integer_dtype(column) else "number",
            "min": float(column.min()),
            "max": float(column.max()),
            "missing_allowed": True,
        }
    return schema


def _sample_record(row: pd.Series) -> dict[str, Any]:
    return {feature: _to_builtin(row[feature]) for feature in REQUIRED_FEATURES}


def finalize_notebook(
    *,
    project_root: Path | str,
    df: pd.DataFrame,
    target: str,
    models: dict[str, dict[str, Any]],
    evaluation: dict[str, dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: Any,
    seed: int,
    test_size: float,
    validation_size: float,
    validation_seed: int,
) -> dict[str, Any]:
    if target != "not.fully.paid":
        raise ValueError("Production evaluation requires target='not.fully.paid'.")
    if list(X_train.columns) != REQUIRED_FEATURES:
        raise ValueError("Notebook model feature order differs from API feature order.")

    candidate_models = [name for name in PRIMARY_MODELS if name in models]
    if candidate_models != PRIMARY_MODELS:
        missing = sorted(set(PRIMARY_MODELS) - set(candidate_models))
        raise KeyError(f"Required production candidates are missing: {missing}")

    result = run_operational_evaluation(
        candidate_models=candidate_models,
        models=models,
        y_train=y_train.to_numpy(),
        y_validation=y_validation.to_numpy(),
        y_test=y_test.to_numpy(),
        validation_indices=X_validation.index.to_list(),
        test_indices=X_test.index.to_list(),
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        cv=cv,
        capacities=CAPACITIES,
        target_recalls=TARGET_RECALLS,
        evaluation=evaluation,
    )

    print(
        "Selección realizada exclusivamente con validación. "
        f"Capacidad hipotética de selección: {result['selection_capacity']:,}."
    )
    display(result["selection_summary"].round(4))
    print("Top-K en validación:")
    display(result["validation_capacity"].round(4))
    print("Top-K en test (evaluación final, no usado para seleccionar):")
    display(result["test_capacity"].round(4))
    print("Umbrales para objetivos de recall: elegidos en validación y aplicados sin cambios a test:")
    display(result["recall_threshold_summary"].round(4))
    print("Umbral por capacidad: threshold de validación aplicado sin reoptimizar a test:")
    display(result["capacity_threshold_summary"].round(4))

    plot_specs = [
        ("Recall@K", "Recall@K", "Recall por capacidad de revisión"),
        (
            "False Positives / Extra Reviews",
            "Falsos positivos",
            "Falsos positivos dentro de Top-K",
        ),
        (
            "Precision@K",
            "Precision@K",
            "Precisión por capacidad de revisión",
        ),
    ]
    for metric, ylabel, title in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        for split, table, linestyle in [
            ("validación", result["validation_capacity"], "-"),
            ("test", result["test_capacity"], "--"),
        ]:
            for model_name in candidate_models:
                rows = table.loc[table["Model"] == model_name].sort_values(
                    "Capacity"
                )
                ax.plot(
                    rows["Capacity"],
                    rows[metric],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{model_name} — {split}",
                )
        ax.set(
            title=title,
            xlabel="Capacidad de revisión K",
            ylabel=ylabel,
        )
        ax.legend(loc="best", fontsize=7)
        plt.tight_layout()
        plt.show()

    selected_model_name = result["selected_model_name"]
    selected_model = models[selected_model_name]["estimator"]
    production_threshold = float(result["selected_validation_threshold"])
    production_test_metrics = result["production_test_metrics"]
    selected_validation_metrics = result["selected_validation_metrics"]
    selected_test_capacity_metrics = result["selected_test_capacity_metrics"]
    print(
        f"Modelo seleccionado para priorización: {selected_model_name}. "
        f"Recall@{result['selection_capacity']} en validación="
        f"{selected_validation_metrics['Recall@K']:.4f}; "
        f"PR-AUC en validación={selected_validation_metrics['PR-AUC']:.4f}; "
        f"CV PR-AUC={selected_validation_metrics['CV PR-AUC mean']:.4f} "
        f"± {selected_validation_metrics['CV PR-AUC std']:.4f}."
    )
    print(
        f"Test, Top-{result['selection_capacity']}: "
        f"Recall@K={selected_test_capacity_metrics['Recall@K']:.4f}, "
        f"TP={int(selected_test_capacity_metrics['TP'])}, "
        f"FP={int(selected_test_capacity_metrics['FP'])}, "
        f"FN={int(selected_test_capacity_metrics['FN'])}."
    )
    print(
        f"Threshold fijado en validación={production_threshold:.10f}; "
        f"al aplicarlo a test: alertas={int(production_test_metrics['Alerts'])}, "
        f"recall={production_test_metrics['Recall']:.4f}, "
        f"FP={int(production_test_metrics['FP'])}, "
        f"FN={int(production_test_metrics['FN'])}."
    )

    preprocessor = selected_model.named_steps["preprocesamiento"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    source_data = df.loc[:, REQUIRED_FEATURES]
    schema = _input_schema(source_data)
    package_versions = {
        package: importlib.metadata.version(package)
        for package in (
            "numpy",
            "pandas",
            "scikit-learn",
            "imbalanced-learn",
            "xgboost",
            "joblib",
        )
    }
    metadata = {
        "model_name": selected_model_name,
        "model_family": models[selected_model_name]["family"],
        "selection_method": "Validation Recall@K; metric hierarchy documented in notebook.",
        "selection_capacity": int(result["selection_capacity"]),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "seed": int(seed),
        "test_size": float(test_size),
        "validation_size": float(validation_size),
        "validation_seed": int(validation_seed),
        "test_indices": [int(index) for index in X_test.index],
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "positive_label": 1,
        "input_schema": schema,
        "schema_bounds_note": (
            "Numeric min/max bounds are the empirical ranges in the supplied "
            "historical dataset, not business or regulatory limits."
        ),
        "sample_record": _sample_record(X_train.iloc[0]),
        "validation_selection_metrics": _to_builtin(selected_validation_metrics),
        "test_top_k_metrics": _to_builtin(selected_test_capacity_metrics),
        "test_threshold_metrics": _to_builtin(production_test_metrics),
        "package_versions": package_versions,
    }
    bundle = {
        "model": selected_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "threshold": production_threshold,
        "metadata": metadata,
        "target": target,
    }

    root = Path(project_root)
    artifacts_dir = root / "artifacts"
    reports_dir = root / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifacts_dir / "credit_risk_model_bundle.joblib")

    result["validation_capacity"].to_csv(
        reports_dir / "capacity_validation.csv", index=False
    )
    result["test_capacity"].to_csv(reports_dir / "capacity_test.csv", index=False)
    result["selection_summary"].to_csv(
        reports_dir / "model_selection_validation.csv", index=False
    )
    result["recall_threshold_summary"].to_csv(
        reports_dir / "recall_thresholds_validation_test.csv", index=False
    )
    result["capacity_threshold_summary"].to_csv(
        reports_dir / "capacity_thresholds_validation_test.csv", index=False
    )
    result["ranked_validation"].to_csv(
        reports_dir / "ranking_validation.csv", index=False
    )
    result["ranked_test"].to_csv(reports_dir / "ranking_test.csv", index=False)
    selection_report = {
        "selected_model": selected_model_name,
        "target": target,
        "selection_capacity": int(result["selection_capacity"]),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "threshold": production_threshold,
        "validation_metrics": _to_builtin(selected_validation_metrics),
        "test_top_k_metrics": _to_builtin(selected_test_capacity_metrics),
        "test_threshold_metrics": _to_builtin(production_test_metrics),
        "package_versions": package_versions,
    }
    (reports_dir / "model_selection.json").write_text(
        json.dumps(selection_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
