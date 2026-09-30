"""Compare production API scores against the notebook's held-out test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import FEATURES, TARGET, app, load_model_bundle


DATA_PATH = ROOT / "data" / "loan_data.csv"
BUNDLE_PATH = ROOT / "artifacts" / "credit_risk_model_bundle.joblib"
REPORT_PATH = ROOT / "reports" / "production_equivalence.json"


def main() -> None:
    bundle = load_model_bundle(BUNDLE_PATH)
    metadata = bundle["metadata"]
    frame = pd.read_csv(DATA_PATH)
    features = frame.loc[:, FEATURES]
    target = frame[TARGET]
    _, test_features, _, _ = train_test_split(
        features,
        target,
        test_size=metadata["test_size"],
        random_state=metadata["seed"],
        stratify=target,
    )
    reconstructed_indices = test_features.index.tolist()
    if reconstructed_indices != metadata["test_indices"]:
        raise AssertionError("The notebook test partition or row order changed.")
    # Use the exact saved row order after checking the deterministic split.
    test_features = features.loc[metadata["test_indices"]]

    preprocessor = bundle["model"].named_steps["preprocesamiento"]
    actual_feature_names = preprocessor.get_feature_names_out().tolist()
    if actual_feature_names != bundle["feature_names"]:
        raise AssertionError("Stored feature names differ from the trained preprocessor.")

    notebook_probabilities = bundle["model"].predict_proba(test_features)[:, 1]
    notebook_predictions = (
        notebook_probabilities >= float(bundle["threshold"])
    ).astype(int)
    payload = test_features.to_dict(orient="records")
    with TestClient(app) as client:
        response = client.post("/predict-file", json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Production API returned {response.status_code}: {response.text}"
            )

    served = response.json()["predictions"]
    ranking = np.argsort(-notebook_probabilities, kind="stable")
    ranked_notebook_probabilities = notebook_probabilities[ranking]
    ranked_notebook_predictions = notebook_predictions[ranking]
    served_probabilities = np.asarray(
        [row["probability_not_fully_paid"] for row in served], dtype=float
    )
    served_predictions = np.asarray([row["prediction"] for row in served], dtype=int)
    served_thresholds = np.asarray([row["threshold"] for row in served], dtype=float)
    np.testing.assert_allclose(
        served_probabilities,
        ranked_notebook_probabilities,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_array_equal(served_predictions, ranked_notebook_predictions)
    np.testing.assert_allclose(
        served_thresholds,
        np.full(len(served), float(bundle["threshold"])),
        rtol=0,
        atol=0,
    )

    report = {
        "model_name": metadata["model_name"],
        "target": TARGET,
        "test_rows": len(test_features),
        "features": len(FEATURES),
        "transformed_features": len(actual_feature_names),
        "threshold": float(bundle["threshold"]),
        "probability_max_abs_difference": float(
            np.max(
                np.abs(served_probabilities - ranked_notebook_probabilities)
            )
        ),
        "predictions_exact_match": bool(
            np.array_equal(served_predictions, ranked_notebook_predictions)
        ),
        "threshold_exact_match": bool(
            np.array_equal(
                served_thresholds,
                np.full(len(served), float(bundle["threshold"])),
            )
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
