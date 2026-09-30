from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


def csv_content(records):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_record(client):
    response = client.get("/sample-json")
    assert response.status_code == 200
    return response.json()


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
    assert response.json()["target"] == "not.fully.paid"


def test_predict_returns_probabilities_threshold_and_review_label(client, sample_record):
    response = client.post("/predict", json=sample_record)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["probability_not_fully_paid"] <= 1
    assert result["probability_not_fully_paid"] + result["probability_fully_paid"] == pytest.approx(1)
    assert result["prediction_label"] in {"review", "no_review"}
    assert result["prediction"] == int(
        result["probability_not_fully_paid"] >= result["threshold"]
    )
    assert "approve" not in result["prediction_label"]
    assert "reject" not in result["prediction_label"]


def test_missing_fico_is_imputed_by_trained_pipeline(client, sample_record):
    record = copy.deepcopy(sample_record)
    record["fico"] = None

    response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert 0 <= response.json()["probability_not_fully_paid"] <= 1


def test_missing_purpose_is_imputed_by_trained_pipeline(client, sample_record):
    record = copy.deepcopy(sample_record)
    record["purpose"] = None

    response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert 0 <= response.json()["probability_not_fully_paid"] <= 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("purpose", "not_a_dataset_category"),
        ("credit.policy", 2),
        ("int.rate", -0.01),
        ("fico", 1000),
    ],
)
def test_invalid_values_return_structured_422(client, sample_record, field, value):
    record = copy.deepcopy(sample_record)
    record[field] = value

    response = client.post("/predict", json=record)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["field"] == field
    assert detail["value"] == value
    assert detail["message"]


def test_negative_numeric_value_is_rejected(client, sample_record):
    record = copy.deepcopy(sample_record)
    record["installment"] = -1

    response = client.post("/predict", json=record)

    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "installment"


def test_target_is_not_an_allowed_input(client, sample_record):
    record = copy.deepcopy(sample_record)
    record["not.fully.paid"] = 1

    response = client.post("/predict", json=record)

    assert response.status_code == 422
    assert response.json()["detail"]["field"] == "not.fully.paid"


def test_predict_file_accepts_object_and_multiple_records(client, sample_record):
    single_response = client.post("/predict-file", json=sample_record)
    records = [sample_record, copy.deepcopy(sample_record)]
    records[1]["fico"] = min(
        records[1]["fico"] + 10,
        827,
    )
    list_response = client.post("/predict-file", json=records)

    assert single_response.status_code == 200
    assert single_response.json()["count"] == 1
    assert list_response.status_code == 200
    result = list_response.json()
    assert result["count"] == 2
    assert [row["rank"] for row in result["predictions"]] == [1, 2]
    scores = [row["probability_not_fully_paid"] for row in result["predictions"]]
    assert scores == sorted(scores, reverse=True)


def test_predict_file_accepts_csv_with_multiple_records(client, sample_record):
    records = [copy.deepcopy(sample_record) for _ in range(2)]
    records[1]["fico"] = min(records[1]["fico"] + 10, 827)

    response = client.post(
        "/predict-file",
        files={"file": ("loans.csv", csv_content(records), "text/csv")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 2
    assert [row["rank"] for row in result["predictions"]] == [1, 2]


def test_prioritize_respects_review_capacity_and_rank(client, sample_record):
    records = [copy.deepcopy(sample_record) for _ in range(3)]
    records[0]["fico"] = 612
    records[1]["fico"] = 707
    records[2]["fico"] = 827

    response = client.post(
        "/prioritize",
        json={"records": records, "review_capacity": 2},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["review_capacity"] == 2
    assert [row["rank"] for row in result["predictions"]] == [1, 2, 3]
    assert [row["prediction_label"] for row in result["predictions"]] == [
        "review",
        "review",
        "no_review",
    ]
    scores = [row["probability_not_fully_paid"] for row in result["predictions"]]
    assert scores == sorted(scores, reverse=True)


def test_prioritize_accepts_csv_with_multiple_records_and_capacity(
    client, sample_record
):
    records = [copy.deepcopy(sample_record) for _ in range(3)]
    records[0]["fico"] = 612
    records[1]["fico"] = 707
    records[2]["fico"] = 827

    response = client.post(
        "/prioritize",
        files={"file": ("loans.csv", csv_content(records), "text/csv")},
        data={"review_capacity": "2"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 3
    assert result["review_capacity"] == 2
    assert [row["rank"] for row in result["predictions"]] == [1, 2, 3]
    assert [row["prediction_label"] for row in result["predictions"]] == [
        "review",
        "review",
        "no_review",
    ]


def test_csv_uploads_are_listed_in_openapi(client):
    schema = client.get("/openapi.json").json()

    assert "multipart/form-data" in schema["paths"]["/predict-file"]["post"][
        "requestBody"
    ]["content"]
    assert "multipart/form-data" in schema["paths"]["/prioritize"]["post"][
        "requestBody"
    ]["content"]


def test_provided_csv_fixture_works_with_both_batch_endpoints(client):
    fixture_path = Path(__file__).parent / "fixtures" / "test_loans_credit_risk.csv"
    csv_bytes = fixture_path.read_bytes()
    upload = ("test_loans_credit_risk.csv", csv_bytes, "text/csv")

    predict_response = client.post("/predict-file", files={"file": upload})
    prioritize_response = client.post(
        "/prioritize",
        files={"file": upload},
        data={"review_capacity": "3"},
    )

    assert predict_response.status_code == 200
    assert predict_response.json()["count"] == 10
    assert prioritize_response.status_code == 200
    assert prioritize_response.json()["count"] == 10
    assert prioritize_response.json()["review_capacity"] == 3


def test_prioritize_rejects_non_integer_csv_review_capacity(client, sample_record):
    response = client.post(
        "/prioritize",
        files={
            "file": (
                "loans.csv",
                csv_content([sample_record]),
                "text/csv",
            )
        },
        data={"review_capacity": "not-an-integer"},
    )

    assert response.status_code == 422


def test_prioritize_caps_capacity_at_record_count(client, sample_record):
    response = client.post(
        "/prioritize",
        json={"records": [sample_record], "review_capacity": 500},
    )

    assert response.status_code == 200
    assert response.json()["review_capacity"] == 1
    assert response.json()["predictions"][0]["prediction_label"] == "review"


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/predict-file",
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_empty_record_list_is_rejected(client):
    response = client.post("/prioritize", json={"records": [], "review_capacity": 1})

    assert response.status_code == 422


def test_sample_json_contains_only_model_features(client):
    record = client.get("/sample-json").json()

    assert len(record) == 13
    assert "not.fully.paid" not in record
    assert set(record) == {
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
    }
