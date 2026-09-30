"""Serve risk-prioritization predictions from the notebook-trained bundle."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from contextlib import asynccontextmanager
from numbers import Real
from pathlib import Path
from typing import Any, NoReturn

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from starlette.datastructures import UploadFile

TARGET = "not.fully.paid"
FEATURES = [
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
INTEGER_FEATURES = {
    "credit.policy",
    "fico",
    "revol.bal",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
}
NUMERIC_FEATURES = set(FEATURES) - {"purpose"}
INPUT_ALIASES = {
    "credit_policy": "credit.policy",
    "int_rate": "int.rate",
    "log_annual_inc": "log.annual.inc",
    "days_with_cr_line": "days.with.cr.line",
    "revol_bal": "revol.bal",
    "revol_util": "revol.util",
    "inq_last_6mths": "inq.last.6mths",
    "delinq_2yrs": "delinq.2yrs",
    "pub_rec": "pub.rec",
}
DEFAULT_BUNDLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "credit_risk_model_bundle.joblib"
)


class LoanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)

    credit_policy: int | float | None = Field(alias="credit.policy")
    purpose: str | None
    int_rate: int | float | None = Field(alias="int.rate")
    installment: int | float | None
    log_annual_inc: int | float | None = Field(alias="log.annual.inc")
    dti: int | float | None
    fico: int | float | None
    days_with_cr_line: int | float | None = Field(alias="days.with.cr.line")
    revol_bal: int | float | None = Field(alias="revol.bal")
    revol_util: int | float | None = Field(alias="revol.util")
    inq_last_6mths: int | float | None = Field(alias="inq.last.6mths")
    delinq_2yrs: int | float | None = Field(alias="delinq.2yrs")
    pub_rec: int | float | None = Field(alias="pub.rec")

    @model_validator(mode="before")
    @classmethod
    def validate_input_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if TARGET in value:
            raise PydanticCustomError(
                "target_not_allowed",
                "El target no se acepta como variable de entrada.",
                {"field": TARGET, "value": value[TARGET]},
            )

        aliases = set(FEATURES)
        unexpected = set(value) - aliases
        if unexpected:
            field_name = sorted(unexpected)[0]
            raise PydanticCustomError(
                "unexpected_feature",
                "Variable no admitida.",
                {"field": field_name, "value": value[field_name]},
            )

        for field_name in NUMERIC_FEATURES:
            if field_name not in value or value[field_name] is None:
                continue
            field_value = value[field_name]
            if isinstance(field_value, bool) or not isinstance(field_value, Real):
                raise PydanticCustomError(
                    "invalid_numeric_type",
                    "Se requiere un valor numérico o null.",
                    {"field": field_name, "value": field_value},
                )
        if "purpose" in value and value["purpose"] is not None and not isinstance(
            value["purpose"], str
        ):
            raise PydanticCustomError(
                "invalid_purpose_type",
                "purpose debe ser una categoría de texto o null.",
                {"field": "purpose", "value": value["purpose"]},
            )
        return value

    @field_validator("*")
    @classmethod
    def require_finite_numbers(cls, value: Any, info: ValidationInfo) -> Any:
        if info.field_name == "purpose" or value is None:
            return value
        if not math.isfinite(float(value)):
            raise PydanticCustomError(
                "non_finite_number",
                "El valor debe ser finito.",
                {
                    "field": INPUT_ALIASES.get(info.field_name, info.field_name),
                    "value": value,
                },
            )
        return value


class PrioritizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[LoanRecord] = Field(min_length=1)
    review_capacity: int | None = Field(default=None, ge=1)


def is_multipart_request(request: Request) -> bool:
    return request.headers.get("content-type", "").split(";", 1)[0].lower() == (
        "multipart/form-data"
    )


async def read_json_request(request: Request) -> Any:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/json" and not content_type.endswith("+json"):
        raise HTTPException(
            status_code=415,
            detail="Envía JSON o un archivo CSV en una solicitud multipart/form-data.",
        )
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail="El cuerpo JSON no es válido.") from exc


def raise_request_validation_error(exc: ValidationError) -> NoReturn:
    raise RequestValidationError(exc.errors()) from exc


def parse_csv_records(content: bytes) -> list[LoanRecord]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="El CSV debe estar codificado como UTF-8.",
        ) from exc

    try:
        reader = csv.DictReader(io.StringIO(text), strict=True)
        if reader.fieldnames is None:
            raise HTTPException(
                status_code=422,
                detail="El CSV debe incluir una fila de encabezados.",
            )
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise HTTPException(
                status_code=422,
                detail="El CSV contiene encabezados duplicados.",
            )

        missing_fields = sorted(set(FEATURES) - set(reader.fieldnames))
        unexpected_fields = sorted(set(reader.fieldnames) - set(FEATURES))
        if missing_fields or unexpected_fields:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_fields": missing_fields,
                    "unexpected_fields": unexpected_fields,
                    "message": "Los encabezados CSV deben coincidir con las features del modelo.",
                },
            )

        records = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "row": row_number,
                        "message": "La fila CSV contiene más valores que encabezados.",
                    },
                )
            values: dict[str, Any] = {}
            for field_name in FEATURES:
                raw_value = row[field_name]
                if field_name == "purpose":
                    values[field_name] = (
                        raw_value.strip()
                        if raw_value is not None and raw_value.strip()
                        else None
                    )
                    continue

                if raw_value is None or not raw_value.strip():
                    values[field_name] = None
                    continue
                try:
                    values[field_name] = float(raw_value)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "row": row_number,
                            "field": field_name,
                            "value": raw_value,
                            "message": "Se requiere un valor numérico o una celda vacía.",
                        },
                    ) from exc

            try:
                records.append(LoanRecord.model_validate(values))
            except ValidationError as exc:
                raise_request_validation_error(exc)
    except csv.Error as exc:
        raise HTTPException(
            status_code=422,
            detail=f"El CSV no tiene un formato válido: {exc}",
        ) from exc
    return records


async def read_csv_upload(request: Request) -> tuple[list[LoanRecord], Any]:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(
            status_code=422,
            detail="Adjunta el CSV en el campo multipart 'file'.",
        )
    return parse_csv_records(await upload.read()), form.get("review_capacity")


def load_model_bundle(bundle_path: Path | str) -> dict[str, Any]:
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el artefacto del modelo: {path}. "
            "Ejecuta el notebook de modelado para generarlo."
        )
    bundle = joblib.load(path)
    required_keys = {"model", "feature_names", "threshold", "metadata", "target"}
    missing_keys = required_keys - set(bundle)
    if missing_keys:
        raise ValueError(
            f"El artefacto del modelo no contiene claves obligatorias: "
            f"{sorted(missing_keys)}"
        )
    if bundle["target"] != TARGET:
        raise ValueError(f"El artefacto debe usar el target {TARGET!r}.")
    if not hasattr(bundle["model"], "predict_proba"):
        raise TypeError("El modelo del artefacto debe implementar predict_proba.")
    threshold = float(bundle["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("El artefacto contiene un umbral fuera de [0, 1].")
    schema = bundle["metadata"].get("input_schema", {})
    if set(schema) != set(FEATURES):
        raise ValueError("El esquema de entrada del artefacto no coincide con FEATURES.")
    return bundle


def validate_record(
    record: LoanRecord, bundle: dict[str, Any]
) -> dict[str, int | float | str | None]:
    values = record.model_dump(by_alias=True)
    schema = bundle["metadata"]["input_schema"]
    categories = schema["purpose"]["categories"]

    purpose = values["purpose"]
    if purpose is not None and purpose not in categories:
        raise HTTPException(
            status_code=422,
            detail={
                "field": "purpose",
                "value": purpose,
                "allowed_values": categories,
                "message": "Categoría purpose no observada en los datos de entrenamiento.",
            },
        )

    for field_name in NUMERIC_FEATURES:
        value = values[field_name]
        if value is None:
            continue
        numeric_value = float(value)
        bounds = schema[field_name]
        if numeric_value < bounds["min"] or numeric_value > bounds["max"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "field": field_name,
                    "value": value,
                    "range": [bounds["min"], bounds["max"]],
                    "message": (
                        f"Valor fuera del rango empírico observado para {field_name}."
                    ),
                },
            )
        if field_name in INTEGER_FEATURES and not numeric_value.is_integer():
            raise HTTPException(
                status_code=422,
                detail={
                    "field": field_name,
                    "value": value,
                    "range": [bounds["min"], bounds["max"]],
                    "message": f"{field_name} debe ser un número entero.",
                },
            )
        values[field_name] = (
            int(numeric_value)
            if field_name in INTEGER_FEATURES
            else numeric_value
        )

    return values


def score_records(
    records: list[LoanRecord], bundle: dict[str, Any]
) -> list[dict[str, Any]]:
    if not records:
        raise HTTPException(
            status_code=422,
            detail={
                "field": "records",
                "value": records,
                "message": "Se requiere al menos un préstamo.",
            },
        )

    validated = [validate_record(record, bundle) for record in records]
    frame = pd.DataFrame(validated, columns=FEATURES)
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    threshold = float(bundle["threshold"])
    results = []
    for values, probability in zip(validated, probabilities, strict=True):
        prediction = int(float(probability) >= threshold)
        results.append(
            {
                "probability_not_fully_paid": float(probability),
                "probability_fully_paid": float(1.0 - probability),
                "prediction": prediction,
                "prediction_label": "review" if prediction else "no_review",
                "threshold": threshold,
                "record": values,
            }
        )
    return results


def sort_and_rank(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        enumerate(predictions),
        key=lambda item: (-item[1]["probability_not_fully_paid"], item[0]),
    )
    ranked = []
    for rank, (_, prediction) in enumerate(ordered, start=1):
        ranked.append({"rank": rank, **prediction})
    return ranked


def create_app(bundle_path: Path | str | None = None) -> FastAPI:
    resolved_bundle_path = Path(
        bundle_path
        or os.environ.get("CREDIT_RISK_MODEL_BUNDLE", DEFAULT_BUNDLE_PATH)
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model_bundle = load_model_bundle(resolved_bundle_path)
        yield

    application = FastAPI(
        title="Credit Risk Review Prioritization API",
        description=(
            "Estima P(not.fully.paid = 1) para priorizar préstamos para revisión "
            "humana. No aprueba ni rechaza préstamos."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        def json_safe(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return repr(value)
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return str(value)

        for error in exc.errors():
            context = error.get("ctx", {})
            if "field" in context:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "field": context["field"],
                            "value": json_safe(context.get("value")),
                            "message": error["msg"],
                        }
                    },
                )
        return JSONResponse(
            status_code=422,
            content={"detail": json_safe(exc.errors())},
        )

    @application.get("/health")
    def health() -> dict[str, Any]:
        bundle = application.state.model_bundle
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": bundle["metadata"]["model_name"],
            "target": bundle["target"],
            "threshold": float(bundle["threshold"]),
        }

    @application.post("/predict")
    def predict(record: LoanRecord) -> dict[str, Any]:
        bundle = application.state.model_bundle
        return score_records([record], bundle)[0]

    @application.post(
        "/predict-file",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "oneOf": [
                                {"$ref": "#/components/schemas/LoanRecord"},
                                {
                                    "type": "array",
                                    "items": {
                                        "$ref": "#/components/schemas/LoanRecord"
                                    },
                                },
                            ]
                        }
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"}
                            },
                        }
                    },
                },
            }
        },
    )
    async def predict_file(request: Request) -> dict[str, Any]:
        if is_multipart_request(request):
            records, _ = await read_csv_upload(request)
        else:
            payload = await read_json_request(request)
            try:
                validated_payload = TypeAdapter(
                    LoanRecord | list[LoanRecord]
                ).validate_python(payload)
            except ValidationError as exc:
                raise_request_validation_error(exc)
            records = (
                validated_payload
                if isinstance(validated_payload, list)
                else [validated_payload]
            )
        scored = score_records(records, application.state.model_bundle)
        ranked = sort_and_rank(scored)
        return {"count": len(ranked), "predictions": ranked}

    @application.post(
        "/prioritize",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["records"],
                            "properties": {
                                "records": {
                                    "type": "array",
                                    "items": {
                                        "$ref": "#/components/schemas/LoanRecord"
                                    },
                                    "minItems": 1,
                                },
                                "review_capacity": {
                                    "anyOf": [
                                        {"type": "integer", "minimum": 1},
                                        {"type": "null"},
                                    ]
                                },
                            },
                        }
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "review_capacity": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                            },
                        }
                    },
                },
            }
        },
    )
    async def prioritize(request: Request) -> dict[str, Any]:
        if is_multipart_request(request):
            records, review_capacity = await read_csv_upload(request)
            payload = {"records": records, "review_capacity": review_capacity}
        else:
            payload = await read_json_request(request)
        try:
            prioritize_request = PrioritizeRequest.model_validate(payload)
        except ValidationError as exc:
            raise_request_validation_error(exc)

        bundle = application.state.model_bundle
        scored = score_records(prioritize_request.records, bundle)
        ranked = sort_and_rank(scored)
        effective_capacity = (
            min(prioritize_request.review_capacity, len(ranked))
            if prioritize_request.review_capacity is not None
            else None
        )
        for prediction in ranked:
            if effective_capacity is not None:
                is_review = prediction["rank"] <= effective_capacity
                prediction["prediction"] = int(is_review)
                prediction["prediction_label"] = (
                    "review" if is_review else "no_review"
                )
            prediction["review_priority"] = (
                "review"
                if effective_capacity is not None
                and prediction["rank"] <= effective_capacity
                else (
                    "no_review"
                    if effective_capacity is not None
                    else prediction["prediction_label"]
                )
            )
        return {
            "count": len(ranked),
            "review_capacity": effective_capacity,
            "predictions": ranked,
        }

    @application.get("/sample-json")
    def sample_json() -> dict[str, Any]:
        return dict(application.state.model_bundle["metadata"]["sample_record"])

    return application


app = create_app()
