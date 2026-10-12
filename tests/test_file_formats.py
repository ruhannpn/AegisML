"""
tests/test_file_formats.py
==========================
Tests for multi-format tabular dataset ingestion:
- CSV, TSV, TXT
- JSON & JSON Lines (.json, .jsonl, .ndjson)
- Apache Parquet (.parquet, .pq)
- Apache Feather / Arrow (.feather, .arrow)
- Microsoft Excel (.xlsx)
- Content-sniffing fallback & error handling
- FastAPI inspection and start pipeline validation endpoints
"""

import io
import json
import pandas as pd
import pytest
from starlette.testclient import TestClient

from dataset_utils import (
    SUPPORTED_EXTENSIONS,
    is_supported_dataset_file,
    load_dataset_from_bytes,
    suggest_dataset_metadata,
)
from server import app


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 38, 42, 55],
        "workclass": ["Private", "Self-emp", "Private", "Federal-gov"],
        "education": ["Bachelors", "Masters", "HS-grad", "Doctorate"],
        "income": ["<=50K", ">50K", "<=50K", ">50K"],
    })


@pytest.fixture
def regression_df():
    return pd.DataFrame({
        "feature_1": [1.2, 3.4, 5.6, 7.8],
        "feature_2": [10, 20, 30, 40],
        "target": [15.2, 28.4, 42.1, 59.9],
    })


# ---------------------------------------------------------------------------
# Unit tests: load_dataset_from_bytes across formats
# ---------------------------------------------------------------------------

def test_load_csv(sample_df):
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    loaded = load_dataset_from_bytes(csv_bytes, "adult_sample.csv")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_tsv(sample_df):
    tsv_bytes = sample_df.to_csv(index=False, sep="\t").encode("utf-8")
    loaded = load_dataset_from_bytes(tsv_bytes, "adult_sample.tsv")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_tab(sample_df):
    tsv_bytes = sample_df.to_csv(index=False, sep="\t").encode("utf-8")
    loaded = load_dataset_from_bytes(tsv_bytes, "adult_sample.tab")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_txt_delimited(sample_df):
    tsv_bytes = sample_df.to_csv(index=False, sep="\t").encode("utf-8")
    loaded = load_dataset_from_bytes(tsv_bytes, "adult_sample.txt")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_json_records(sample_df):
    json_bytes = sample_df.to_json(orient="records").encode("utf-8")
    loaded = load_dataset_from_bytes(json_bytes, "adult_sample.json")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_json_lines(sample_df):
    jsonl_bytes = sample_df.to_json(orient="records", lines=True).encode("utf-8")
    loaded = load_dataset_from_bytes(jsonl_bytes, "adult_sample.jsonl")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_ndjson(sample_df):
    ndjson_bytes = sample_df.to_json(orient="records", lines=True).encode("utf-8")
    loaded = load_dataset_from_bytes(ndjson_bytes, "adult_sample.ndjson")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_json_nested_data_key(sample_df):
    raw_dict = {"data": sample_df.to_dict(orient="records")}
    json_bytes = json.dumps(raw_dict).encode("utf-8")
    loaded = load_dataset_from_bytes(json_bytes, "dataset.json")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_parquet(sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)
    loaded = load_dataset_from_bytes(buf.getvalue(), "adult_sample.parquet")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_pq_extension(sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)
    loaded = load_dataset_from_bytes(buf.getvalue(), "adult_sample.pq")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_feather(sample_df):
    buf = io.BytesIO()
    sample_df.to_feather(buf)
    loaded = load_dataset_from_bytes(buf.getvalue(), "adult_sample.feather")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_arrow_extension(sample_df):
    buf = io.BytesIO()
    sample_df.to_feather(buf)
    loaded = load_dataset_from_bytes(buf.getvalue(), "adult_sample.arrow")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_excel(sample_df):
    buf = io.BytesIO()
    sample_df.to_excel(buf, index=False)
    loaded = load_dataset_from_bytes(buf.getvalue(), "adult_sample.xlsx")
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


# ---------------------------------------------------------------------------
# Sniffing fallback & error handling
# ---------------------------------------------------------------------------

def test_sniff_parquet_without_extension(sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)
    # No extension provided
    loaded = load_dataset_from_bytes(buf.getvalue(), "uploaded_blob")
    assert list(loaded.columns) == list(sample_df.columns)


def test_sniff_feather_without_extension(sample_df):
    buf = io.BytesIO()
    sample_df.to_feather(buf)
    loaded = load_dataset_from_bytes(buf.getvalue(), "uploaded_blob")
    assert list(loaded.columns) == list(sample_df.columns)


def test_sniff_json_without_extension(sample_df):
    json_bytes = sample_df.to_json(orient="records").encode("utf-8")
    loaded = load_dataset_from_bytes(json_bytes, "uploaded_blob")
    assert list(loaded.columns) == list(sample_df.columns)


def test_empty_bytes_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        load_dataset_from_bytes(b"", "data.csv")


def test_unsupported_file_extension():
    with pytest.raises(ValueError, match="Unsupported or unparseable"):
        load_dataset_from_bytes(b"some random binary\x00\x01\x02", "image.png")


def test_corrupted_json_raises_value_error():
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        load_dataset_from_bytes(b"[{'unclosed':", "data.json")


def test_is_supported_dataset_file():
    assert is_supported_dataset_file("data.csv") is True
    assert is_supported_dataset_file("DATA.CSV") is True
    assert is_supported_dataset_file("data.parquet") is True
    assert is_supported_dataset_file("data.xlsx") is True
    assert is_supported_dataset_file("data.jsonl") is True
    assert is_supported_dataset_file("model.joblib") is False
    assert is_supported_dataset_file("") is False
    assert is_supported_dataset_file(None) is False


# ---------------------------------------------------------------------------
# Metadata suggestions
# ---------------------------------------------------------------------------

def test_suggest_dataset_metadata_classification(sample_df):
    meta = suggest_dataset_metadata(sample_df)
    assert meta["columns"] == ["age", "workclass", "education", "income"]
    assert meta["shape"] == {"rows": 4, "columns": 4}
    assert meta["suggested_target"] == "income"
    assert meta["suggested_task"] == "classification"


def test_suggest_dataset_metadata_regression(regression_df):
    meta = suggest_dataset_metadata(regression_df)
    assert meta["suggested_target"] == "target"
    assert meta["suggested_task"] == "regression"


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_endpoint_inspect_dataset_parquet(client, sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)

    response = client.post(
        "/api/dataset/inspect",
        files={"file": ("adult.parquet", buf.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "adult.parquet"
    assert data["columns"] == ["age", "workclass", "education", "income"]
    assert data["suggested_target"] == "income"
    assert data["suggested_task"] == "classification"


def test_endpoint_inspect_dataset_excel(client, regression_df):
    buf = io.BytesIO()
    regression_df.to_excel(buf, index=False)

    response = client.post(
        "/api/dataset/inspect",
        files={"file": ("housing.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "housing.xlsx"
    assert data["columns"] == ["feature_1", "feature_2", "target"]
    assert data["suggested_target"] == "target"
    assert data["suggested_task"] == "regression"


def test_endpoint_inspect_dataset_invalid_file(client):
    response = client.post(
        "/api/dataset/inspect",
        files={"file": ("corrupt.parquet", b"not a parquet file", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_endpoint_start_pipeline_rejects_missing_target_across_formats(client, sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)

    response = client.post(
        "/api/pipeline/start",
        files={"file": ("adult.parquet", buf.getvalue(), "application/octet-stream")},
        data={"target_column": "non_existent_column", "task_type": "classification"},
    )
    assert response.status_code == 400
    assert "not found in dataset headers" in response.json()["detail"]


def test_endpoint_start_pipeline_rejects_missing_protected_attribute(client, sample_df):
    buf = io.BytesIO()
    sample_df.to_parquet(buf, index=False)

    response = client.post(
        "/api/pipeline/start",
        files={"file": ("adult.parquet", buf.getvalue(), "application/octet-stream")},
        data={
            "target_column": "income",
            "task_type": "classification",
            "protected_attributes": "unknown_col,also_fake",
        },
    )
    assert response.status_code == 400
    assert "Protected attribute column(s) not found in dataset headers" in response.json()["detail"]
