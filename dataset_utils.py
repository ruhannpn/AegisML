"""
dataset_utils.py
================
Shared data-loading utilities for the AI-Governed Multi-Agent Platform.
Imported by both test_planner.py and test_data_agent.py to avoid duplication.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Optional
import pandas as pd

ADULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]


import io
import json
from pathlib import Path
from typing import Any, Optional

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".tab",
    ".txt",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".pq",
    ".feather",
    ".arrow",
    ".xlsx",
    ".xls",
}


def is_supported_dataset_file(filename: Optional[str]) -> bool:
    """Return True if filename has an extension recognized by load_dataset_from_bytes."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def load_dataset_from_bytes(contents: bytes, filename: Optional[str] = None) -> pd.DataFrame:
    """
    Load a tabular DataFrame from in-memory file bytes, dispatching by extension
    with robust content-sniffing fallback.

    Supported formats:
    - CSV (.csv)
    - Tab-delimited text (.tsv, .tab, .txt)
    - JSON & JSON Lines (.json, .jsonl, .ndjson)
    - Apache Parquet (.parquet, .pq)
    - Apache Feather / Arrow IPC (.feather, .arrow)
    - Microsoft Excel (.xlsx, .xls)

    Raises
    ------
    ValueError
        If the file is empty, cannot be parsed, or the format is unsupported.
    """
    if not contents or len(contents) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    ext = Path(filename).suffix.lower() if filename else ""
    bio = io.BytesIO(contents)
    df: Optional[pd.DataFrame] = None
    parse_errors: list[str] = []

    if ext == ".csv":
        try:
            df = pd.read_csv(bio)
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    elif ext in {".tsv", ".tab"}:
        try:
            df = pd.read_csv(bio, sep="\t")
        except Exception as exc:
            raise ValueError(f"Failed to parse TSV file: {exc}") from exc

    elif ext == ".txt":
        try:
            df = pd.read_csv(bio, sep=None, engine="python")
        except Exception:
            bio.seek(0)
            try:
                df = pd.read_csv(bio, sep="\t")
            except Exception:
                bio.seek(0)
                try:
                    df = pd.read_csv(bio, sep=",")
                except Exception as exc:
                    raise ValueError(f"Failed to parse delimited text file: {exc}") from exc

    elif ext in {".parquet", ".pq"}:
        try:
            df = pd.read_parquet(bio)
        except Exception as exc:
            raise ValueError(f"Failed to parse Parquet file: {exc}") from exc

    elif ext in {".feather", ".arrow"}:
        try:
            df = pd.read_feather(bio)
        except Exception as exc:
            raise ValueError(f"Failed to parse Feather/Arrow file: {exc}") from exc

    elif ext in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(bio)
        except Exception as exc:
            raise ValueError(f"Failed to parse Excel file: {exc}") from exc

    elif ext in {".jsonl", ".ndjson"}:
        try:
            df = pd.read_json(bio, lines=True)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON Lines file: {exc}") from exc

    elif ext == ".json":
        try:
            raw_str = contents.decode("utf-8")
            parsed_json = json.loads(raw_str)
            if isinstance(parsed_json, dict) and "data" in parsed_json and isinstance(parsed_json["data"], list):
                df = pd.DataFrame(parsed_json["data"])
            elif isinstance(parsed_json, dict) and "records" in parsed_json and isinstance(parsed_json["records"], list):
                df = pd.DataFrame(parsed_json["records"])
            elif isinstance(parsed_json, list):
                df = pd.DataFrame(parsed_json)
            else:
                bio.seek(0)
                df = pd.read_json(bio)
        except Exception as e1:
            parse_errors.append(f"Standard JSON parse error: {e1}")
            bio.seek(0)
            try:
                df = pd.read_json(bio, lines=True)
            except Exception as e2:
                parse_errors.append(f"JSON Lines parse error: {e2}")
                raise ValueError(f"Failed to parse JSON file: {'; '.join(parse_errors)}") from e2

    else:
        # Sniff format from header / magic bytes
        if contents.startswith(b"PAR1"):
            try:
                df = pd.read_parquet(bio)
            except Exception as exc:
                raise ValueError(f"Failed to parse Parquet content: {exc}") from exc
        elif contents.startswith(b"ARROW1"):
            try:
                df = pd.read_feather(bio)
            except Exception as exc:
                raise ValueError(f"Failed to parse Feather/Arrow content: {exc}") from exc
        elif contents.startswith(b"PK\x03\x04"):
            try:
                df = pd.read_excel(bio)
            except Exception as exc:
                raise ValueError(f"Failed to parse Excel content: {exc}") from exc
        else:
            stripped = contents.lstrip()
            if stripped.startswith(b"{") or stripped.startswith(b"["):
                try:
                    df = pd.read_json(bio)
                except Exception:
                    bio.seek(0)
                    try:
                        df = pd.read_json(bio, lines=True)
                    except Exception:
                        pass
            if df is None:
                bio.seek(0)
                try:
                    df = pd.read_csv(bio)
                except Exception as exc:
                    formats_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                    raise ValueError(
                        f"Unsupported or unparseable format for file '{filename}'. "
                        f"Supported extensions: {formats_str}. (Parser error: {exc})"
                    ) from exc

    if ext and ext not in SUPPORTED_EXTENSIONS:
        formats_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported or unparseable format for file '{filename}'. "
            f"Supported extensions: {formats_str}."
        )

    if df is None or df.empty or len(df.columns) == 0:
        raise ValueError("Parsed dataset is empty or contains no columns.")

    # Ensure all column names are string labels
    df.columns = [str(c) for c in df.columns]
    return df


def suggest_dataset_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extract column list, shape, suggested target column, and suggested task type.
    """
    cols = list(df.columns)
    default_col = cols[-1] if cols else ""
    for preferred in ["income", "quality", "Unemployment", "target", "label", "class"]:
        matched = next((c for c in cols if c.lower() == preferred.lower()), None)
        if matched:
            default_col = matched
            break

    suggested_task = "classification"
    if default_col and default_col in df.columns:
        series = df[default_col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_float_dtype(series) or series.nunique() > 15:
                suggested_task = "regression"

    return {
        "columns": cols,
        "shape": {"rows": int(len(df)), "columns": int(len(cols))},
        "suggested_target": default_col,
        "suggested_task": suggested_task,
        "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
    }


def load_adult_dataset() -> pd.DataFrame:
    """
    Load the UCI Adult Income dataset from the UCI repository.

    Missing values are stored as ' ?' in the raw CSV. With skipinitialspace=True,
    the leading space is stripped BEFORE na_values is checked, converting ' ?' -> '?'.
    We therefore catch both forms in na_values to be safe.

    Returns
    -------
    pd.DataFrame with 15 columns and ~32,561 rows.
    """
    print("Loading UCI Adult Income dataset from UCI repository...")
    df = pd.read_csv(
        ADULT_URL,
        header=None,
        names=ADULT_COLUMNS,
        na_values=["?", " ?"],   # catch both pre- and post-strip forms
        skipinitialspace=True,
    )
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print(f"  Null counts:\n{null_counts}\n")
    else:
        print("  No nulls detected.\n")
    return df


load_tabular_data = load_dataset_from_bytes

