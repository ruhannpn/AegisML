"""
dataset_utils.py
================
Shared data-loading utilities for the AI-Governed Multi-Agent Platform.
Imported by both test_planner.py and test_data_agent.py to avoid duplication.
"""

from __future__ import annotations

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
