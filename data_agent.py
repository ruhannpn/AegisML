"""
data_agent.py
=============
Step 2 of the AI-Governed Multi-Agent Platform.

Exposes a single public function:
    run_data_agent(df, plan, target_column, task_type) -> dict

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic.
  - All preprocessing logic is in pre-built functions.
  - The plan's free-text steps are parsed ONLY via keyword matching
    to log hints — never to generate or execute code.

PREPROCESSING PIPELINE (applied in order):
  1. Drop feature columns with > 50% missing
  2. Drop rows with null target (labels cannot be imputed)
  3. Impute remaining nulls:  numeric → median, categorical → mode
  4. Label-encode target column (classification only)
  5. Encode categoricals:
       - One-hot encode if unique values < OHE_CARDINALITY_LIMIT (10)
       - Frequency encode (value → relative frequency) otherwise
  6. StandardScale all numeric, non-boolean feature columns

QUALITY CHECK LOGIC:
  quality_check_passed = True only if ALL of:
    1. missing_pct_after_cleaning < 5%        (spec condition)
    2. No column has any unresolved nulls      (spec condition)
    3. rows_dropped_pct < 30%                 (extension: guards against
       null-label data decimation — if >30% of rows have no label,
       the remaining training set is untrustworthy)

  If quality_check_passed = False, the LangGraph wiring (Step 3) will
  route execution back to the Planner Agent for a revised plan.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Module-level thresholds — easy to tune without hunting through logic
# ---------------------------------------------------------------------------

COLUMN_DROP_NULL_THRESHOLD = 0.50   # drop column if null_pct > 50%
COLUMN_HIGH_NULL_WARNING_THRESHOLD = 0.35 # unhandled if null_pct > 35% and not dropped by plan
ROW_DROP_RATIO_LIMIT = 0.30         # quality fails if > 30% of original rows dropped
OHE_CARDINALITY_LIMIT = 10          # OHE if unique < 10, else frequency encode
NULL_PCT_QUALITY_LIMIT = 5.0        # quality fails if missing_pct >= 5%

# ---------------------------------------------------------------------------
# Plan parsing (keyword matching only — no code generation)
# ---------------------------------------------------------------------------


def _parse_plan_steps(steps: list[str], columns: list[str]) -> dict:
    """
    Extract processing hints from the plan's free-text preprocessing steps
    via keyword matching.
    """
    hints = {
        "smote_mentioned": False,
        "class_weight_mentioned": False,
        "frequency_encoding_mentioned": False,
        "explicit_drop_columns": [],
    }
    for step in steps:
        sl = step.lower()
        if any(kw in sl for kw in ("smote", "oversample", "adasyn", "oversampl")):
            hints["smote_mentioned"] = True
        if any(kw in sl for kw in ("class weight", "class_weight")):
            hints["class_weight_mentioned"] = True
        if "frequency" in sl:
            hints["frequency_encoding_mentioned"] = True
        if any(kw in sl for kw in ("drop", "remove", "exclude")):
            for col in columns:
                if col.lower() in sl:
                    hints["explicit_drop_columns"].append(col)
    return hints


# ---------------------------------------------------------------------------
# Pre-built preprocessing functions (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _drop_high_null_columns(
    df: pd.DataFrame,
    target_column: str,
    explicit_drop_columns: list[str],
    actions: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop feature columns where null_pct > COLUMN_DROP_NULL_THRESHOLD, or columns
    explicitly instructed to be dropped in the plan's preprocessing steps.
    """
    dropped: list[str] = []
    for col in list(df.columns):
        if col == target_column:
            continue
        null_pct = df[col].isnull().sum() / len(df)
        if null_pct > COLUMN_DROP_NULL_THRESHOLD or col in explicit_drop_columns:
            reason = (
                f"plan instruction ('drop {col}')"
                if col in explicit_drop_columns
                else f"{null_pct * 100:.1f}% missing > {COLUMN_DROP_NULL_THRESHOLD * 100:.0f}% threshold"
            )
            actions.append(f"Dropped column '{col}' ({reason})")
            dropped.append(col)
    if not dropped:
        actions.append("No columns exceeded drop threshold or specified for dropping — all retained")
    return df.drop(columns=dropped), dropped


def _drop_null_target_rows(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where the target is null.
    Labels cannot be imputed — missing labels are untrainable.
    """
    n_before = len(df)
    df = df.dropna(subset=[target_column])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        actions.append(
            f"Dropped {n_dropped:,} rows with null target '{target_column}' "
            f"({n_dropped / n_before * 100:.1f}% of rows)"
        )
    return df, n_dropped


def _impute_columns(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> pd.DataFrame:
    """
    Fill nulls in all remaining feature columns:
      - Numeric  → column median
      - Object   → column mode (most frequent non-null value)
    """
    for col in df.columns:
        if col == target_column:
            continue
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            actions.append(
                f"Imputed '{col}' with median {fill_val:.4g} "
                f"({null_count:,} nulls filled)"
            )
        else:
            mode_series = df[col].mode()
            if mode_series.empty:
                # Guard: column is entirely null below the drop threshold.
                # Shouldn't occur with 50% drop rule, but log explicitly.
                actions.append(
                    f"WARNING: Cannot impute '{col}' — no non-null values found. "
                    f"Column will have unresolved nulls."
                )
                continue
            fill_val = str(mode_series.iloc[0])
            df[col] = df[col].fillna(fill_val)
            actions.append(
                f"Imputed '{col}' with mode '{fill_val}' "
                f"({null_count:,} nulls filled)"
            )
    return df


def _label_encode_target(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> pd.DataFrame:
    """
    Convert a categorical target to integer codes (0, 1, …).
    Classes are sorted lexicographically for deterministic mapping.
    Numeric targets are left unchanged.
    """
    if pd.api.types.is_numeric_dtype(df[target_column]):
        return df  # already numeric (regression or previously encoded)

    classes = sorted(df[target_column].dropna().unique())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[target_column] = df[target_column].map(mapping)
    mapping_str = ", ".join(f"'{k}'→{v}" for k, v in mapping.items())
    actions.append(
        f"Label-encoded target '{target_column}': {mapping_str}"
    )
    return df


def _encode_categoricals(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> pd.DataFrame:
    """
    Encode all object-dtype feature columns:
      - One-hot encoding  if nunique < OHE_CARDINALITY_LIMIT
      - Frequency encoding otherwise (value → relative frequency in [0, 1])
    """
    object_cols = [
        col for col in df.columns
        if col != target_column and pd.api.types.is_object_dtype(df[col])
    ]

    ohe_cols = [c for c in object_cols if df[c].nunique() < OHE_CARDINALITY_LIMIT]
    freq_cols = [c for c in object_cols if df[c].nunique() >= OHE_CARDINALITY_LIMIT]

    # Capture nunique BEFORE get_dummies reshapes the df
    ohe_nunique = {col: df[col].nunique() for col in ohe_cols}

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=bool)
        for col in ohe_cols:
            n = ohe_nunique[col]
            actions.append(
                f"One-hot encoded '{col}' "
                f"({n} unique values → {n} new boolean columns)"
            )

    for col in freq_cols:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[col] = df[col].map(freq_map).astype(float)
        actions.append(
            f"Frequency-encoded '{col}' "
            f"({len(freq_map)} unique values → relative frequency [0.0–1.0])"
        )

    return df


def _scale_numeric_features(
    df: pd.DataFrame,
    target_column: str,
    actions: list[str],
) -> pd.DataFrame:
    """
    Apply StandardScaler to all numeric, non-boolean feature columns.
    Skips: target column and boolean OHE columns (True/False — no scaling needed).
    """
    numeric_feature_cols = [
        col for col in df.columns
        if col != target_column
        and pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]
    if not numeric_feature_cols:
        actions.append("No numeric feature columns to scale")
        return df

    scaler = StandardScaler()
    df[numeric_feature_cols] = scaler.fit_transform(df[numeric_feature_cols])
    actions.append(
        f"Applied StandardScaler to {len(numeric_feature_cols)} "
        f"numeric feature(s): {numeric_feature_cols}"
    )
    return df


def _compute_quality_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    rows_dropped: int,
    columns_dropped: list[str],
) -> tuple[dict, bool]:
    """
    Compute quality_report and quality_check_passed.

    quality_check_passed = True only when ALL conditions hold:
      1. missing_pct_after_cleaning < NULL_PCT_QUALITY_LIMIT (5%)
      2. No column in cleaned_df has any null values
      3. rows_dropped_pct < ROW_DROP_RATIO_LIMIT * 100 (30%)
      4. No feature column with > 35% missing values remains unhandled
         (if missing > 35% and not dropped by default threshold or plan, quality fails)
    """
    n_original = len(original_df)
    null_cells = int(cleaned_df.isnull().sum().sum())
    total_cells = cleaned_df.size
    missing_pct = (
        round(float(null_cells / total_cells * 100), 4) if total_cells > 0 else 0.0
    )

    unresolved_null_cols = [
        col for col in cleaned_df.columns if cleaned_df[col].isnull().any()
    ]

    # Feature columns with > 35% missing in original data that were not dropped
    unhandled_high_null = []
    for col in original_df.columns:
        if col == target_column or col in columns_dropped:
            continue
        col_null_pct = original_df[col].isnull().sum() / len(original_df)
        if col_null_pct >= COLUMN_HIGH_NULL_WARNING_THRESHOLD:
            unhandled_high_null.append(f"{col} ({col_null_pct * 100:.1f}% missing)")

    # Class balance ratio on the cleaned target (classification only)
    class_balance_ratio = None
    if task_type == "classification" and target_column in cleaned_df.columns:
        vc = cleaned_df[target_column].value_counts()
        if len(vc) >= 2:
            class_balance_ratio = round(float(vc.iloc[0]) / float(vc.iloc[-1]), 4)

    rows_dropped_pct = round(
        rows_dropped / n_original * 100, 2
    ) if n_original > 0 else 0.0

    quality_report = {
        "missing_pct_after_cleaning": missing_pct,
        "class_balance_ratio": class_balance_ratio,
        "rows_dropped": rows_dropped,
        "rows_dropped_pct": rows_dropped_pct,
        "columns_dropped": columns_dropped,
        "unresolved_null_columns": unresolved_null_cols,  # diagnostic
        "unhandled_high_null_columns": unhandled_high_null,
    }

    quality_check_passed = (
        missing_pct < NULL_PCT_QUALITY_LIMIT           # spec condition 1
        and len(unresolved_null_cols) == 0              # spec condition 2
        and rows_dropped_pct < ROW_DROP_RATIO_LIMIT * 100  # condition 3
        and len(unhandled_high_null) == 0               # condition 4
    )

    return quality_report, quality_check_passed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_data_agent(
    df: pd.DataFrame,
    plan: dict,
    target_column: str,
    task_type: str,
) -> dict:
    """
    Deterministic data cleaning pipeline. Zero LLM calls.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    df = df.copy()
    original_df = df.copy()
    actions: list[str] = []
    columns_dropped_total: list[str] = []

    # Step 0 — Parse plan hints (keyword matching; no code generation)
    hints = _parse_plan_steps(
        plan.get("recommended_preprocessing_steps", []),
        list(df.columns),
    )

    # Step 1 — Drop columns > 50% missing or explicitly dropped by plan
    df, dropped = _drop_high_null_columns(
        df,
        target_column,
        hints["explicit_drop_columns"],
        actions,
    )
    columns_dropped_total.extend(dropped)

    # Step 2 — Drop rows with null target
    df, rows_dropped = _drop_null_target_rows(df, target_column, actions)

    # Step 3 — Impute remaining nulls
    df = _impute_columns(df, target_column, actions)

    # Step 4 — Label-encode target (classification only)
    if task_type == "classification":
        df = _label_encode_target(df, target_column, actions)

    # Steps 5 & 6 — Encode categoricals
    df = _encode_categoricals(df, target_column, actions)

    # Step 7 — Scale numeric features
    df = _scale_numeric_features(df, target_column, actions)

    # Step 8 — Imbalance note (flag only — SMOTE deferred to Training Agent)
    if task_type == "classification":
        vc = original_df[target_column].dropna().value_counts()
        if len(vc) >= 2:
            ratio = float(vc.iloc[0]) / float(vc.iloc[-1])
            if ratio > 1.5 or hints["smote_mentioned"] or hints["class_weight_mentioned"]:
                actions.append(
                    f"NOTE: Class imbalance — {ratio:.2f}:1 ratio "
                    f"(majority:minority). SMOTE/class_weight deferred to "
                    f"Training Agent."
                )

    # Step 9 — Quality check
    quality_report, quality_check_passed = _compute_quality_report(
        original_df=original_df,
        cleaned_df=df,
        target_column=target_column,
        task_type=task_type,
        rows_dropped=rows_dropped,
        columns_dropped=columns_dropped_total,
    )

    return {
        "cleaned_df": df,
        "quality_check_passed": quality_check_passed,
        "quality_report": quality_report,
        "actions_taken": actions,
    }
