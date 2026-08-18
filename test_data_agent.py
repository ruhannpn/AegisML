"""
test_data_agent.py
==================
Standalone test for run_data_agent().

TEST 1 — Normal path:
  Loads UCI Adult Income dataset, calls the REAL plan_pipeline() to get
  a live plan, then runs run_data_agent() and verifies the output.

TEST 2 — Failure path:
  Corrupts the dataset:
    • Nulls injected into 35% of target rows  → rows_dropped_pct > 30%
    • 60% nulls in 3 feature columns          → those columns dropped
  Confirms quality_check_passed = False.
  Uses a dummy plan (no API call needed for the failure path).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from dataset_utils import load_adult_dataset
from planner_agent import plan_pipeline
from data_agent import run_data_agent


# ---------------------------------------------------------------------------
# TEST 1 — Normal path
# ---------------------------------------------------------------------------


def run_normal_test() -> None:
    print("=" * 60)
    print("TEST 1: Normal path — UCI Adult Income dataset")
    print("=" * 60)

    df = load_adult_dataset()

    print("Calling plan_pipeline() to get the real plan...")
    plan = plan_pipeline(df, target_column="income", task_type="classification")
    print("Preprocessing steps from plan:")
    for step in plan["recommended_preprocessing_steps"]:
        print(f"  • {step}")

    print("\nRunning run_data_agent()...")
    result = run_data_agent(
        df=df,
        plan=plan,
        target_column="income",
        task_type="classification",
    )

    cleaned_df = result["cleaned_df"]
    quality_report = result["quality_report"]
    actions = result["actions_taken"]

    print("\n--- Actions Taken ---")
    for action in actions:
        print(f"  • {action}")

    print("\n--- Quality Report ---")
    # Exclude the diagnostics list for cleaner JSON print
    printable_report = {
        k: v for k, v in quality_report.items()
        if k != "unresolved_null_columns"
    }
    print(json.dumps(printable_report, indent=2))

    print(f"\ncleaned_df shape  : {cleaned_df.shape}")
    print(f"quality_check_passed : {result['quality_check_passed']}")

    # --- Assertions ---
    print("\nRunning assertions...")

    assert result["quality_check_passed"] is True, (
        f"Expected quality_check_passed=True. Report: {quality_report}"
    )
    print("  [PASS] quality_check_passed is True")

    null_counts = cleaned_df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"cleaned_df still has nulls in: {cols_with_nulls.to_dict()}"
    )
    print("  [PASS] cleaned_df has zero nulls in all columns")

    assert "income" in cleaned_df.columns, "Target column 'income' missing"
    assert cleaned_df["income"].isin([0, 1]).all(), (
        "Target 'income' should be label-encoded to 0/1"
    )
    print("  [PASS] Target 'income' label-encoded to 0/1")

    assert cleaned_df.shape[0] > 0, "cleaned_df is empty (no rows)"
    assert cleaned_df.shape[1] > 1, "cleaned_df has no feature columns"
    print(f"  [PASS] cleaned_df: {cleaned_df.shape[0]:,} rows, "
          f"{cleaned_df.shape[1]} columns")

    assert quality_report["missing_pct_after_cleaning"] == 0.0, (
        f"Expected 0.0% missing, got {quality_report['missing_pct_after_cleaning']}%"
    )
    print("  [PASS] missing_pct_after_cleaning == 0.0%")

    assert quality_report["class_balance_ratio"] is not None, (
        "class_balance_ratio should be set for classification"
    )
    print(f"  [PASS] class_balance_ratio = {quality_report['class_balance_ratio']}")

    assert len(actions) > 0, "actions_taken should not be empty"
    print(f"  [PASS] {len(actions)} actions logged")

    print("\n✅ TEST 1 PASSED\n")


# ---------------------------------------------------------------------------
# TEST 2 — Failure path
# ---------------------------------------------------------------------------


def run_failure_test() -> None:
    print("=" * 60)
    print("TEST 2: Failure path — corrupted dataset")
    print("=" * 60)

    df = load_adult_dataset()
    corrupt_df = df.copy()

    np.random.seed(42)

    # Null out 35% of target rows → forces rows_dropped_pct > 30%
    # → quality_check_passed = False (rows_dropped extended condition)
    target_null_idx = np.random.choice(
        corrupt_df.index, size=int(len(corrupt_df) * 0.35), replace=False
    )
    corrupt_df.loc[target_null_idx, "income"] = np.nan

    # Also inject 60% nulls into 3 feature columns → they'll be dropped (> 50%)
    for col in ["workclass", "occupation", "native-country"]:
        col_null_idx = np.random.choice(
            corrupt_df.index, size=int(len(corrupt_df) * 0.60), replace=False
        )
        corrupt_df.loc[col_null_idx, col] = np.nan

    print(f"Corrupted df statistics:")
    print(f"  income null count   : {corrupt_df['income'].isnull().sum():,} "
          f"({corrupt_df['income'].isnull().mean() * 100:.1f}%)")
    print(f"  workclass null count: {corrupt_df['workclass'].isnull().sum():,} "
          f"({corrupt_df['workclass'].isnull().mean() * 100:.1f}%)")
    print(f"  occupation null count: {corrupt_df['occupation'].isnull().sum():,} "
          f"({corrupt_df['occupation'].isnull().mean() * 100:.1f}%)")

    # Dummy plan — no LLM call needed for the failure test
    dummy_plan = {
        "data_quality_concerns": [
            "High null rate in target 'income': 35.0% missing",
            "High null rate in 'workclass': 60.0% missing",
            "High null rate in 'occupation': 60.0% missing",
            "High null rate in 'native-country': 60.0% missing",
        ],
        "recommended_preprocessing_steps": [
            "Impute missing values using mode",
            "Apply one-hot encoding to categoricals",
            "Scale numeric features",
        ],
        "recommended_models": ["LogisticRegression", "RandomForest"],
        "sensitive_attribute_candidates": ["sex", "race", "age"],
        "reasoning": "Dummy plan used for failure-path testing only.",
    }

    print("\nRunning run_data_agent() on corrupted dataset...")
    result = run_data_agent(
        df=corrupt_df,
        plan=dummy_plan,
        target_column="income",
        task_type="classification",
    )

    print("\n--- Actions Taken ---")
    for action in result["actions_taken"]:
        print(f"  • {action}")

    print("\n--- Quality Report ---")
    printable_report = {
        k: v for k, v in result["quality_report"].items()
        if k != "unresolved_null_columns"
    }
    print(json.dumps(printable_report, indent=2))
    print(f"\nquality_check_passed: {result['quality_check_passed']}")

    # --- Assertions ---
    print("\nRunning assertions...")

    assert result["quality_check_passed"] is False, (
        "Expected quality_check_passed=False for corrupted dataset "
        f"(rows_dropped_pct={result['quality_report']['rows_dropped_pct']}%)"
    )
    print("  [PASS] quality_check_passed correctly returns False")

    dropped_pct = result["quality_report"]["rows_dropped_pct"]
    assert dropped_pct > 30.0, (
        f"Expected rows_dropped_pct > 30%, got {dropped_pct}%"
    )
    print(f"  [PASS] rows_dropped_pct = {dropped_pct}% "
          f"(triggers quality failure threshold)")

    assert len(result["quality_report"]["columns_dropped"]) > 0, (
        "Expected feature columns to be dropped (60% null)"
    )
    print(f"  [PASS] columns_dropped = {result['quality_report']['columns_dropped']}")

    # Even in the failure case, the remaining data should be cleanly imputed
    null_counts = result["cleaned_df"].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"cleaned_df should have zero nulls after imputation, "
        f"got nulls in: {cols_with_nulls.to_dict()}"
    )
    print("  [PASS] cleaned_df has zero nulls despite corruption "
          "(remaining data cleaned successfully)")

    print("\n✅ TEST 2 PASSED\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        run_normal_test()
        run_failure_test()
        print("🎉 All Data Agent tests passed.")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
