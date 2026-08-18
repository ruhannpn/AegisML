"""
test_fairness_agent.py
======================
Standalone test for run_fairness_agent().

Uses the real pipeline:
  load_adult_dataset() -> plan_pipeline() -> run_data_agent() -> run_training_agent() -> run_fairness_agent()

TEST 1 — Classification fairness evaluation:
  - Verifies fairness metrics (Disparate Impact, Demographic Parity Difference)
    computed across candidate sensitive attributes.
  - Confirms OHE attributes like 'sex' and 'race' are correctly reconstructed and evaluated.
  - Asserts fairness_report structure and group details.

TEST 2 — Regression error handling:
  - Confirms NotImplementedError is raised when task_type == "regression".
"""

from __future__ import annotations

import json
import os
import sys

# Load .env if GROQ_API_KEY is not already in environment
if "GROQ_API_KEY" not in os.environ and os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

from dataset_utils import load_adult_dataset
from planner_agent import plan_pipeline
from data_agent import run_data_agent
from training_agent import run_training_agent
from fairness_agent import run_fairness_agent


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_fairness_classification() -> None:
    print_section("TEST 1: Classification Fairness Evaluation — Adult Income Dataset")

    # --- Data loading & planning ---
    df = load_adult_dataset()
    print("\nCalling plan_pipeline()...")
    plan = plan_pipeline(df, target_column="income", task_type="classification")
    print(f"  sensitive_attribute_candidates: {plan['sensitive_attribute_candidates']}")

    # --- Data cleaning ---
    print("\nCalling run_data_agent()...")
    data_result = run_data_agent(
        df=df,
        plan=plan,
        target_column="income",
        task_type="classification",
    )
    cleaned_df = data_result["cleaned_df"]
    print(f"  cleaned_df shape: {cleaned_df.shape}")

    # --- Training ---
    print("\nCalling run_training_agent()...")
    training_result = run_training_agent(
        cleaned_df=cleaned_df,
        target_column="income",
        task_type="classification",
        recommended_models=plan["recommended_models"],
    )
    fitted_model = training_result["_fitted_model"]
    selected_name = training_result["selected_model_name"]
    print(f"  Selected model: {selected_name}")

    # --- Fairness Agent ---
    print("\nCalling run_fairness_agent()...")
    fairness_result = run_fairness_agent(
        cleaned_df=cleaned_df,
        fitted_model=fitted_model,
        target_column="income",
        sensitive_attribute_candidates=plan["sensitive_attribute_candidates"],
        task_type="classification",
    )

    print("\n--- Actions Taken ---")
    for action in fairness_result["actions_taken"]:
        print(f"  • {action}")

    print("\n--- Fairness Report ---")
    for entry in fairness_result["fairness_report"]:
        status = "❌ VIOLATION" if entry["violation"] else "✅ PASS"
        gd = entry["group_details"]
        print(
            f"  {entry['attribute']:<20} | {status} | "
            f"DI: {entry['disparate_impact']:.4f} | DPD: {entry['demographic_parity_difference']:.4f} | "
            f"[{gd['group_a']}: {gd['group_a_positive_rate']:.4f} vs {gd['group_b']}: {gd['group_b_positive_rate']:.4f}]"
        )

    print(f"\nOverall Fairness Passed: {fairness_result['overall_fairness_passed']}")
    print(f"Attributes Skipped      : {fairness_result['attributes_skipped']}")

    # --- Assertions ---
    print("\nRunning assertions...")

    report = fairness_result["fairness_report"]
    assert len(report) >= 1, "Expected at least 1 evaluated attribute in fairness_report"
    print(f"  [PASS] fairness_report contains {len(report)} evaluated attribute(s)")

    evaluated_attrs = [entry["attribute"] for entry in report]
    # Check that sex and/or race were successfully evaluated
    has_sex_or_race = any(attr in ("sex", "race") for attr in evaluated_attrs)
    assert has_sex_or_race, f"Expected 'sex' or 'race' in evaluated attributes, got: {evaluated_attrs}"
    print(f"  [PASS] Successfully evaluated key sensitive attributes: {evaluated_attrs}")

    for entry in report:
        assert "attribute" in entry
        assert "disparate_impact" in entry
        assert "demographic_parity_difference" in entry
        assert "violation" in entry
        assert "group_details" in entry

        gd = entry["group_details"]
        for key in ("group_a", "group_a_positive_rate", "group_b", "group_b_positive_rate"):
            assert key in gd, f"Missing key '{key}' in group_details"

    print("  [PASS] All fairness_report entries have complete schema and group_details")

    assert isinstance(fairness_result["overall_fairness_passed"], bool)
    print("  [PASS] overall_fairness_passed is boolean")

    print("\n✅ TEST 1 PASSED\n")
    return cleaned_df, fitted_model


def test_fairness_regression_not_implemented(cleaned_df, fitted_model) -> None:
    print_section("TEST 2: Regression Task Raises NotImplementedError")

    try:
        run_fairness_agent(
            cleaned_df=cleaned_df,
            fitted_model=fitted_model,
            target_column="income",
            sensitive_attribute_candidates=["sex", "race"],
            task_type="regression",
        )
        assert False, "Expected NotImplementedError for task_type='regression'"
    except NotImplementedError as exc:
        print(f"  [PASS] Correctly caught expected error: {exc}")

    print("\n✅ TEST 2 PASSED\n")


if __name__ == "__main__":
    try:
        cleaned_df, fitted_model = test_fairness_classification()
        test_fairness_regression_not_implemented(cleaned_df, fitted_model)
        print("🎉 All Fairness Agent tests passed.")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
