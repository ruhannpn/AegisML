"""
test_planner.py
===============
Standalone test for plan_pipeline() using the UCI Adult Income dataset.

The Adult dataset is ideal for this project because it has:
  - Missing values (marked as " ?" in the CSV)
  - Moderate class imbalance (~3:1, <=50K vs >50K)
  - Natural sensitive attributes (sex, race, age, marital-status, etc.)

Usage:
    export GROQ_API_KEY=your_key_here
    python test_planner.py
"""

import json
import sys

from dataset_utils import load_adult_dataset
from planner_agent import plan_pipeline, REQUIRED_PLAN_KEYS


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def run_test() -> None:
    df = load_adult_dataset()

    print("=" * 60)
    print("Calling plan_pipeline()  (task_type='classification')")
    print("=" * 60)

    plan = plan_pipeline(
        df=df,
        target_column="income",
        task_type="classification",
    )

    # Pretty-print the returned JSON plan
    print("\n--- Returned Plan ---")
    print(json.dumps(plan, indent=2))
    print("---------------------\n")

    # Assertions
    print("Running assertions...")

    # 1. All required keys present
    for key in REQUIRED_PLAN_KEYS:
        assert key in plan, f"Missing required key: {key!r}"
    print("  [PASS] All required keys present:", sorted(REQUIRED_PLAN_KEYS))

    # 2. List fields are actually lists
    for list_field in (
        "data_quality_concerns",
        "recommended_preprocessing_steps",
        "recommended_models",
        "sensitive_attribute_candidates",
    ):
        assert isinstance(plan[list_field], list), (
            f"Expected list for {list_field!r}, got {type(plan[list_field])}"
        )
    print("  [PASS] All list fields are lists.")

    # 3. Non-empty list fields
    # The agent pre-computes quality flags deterministically, so the LLM is
    # explicitly instructed to populate data_quality_concerns. With a known-
    # imbalanced dataset like Adult, this must always be non-empty.
    assert len(plan["data_quality_concerns"]) > 0, "data_quality_concerns is empty"
    assert len(plan["recommended_models"]) >= 2, "Need at least 2 recommended models"
    print("  [PASS] Non-empty list fields.")

    # 4. Reasoning is a non-empty string
    assert isinstance(plan["reasoning"], str) and len(plan["reasoning"].strip()) > 10, (
        "reasoning must be a non-trivial string"
    )
    print("  [PASS] reasoning is a non-empty string.")

    # 5. Sensitive attribute candidates should include known sensitive columns
    #    from Adult dataset (at minimum 'sex' or 'race' should appear)
    candidates_lower = [c.lower() for c in plan["sensitive_attribute_candidates"]]
    known_sensitive = {"sex", "race", "age", "marital-status", "native-country"}
    found = known_sensitive & set(candidates_lower)
    assert len(found) > 0, (
        f"Expected at least one known sensitive attribute from {known_sensitive}, "
        f"got: {plan['sensitive_attribute_candidates']}"
    )
    print(f"  [PASS] Sensitive attribute candidates include known columns: {found}")

    # 6. Recommended models should be 2-3
    assert 2 <= len(plan["recommended_models"]) <= 3, (
        f"Expected 2-3 recommended models, got {len(plan['recommended_models'])}"
    )
    print(f"  [PASS] Got {len(plan['recommended_models'])} recommended models: "
          f"{plan['recommended_models']}")

    print("\n✅  All assertions passed. Planner Agent is working correctly.\n")


if __name__ == "__main__":
    try:
        run_test()
    except Exception as exc:
        print(f"\n❌  Test failed: {exc}", file=sys.stderr)
        sys.exit(1)
