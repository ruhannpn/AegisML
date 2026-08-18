"""
test_training_agent.py
======================
Standalone test for run_training_agent().

Uses the real pipeline:
  load_adult_dataset() -> plan_pipeline() -> run_data_agent() -> run_training_agent()

TEST 1 — Full pipeline:
  Verifies leaderboard structure, selected model metrics, and SHAP summary
  on the real cleaned Adult Income dataset.

TEST 2 — Unknown model name:
  Passes a model name not in the registry ("FakeBoost") alongside valid ones.
  Confirms it is skipped with a warning, not a crash.
"""

from __future__ import annotations

import json
import os
import sys

# Load .env if GROQ_API_KEY not already in environment
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


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# TEST 1 — Full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline() -> None:
    print_section("TEST 1: Full pipeline — Adult Income dataset")

    # --- Data loading & planning ---
    df = load_adult_dataset()
    print("\nCalling plan_pipeline()...")
    plan = plan_pipeline(df, target_column="income", task_type="classification")
    print(f"  recommended_models: {plan['recommended_models']}")

    # --- Data cleaning ---
    print("\nCalling run_data_agent()...")
    data_result = run_data_agent(
        df=df,
        plan=plan,
        target_column="income",
        task_type="classification",
    )
    assert data_result["quality_check_passed"], (
        f"Data agent quality check failed: {data_result['quality_report']}"
    )
    cleaned_df = data_result["cleaned_df"]
    print(f"  cleaned_df shape: {cleaned_df.shape}")

    # --- Training ---
    print("\nCalling run_training_agent()...")
    result = run_training_agent(
        cleaned_df=cleaned_df,
        target_column="income",
        task_type="classification",
        recommended_models=plan["recommended_models"],
    )

    # --- Print results ---
    print("\n--- Actions Taken ---")
    for action in result["actions_taken"]:
        print(f"  • {action}")

    print("\n--- Leaderboard ---")
    for i, entry in enumerate(result["leaderboard"]):
        status = "OK" if entry["trained_successfully"] else "SKIP"
        metrics_str = (
            ", ".join(f"{k}={v}" for k, v in entry["metrics"].items())
            if entry["metrics"]
            else entry.get("skip_reason", "")
        )
        print(f"  {i+1}. [{status}] {entry['model_name']}: {metrics_str}")

    print(f"\n--- Selected Model: {result['selected_model_name']} ---")
    print(json.dumps(result["selected_model_metrics"], indent=2))

    print("\n--- SHAP Summary (top features) ---")
    for entry in result["shap_summary"]:
        print(f"  {entry['feature']}: {entry['importance']:.6f}")

    # --- Assertions ---
    print("\nRunning assertions...")

    lb = result["leaderboard"]
    successful = [e for e in lb if e["trained_successfully"]]
    assert len(successful) >= 1, "Expected at least 1 successfully trained model"
    print(f"  [PASS] leaderboard has {len(successful)} successfully trained model(s)")

    metrics = result["selected_model_metrics"]
    for key in ("accuracy", "f1", "auc_roc"):
        assert key in metrics, f"Missing key '{key}' in selected_model_metrics"
        assert metrics[key] is not None, f"'{key}' is None"
    print(f"  [PASS] selected_model_metrics has all expected classification keys")

    assert result["selected_model_name"] == successful[0]["model_name"], (
        "selected_model_name must match the top of the sorted leaderboard"
    )
    print(f"  [PASS] selected_model_name == leaderboard[0]: '{result['selected_model_name']}'")

    assert isinstance(result["shap_summary"], list)
    assert len(result["shap_summary"]) <= 5, (
        f"shap_summary should have at most 5 entries, got {len(result['shap_summary'])}"
    )
    if result["shap_summary"]:
        entry = result["shap_summary"][0]
        assert "feature" in entry and "importance" in entry
    print(f"  [PASS] shap_summary has {len(result['shap_summary'])} entries (≤ 5)")

    # Primary metric: top model should have best AUC-ROC of all successful ones
    if len(successful) > 1:
        top_auc = result["selected_model_metrics"].get("auc_roc") or 0
        for other in successful[1:]:
            other_auc = other["metrics"].get("auc_roc") or 0
            assert top_auc >= other_auc, (
                f"Leaderboard not sorted: top AUC-ROC {top_auc} < other {other_auc}"
            )
        print("  [PASS] Leaderboard correctly sorted by AUC-ROC (descending)")

    print("\n✅ TEST 1 PASSED\n")
    return result


# ---------------------------------------------------------------------------
# TEST 2 — Unknown model name does not crash
# ---------------------------------------------------------------------------


def test_unknown_model_skipped(previous_cleaned_df) -> None:
    print_section("TEST 2: Unknown model name is skipped, not a crash")

    mixed_models = ["FakeBoost", "LogisticRegression", "RandomForest"]
    print(f"  recommended_models: {mixed_models}")
    print("  Expecting 'FakeBoost' to be skipped with a warning...")

    result = run_training_agent(
        cleaned_df=previous_cleaned_df,
        target_column="income",
        task_type="classification",
        recommended_models=mixed_models,
    )

    print("\n--- Leaderboard ---")
    for entry in result["leaderboard"]:
        status = "OK" if entry["trained_successfully"] else "SKIP"
        print(f"  [{status}] {entry['model_name']}: {entry.get('skip_reason', entry.get('metrics', {}))}")

    print("\nRunning assertions...")

    lb = result["leaderboard"]
    fake_entries = [e for e in lb if "FakeBoost" in e["model_name"]]
    assert len(fake_entries) == 1, "Expected FakeBoost in leaderboard"
    assert not fake_entries[0]["trained_successfully"], (
        "FakeBoost should have trained_successfully=False"
    )
    print("  [PASS] 'FakeBoost' correctly marked as failed with skip_reason")

    successful = [e for e in lb if e["trained_successfully"]]
    assert len(successful) >= 1, "Expected at least 1 model to succeed despite FakeBoost"
    print(f"  [PASS] {len(successful)} valid model(s) trained successfully despite unknown model")

    skip_logged = any("FakeBoost" in a for a in result["actions_taken"])
    assert skip_logged, "Expected FakeBoost skip to appear in actions_taken"
    print("  [PASS] Skip warning logged in actions_taken")

    print("\n✅ TEST 2 PASSED\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        t1_result = test_full_pipeline()
        # Reuse the same cleaned_df from TEST 1 (plan_pipeline already called)
        df = load_adult_dataset()
        plan = plan_pipeline(df, target_column="income", task_type="classification")
        data_result = run_data_agent(df, plan, "income", "classification")
        cleaned_df = data_result["cleaned_df"]
        test_unknown_model_skipped(cleaned_df)
        print("🎉 All Training Agent tests passed.")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
