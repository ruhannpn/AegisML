"""
test_graph_step1.py
===================
Tests the two-node LangGraph pipeline (Planner + Data Agent).

NOTE: DataFrames are stored in state as pickled bytes (df_bytes /
cleaned_df_bytes). Use graph_state.bytes_to_df() to reconstruct them.

TEST 1 — Happy path:
  Normal Adult Income dataset → quality_check_passed = True → END.

TEST 2 — Retry path:
  Corrupted dataset (35% null targets + 60% null in 3 feature columns) →
  quality fails every attempt → loop executes → retry cap hit →
  unresolved_quality_issue = True → END.
  Confirms retry_count == MAX_RETRIES to prove the loop actually ran.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

# Load .env if GROQ_API_KEY is not already exported in environment
if "GROQ_API_KEY" not in os.environ and os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

import numpy as np

from dataset_utils import load_adult_dataset
from graph_state import df_to_bytes, bytes_to_df, bytes_to_model
from pipeline_graph import graph, MAX_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh_thread() -> dict:
    """Each invocation needs a unique thread_id for MemorySaver isolation."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def make_initial_state(df, target_column="income", task_type="classification") -> dict:
    return {
        "df_bytes": df_to_bytes(df),
        "target_column": target_column,
        "task_type": task_type,
        "plan": None,
        "data_agent_result": None,
        "cleaned_df_bytes": None,
        "last_failure_reason": None,
        "retry_count": 0,
        "unresolved_quality_issue": False,
        "training_result": None,
        "selected_model_bytes": None,
        "fairness_result": None,
    }


def print_final_state(state: dict) -> None:
    print("\n--- Final State Summary ---")
    print(f"  retry_count             : {state.get('retry_count', 0)}")
    print(f"  unresolved_quality_issue: {state.get('unresolved_quality_issue', False)}")

    result = state.get("data_agent_result") or {}
    if result:
        report = result.get("quality_report", {})
        print(f"  quality_check_passed    : {result.get('quality_check_passed')}")
        print(f"  missing_pct_after_clean : {report.get('missing_pct_after_cleaning')}%")
        print(f"  rows_dropped            : {report.get('rows_dropped')} "
              f"({report.get('rows_dropped_pct')}%)")
        print(f"  columns_dropped         : {report.get('columns_dropped')}")

    if state.get("cleaned_df_bytes"):
        cleaned = bytes_to_df(state["cleaned_df_bytes"])
        print(f"  cleaned_df shape        : {cleaned.shape}")

    if state.get("plan"):
        print(f"  plan.recommended_models : {state['plan'].get('recommended_models')}")

    train_res = state.get("training_result") or {}
    if train_res:
        print(f"  selected_model_name     : {train_res.get('selected_model_name')}")
        print(f"  selected_model_metrics  : {train_res.get('selected_model_metrics')}")
        shap_sum = train_res.get("shap_summary", [])
        if shap_sum:
            top_feats = ", ".join(f"{e['feature']} ({e['importance']:.4f})" for e in shap_sum[:3])
            print(f"  top_shap_features       : {top_feats}")

    if state.get("selected_model_bytes"):
        print(f"  selected_model_bytes    : {len(state['selected_model_bytes']):,} bytes")

    fairness_res = state.get("fairness_result") or {}
    if fairness_res:
        print(f"  overall_fairness_passed : {fairness_res.get('overall_fairness_passed')}")
        report = fairness_res.get("fairness_report", [])
        print(f"  fairness_evaluated_attrs: {[e['attribute'] for e in report]}")
        print(f"  fairness_skipped_attrs  : {fairness_res.get('attributes_skipped', [])}")


# ---------------------------------------------------------------------------
# TEST 1 — Happy path
# ---------------------------------------------------------------------------


def test_happy_path() -> None:
    print_section("TEST 1: Happy path — normal Adult Income dataset (Planner → Data → Training → Fairness)")

    df = load_adult_dataset()

    print("\nInvoking graph...")
    final_state = graph.invoke(
        make_initial_state(df),
        config=fresh_thread(),
    )

    print_final_state(final_state)

    # Assertions
    print("\nRunning assertions...")

    result = final_state.get("data_agent_result") or {}
    assert result.get("quality_check_passed") is True, (
        f"Expected quality_check_passed=True, got: {result.get('quality_check_passed')}"
    )
    print("  [PASS] quality_check_passed is True")

    assert final_state.get("retry_count", 0) == 0, (
        f"Expected retry_count == 0, got: {final_state.get('retry_count')}"
    )
    print("  [PASS] retry_count == 0 (no retries needed)")

    assert final_state.get("unresolved_quality_issue", False) is False, (
        "unresolved_quality_issue should be False"
    )
    print("  [PASS] unresolved_quality_issue is False")

    assert final_state.get("cleaned_df_bytes") is not None
    cleaned_df = bytes_to_df(final_state["cleaned_df_bytes"])
    assert cleaned_df.isnull().sum().sum() == 0, "cleaned_df still contains nulls"
    print(f"  [PASS] cleaned_df: {cleaned_df.shape[0]:,} rows, {cleaned_df.shape[1]} cols, 0 nulls")

    assert final_state.get("plan") is not None
    assert len(final_state["plan"].get("recommended_models", [])) >= 2
    print(f"  [PASS] plan populated: models = {final_state['plan']['recommended_models']}")

    # Training Agent assertions
    train_res = final_state.get("training_result") or {}
    assert train_res, "training_result is missing or empty"
    assert "leaderboard" in train_res and len(train_res["leaderboard"]) >= 1
    assert "selected_model_name" in train_res
    assert train_res["selected_model_name"] in final_state["plan"]["recommended_models"]
    print(f"  [PASS] training_result populated: selected model = '{train_res['selected_model_name']}'")

    metrics = train_res.get("selected_model_metrics", {})
    for k in ("accuracy", "f1", "auc_roc"):
        assert k in metrics and metrics[k] is not None
    print(f"  [PASS] selected_model_metrics populated: {metrics}")

    assert train_res.get("shap_summary") and len(train_res["shap_summary"]) <= 5
    print(f"  [PASS] shap_summary generated ({len(train_res['shap_summary'])} features)")

    # Model deserialisation and predict test
    model_bytes = final_state.get("selected_model_bytes")
    assert model_bytes is not None and len(model_bytes) > 0, "selected_model_bytes missing or empty"
    model = bytes_to_model(model_bytes)
    assert hasattr(model, "predict"), "Deserialised model object lacks .predict() method"

    # Test prediction on a sample of 5 rows
    X_sample = cleaned_df.drop(columns=["income"]).head(5)
    preds = model.predict(X_sample)
    assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
    print(f"  [PASS] deserialised model predicts successfully on sample: preds={preds.tolist()}")

    # Fairness Agent assertions
    fairness_res = final_state.get("fairness_result") or {}
    assert fairness_res, "fairness_result is missing or empty"
    assert "fairness_report" in fairness_res
    assert len(fairness_res["fairness_report"]) >= 1, "fairness_report should have at least 1 evaluated attribute"
    assert isinstance(fairness_res.get("overall_fairness_passed"), bool), "overall_fairness_passed should be bool"
    print(f"  [PASS] fairness_result populated: evaluated {len(fairness_res['fairness_report'])} attribute(s), "
          f"overall_fairness_passed={fairness_res['overall_fairness_passed']}")

    print("\n✅ TEST 1 PASSED\n")


# ---------------------------------------------------------------------------
# TEST 2 — Retry path
# ---------------------------------------------------------------------------


def test_retry_path() -> None:
    print_section("TEST 2: Retry path — corrupted dataset")

    df = load_adult_dataset()
    corrupt_df = df.copy()
    np.random.seed(42)

    # Null out 35% of target (income) → rows_dropped_pct = 35% > 30% → fails every attempt
    target_null_idx = np.random.choice(
        corrupt_df.index, size=int(len(corrupt_df) * 0.35), replace=False
    )
    corrupt_df.loc[target_null_idx, "income"] = np.nan

    # 60% nulls in 3 feature columns → they get dropped (> 50% threshold)
    for col in ["workclass", "occupation", "native-country"]:
        col_null_idx = np.random.choice(
            corrupt_df.index, size=int(len(corrupt_df) * 0.60), replace=False
        )
        corrupt_df.loc[col_null_idx, col] = np.nan

    print(f"\nCorrupted df: {corrupt_df['income'].isnull().mean()*100:.1f}% null targets "
          f"→ rows_dropped_pct will be ~35% (> 30% threshold) on every attempt")
    print(f"Expected: retry loop fires {MAX_RETRIES}x → "
          f"unresolved_quality_issue=True")

    print("\nInvoking graph (up to 3 Groq calls)...")
    final_state = graph.invoke(
        make_initial_state(corrupt_df),
        config=fresh_thread(),
    )

    print_final_state(final_state)

    # Assertions
    print("\nRunning assertions...")

    retry_count = final_state.get("retry_count", 0)
    assert retry_count >= 1, (
        f"Expected at least 1 retry — loop did not execute. retry_count={retry_count}"
    )
    print(f"  [PASS] retry_count = {retry_count} (loop executed)")

    assert retry_count == MAX_RETRIES, (
        f"Expected retry_count == MAX_RETRIES ({MAX_RETRIES}), got {retry_count}"
    )
    print(f"  [PASS] retry_count == MAX_RETRIES ({MAX_RETRIES}) — cap reached")

    assert final_state.get("unresolved_quality_issue") is True, (
        "Expected unresolved_quality_issue=True after retry cap"
    )
    print("  [PASS] unresolved_quality_issue=True — no silent failure")

    result = final_state.get("data_agent_result") or {}
    assert result.get("quality_check_passed") is False
    print("  [PASS] quality_check_passed correctly remains False")

    last_failure = final_state.get("last_failure_reason") or {}
    assert last_failure.get("rows_dropped_pct", 0) > 30, (
        f"Expected rows_dropped_pct > 30%, got {last_failure.get('rows_dropped_pct')}"
    )
    print(f"  [PASS] last_failure_reason.rows_dropped_pct = "
          f"{last_failure['rows_dropped_pct']}%")

    print("\n✅ TEST 2 PASSED\n")


# ---------------------------------------------------------------------------
# TEST 3 — Fixable retry path
# ---------------------------------------------------------------------------


def test_fixable_retry_path() -> None:
    print_section("TEST 3: Fixable retry path — feature high nulls recoverable by plan revision")

    df = load_adult_dataset()
    fixable_df = df.copy()
    np.random.seed(42)

    # Inject 42% missingness into 'occupation' (42% <= 50% default drop threshold, but > 35% warning threshold)
    # Target 'income' is left 100% clean (0% missing).
    null_idx = np.random.choice(
        fixable_df.index, size=int(len(fixable_df) * 0.42), replace=False
    )
    fixable_df.loc[null_idx, "occupation"] = np.nan

    print(f"\nFixable dataset: 'occupation' has 42% nulls (target 'income' clean)")
    print(f"Expected: Pass 1 fails (unhandled 42% null feature) → Retry 1 feedback → "
          f"Planner revises plan to drop 'occupation' → Pass 2 succeeds → END with quality_check_passed=True")

    print("\nInvoking graph...")
    final_state = graph.invoke(
        make_initial_state(fixable_df),
        config=fresh_thread(),
    )

    print_final_state(final_state)

    # Assertions
    print("\nRunning assertions...")

    retry_count = final_state.get("retry_count", 0)
    assert retry_count == 1, (
        f"Expected exactly 1 retry to fix the plan, got retry_count={retry_count}"
    )
    print("  [PASS] retry_count == 1 (retry fired and succeeded)")

    result = final_state.get("data_agent_result") or {}
    assert result.get("quality_check_passed") is True, (
        f"Expected quality_check_passed=True after plan revision, got: {result.get('quality_check_passed')}"
    )
    print("  [PASS] quality_check_passed is True after plan revision")

    assert final_state.get("unresolved_quality_issue", False) is False, (
        "unresolved_quality_issue should be False when retry succeeds"
    )
    print("  [PASS] unresolved_quality_issue is False")

    cols_dropped = result.get("quality_report", {}).get("columns_dropped", [])
    assert "occupation" in cols_dropped, (
        f"Expected 'occupation' to be dropped in revised plan, got columns_dropped: {cols_dropped}"
    )
    print(f"  [PASS] 'occupation' was explicitly dropped by revised plan: {cols_dropped}")

    print("\n✅ TEST 3 PASSED\n")


# ---------------------------------------------------------------------------
# Mermaid diagram
# ---------------------------------------------------------------------------


def print_mermaid() -> None:
    print_section("Graph Structure (Mermaid)")
    print(graph.get_graph().draw_mermaid())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        print_mermaid()
        test_happy_path()
        test_retry_path()
        test_fixable_retry_path()
        print("🎉 All graph tests passed.")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

