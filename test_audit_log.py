"""
test_audit_log.py
=================
Test suite for Audit Log module (audit_log.py) and pipeline integration.

Runs a full pipeline invocation through human approval, then verifies:
  1. Audit entries are created in audit_log.db (separate from pipeline_state.db).
  2. get_audit_trail(run_id) returns chronologically ordered event dicts.
  3. All major stages exist: planner_run, data_agent_run, training_run, fairness_run, human_decision, final_outcome.
  4. event_source correctly distinguishes 'automated' vs 'human_reviewer'.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

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

from langgraph.types import Command
from dataset_utils import load_adult_dataset
from graph_state import df_to_bytes
from pipeline_graph import graph
from audit_log import get_audit_trail, DEFAULT_AUDIT_DB


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def fresh_thread() -> dict:
    return {"configurable": {"thread_id": f"thread-audit-{uuid.uuid4().hex[:6]}"}}


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
        "human_decision": None,
        "rejection_reroute_count": 0,
        "unresolved_human_rejection": False,
    }


def test_full_pipeline_audit_trail() -> None:
    print_section("TEST: Full Pipeline Audit Trail Verification")

    df = load_adult_dataset()
    config = fresh_thread()
    thread_id = config["configurable"]["thread_id"]
    print(f"Run ID / Thread ID: '{thread_id}'")

    print("\nStep 1: Invoking initial graph pipeline...")
    graph.invoke(make_initial_state(df), config=config)

    print("\nStep 2: Resuming with human approval ('approve')...")
    resumed = graph.invoke(Command(resume="approve"), config=config)

    assert resumed.get("human_decision") == "approve"
    print("  [PASS] Pipeline completed successfully with human approval")

    print(f"\nStep 3: Retrieving audit trail for run_id='{thread_id}' from {DEFAULT_AUDIT_DB}...")
    entries = get_audit_trail(thread_id)

    print(f"\n--- Retrieved Audit Trail ({len(entries)} entries) ---")
    for i, entry in enumerate(entries, 1):
        print(
            f"  {i}. [{entry['timestamp']}] [{entry['event_source'].upper()}] "
            f"event_type='{entry['event_type']}': {entry['summary']}"
        )

    # --- Assertions ---
    print("\nRunning assertions...")

    assert len(entries) >= 5, f"Expected at least 5 audit entries, got {len(entries)}"
    print(f"  [PASS] audit trail contains {len(entries)} entries (≥ 5)")

    # Timestamps are in non-decreasing chronological order
    timestamps = [e["timestamp"] for e in entries]
    assert timestamps == sorted(timestamps), "Audit entries are not in chronological order"
    print("  [PASS] All entry timestamps are in strict chronological order")

    # Check event types present
    event_types = [e["event_type"] for e in entries]
    expected_types = {"planner_run", "data_agent_run", "training_run", "fairness_run", "human_decision", "final_outcome"}
    missing_types = expected_types - set(event_types)
    assert not missing_types, f"Missing expected event types: {missing_types}"
    print(f"  [PASS] All expected event types present: {sorted(list(expected_types))}")

    # Check event_source values
    for entry in entries:
        if entry["event_type"] == "human_decision":
            assert entry["event_source"] == "human_reviewer", (
                f"Expected event_source='human_reviewer' for human_decision, got {entry['event_source']}"
            )
        else:
            assert entry["event_source"] == "automated", (
                f"Expected event_source='automated' for {entry['event_type']}, got {entry['event_source']}"
            )
    print("  [PASS] event_source correctly distinguishes 'automated' vs 'human_reviewer'")

    # Check final_outcome event
    final_entry = entries[-1]
    assert final_entry["event_type"] == "final_outcome"
    assert final_entry["details"].get("status") == "APPROVED"
    print(f"  [PASS] Final outcome event recorded as APPROVED: {final_entry['summary']}")

    print("\n✅ AUDIT TRAIL TEST A PASSED\n")


def test_rejection_reroute_audit_trail(df) -> None:
    print_section("TEST B: Audit Trail for Human Rejection & Reroute Run")

    config = fresh_thread()
    thread_id = config["configurable"]["thread_id"]
    print(f"Run ID / Thread ID: '{thread_id}'")

    print("\nStep 1: Invoking initial graph pipeline...")
    graph.invoke(make_initial_state(df), config=config)

    print("\nStep 2: Resuming with human rejection ('reject_data_quality')...")
    graph.invoke(Command(resume="reject_data_quality"), config=config)

    print("\nStep 3: Resuming 2nd pass with human approval ('approve')...")
    resumed = graph.invoke(Command(resume="approve"), config=config)
    assert resumed.get("human_decision") == "approve"

    print(f"\nStep 4: Retrieving audit trail for run_id='{thread_id}'...")
    entries = get_audit_trail(thread_id)

    print(f"\n--- Retrieved Audit Trail ({len(entries)} entries) ---")
    for i, entry in enumerate(entries, 1):
        print(
            f"  {i}. [{entry['timestamp']}] [{entry['event_source'].upper()}] "
            f"event_type='{entry['event_type']}': {entry['summary']}"
        )

    # Verify multiple planner runs and human decisions exist
    planner_runs = [e for e in entries if e["event_type"] == "planner_run"]
    human_decisions = [e for e in entries if e["event_type"] == "human_decision"]
    final_outcomes = [e for e in entries if e["event_type"] == "final_outcome"]

    assert len(planner_runs) == 2, f"Expected 2 planner_run events, got {len(planner_runs)}"
    assert len(human_decisions) == 2, f"Expected 2 human_decision events, got {len(human_decisions)}"
    assert len(final_outcomes) == 1, f"Expected 1 final_outcome event, got {len(final_outcomes)}"
    assert final_outcomes[0]["details"].get("status") == "APPROVED"

    print("  [PASS] Confirmed reroute pass created 2nd planner_run & 2nd human_decision with timestamps")
    print("\n✅ AUDIT TRAIL TEST B PASSED\n")


def test_rejection_cap_audit_trail(df) -> None:
    print_section("TEST C: Audit Trail for Human Rejection Cap Termination")

    config = fresh_thread()
    thread_id = config["configurable"]["thread_id"]
    print(f"Run ID / Thread ID: '{thread_id}'")

    print("\nStep 1: Invoking initial graph pipeline...")
    graph.invoke(make_initial_state(df), config=config)

    print("\nStep 2: Rejecting 3 times until cap reached...")
    graph.invoke(Command(resume="reject_data_quality"), config=config)
    graph.invoke(Command(resume="reject_data_quality"), config=config)
    final_res = graph.invoke(Command(resume="reject_data_quality"), config=config)

    assert final_res.get("unresolved_human_rejection") is True

    print(f"\nStep 3: Retrieving audit trail for run_id='{thread_id}'...")
    entries = get_audit_trail(thread_id)

    print(f"\n--- Retrieved Audit Trail ({len(entries)} entries) ---")
    for i, entry in enumerate(entries, 1):
        print(
            f"  {i}. [{entry['timestamp']}] [{entry['event_source'].upper()}] "
            f"event_type='{entry['event_type']}': {entry['summary']}"
        )

    final_outcomes = [e for e in entries if e["event_type"] == "final_outcome"]
    assert len(final_outcomes) == 1, f"Expected 1 final_outcome event, got {len(final_outcomes)}"
    assert final_outcomes[0]["details"].get("status") == "HUMAN_REJECTION_CAP_REACHED"
    print(f"  [PASS] Confirmed cap termination recorded final_outcome status: '{final_outcomes[0]['details'].get('status')}'")
    print("\n✅ AUDIT TRAIL TEST C PASSED\n")


if __name__ == "__main__":
    try:
        df = load_adult_dataset()
        test_full_pipeline_audit_trail()
        test_rejection_reroute_audit_trail(df)
        test_rejection_cap_audit_trail(df)
        print("🎉 All Audit Log tests (happy path, reroutes, cap termination) passed successfully.")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
