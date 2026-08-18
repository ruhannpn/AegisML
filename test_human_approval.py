"""
test_human_approval.py
======================
Test suite for Human-Approval gate (interrupt/resume) and SqliteSaver checkpointer.

TEST A — Happy path approval:
  Planner → Data → Training → Fairness → human_approval_node (PAUSE) → Command(resume="approve") → END

TEST B — Reject data quality reroute:
  human_approval_node → Command(resume="reject_data_quality") → planner_node (with human failure context)
  → data_agent_node → training_node → fairness_node → human_approval_node (PAUSE 2) → approve → END

TEST C — Reject model/fairness reroute:
  human_approval_node → Command(resume="reject_model_or_fairness") → training_node (DIRECT, skipping planner & data)
  → fairness_node → human_approval_node (PAUSE 2) → approve → END
  Asserts data_agent_result actions_taken is UNCHANGED (proves Data Agent was skipped).

TEST D — Rejection cap safety:
  Repeatedly reject until MAX_HUMAN_REROUTES cap (2) is reached → unresolved_human_rejection = True → END.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any

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
from graph_state import df_to_bytes, bytes_to_df, bytes_to_model
from pipeline_graph import graph


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def fresh_thread() -> dict:
    return {"configurable": {"thread_id": f"thread-human-{uuid.uuid4().hex[:6]}"}}


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


# ---------------------------------------------------------------------------
# TEST A: Happy Path Approval
# ---------------------------------------------------------------------------


def test_happy_path_approval(df) -> None:
    print_section("TEST A: Happy Path Approval (pause at gate → approve → END)")

    config = fresh_thread()
    thread_id = config["configurable"]["thread_id"]
    print(f"Step 1: Invoking graph initially ({thread_id})...")

    graph.invoke(make_initial_state(df), config=config)

    print("\nStep 2: Verifying pause at human_approval_node...")
    snapshot = graph.get_state(config)
    print(f"  • Queued next node(s) : {snapshot.next}")

    assert "human_approval_node" in snapshot.next, "Expected graph to pause at human_approval_node"
    assert snapshot.values.get("human_decision") is None, "human_decision should be None before resume"

    tasks = snapshot.tasks
    assert len(tasks) > 0 and len(tasks[0].interrupts) > 0
    payload = tasks[0].interrupts[0].value
    print(f"  • Interrupt payload question : '{payload.get('question')}'")
    print(f"  • Selected model for review : '{payload.get('selected_model_name')}'")
    print(f"  • Overall fairness passed    : {payload.get('overall_fairness_passed')}")
    print("  [PASS] Confirmed graph paused at human_approval_node with complete review payload")

    print("\nStep 3: Resuming with Command(resume='approve')...")
    resumed = graph.invoke(Command(resume="approve"), config=config)

    final_snapshot = graph.get_state(config)
    assert not final_snapshot.next, "Graph should reach END after approval"
    assert resumed.get("human_decision") == "approve"
    assert resumed.get("rejection_reroute_count") == 0
    print("  [PASS] Reached END with human_decision='approve'")
    print("\n✅ TEST A PASSED\n")


# ---------------------------------------------------------------------------
# TEST B: Reject Data Quality Reroute
# ---------------------------------------------------------------------------


def test_reject_data_quality(df) -> None:
    print_section("TEST B: Reject Data Quality Reroute (human rejects data → planner reroute)")

    config = fresh_thread()
    print("Step 1: Invoking graph initially...")
    graph.invoke(make_initial_state(df), config=config)

    snapshot1 = graph.get_state(config)
    assert "human_approval_node" in snapshot1.next

    print("\nStep 2: Resuming with Command(resume='reject_data_quality')...")
    graph.invoke(Command(resume="reject_data_quality"), config=config)

    print("\nStep 3: Verifying graph paused at human_approval_node for the 2nd time...")
    snapshot2 = graph.get_state(config)
    assert "human_approval_node" in snapshot2.next, "Expected graph to pause at human_approval_node a 2nd time"

    reroute_count = snapshot2.values.get("rejection_reroute_count", 0)
    assert reroute_count == 1, f"Expected rejection_reroute_count == 1, got {reroute_count}"
    print(f"  [PASS] rejection_reroute_count == {reroute_count} (reroute executed)")

    print("\nStep 4: Approving 2nd review pass...")
    final_res = graph.invoke(Command(resume="approve"), config=config)
    assert final_res.get("human_decision") == "approve"
    print("  [PASS] 2nd pass approved successfully")
    print("\n✅ TEST B PASSED\n")


# ---------------------------------------------------------------------------
# TEST C: Reject Model / Fairness Reroute (Direct to Training)
# ---------------------------------------------------------------------------


def test_reject_model_or_fairness(df) -> None:
    print_section("TEST C: Reject Model/Fairness (direct to training_node, skip Data Agent)")

    config = fresh_thread()
    print("Step 1: Invoking graph initially...")
    graph.invoke(make_initial_state(df), config=config)

    snapshot1 = graph.get_state(config)
    assert "human_approval_node" in snapshot1.next

    # Record data_agent_result before rejection
    data_res_before = snapshot1.values.get("data_agent_result")
    actions_before = list(data_res_before.get("actions_taken", [])) if data_res_before else []

    print("\nStep 2: Resuming with Command(resume='reject_model_or_fairness')...")
    graph.invoke(Command(resume="reject_model_or_fairness"), config=config)

    print("\nStep 3: Verifying graph paused at human_approval_node for 2nd time...")
    snapshot2 = graph.get_state(config)
    assert "human_approval_node" in snapshot2.next

    data_res_after = snapshot2.values.get("data_agent_result")
    actions_after = list(data_res_after.get("actions_taken", [])) if data_res_after else []

    # Proves Data Agent was skipped: actions_taken in data_agent_result is strictly identical
    assert actions_before == actions_after, (
        f"Data Agent actions changed — Data Agent was re-executed! Before={actions_before}, After={actions_after}"
    )
    print(f"  [PASS] Data Agent was SKIPPED! Data agent actions identical before ({len(actions_before)}) and after ({len(actions_after)})")

    reroute_count = snapshot2.values.get("rejection_reroute_count", 0)
    assert reroute_count == 1

    print("\nStep 4: Approving 2nd review pass...")
    final_res = graph.invoke(Command(resume="approve"), config=config)
    assert final_res.get("human_decision") == "approve"
    print("  [PASS] Direct training reroute succeeded and approved")
    print("\n✅ TEST C PASSED\n")


# ---------------------------------------------------------------------------
# TEST D: Rejection Cap Safety
# ---------------------------------------------------------------------------


def test_rejection_cap_safety(df) -> None:
    print_section("TEST D: Rejection Cap Safety (rejection_reroute_count cap = 2)")

    config = fresh_thread()
    print("Step 1: Invoking graph initially...")
    graph.invoke(make_initial_state(df), config=config)

    # Rejection 1
    print("\nStep 2: Rejection 1 ('reject_data_quality')...")
    graph.invoke(Command(resume="reject_data_quality"), config=config)

    # Rejection 2
    print("\nStep 3: Rejection 2 ('reject_data_quality')...")
    graph.invoke(Command(resume="reject_data_quality"), config=config)

    # Rejection 3 — should hit cap (MAX_HUMAN_REROUTES = 2) and route to END
    print("\nStep 4: Rejection 3 ('reject_data_quality') — expecting cap to be hit...")
    final_res = graph.invoke(Command(resume="reject_data_quality"), config=config)

    final_snapshot = graph.get_state(config)
    assert not final_snapshot.next, "Graph should terminate at END after reaching rejection cap"

    assert final_res.get("unresolved_human_rejection") is True, (
        "Expected unresolved_human_rejection=True after cap reached"
    )
    print(f"  [PASS] unresolved_human_rejection={final_res.get('unresolved_human_rejection')}")
    print(f"  [PASS] rejection_reroute_count={final_res.get('rejection_reroute_count')}")
    print("  [PASS] Graph safely terminated without infinite human-rejection loop")
    print("\n✅ TEST D PASSED\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        df = load_adult_dataset()
        test_happy_path_approval(df)
        test_reject_data_quality(df)
        test_reject_model_or_fairness(df)
        test_rejection_cap_safety(df)
        print("🎉 All Human Approval gate tests passed successfully.")
    except Exception as exc:
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
