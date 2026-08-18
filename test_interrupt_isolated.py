"""
test_interrupt_isolated.py
===========================
Isolated test demonstrating LangGraph's interrupt() and Command(resume=...) mechanism.

CONCEPT & DESIGN NOTES:
  1. Why a Checkpointer (MemorySaver) is REQUIRED for Interrupt/Resume:
     - Without a checkpointer, a graph execution is a single ephemeral Python call.
     - When interrupt() is called, LangGraph pauses execution and creates a state checkpoint.
     - The checkpointer persists the exact node state, task queue, and interrupt payloads
       to memory (or SQLite/Postgres in production).
     - When Command(resume=...) is called later, the checkpointer retrieves the saved frame
       and injects the human response directly into the interrupted line of code.

  2. What thread_id does and why it's needed:
     - thread_id (passed in config={'configurable': {'thread_id': '...'}}) acts as the
       unique session identifier for a specific graph run.
     - It allows LangGraph to locate the exact checkpoint associated with a user or run,
       enabling async resumption seconds, hours, or days later across completely separate
       API/CLI requests.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------------------------
# 1. State Schema
# ---------------------------------------------------------------------------


class ToyState(TypedDict, total=False):
    counter: int
    human_decision: Optional[str]


# ---------------------------------------------------------------------------
# 2. Graph Nodes
# ---------------------------------------------------------------------------


def step_one(state: ToyState) -> dict:
    current = state.get("counter", 0)
    new_val = current + 1
    print(f"  [Node 1: step_one] Counter: {current} → {new_val}")
    return {"counter": new_val}


def approval_gate(state: ToyState) -> dict:
    print(f"  [Node 2: approval_gate] Reached gate. Current counter = {state['counter']}")
    print("  [Node 2: approval_gate] Calling interrupt(). Graph will PAUSE now...")

    # Pauses graph execution here and emits payload to checkpointer
    decision = interrupt({
        "question": "Approve pipeline execution?",
        "current_counter": state["counter"],
    })

    print(f"  [Node 2: approval_gate] RESUMED! Received human decision = '{decision}'")
    return {"human_decision": decision}


def step_two(state: ToyState) -> dict:
    current = state.get("counter", 0)
    new_val = current + 1
    print(f"  [Node 3: step_two] Reading human_decision = '{state.get('human_decision')}'")
    print(f"  [Node 3: step_two] Counter: {current} → {new_val}")
    return {"counter": new_val}


# ---------------------------------------------------------------------------
# 3. Build & Compile Toy Graph
# ---------------------------------------------------------------------------


def build_toy_graph():
    builder = StateGraph(ToyState)

    builder.add_node("step_one", step_one)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("step_two", step_two)

    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "approval_gate")
    builder.add_edge("approval_gate", "step_two")
    builder.add_edge("step_two", END)

    # MemorySaver checkpointer is REQUIRED for interrupt/resume
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


toy_graph = build_toy_graph()


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ---------------------------------------------------------------------------
# 5. Test Suite
# ---------------------------------------------------------------------------


def test_immediate_resume() -> None:
    print_section("RUN 1: Immediate Interrupt & Resumption Test")

    thread_id = f"thread-immediate-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Step 1: Invoking graph initially (thread_id='{thread_id}')...")
    toy_graph.invoke({"counter": 0, "human_decision": None}, config=config)

    # Inspect graph state at the pause boundary
    print("\nStep 2: Checking state after pause...")
    state_snapshot = toy_graph.get_state(config)

    print(f"  • Current next node(s) queued : {state_snapshot.next}")
    print(f"  • Current counter in state   : {state_snapshot.values.get('counter')}")
    print(f"  • human_decision in state    : {state_snapshot.values.get('human_decision')}")

    # Confirm step_two has NOT run yet
    assert state_snapshot.values.get("counter") == 1, "Counter should be 1 (step_one only)"
    assert state_snapshot.values.get("human_decision") is None, "human_decision should be None before resume"
    assert "approval_gate" in state_snapshot.next, "Graph should be paused at approval_gate"

    # Extract interrupt payload
    interrupts = state_snapshot.tasks[0].interrupts if state_snapshot.tasks else ()
    assert len(interrupts) > 0, "Expected active interrupt payload in checkpointer"
    payload = interrupts[0].value
    print(f"  • Interrupt payload retrieved : {payload}")
    assert payload["question"] == "Approve pipeline execution?"
    print("  [PASS] Confirmed graph is safely PAUSED at approval_gate")

    # Simulate human approval
    print("\nStep 3: Simulating human approval by calling Command(resume='approved')...")
    resumed_state = toy_graph.invoke(Command(resume="approved"), config=config)

    print("\nStep 4: Verifying final state after resume...")
    print(f"  • Final counter               : {resumed_state.get('counter')}")
    print(f"  • Final human_decision        : {resumed_state.get('human_decision')}")

    assert resumed_state.get("counter") == 2, f"Expected counter=2, got {resumed_state.get('counter')}"
    assert resumed_state.get("human_decision") == "approved", "Expected human_decision='approved'"

    # Verify next is empty (reached END)
    final_snapshot = toy_graph.get_state(config)
    assert not final_snapshot.next, "Graph should have reached END (next is empty)"
    print("  [PASS] step_two executed successfully after resume")
    print("\n✅ RUN 1 PASSED")


def test_delayed_resume() -> None:
    print_section("RUN 2: Delayed Resumption Test (Simulating Multi-Second Delay)")

    thread_id = f"thread-delayed-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Step 1: Invoking graph initially (thread_id='{thread_id}')...")
    toy_graph.invoke({"counter": 0, "human_decision": None}, config=config)

    print("\nStep 2: Simulating a 2-second real-time delay (e.g. human reading notification)...")
    time.sleep(2.0)
    print("  ...2 seconds elapsed.")

    print("\nStep 3: Re-fetching state snapshot from checkpointer after delay...")
    snapshot = toy_graph.get_state(config)
    print(f"  • Re-fetched counter from checkpoint : {snapshot.values.get('counter')}")
    print(f"  • Next queued node                   : {snapshot.next}")
    assert snapshot.values.get("counter") == 1, "State should be preserved across delay"
    assert "approval_gate" in snapshot.next, "State should remain paused at approval_gate"
    print("  [PASS] Confirmed state genuinely persisted across time delay")

    print("\nStep 4: Simulating human rejection by calling Command(resume='rejected')...")
    resumed_state = toy_graph.invoke(Command(resume="rejected"), config=config)

    print("\nStep 5: Verifying final state after delayed resume...")
    print(f"  • Final counter        : {resumed_state.get('counter')}")
    print(f"  • Final human_decision : {resumed_state.get('human_decision')}")

    assert resumed_state.get("counter") == 2
    assert resumed_state.get("human_decision") == "rejected"
    print("\n✅ RUN 2 PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_immediate_resume()
        test_delayed_resume()
        print("\n🎉 All isolated interrupt/resume tests passed successfully.")
    except Exception as exc:
        import sys
        import traceback
        print(f"\n❌ Test failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
