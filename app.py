"""
app.py
======
Streamlit UI for the AI-Governed Multi-Agent Platform.

Provides a premium interactive web interface for:
  1. Dataset upload, target/task selection, and optional Business Objective input
  2. Executing the LangGraph pipeline (Planner → Data → Training → Fairness)
  3. Interactive Governance Approval interrupt boundary (reviewing plan, quality report,
     leaderboard, fairness metrics, submitting approve/reject decisions)
  4. Automatic reroute handling (data quality rejection -> planner, model rejection -> training)
  5. Governance Audit Trail visualization (chronological event log from audit_log.db)
"""

from __future__ import annotations

import json
import os
import uuid
import pandas as pd
import streamlit as st
from langgraph.types import Command

# Auto-load .env if GROQ_API_KEY is not already in environment
if "GROQ_API_KEY" not in os.environ:
    env_paths = [".env", os.path.join(os.path.dirname(__file__), ".env")]
    for path in env_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[7:]
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip("'\"")

from graph_state import df_to_bytes, bytes_to_df, bytes_to_model
from pipeline_graph import graph
from audit_log import get_audit_trail


# ---------------------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Governance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetics
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #4CAF50;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #E0E0E0;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9E9E9E;
        margin-bottom: 1.5rem;
    }
    .badge-pass {
        background-color: #1b5e20;
        color: #81c784;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-fail {
        background-color: #b71c1c;
        color: #ef9a9a;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .card-box {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ AI-Governed Multi-Agent Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An enterprise governance architecture orchestrating '
    '<b>Planner</b>, <b>Data</b>, <b>Training</b>, and <b>Fairness</b> agents into a '
    'human-in-the-loop workflow via LangGraph.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = f"run-{uuid.uuid4().hex[:8]}"

if "pipeline_started" not in st.session_state:
    st.session_state["pipeline_started"] = False

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

if "last_error_details" not in st.session_state:
    st.session_state["last_error_details"] = None


# ---------------------------------------------------------------------------
# Sidebar Controls & Architecture
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Session Controls")
    st.info(f"**Current Run ID:** `{st.session_state['thread_id']}`")

    # API Key status check
    api_key_status = "✅ Set" if os.environ.get("GROQ_API_KEY") else "❌ Missing"
    st.markdown(f"**Groq API Key Status:** `{api_key_status}`")

    if st.button("🔄 Start New Run", use_container_width=True):
        st.session_state["thread_id"] = f"run-{uuid.uuid4().hex[:8]}"
        st.session_state["pipeline_started"] = False
        st.session_state["last_error"] = None
        st.session_state["last_error_details"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧩 Agent DAG Architecture")
    st.markdown("1. 🧠 **Planner Agent** *(LLM - Groq)*")
    st.markdown("2. 🧹 **Data Agent** *(Deterministic Pandas)*")
    st.markdown("3. 🏆 **Training Agent** *(Deterministic Sklearn/XGB)*")
    st.markdown("4. ⚖️ **Fairness Agent** *(Deterministic DI/DPD)*")
    st.markdown("5. ⏸️ **Human Approval Gate** *(LangGraph Interrupt)*")
    st.markdown("6. 📜 **Audit Log** *(SQLite DB)*")


# ---------------------------------------------------------------------------
# Step 1: Configuration & Upload Section
# ---------------------------------------------------------------------------

thread_id = st.session_state["thread_id"]
config = {"configurable": {"thread_id": thread_id}}

st.subheader("1. Dataset & Pipeline Setup")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Loaded CSV: **{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns")

        with st.expander("Preview Raw Dataset", expanded=False):
            st.dataframe(df_raw.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox(
                "Select Target / Label Column",
                options=list(df_raw.columns),
                index=len(df_raw.columns) - 1 if "income" not in df_raw.columns else list(df_raw.columns).index("income"),
            )
        with col2:
            task_type = st.radio(
                "Select Task Type",
                options=["classification", "regression"],
                index=0,
                horizontal=True,
            )

        # Business Objective Input
        business_objective = st.text_area(
            "Business Objective & Governance Constraints (Optional)",
            placeholder="e.g. Optimize for high precision, ensure strict algorithmic fairness across gender/race, prioritize model interpretability...",
            help="Your business objective will be passed to the Planner Agent to align preprocessing and model selection recommendations.",
        )

        # Pre-Run Validation Checks
        validation_passed = True
        if target_column not in df_raw.columns:
            st.error(f"❌ Target column '{target_column}' is not present in the uploaded dataset columns.")
            validation_passed = False

        if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
            st.error("❌ `GROQ_API_KEY` is not populated. Please add `GROQ_API_KEY=your_key` to `.env` file.")
            validation_passed = False

        if validation_passed and not st.session_state["pipeline_started"]:
            if st.button("🚀 Run Governed Pipeline", type="primary", use_container_width=True):
                st.session_state["pipeline_started"] = True
                st.session_state["last_error"] = None
                st.session_state["last_error_details"] = None

                initial_state = {
                    "df_bytes": df_to_bytes(df_raw),
                    "target_column": target_column,
                    "task_type": task_type,
                    "business_objective": business_objective.strip(),
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

                with st.spinner("Executing Pipeline DAG (Planner → Data → Training → Fairness)..."):
                    try:
                        graph.invoke(initial_state, config=config)
                    except Exception as exc:
                        import traceback
                        st.session_state["last_error"] = str(exc)
                        st.session_state["last_error_details"] = traceback.format_exc()
                st.rerun()

    except Exception as exc:
        st.error(f"Failed to parse CSV file: {exc}")
else:
    st.info("Please upload a CSV dataset to begin (e.g. UCI Adult Income dataset).")


# Display runtime errors if any occurred
if st.session_state["last_error"]:
    st.error(f"❌ **Pipeline Execution Error:** {st.session_state['last_error']}")
    if st.session_state["last_error_details"]:
        with st.expander("Detailed Error Traceback"):
            st.code(st.session_state["last_error_details"])


# ---------------------------------------------------------------------------
# Step 2 & 3: Pipeline Status & Governance Interrupt Boundary
# ---------------------------------------------------------------------------

if st.session_state["pipeline_started"]:
    st.markdown("---")
    st.subheader("2. Governance Review & Pipeline Status")

    snapshot = graph.get_state(config)
    next_nodes = snapshot.next

    # --- Scenario A: Paused at Human Approval Gate ---
    if "human_approval_node" in next_nodes:
        st.warning("⏳ **Pipeline Paused at Human Approval Gate** — Mandatory Governance Audit Required")

        if snapshot.tasks and snapshot.tasks[0].interrupts:
            payload = snapshot.tasks[0].interrupts[0].value
            plan_sum = payload.get("plan_summary", {})
            data_actions = payload.get("data_agent_actions", [])
            quality_rep = payload.get("quality_report", {})
            leaderboard = payload.get("leaderboard", [])
            fairness_rep = payload.get("fairness_report", [])
            skipped_attrs = payload.get("attributes_skipped", [])
            overall_fairness = payload.get("overall_fairness_passed", False)
            unresolved_quality = payload.get("unresolved_quality_issue", False)

            if unresolved_quality:
                st.error("⚠️ **Quality Concern Warning:** Automated data cleaning hit retry limits with unresolved quality concerns.")

            tab1, tab2, tab3, tab4 = st.tabs([
                "🧠 Planner Proposal",
                "🧹 Data Cleaning Report",
                "🏆 Model Leaderboard",
                "⚖️ Fairness Assessment",
            ])

            # Tab 1: Planner Proposal
            with tab1:
                st.markdown("### Planner Agent Proposal")
                st.markdown(f"**Reasoning:** {plan_sum.get('reasoning', 'N/A')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Data Quality Concerns Identified:**")
                    for c in plan_sum.get("data_quality_concerns", []):
                        st.markdown(f"- `{c}`")
                    st.markdown("**Recommended Models:**")
                    for m in plan_sum.get("recommended_models", []):
                        st.markdown(f"- `{m}`")

                with col2:
                    st.markdown("**Recommended Preprocessing Steps:**")
                    for s in plan_sum.get("recommended_preprocessing_steps", []):
                        st.markdown(f"- `{s}`")
                    st.markdown("**Sensitive Attribute Candidates:**")
                    for a in plan_sum.get("sensitive_attribute_candidates", []):
                        st.markdown(f"- `{a}`")

            # Tab 2: Data Agent Actions & Quality Report
            with tab2:
                st.markdown("### Data Agent Actions & Quality Report")
                qcol1, qcol2, qcol3 = st.columns(3)
                qcol1.metric("Missing % After Cleaning", f"{quality_rep.get('missing_pct_after_cleaning', 0)}%")
                qcol2.metric("Rows Dropped", f"{quality_rep.get('rows_dropped', 0):,} ({quality_rep.get('rows_dropped_pct', 0)}%)")
                qcol3.metric("Columns Dropped", f"{len(quality_rep.get('columns_dropped', []))}")

                if quality_rep.get("columns_dropped"):
                    st.warning(f"Columns explicitly dropped: `{quality_rep['columns_dropped']}`")

                st.markdown("**Executed Preprocessing Actions:**")
                for act in data_actions:
                    st.markdown(f"• {act}")

            # Tab 3: Model Leaderboard
            with tab3:
                st.markdown("### Model Training Leaderboard")
                selected_name = payload.get("selected_model_name")
                st.success(f"**Selected Best Model:** `{selected_name}`")

                if leaderboard:
                    is_regression = any("rmse" in e.get("metrics", {}) for e in leaderboard)
                    df_board = []
                    for rank, entry in enumerate(leaderboard, 1):
                        m_name = entry["model_name"]
                        metrics = entry.get("metrics", {})
                        is_selected = "⭐ Winner" if m_name == selected_name else ""

                        if is_regression:
                            row = {
                                "Rank": rank,
                                "Model Name": m_name,
                                "Selection": is_selected,
                                "RMSE (↓)": f"{metrics.get('rmse', 0):.4f}" if metrics.get('rmse') is not None else "N/A",
                                "MAE (↓)": f"{metrics.get('mae', 0):.4f}" if metrics.get('mae') is not None else "N/A",
                                "R² Score (↑)": f"{metrics.get('r2', 0):.4f}" if metrics.get('r2') is not None else "N/A",
                                "Status": "Trained" if entry.get("trained_successfully") else "Failed",
                            }
                        else:
                            row = {
                                "Rank": rank,
                                "Model Name": m_name,
                                "Selection": is_selected,
                                "Accuracy": f"{metrics.get('accuracy', 0):.4f}" if metrics.get('accuracy') is not None else "N/A",
                                "F1 Score": f"{metrics.get('f1', 0):.4f}" if metrics.get('f1') is not None else "N/A",
                                "AUC-ROC": f"{metrics.get('auc_roc', 0):.4f}" if metrics.get('auc_roc') is not None else "N/A",
                                "Status": "Trained" if entry.get("trained_successfully") else "Failed",
                            }
                        df_board.append(row)
                    st.dataframe(pd.DataFrame(df_board), use_container_width=True)

            # Tab 4: Fairness Assessment
            with tab4:
                st.markdown("### Algorithmic Fairness Assessment")
                if overall_fairness:
                    st.markdown('<span class="badge-pass">🟢 OVERALL FAIRNESS STATUS: PASSED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-fail">🔴 OVERALL FAIRNESS STATUS: VIOLATION DETECTED</span>', unsafe_allow_html=True)

                st.markdown("")
                if fairness_rep:
                    df_fair = []
                    for entry in fairness_rep:
                        attr = entry["attribute"]
                        di = entry["disparate_impact"]
                        dpd = entry["demographic_parity_difference"]
                        viol = entry["violation"]
                        gd = entry.get("group_details", {})

                        status_str = "🔴 VIOLATION" if viol else "🟢 PASSED"
                        rates_str = f"Max: {gd.get('group_a')} ({gd.get('group_a_positive_rate')}) vs Min: {gd.get('group_b')} ({gd.get('group_b_positive_rate')})"

                        df_fair.append({
                            "Attribute": attr,
                            "Status": status_str,
                            "Disparate Impact (≥ 0.80)": f"{di:.4f}",
                            "Demographic Parity Diff (≤ 0.10)": f"{dpd:.4f}",
                            "Sub-Group Positive Rates": rates_str,
                        })
                    st.dataframe(pd.DataFrame(df_fair), use_container_width=True)

                if skipped_attrs:
                    with st.expander("Attributes Skipped & Audit Reasons"):
                        for sk in skipped_attrs:
                            st.markdown(f"- `{sk}`")

            # --- Governance Decision Input Form ---
            st.markdown("---")
            st.markdown("### ✍️ Submit Governance Decision")

            decision_labels = {
                "approve": "🟢 Approve — Deploy model to production",
                "reject_data_quality": "🔴 Reject — Data quality concerns (Reroute to Planner)",
                "reject_model_or_fairness": "🔴 Reject — Model choice or fairness concerns (Reroute directly to Training)",
            }

            chosen_label = st.radio(
                "Select Governance Action",
                options=list(decision_labels.values()),
                index=0,
            )
            # Map selected human label back to exact backend value
            label_to_val = {v: k for k, v in decision_labels.items()}
            chosen_val = label_to_val[chosen_label]

            if st.button("Submit Governance Decision", type="primary", use_container_width=True):
                with st.spinner(f"Submitting decision '{chosen_val}' and executing graph reroute..."):
                    try:
                        graph.invoke(Command(resume=chosen_val), config=config)
                        st.session_state["last_error"] = None
                        st.session_state["last_error_details"] = None
                    except Exception as exc:
                        import traceback
                        st.session_state["last_error"] = str(exc)
                        st.session_state["last_error_details"] = traceback.format_exc()
                st.rerun()

    # --- Scenario B: Terminal State (Completed / Capped) ---
    elif not next_nodes:
        vals = snapshot.values
        if vals.get("unresolved_human_rejection"):
            st.error("🛑 **Pipeline Halted:** Human rejection cap reached (2 max reroutes used). Governance approval failed.")
        elif vals.get("unresolved_quality_issue"):
            st.error("🛑 **Pipeline Halted:** Data Agent retry cap reached. Automated cleaning failed to resolve quality issues.")
        elif vals.get("human_decision") == "approve":
            st.success("🎉 **Pipeline Approved!** Governance approval granted — model and data passed all audit checks.")

    # -----------------------------------------------------------------------
    # Step 4: Governance Audit Trail View
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Governance Audit Trail")

    trail = get_audit_trail(thread_id)
    if trail:
        st.markdown(f"**Total Registered Audit Events:** {len(trail)}")

        audit_display = []
        for entry in trail:
            icon = "👤" if entry["event_source"] == "human_reviewer" else "🤖"
            audit_display.append({
                "ID": entry["id"],
                "Timestamp (UTC)": entry["timestamp"],
                "Source": f"{icon} {entry['event_source']}",
                "Event Type": entry["event_type"],
                "Summary": entry["summary"],
            })

        st.dataframe(pd.DataFrame(audit_display), use_container_width=True)

        with st.expander("Inspect Detailed Event JSON Records"):
            for entry in trail:
                icon = "👤" if entry["event_source"] == "human_reviewer" else "🤖"
                st.markdown(f"**Event #{entry['id']} — {icon} `{entry['event_type']}`** ({entry['timestamp']})")
                st.json(entry["details"])
    else:
        st.info("No audit entries logged yet for this run.")
