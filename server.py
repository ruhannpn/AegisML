"""
server.py
=========
FastAPI REST Server for the AI-Governed Multi-Agent Data Science Platform.

Provides RESTful API endpoints for:
  - Starting initial pipeline execution (/api/pipeline/start)
  - Submitting human governance decisions (/api/pipeline/resume)
  - Inspecting pipeline status (/api/pipeline/status/{thread_id})
  - Retrieving audit log trail (/api/pipeline/audit/{thread_id})
  - Serving static Web UI dashboard (/)
"""

from __future__ import annotations

import os
import uuid
import io
import pandas as pd
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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

from langgraph.types import Command
from graph_state import df_to_bytes
from pipeline_graph import graph
from audit_log import get_audit_trail

app = FastAPI(
    title="AI Multi-Agent Governance API",
    description="REST backend for LangGraph Multi-Agent Data Science Governance Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str


def _build_pipeline_response(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    next_nodes = list(snapshot.next)

    is_paused = "human_approval_node" in next_nodes
    payload = None
    if is_paused and snapshot.tasks and snapshot.tasks[0].interrupts:
        payload = snapshot.tasks[0].interrupts[0].value

    return {
        "thread_id": thread_id,
        "status": "paused" if is_paused else ("completed" if not next_nodes else "running"),
        "next_nodes": next_nodes,
        "review_payload": payload,
        "values": {
            "unresolved_human_rejection": snapshot.values.get("unresolved_human_rejection", False),
            "unresolved_quality_issue": snapshot.values.get("unresolved_quality_issue", False),
            "human_decision": snapshot.values.get("human_decision"),
            "retry_count": snapshot.values.get("retry_count", 0),
            "rejection_reroute_count": snapshot.values.get("rejection_reroute_count", 0),
        },
    }


@app.post("/api/pipeline/start")
async def start_pipeline(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...),
    business_objective: Optional[str] = Form(None),
):
    """
    Start initial pipeline execution for an uploaded dataset.
    """
    if task_type not in ("classification", "regression"):
        raise HTTPException(status_code=400, detail="task_type must be 'classification' or 'regression'")

    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {exc}")

    if target_column not in df_raw.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found in CSV. Available columns: {list(df_raw.columns)}",
        )

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set in environment or .env file.",
        )

    thread_id = f"run-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "df_bytes": df_to_bytes(df_raw),
        "target_column": target_column,
        "task_type": task_type,
        "business_objective": business_objective.strip() if business_objective and business_objective.strip() else None,
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
        "rejected_models": [],
    }

    try:
        graph.invoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {exc}")

    return _build_pipeline_response(thread_id)


@app.post("/api/pipeline/resume")
async def resume_pipeline(req: ResumeRequest):
    """
    Submit human governance decision to resume graph execution.
    """
    allowed = ("approve", "reject_data_quality", "reject_model_or_fairness")
    if req.decision not in allowed:
        raise HTTPException(status_code=400, detail=f"Decision must be one of {allowed}")

    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        graph.invoke(Command(resume=req.decision), config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error resuming graph execution: {exc}")

    return _build_pipeline_response(req.thread_id)


@app.get("/api/pipeline/status/{thread_id}")
async def get_pipeline_status(thread_id: str):
    """
    Get current snapshot status for a given thread_id.
    """
    return _build_pipeline_response(thread_id)


@app.get("/api/pipeline/audit/{thread_id}")
async def get_audit_log(thread_id: str):
    """
    Retrieve chronological audit log trail for a given thread_id.
    """
    trail = get_audit_trail(thread_id)
    return {"thread_id": thread_id, "entries": trail}


# Serve static web frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
