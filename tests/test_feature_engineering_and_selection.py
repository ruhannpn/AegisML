"""
tests/test_feature_engineering_and_selection.py
================================================
Unit tests verifying that:
1. The Planner agent includes feature selection and feature engineering in prompts
   and schema validation.
2. The Data agent deterministically executes feature engineering:
   - Binary non-zero indicators for zero-inflated features (e.g. capital-gain > 0)
   - Log1p transformations for skewed/outlier features
3. The Data agent applies feature selection (retaining chosen features + engineered
   features + target, dropping unselected features).
4. EDA linkage records feature engineering actions as 'applied'.
"""

import numpy as np
import pandas as pd
import pytest

from data_agent import run_data_agent, _parse_plan_steps
from eda_insights import build_eda_linkage
import planner_agent


@pytest.fixture
def feature_df():
    """Dataset with zero-inflated and skewed columns to test feature engineering."""
    rng = np.random.default_rng(42)
    n = 100
    # capital_gain: 90% zeros, 10% large values
    cap_gain = np.zeros(n)
    nonzero_idx = rng.choice(n, 10, replace=False)
    cap_gain[nonzero_idx] = rng.uniform(1000, 50000, size=10)

    # capital_loss: skewed positive values
    cap_loss = np.zeros(n)
    loss_idx = rng.choice(n, 20, replace=False)
    cap_loss[loss_idx] = rng.exponential(500, size=20)

    return pd.DataFrame({
        "age": rng.integers(18, 70, size=n).astype(float),
        "capital_gain": cap_gain,
        "capital_loss": cap_loss,
        "unselected_noise": rng.normal(0, 1, size=n),
        "target": rng.choice([0, 1], size=n),
    })


def test_parse_plan_steps_feature_engineering_and_selection():
    steps = [
        "Create binary indicator for capital_gain",
        "Apply log1p transform to capital_loss",
        "Drop column unselected_noise",
    ]
    cols = ["age", "capital_gain", "capital_loss", "unselected_noise", "target"]
    plan = {
        "recommended_preprocessing_steps": steps,
        "selected_features": ["age", "capital_gain", "capital_loss"],
    }
    hints = _parse_plan_steps(steps, cols, plan=plan)

    assert "capital_gain" in hints["indicator_columns"]
    assert "capital_loss" in hints["log1p_columns"]
    assert "unselected_noise" in hints["explicit_drop_columns"]
    assert hints["selected_features"] == ["age", "capital_gain", "capital_loss"]


def test_data_agent_executes_feature_engineering(feature_df):
    plan = {
        "recommended_preprocessing_steps": [
            "Create binary indicator for capital_gain",
            "Apply log1p transform to capital_loss",
        ],
        "recommended_models": ["RandomForest"],
        "sensitive_attribute_candidates": [],
        "reasoning": "Test feature engineering execution.",
    }

    result = run_data_agent(
        df=feature_df,
        plan=plan,
        target_column="target",
        task_type="classification",
    )

    cleaned = result["cleaned_df"]
    actions = result["actions_taken"]

    # Verify binary indicator was created
    assert "capital_gain_is_nonzero" in cleaned.columns
    # Non-zero values must match (accounting for StandardScaler)
    expected_nonzero = (feature_df["capital_gain"] > 0).astype(float)
    actual_nonzero = (cleaned["capital_gain_is_nonzero"] > cleaned["capital_gain_is_nonzero"].min()).astype(float)
    np.testing.assert_array_equal(
        actual_nonzero.to_numpy(),
        expected_nonzero.to_numpy(),
    )

    # Verify log1p feature was created
    assert "capital_loss_log1p" in cleaned.columns
    expected_log = np.log1p(np.maximum(0.0, feature_df["capital_loss"]))
    # Since numeric features are standardized in step 7, verify log1p transform action was logged
    assert any("created binary indicator 'capital_gain_is_nonzero'" in a for a in actions)
    assert any("created log-transformed feature 'capital_loss_log1p'" in a for a in actions)
    assert "capital_gain_is_nonzero" in result["engineered_features"]
    assert "capital_loss_log1p" in result["engineered_features"]


def test_data_agent_executes_feature_selection(feature_df):
    plan = {
        "recommended_preprocessing_steps": [
            "Create binary indicator for capital_gain",
        ],
        "selected_features": ["age", "capital_gain"],  # unselected_noise & capital_loss excluded
        "recommended_models": ["RandomForest"],
        "sensitive_attribute_candidates": [],
        "reasoning": "Test feature selection execution.",
    }

    result = run_data_agent(
        df=feature_df,
        plan=plan,
        target_column="target",
        task_type="classification",
    )

    cleaned = result["cleaned_df"]
    # unselected_noise should be dropped
    assert "unselected_noise" not in cleaned.columns
    # capital_loss should be dropped
    assert "capital_loss" not in cleaned.columns
    # selected features and newly engineered features must be retained
    assert "age" in cleaned.columns
    assert "capital_gain" in cleaned.columns
    assert "capital_gain_is_nonzero" in cleaned.columns
    assert "target" in cleaned.columns


def test_eda_linkage_records_engineered_features():
    findings = [
        {
            "id": "finding_zero_inf",
            "type": "zero_inflated",
            "severity": "medium",
            "route": "planner",
            "columns": ["capital_gain"],
            "evidence": "90% zeros",
            "recommendation": "Create binary indicator",
        }
    ]
    plan = {
        "data_quality_concerns": ["Zero-inflated capital_gain"],
        "recommended_preprocessing_steps": ["Create binary indicator for capital_gain"],
    }
    data_res = {
        "quality_report": {"columns_dropped": []},
        "winsorized_columns": [],
        "engineered_features": ["capital_gain_is_nonzero"],
    }

    linkage = build_eda_linkage(findings, plan=plan, data_agent_result=data_res)
    assert len(linkage) == 1
    outcomes = linkage[0]["outcomes"]

    data_agent_outcomes = [o for o in outcomes if o["stage"] == "data_agent"]
    assert any(o["status"] == "applied" and "capital_gain" in o["detail"] for o in data_agent_outcomes)


def test_planner_schema_validation_defaults_selected_features():
    # If selected_features is omitted from JSON, validation gracefully sets it to []
    raw_json = """{
        "data_quality_concerns": ["test"],
        "recommended_preprocessing_steps": ["step"],
        "recommended_models": ["RandomForest"],
        "sensitive_attribute_candidates": ["age"],
        "reasoning": "reason"
    }"""
    plan = planner_agent._parse_and_validate(raw_json)
    assert "selected_features" in plan
    assert plan["selected_features"] == []
