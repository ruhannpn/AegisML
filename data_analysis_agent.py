"""
data_analysis_agent.py
======================
Autonomous Data Analysis & Exploratory Data Profiling Agent for AegisML.
Generates comprehensive EDA statistics, correlation matrices, outlier metrics,
missingness profiles, target distributions, and chart visualization payloads.
"""

from typing import Any
import numpy as np
import pandas as pd


def analyze_raw_dataset(df: pd.DataFrame, target_column: str = "") -> dict[str, Any]:
    """
    Perform comprehensive Exploratory Data Analysis (EDA) on a raw pandas DataFrame.

    Returns
    -------
    dict
        Contains: summary, column_profiles, correlations, outliers, target_analysis, charts
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    null_cells = int(df.isnull().sum().sum())
    clean_cells = total_cells - null_cells
    missing_pct = round((null_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    # 1. Column Profiles
    column_profiles = []
    outlier_counts = {}

    for col in df.columns:
        s = df[col]
        null_cnt = int(s.isnull().sum())
        null_p = round((null_cnt / n_rows * 100), 2) if n_rows > 0 else 0.0
        unique_cnt = int(s.nunique())
        dtype_str = str(s.dtype)

        col_stat = {
            "name": col,
            "dtype": dtype_str,
            "null_count": null_cnt,
            "null_pct": null_p,
            "cardinality": unique_cnt,
            "is_numeric": col in numeric_cols,
        }

        if col in numeric_cols:
            s_clean = s.dropna()
            if len(s_clean) > 0:
                col_stat["min"] = round(float(s_clean.min()), 4)
                col_stat["max"] = round(float(s_clean.max()), 4)
                col_stat["mean"] = round(float(s_clean.mean()), 4)
                col_stat["std"] = round(float(s_clean.std()), 4)

                # IQR Outlier Detection
                q1 = float(s_clean.quantile(0.25))
                q3 = float(s_clean.quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = int(((s_clean < lower_bound) | (s_clean > upper_bound)).sum())
                col_stat["outliers"] = outliers
                if outliers > 0:
                    outlier_counts[col] = outliers
            else:
                col_stat.update({"min": "N/A", "max": "N/A", "mean": "N/A", "std": "N/A", "outliers": 0})
        else:
            top_vals = s.value_counts().head(5).to_dict()
            col_stat["top_categories"] = {str(k): int(v) for k, v in top_vals.items()}

        column_profiles.append(col_stat)

    # 2. Correlation Analysis
    correlations = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val) and abs(val) >= 0.20:
                    correlations.append({
                        "feature_a": c1,
                        "feature_b": c2,
                        "correlation": round(float(val), 4)
                    })
        correlations = sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)[:10]

    # 3. Target Distribution & Chart Payloads
    target_info = {}
    chart_target_labels = []
    chart_target_values = []

    if target_column and target_column in df.columns:
        ts = df[target_column].dropna()
        if target_column in numeric_cols and ts.nunique() > 15:
            counts, bin_edges = np.histogram(ts, bins=10)
            chart_target_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(counts))]
            chart_target_values = [int(c) for c in counts]
            target_info = {
                "type": "numeric",
                "min": round(float(ts.min()), 4),
                "max": round(float(ts.max()), 4),
                "mean": round(float(ts.mean()), 4),
                "std": round(float(ts.std()), 4),
                "skew": round(float(ts.skew()), 4),
            }
        else:
            vc = ts.value_counts().head(10).to_dict()
            chart_target_labels = [str(k) for k in vc.keys()]
            chart_target_values = [int(v) for v in vc.values()]
            target_info = {
                "type": "categorical",
                "counts": {str(k): int(v) for k, v in vc.items()},
                "unique_classes": int(ts.nunique())
            }

    return {
        "summary": {
            "total_rows": n_rows,
            "total_columns": n_cols,
            "missing_cells": null_cells,
            "clean_cells": clean_cells,
            "missing_pct": missing_pct,
            "numeric_columns_count": len(numeric_cols),
            "categorical_columns_count": len(categorical_cols),
            "outlier_columns_count": len(outlier_counts),
        },
        "column_profiles": column_profiles,
        "correlations": correlations,
        "outliers": outlier_counts,
        "target_analysis": target_info,
        "charts": {
            "target_labels": chart_target_labels,
            "target_values": chart_target_values,
            "missingness": [clean_cells, null_cells],
            "correlation_labels": [f"{c['feature_a']} vs {c['feature_b']}" for c in correlations],
            "correlation_values": [c['correlation'] for c in correlations],
        }
    }
