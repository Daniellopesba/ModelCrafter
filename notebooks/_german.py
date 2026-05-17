"""Loader for the UCI German Credit dataset (raw file in data/raw/german.data).

Decodes the symbolic A-codes into readable category labels and the 1/2
target into a 0/1 ``default`` flag (the UCI convention is 1 = good,
2 = bad — we flip to the credit-risk convention 1 = default).

This helper exists so the example notebooks can focus on modelling
rather than data wrangling. The data dictionary lives in
``data/raw/german.doc``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since_years",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "n_existing_credits",
    "job",
    "n_dependents",
    "telephone",
    "foreign_worker",
    "target_raw",
]

# Decode tables — see data/raw/german.doc.
_DECODE = {
    "checking_status": {
        "A11": "<0DM",
        "A12": "0-200DM",
        "A13": ">=200DM",
        "A14": "no_account",
    },
    "credit_history": {
        "A30": "none_or_paid",
        "A31": "paid_at_this_bank",
        "A32": "paid_so_far",
        "A33": "delays_in_past",
        "A34": "critical",
    },
    "purpose": {
        "A40": "car_new",
        "A41": "car_used",
        "A42": "furniture",
        "A43": "radio_tv",
        "A44": "appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings": {
        "A61": "<100DM",
        "A62": "100-500DM",
        "A63": "500-1000DM",
        "A64": ">=1000DM",
        "A65": "unknown",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "<1yr",
        "A73": "1-4yr",
        "A74": "4-7yr",
        "A75": ">=7yr",
    },
    "personal_status_sex": {
        "A91": "male_div",
        "A92": "female_div_mar",
        "A93": "male_single",
        "A94": "male_mar_wid",
        "A95": "female_single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co_applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real_estate",
        "A122": "building_society",
        "A123": "car_other",
        "A124": "none",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "free",
    },
    "job": {
        "A171": "unskilled_nonres",
        "A172": "unskilled_resident",
        "A173": "skilled",
        "A174": "management",
    },
    "telephone": {"A191": "none", "A192": "registered"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def load_german(path: str | Path | None = None) -> pd.DataFrame:
    """Return the tidied German Credit frame.

    Categorical columns are decoded to readable labels; the target is
    flipped to ``default`` (1 = bad, 0 = good).
    """
    if path is None:
        # data/raw/german.data sits at the repo root; this file lives at
        # notebooks/_german.py, so walk up one level.
        path = Path(__file__).resolve().parent.parent / "data" / "raw" / "german.data"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    for col, mapping in _DECODE.items():
        df[col] = df[col].map(mapping).astype("object")
    # 1 = good, 2 = bad → default flag (1 = bad).
    df["default"] = (df["target_raw"] == 2).astype(int)
    df = df.drop(columns=["target_raw"])
    return df


def add_synthetic_origination(
    df: pd.DataFrame,
    *,
    start: str = "2018-01-01",
    end: str = "2022-12-31",
    seed: int = 0,
) -> pd.DataFrame:
    """Attach a synthetic ``origination_dt`` so the temporal CV machinery has
    a time column to chew on.

    The dates are random uniform over ``[start, end]`` with a fixed seed
    so the demo is reproducible. The result is *not* a real time series —
    the labels are exchangeable across the date axis — but it suffices to
    exercise the splitters and the performance-over-time API.
    """
    rng = np.random.default_rng(seed)
    span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    offsets = rng.integers(0, span_days + 1, size=len(df))
    dates = pd.to_datetime(start) + pd.to_timedelta(offsets, unit="D")
    out = df.copy()
    out["origination_dt"] = dates.sort_values().to_numpy()  # sort so the index has chronology
    return out.sort_values("origination_dt").reset_index(drop=True)
