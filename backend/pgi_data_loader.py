# pgi_data_loader.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from pgi_model import PGIModel

def _read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read CSV {path}: {e}")
        return pd.DataFrame()

def _find_file_by_tokens(tokens, files_in_dir):
    """
    Return the first filename in files_in_dir that contains ALL tokens (case-insensitive).
    tokens: list of strings (substrings to match)
    """
    tokens = [t.lower() for t in tokens if t]
    for fname in files_in_dir:
        low = fname.lower()
        if all(t in low for t in tokens):
            return fname
    return None

def load_pgi_dataset():
    st.info("Searching folder for ILOSTAT CSVs (flexible filename matching)...", icon="🔎")

    # mapping of indicator -> list of tokens likely present in file name
    expected = {
        "Earnings": ["earn"],
        "EmpPop": ["employment", "population", "pop"],
        "Informal": ["informal", "informality"],
        "LFPR": ["labour", "force", "participation", "lfpr"],
        "Unemp": ["unemp", "unemployment"],
        "Poverty": ["poverty", "working poverty", "working_poverty"],
        "NEET": ["neet", "youth", "youth neet"]
    }

    # list files in current working dir
    cwd_files = [f for f in os.listdir(".") if os.path.isfile(f)]
    # prioritise exact matches if present
    dfs = {}

    for key, tokens in expected.items():
        # try exact filename first (common case)
        exact_names = [
            f"{key}.csv",
            f"{key}.CSV",
            # also try common explicit names
            "Earnings.csv" if key=="Earnings" else None,
            "Employment to population ratio.csv" if key=="EmpPop" else None,
            "Informal employment rate.csv" if key=="Informal" else None,
            "Labour Force participation rate.csv" if key=="LFPR" else None,
            "Unemployment rate.csv" if key=="Unemp" else None,
            "working poverty rate.csv" if key=="Poverty" else None,
            "YOUTH NEET rate.csv" if key=="NEET" else None
        ]
        found = None
        for en in [n for n in exact_names if n]:
            if en in cwd_files:
                found = en
                break

        # if not exact, try token-based fuzzy match
        if not found:
            found = _find_file_by_tokens(tokens, cwd_files)

        if not found:
            st.info(f"No file found for '{key}' (tokens={tokens}). This indicator will use neutral defaults.", icon="ℹ️")
            continue

        
        df = _read_csv_safe(found)
        if df.empty:
            st.warning(f"File {found} was empty or couldn't be parsed; skipping.", icon="⚠️")
            continue

        # attempt to find Area, Year, and a numeric value column
        cols_lower = [c.lower() for c in df.columns]
        # find area column
        area_col = next((c for c in df.columns if c.lower() in ("area", "country", "country or area", "location", "geo")), None)
        # find year column
        year_col = next((c for c in df.columns if "year" in c.lower() or c.lower() in ("time", "period")), None)
        # find total/value column: prefer "total" or a numeric column
        total_col = next((c for c in df.columns if c.lower() in ("total","value","observed_value","obs_value")), None)
        if total_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # prefer columns that are not year if possible
            if year_col and year_col in numeric_cols and len(numeric_cols) > 1:
                numeric_cols = [c for c in numeric_cols if c != year_col]
            total_col = numeric_cols[0] if numeric_cols else None

        # fallback inference if necessary
        if area_col is None:
            str_cols = df.select_dtypes(include=['object']).columns.tolist()
            area_col = str_cols[0] if str_cols else None
        if year_col is None:
            # try find integer-like column
            cand = next((c for c in df.columns if df[c].dropna().apply(lambda v: isinstance(v, (int, np.integer))).all()), None)
            year_col = cand
        if not (area_col and year_col and total_col):
            st.warning(f"Could not infer columns Area/Year/Value in {found}. Skipping this file.", icon="⚠️")
            continue

        small = df[[area_col, year_col, total_col]].copy()
        small.columns = ["Area", "Year", "Total"]
        small["Area"] = small["Area"].astype(str).str.strip()
        # coerce Year to int where possible
        try:
            small["Year"] = small["Year"].astype(int)
        except Exception:
            small["Year"] = pd.to_numeric(small["Year"], errors="coerce").astype('Int64')
        small["Total"] = pd.to_numeric(small["Total"], errors="coerce")
        small.dropna(subset=["Area", "Year"], inplace=True)
        dfs[key] = small.rename(columns={"Total": key})

    if not dfs:
        st.error("No usable ILOSTAT CSVs found (after flexible matching). Please place CSVs in the app folder or upload them.", icon="❌")
        return pd.DataFrame()

    # Merge available dfs (outer join to preserve rows)
    # Prefer Earnings as base if present
    if "Earnings" in dfs:
        data = dfs["Earnings"].copy()
    else:
        first_key = list(dfs.keys())[0]
        st.warning(f"'Earnings' not found; using '{first_key}' as base; note PGI requires earnings (P0).", icon="⚠️")
        data = dfs[first_key].copy()

    for k, d in dfs.items():
        if k == ("Earnings" if "Earnings" in dfs else first_key):
            continue
        data = data.merge(d, on=["Area", "Year"], how="outer")

    # Normalize columns safely (min==max guard)
    num_cols = ["Earnings", "EmpPop", "Informal", "LFPR", "Unemp", "Poverty", "NEET"]
    for col in num_cols:
        if col in data.columns:
            vals = pd.to_numeric(data[col], errors="coerce")
            if vals.dropna().shape[0] > 1 and vals.max() != vals.min():
                data[col + "_norm"] = (vals - vals.min()) / (vals.max() - vals.min())
            else:
                data[col + "_norm"] = 0.5
        else:
            data[col + "_norm"] = 0.5

    # Compute automation proxy A (consistent with ERI)
    data["A"] = ((data["Unemp_norm"].fillna(0.5)) + (1 - data["EmpPop_norm"].fillna(0.5))) / 2

    # Ensure Earnings numeric column exists
    data["Earnings"] = pd.to_numeric(data.get("Earnings", np.nan), errors="coerce")

    # Compute PGI_raw using PGIModel (alpha default 0.4 — you can expose as param)
    model = PGIModel(alpha=0.4)
    def safe_pgi(row):
        p0 = row.get("Earnings")
        a = row.get("A", 0.0)
        if pd.isna(p0):
            return np.nan
        try:
            return model.compute_pgi_raw(float(p0), float(a))
        except Exception:
            return np.nan

    data["PGI_raw"] = data.apply(safe_pgi, axis=1)

    # PGI_pct = (P - P0)/P0
    def safe_pct(row):
        p0 = row.get("Earnings")
        p = row.get("PGI_raw")
        if pd.isna(p0) or pd.isna(p) or p0 == 0:
            return np.nan
        return (p - p0) / p0

    data["PGI_pct"] = data.apply(safe_pct, axis=1)
    data["PGI_pct"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Robust clipping and scale to 0..1 (1st-99th percentile)
    if data["PGI_pct"].dropna().shape[0] > 0:
        p1 = float(data["PGI_pct"].quantile(0.01))
        p99 = float(data["PGI_pct"].quantile(0.99))
        denom = p99 - p1 if (p99 - p1) != 0 else 1.0
        data["PGI_index"] = data["PGI_pct"].clip(lower=p1, upper=p99).fillna(p1).apply(lambda v: (v - p1) / denom)
    else:
        st.warning("No PGI_pct values available (likely missing earnings). Setting PGI_index=0 for visualization.", icon="⚠️")
        data["PGI_index"] = 0.0

    data["PGI_index"] = pd.to_numeric(data["PGI_index"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    out = data[["Area", "Year", "A", "Earnings", "PGI_raw", "PGI_pct", "PGI_index"]].copy()
    out = out.sort_values(by=["Area", "Year"]).reset_index(drop=True)
    st.success(f"PGI loader prepared {out.shape[0]} rows from {len(dfs)} source files.", icon="✅")
    return out
