# edm_data_loader.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from edm_model import EDMModel

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

def load_edm_dataset():
    st.info("Searching folder for ILOSTAT CSVs (flexible filename matching)...", icon="🔍")

    # mapping of indicator -> list of tokens likely present in file name
    expected = {
        "EmpPop": ["employment", "population", "ratio"],
        "Unemp": ["unemployment", "rate"],
        "LFPR": ["labour", "force", "participation", "lfpr"],
        "Informal": ["informal", "informality"],
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
            "Employment to population ratio.csv" if key=="EmpPop" else None,
            "Unemployment rate.csv" if key=="Unemp" else None,
            "Labour Force participation rate.csv" if key=="LFPR" else None,
            "Informal employment rate.csv" if key=="Informal" else None,
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
    # Start with first available dataset
    first_key = list(dfs.keys())[0]
    data = dfs[first_key].copy()

    for k, d in dfs.items():
        if k == first_key:
            continue
        data = data.merge(d, on=["Area", "Year"], how="outer")

    # Normalize columns safely (min==max guard)
    num_cols = ["EmpPop", "Unemp", "LFPR", "Informal", "Poverty", "NEET"]
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
    # A combines unemployment and inverse employment-to-population ratio
    data["A"] = ((data["Unemp_norm"].fillna(0.5)) + (1 - data["EmpPop_norm"].fillna(0.5))) / 2

    # ========================================================================
    # Derive Employment from EmpPop Ratio and Population
    # ========================================================================
    # Employment = EmpPop_ratio * Population
    # We estimate Population from World Bank or use a proxy
    
    # Create a mapping of countries to approximate populations (simplified)
    population_proxy = {
        "India": 1417173000,
        "China": 1425887337,
        "United States": 338289857,
        "Indonesia": 277534122,
        "Pakistan": 240485658,
        "Brazil": 215313498,
        "Nigeria": 223804632,
        "Bangladesh": 171186372,
        "Russia": 144444359,
        "Mexico": 128932753,
        "Japan": 123294513,
        "Ethiopia": 130000000,
        "Philippines": 120595548,
        "Egypt": 110000000,
        "Germany": 84405100,
        "Vietnam": 98186856,
        "DR Congo": 99010000,
        "Turkey": 86749700,
        "Iran": 91567416,
        "Thailand": 71801915,
        "United Kingdom": 67736802,
        "Tanzania": 65497748,
        "France": 68017000,
        "South Africa": 60142978,
        "Kenya": 54054487,
        "Myanmar": 54732500,
        "Sudan": 47753632,
        "Uganda": 48582220,
        "Angola": 36815961,
        "Algeria": 44945000,
        "Iraq": 43533592,
        "Canada": 39742154,
        "Afghanistan": 42972958,
        "Ukraine": 38000000,
        "Saudi Arabia": 36408820,
        "Morocco": 38081755,
        "Uzbekistan": 35896996,
        "Malaysia": 34305500,
        "Yemen": 34449825,
        "Peru": 34352719,
        "Angola": 36815961,
        "Australia": 26603400,
        "Colombia": 52085168,
        "Sri Lanka": 22156000,
        "Syria": 22125490,
        "Poland": 37654000,
        "Romania": 18970458,
        "Chile": 19600000,
        "Kazakhstan": 20331129,
        "Tajikistan": 10143200,
        "Netherlands": 17750000,
        "South Korea": 51908400,
        "Greece": 10640801,
        "Portugal": 10426199,
        "Austria": 9108202,
        "Hungary": 9673107,
        "Sweden": 10549347,
        "Azerbaijan": 10139177,
        "Belgium": 11690814,
        "Tunisia": 12356117,
        "Cuba": 10500981,
        "Czech Republic": 10510785,
        "Greece": 10640801,
        "Israel": 9656842,
        "Switzerland": 8776000,
        "Bulgaria": 6840000,
        "Serbia": 6690121,
        "Hong Kong": 7685600,
        "Denmark": 5903037,
        "Singapore": 5917600,
        "Slovakia": 5460721,
        "Finland": 5571665,
        "Norway": 5547933,
        "Ireland": 5127900,
        "New Zealand": 5228100,
        "Costa Rica": 5180829,
        "Lebanon": 5489094,
        "Panama": 4408581,
        "Iceland": 397413,
        "Luxembourg": 683201,
    }

    # Estimate population from country name if not in ILOSTAT
    def get_population(area):
        # Try exact match first
        if area in population_proxy:
            return population_proxy[area]
        # Try partial match
        for key, val in population_proxy.items():
            if key.lower() in area.lower() or area.lower() in key.lower():
                return val
        # Default fallback: return None (will skip calculation)
        return None

    data["Population"] = data["Area"].apply(get_population)
    
    # Calculate Employment = (EmpPop / 100) * Population
    # EmpPop is typically in percentage (0-100), so divide by 100
    def calc_employment(row):
        emp_pop = row.get("EmpPop")
        pop = row.get("Population")
        
        if pd.isna(emp_pop) or pd.isna(pop) or pop is None:
            return np.nan
        
        # Assume EmpPop is in percentage
        try:
            emp_ratio = float(emp_pop) / 100.0 if float(emp_pop) > 1 else float(emp_pop)
            return emp_ratio * pop
        except Exception:
            return np.nan
    
    data["Employment"] = data.apply(calc_employment, axis=1)

    st.info(f"✓ Calculated Employment from Employment-to-Population ratio and population proxy. {data['Employment'].notna().sum()} rows with valid employment data.", icon="✅")

    # Time horizon: compute years from first available year per country
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["YearMin"] = data.groupby("Area")["Year"].transform("min")
    data["TimeYears"] = (data["Year"] - data["YearMin"]).fillna(0)

    # Compute EDM_raw using EDMModel (beta default 0.3)
    model = EDMModel(beta=0.3)
    def safe_edm(row):
        d0 = row.get("Employment")
        a = row.get("A", 0.0)
        t = row.get("TimeYears", 0.0)
        if pd.isna(d0) or d0 <= 0:
            return np.nan
        try:
            return model.compute_edm_raw(float(d0), float(a), float(t))
        except Exception:
            return np.nan

    data["EDM_raw"] = data.apply(safe_edm, axis=1)

    # EDM_pct = (D(t) - D₀)/D₀
    def safe_pct(row):
        d0 = row.get("Employment")
        d = row.get("EDM_raw")
        if pd.isna(d0) or pd.isna(d) or d0 == 0:
            return np.nan
        return (d - d0) / d0

    data["EDM_pct"] = data.apply(safe_pct, axis=1)
    data["EDM_pct"] = data["EDM_pct"].replace([np.inf, -np.inf], np.nan)

    # Robust clipping and scale to 0..1 (1st-99th percentile)
    if data["EDM_pct"].dropna().shape[0] > 0:
        p1 = float(data["EDM_pct"].quantile(0.01))
        p99 = float(data["EDM_pct"].quantile(0.99))
        denom = p99 - p1 if (p99 - p1) != 0 else 1.0
        data["EDM_index"] = data["EDM_pct"].clip(lower=p1, upper=p99).fillna(p1).apply(lambda v: (v - p1) / denom)
    else:
        st.warning("No EDM_pct values available (likely missing employment). Setting EDM_index=0 for visualization.", icon="⚠️")
        data["EDM_index"] = 0.0

    data["EDM_index"] = pd.to_numeric(data["EDM_index"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    out = data[["Area", "Year", "A", "Employment", "Population", "EmpPop", "Unemp", "TimeYears", "EDM_raw", "EDM_pct", "EDM_index"]].copy()
    out = out.sort_values(by=["Area", "Year"]).reset_index(drop=True)
    st.success(f"EDM loader prepared {out.shape[0]} rows from {len(dfs)} source files.", icon="✅")
    return out