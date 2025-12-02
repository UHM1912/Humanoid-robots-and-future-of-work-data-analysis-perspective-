import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from eri_model import ERIModel

def load_ilostat_data():
    # Filenames (adjust if needed)
    files = {
        "Earnings": "Earnings.csv",
        "EmpPop": "Employment to population ratio.csv",
        "Informal": "Informal employment rate.csv",
        "LFPR": "Labour Force participation rate.csv",
        "Unemp": "Unemployment rate.csv",
        "Poverty": "working poverty rate.csv",
        "NEET": "YOUTH NEET rate.csv"
    }

    dfs = {}
    for key, path in files.items():
        df = pd.read_csv(path)
        df = df[["Area", "Year", "Total"]].copy()
        df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
        df.dropna(subset=["Total"], inplace=True)
        df["Area"] = df["Area"].astype(str).str.strip()
        df["Year"] = df["Year"].astype(int)
        dfs[key] = df.rename(columns={"Total": key})

    # Merge all datasets on Area + Year (outer join to keep all)
    data = dfs["Earnings"]
    for key in ["EmpPop", "Informal", "LFPR", "Unemp", "Poverty", "NEET"]:
        data = data.merge(dfs[key], on=["Area", "Year"], how="outer")

    # Drop rows with too many NaNs (less than 3 valid indicators)
    data = data.dropna(thresh=4)

    # Normalize available numeric columns
    scaler = MinMaxScaler()
    num_cols = ["Earnings", "EmpPop", "Informal", "LFPR", "Unemp", "Poverty", "NEET"]
    for col in num_cols:
        if col in data.columns:
            valid = data[col].dropna()
            if len(valid) > 0:
                data[col + "_norm"] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # Compute A, W, S (use fillna to avoid NaN)
    data["A"] = ((data["Unemp_norm"].fillna(0.5)) + (1 - data["EmpPop_norm"].fillna(0.5))) / 2
    data["W"] = data["Earnings_norm"].fillna(0.5)
    data["S"] = (
        (data["LFPR_norm"].fillna(0.5)
         + (1 - data["Informal_norm"].fillna(0.5))
         + (1 - data["Poverty_norm"].fillna(0.5))
         + (1 - data["NEET_norm"].fillna(0.5))) / 4
    )

    # Compute ERI
    model = ERIModel()
    data["ERI"] = data.apply(lambda row: model.compute_eri(row["A"], row["W"], row["S"]), axis=1)

    # Clean output
    data = data.dropna(subset=["ERI"])
    data = data.sort_values(by=["Area", "Year"])

    return data[["Area", "Year", "A", "W", "S", "ERI"]]
