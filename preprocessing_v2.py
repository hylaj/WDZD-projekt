import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "openaq_data_v2"
OUTPUT_DIR = "openaq_clean_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WHO_LIMITS = {
    "pm25": 15.0,
    "pm10": 45.0,
    "no2":  25.0,
    "o3":   100.0,
}

PHYSICAL_LIMITS = {
    "pm25": (0, 1000),
    "pm10": (0, 2000),
    "no2":  (0, 2000),
    "o3":   (0, 500),
}

IQR_MULTIPLIER = 3.0
DATE_FROM = pd.Timestamp("2022-01-01")
DATE_TO   = pd.Timestamp("2025-12-31")

SEASON_MAP = {
    12: "Zima",  1: "Zima",  2: "Zima",
    3: "Wiosna", 4: "Wiosna", 5: "Wiosna",
    6: "Lato",   7: "Lato",   8: "Lato",
    9: "Jesień", 10: "Jesień", 11: "Jesień",
}

df_stations = pd.read_parquet(os.path.join(INPUT_DIR, "stations.parquet"))
df_meas     = pd.read_parquet(os.path.join(INPUT_DIR, "measurements_daily.parquet"))
df_profiles = pd.read_parquet(os.path.join(INPUT_DIR, "station_profiles.parquet"))

df_stations = df_stations.dropna(subset=["latitude", "longitude"])
df_stations = df_stations.drop_duplicates(subset=["location_id"])

mask_eu = (
    df_stations["latitude"].between(34, 72) &
    df_stations["longitude"].between(-25, 45)
)
df_stations = df_stations[mask_eu].reset_index(drop=True)

valid_ids = set(df_stations["location_id"])

df_meas["date"] = pd.to_datetime(df_meas["date"], errors="coerce")
df_meas = df_meas.dropna(subset=["date", "value"])

known_params = list(PHYSICAL_LIMITS.keys())
df_meas["parameter"] = df_meas["parameter"].str.lower().str.strip()
df_meas = df_meas[df_meas["parameter"].isin(known_params)]
df_meas = df_meas[df_meas["date"].between(DATE_FROM, DATE_TO)]
df_meas = df_meas[df_meas["value"] >= 0]

for param, (lo, hi) in PHYSICAL_LIMITS.items():
    mask = (df_meas["parameter"] == param) & ~df_meas["value"].between(lo, hi)
    df_meas = df_meas[~mask]

df_meas = df_meas.reset_index(drop=True)

def iqr_mask(g):
    q1 = g["value"].quantile(0.25)
    q3 = g["value"].quantile(0.75)
    iqr = q3 - q1
    return pd.Series(
        g["value"].between(q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr),
        index=g.index
    )

mask_keep = df_meas.groupby(
    ["location_id", "parameter"], group_keys=False
).apply(iqr_mask)
mask_keep = mask_keep.reindex(df_meas.index, fill_value=False)
df_meas = df_meas[mask_keep].reset_index(drop=True)
df_meas = df_meas[df_meas["location_id"].isin(valid_ids)]

df_meas["year"]   = df_meas["date"].dt.year
df_meas["month"]  = df_meas["date"].dt.month
df_meas["season"] = df_meas["month"].map(SEASON_MAP)
df_meas["above_who"] = df_meas.apply(
    lambda r: int(r["value"] > WHO_LIMITS.get(r["parameter"], np.inf)),
    axis=1
)

df_profiles.columns = [c.lower().strip() for c in df_profiles.columns]
df_profiles = df_profiles[df_profiles["location_id"].isin(valid_ids)]

param_cols = [c for c in known_params if c in df_profiles.columns]
df_profiles = df_profiles.dropna(subset=param_cols, thresh=2)

for col in param_cols:
    df_profiles[col] = df_profiles[col].fillna(df_profiles[col].median())
    cap = df_profiles[col].quantile(0.995)
    df_profiles[col] = df_profiles[col].clip(upper=cap)

df_base = df_meas[["location_id", "parameter", "value", "month"]].copy()

feat_mean = (
    df_base.groupby(["location_id", "parameter"])["value"]
    .mean().unstack("parameter").add_prefix("mean_")
)
feat_std = (
    df_base.groupby(["location_id", "parameter"])["value"]
    .std().unstack("parameter").add_prefix("std_")
)

df_winter = df_base[df_base["month"].isin([12, 1, 2])]
df_summer = df_base[df_base["month"].isin([6, 7, 8])]

feat_winter = (
    df_winter.groupby(["location_id", "parameter"])["value"]
    .mean().unstack("parameter").add_prefix("winter_")
)
feat_summer = (
    df_summer.groupby(["location_id", "parameter"])["value"]
    .mean().unstack("parameter").add_prefix("summer_")
)

df_ext = feat_mean.join([feat_std, feat_winter, feat_summer], how="outer").reset_index()

meta_cols = ["location_id", "station_name", "city",
             "country_code", "country_name", "latitude", "longitude"]
df_ext = df_ext.merge(df_stations[meta_cols], on="location_id", how="left")

feature_cols = [c for c in df_ext.columns if c.startswith(("mean_", "std_", "winter_", "summer_"))]
df_ext = df_ext.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)

for col in feature_cols:
    df_ext[col] = df_ext[col].fillna(df_ext[col].median())
    cap = df_ext[col].quantile(0.995)
    df_ext[col] = df_ext[col].clip(upper=cap)

df_monthly = (
    df_meas
    .groupby(["country_code", "country_name", "year", "month", "season", "parameter"])
    ["value"]
    .agg(
        mean="mean",
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        max="max",
        count="count",
        pct_above_who=lambda x: (
            (x > WHO_LIMITS.get(
                df_meas.loc[x.index, "parameter"].iloc[0], np.inf
            )).mean() * 100
        )
    )
    .reset_index()
)

df_stations.to_parquet(os.path.join(OUTPUT_DIR, "stations_clean.parquet"), index=False)
df_meas.to_parquet(os.path.join(OUTPUT_DIR, "measurements_clean.parquet"), index=False)
df_profiles.to_parquet(os.path.join(OUTPUT_DIR, "station_profiles_clean.parquet"), index=False)
df_ext.to_parquet(os.path.join(OUTPUT_DIR, "station_profiles_extended.parquet"), index=False)
df_monthly.to_parquet(os.path.join(OUTPUT_DIR, "monthly_aggregates.parquet"), index=False)

df_stations.to_csv(
    os.path.join(OUTPUT_DIR, "stations_clean.csv"), index=False, encoding="utf-8-sig")
df_ext.to_csv(
    os.path.join(OUTPUT_DIR, "station_profiles_extended.csv"), index=False, encoding="utf-8-sig")

cols_b_daily = [
    "date", "location_id", "station_name", "city",
    "country_code", "country_name", "latitude", "longitude",
    "parameter", "value", "unit", "year", "month", "season", "above_who"
]
df_meas[cols_b_daily].to_csv(
    os.path.join(OUTPUT_DIR, "measurements_daily.csv"), index=False, encoding="utf-8-sig")
df_monthly.to_csv(
    os.path.join(OUTPUT_DIR, "monthly_aggregates.csv"), index=False, encoding="utf-8-sig")

print(f"Kraje: {df_meas['country_code'].nunique()}")
print(f"Stacje: {df_meas['location_id'].nunique()}")
print(f"Rekordy: {len(df_meas):,}")
print(f"Zakres dat: {df_meas['date'].min().date()} → {df_meas['date'].max().date()}")
