"""
Preprocessing danych OpenAQ — Globalna jakość powietrza (Europa)
=================================================================
Wejście:  openaq_data/stations.parquet
          openaq_data/measurements_daily.parquet
          openaq_data/station_profiles.parquet

Wyjście:  openaq_clean/stations_clean.parquet
          openaq_clean/measurements_clean.parquet
          openaq_clean/station_profiles_clean.parquet
          openaq_clean/monthly_aggregates.parquet
          openaq_clean/preprocessing_report.txt
"""

import pandas as pd
import numpy as np
import os

INPUT_DIR  = "openaq_data"   # ← zmień na "openaq_data" dla starych danych
OUTPUT_DIR = "openaq_clean"  # ← zmień na "openaq_clean" dla starych danych
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Progi fizyczne dla każdego parametru (wartości niemożliwe = błąd czujnika)
PHYSICAL_LIMITS = {
    "pm25": (0, 1000),   # µg/m³ — absolutne maksimum w skrajnych pożarach
    "pm10": (0, 2000),
    "no2":  (0, 2000),
    "o3":   (0, 500),
}

# Progi IQR do wykrywania outlierów per stacja (mnożnik)
IQR_MULTIPLIER = 3.0

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

# ─────────────────────────────────────────────
# 1. WCZYTANIE DANYCH
# ─────────────────────────────────────────────

log("=" * 60)
log("  PREPROCESSING — OpenAQ Europa")
log("=" * 60)

df_stations  = pd.read_parquet(os.path.join(INPUT_DIR, "stations.parquet"))
df_meas      = pd.read_parquet(os.path.join(INPUT_DIR, "measurements_daily.parquet"))
df_profiles  = pd.read_parquet(os.path.join(INPUT_DIR, "station_profiles.parquet"))

log(f"\n[1] Wczytano dane:")
log(f"    stations:      {len(df_stations):>7,} wierszy")
log(f"    measurements:  {len(df_meas):>7,} wierszy")
log(f"    profiles:      {len(df_profiles):>7,} wierszy")

# ─────────────────────────────────────────────
# 2. CZYSZCZENIE STACJI
# ─────────────────────────────────────────────

log(f"\n[2] Czyszczenie stacji...")

n_before = len(df_stations)

# Usuń stacje bez współrzędnych
df_stations = df_stations.dropna(subset=["latitude", "longitude"])

# Usuń duplikaty location_id
df_stations = df_stations.drop_duplicates(subset=["location_id"])

# Sanity check współrzędnych (Europa: lat 34–72, lon -25–45)
mask_valid_coords = (
    df_stations["latitude"].between(34, 72) &
    df_stations["longitude"].between(-25, 45)
)
n_bad_coords = (~mask_valid_coords).sum()
df_stations = df_stations[mask_valid_coords].reset_index(drop=True)

log(f"    Przed:         {n_before} stacji")
log(f"    Złe współrz.:  {n_bad_coords} usuniętych")
log(f"    Po:            {len(df_stations)} stacji")

# ─────────────────────────────────────────────
# 3. CZYSZCZENIE POMIARÓW
# ─────────────────────────────────────────────

log(f"\n[3] Czyszczenie pomiarów...")

n_before = len(df_meas)

# Upewnij się że date jest datetime
df_meas["date"] = pd.to_datetime(df_meas["date"], errors="coerce")

# Usuń wiersze z brakującą datą lub wartością
df_meas = df_meas.dropna(subset=["date", "value"])

# Zostaw tylko znane parametry
known_params = list(PHYSICAL_LIMITS.keys())
df_meas = df_meas[df_meas["parameter"].isin(known_params)]

# Ujednolicenie nazwy parametru (lowercase, strip)
df_meas["parameter"] = df_meas["parameter"].str.lower().str.strip()

# Usuń wartości poniżej zera (błąd czujnika)
n_negative = (df_meas["value"] < 0).sum()
df_meas = df_meas[df_meas["value"] >= 0]

# Filtry fizyczne per parametr
n_physical = 0
for param, (lo, hi) in PHYSICAL_LIMITS.items():
    mask = (df_meas["parameter"] == param) & ~df_meas["value"].between(lo, hi)
    n_physical += mask.sum()
    df_meas = df_meas[~((df_meas["parameter"] == param) & ~df_meas["value"].between(lo, hi))]

log(f"    Ujemne wartości usunięte:  {n_negative:,}")
log(f"    Poza limitami fizycznymi:  {n_physical:,}")

# Outlier removal — IQR per (location_id, parameter)
log(f"    Usuwanie outlierów (IQR×{IQR_MULTIPLIER})...")

def remove_outliers_iqr(group):
    q1 = group["value"].quantile(0.25)
    q3 = group["value"].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - IQR_MULTIPLIER * iqr
    hi = q3 + IQR_MULTIPLIER * iqr
    return group[group["value"].between(lo, hi)]

# Pandas 2.x — include_groups=False żeby uniknąć deprecation warning,
# potem merge z powrotem żeby odzyskać wszystkie kolumny
idx_before = df_meas.index
df_meas = df_meas.reset_index(drop=True)
mask_keep = (
    df_meas
    .groupby(["location_id", "parameter"], group_keys=False)
    .apply(lambda g: pd.Series(
        g["value"].between(
            g["value"].quantile(0.25) - IQR_MULTIPLIER * (g["value"].quantile(0.75) - g["value"].quantile(0.25)),
            g["value"].quantile(0.75) + IQR_MULTIPLIER * (g["value"].quantile(0.75) - g["value"].quantile(0.25))
        ),
        index=g.index
    ))
)
df_meas = df_meas[mask_keep].reset_index(drop=True)

# Zostaw tylko stacje które przetrwały czyszczenie stacji
valid_ids = set(df_stations["location_id"])
df_meas = df_meas[df_meas["location_id"].isin(valid_ids)]

# Dodaj kolumny pomocnicze
df_meas["year"]    = df_meas["date"].dt.year
df_meas["month"]   = df_meas["date"].dt.month
df_meas["season"]  = df_meas["month"].map({
    12: "Zima", 1: "Zima", 2: "Zima",
    3:  "Wiosna", 4: "Wiosna", 5: "Wiosna",
    6:  "Lato",   7: "Lato",   8: "Lato",
    9:  "Jesień", 10: "Jesień", 11: "Jesień",
})

n_after = len(df_meas)
log(f"    Przed:         {n_before:,} rekordów")
log(f"    Po:            {n_after:,} rekordów")
log(f"    Usunięto:      {n_before - n_after:,} ({(n_before-n_after)/n_before*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. CZYSZCZENIE PROFILI STACJI (pod PCA/UMAP)
# ─────────────────────────────────────────────

log(f"\n[4] Czyszczenie profili stacji...")

n_before = len(df_profiles)

# Ujednolicenie nazw kolumn parametrów
df_profiles.columns = [c.lower().strip() for c in df_profiles.columns]

# Zostaw tylko stacje z ważnymi współrzędnymi
df_profiles = df_profiles[df_profiles["location_id"].isin(valid_ids)]

# Kolumny parametrów które faktycznie istnieją w profilu
param_cols = [c for c in known_params if c in df_profiles.columns]
log(f"    Parametry w profilu:  {param_cols}")

# Zachowaj wiersze które mają przynajmniej 2 parametry (nie same NaN)
df_profiles = df_profiles.dropna(subset=param_cols, thresh=2)

# Wypełnij brakujące parametry medianą kolumny (nie usuwaj stacji całkowicie)
for col in param_cols:
    median_val = df_profiles[col].median()
    n_filled = df_profiles[col].isna().sum()
    df_profiles[col] = df_profiles[col].fillna(median_val)
    if n_filled > 0:
        log(f"    Uzupełniono {col}: {n_filled} braków → mediana ({median_val:.2f})")

# Usuń outlierów na poziomie profili (> 99.5 percentyla)
for col in param_cols:
    cap = df_profiles[col].quantile(0.995)
    n_capped = (df_profiles[col] > cap).sum()
    df_profiles[col] = df_profiles[col].clip(upper=cap)
    if n_capped > 0:
        log(f"    Przycięto {col} do {cap:.2f}: {n_capped} stacji")

log(f"    Przed:  {n_before} profili")
log(f"    Po:     {len(df_profiles)} profili")

# ─────────────────────────────────────────────
# 5. AGREGATY MIESIĘCZNE (pod wizualizacje)
# ─────────────────────────────────────────────

log(f"\n[5] Tworzenie agregatów miesięcznych...")

df_monthly = (
    df_meas
    .groupby(["country_code", "country_name", "year", "month", "season", "parameter"])
    ["value"]
    .agg(mean="mean", median="median", p25=lambda x: x.quantile(0.25),
         p75=lambda x: x.quantile(0.75), count="count")
    .reset_index()
)

log(f"    Agregaty miesięczne:  {len(df_monthly):,} wierszy")

# ─────────────────────────────────────────────
# 6. ZAPIS
# ─────────────────────────────────────────────

log(f"\n[6] Zapisywanie plików...")

df_stations.to_parquet(os.path.join(OUTPUT_DIR, "stations_clean.parquet"), index=False)
log(f"    stations_clean.parquet       ({len(df_stations):,} stacji)")

df_meas.to_parquet(os.path.join(OUTPUT_DIR, "measurements_clean.parquet"), index=False)
log(f"    measurements_clean.parquet   ({len(df_meas):,} rekordów)")

df_profiles.to_parquet(os.path.join(OUTPUT_DIR, "station_profiles_clean.parquet"), index=False)
log(f"    station_profiles_clean.parquet ({len(df_profiles):,} stacji)")

df_monthly.to_parquet(os.path.join(OUTPUT_DIR, "monthly_aggregates.parquet"), index=False)
log(f"    monthly_aggregates.parquet   ({len(df_monthly):,} wierszy)")

# ─────────────────────────────────────────────
# 7. STATYSTYKI KOŃCOWE
# ─────────────────────────────────────────────

log(f"\n{'='*60}")
log("  PODSUMOWANIE")
log(f"{'='*60}")

log(f"\nKraje w danych: {df_meas['country_code'].nunique()}")
log(f"Stacje:         {df_meas['location_id'].nunique()}")
log(f"Zakres dat:     {df_meas['date'].min().date()} → {df_meas['date'].max().date()}")
log(f"\nLiczba rekordów per parametr:")
for p, cnt in df_meas.groupby("parameter")["value"].count().items():
    log(f"    {p:<8}: {cnt:>10,}")

log(f"\nŚrednie wartości per parametr (mediana globalnie):")
for p, val in df_meas.groupby("parameter")["value"].median().items():
    log(f"    {p:<8}: {val:.2f} µg/m³")

log(f"\nPokrycie krajów (top 10 wg liczby rekordów):")
top_countries = (
    df_meas.groupby("country_name")["value"]
    .count()
    .sort_values(ascending=False)
    .head(10)
)
for country, cnt in top_countries.items():
    log(f"    {country:<30}: {cnt:>8,}")

# Zapis raportu tekstowego
report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log(f"\n  Raport zapisany → {report_path}")
log("  Preprocessing zakończony!\n")
