"""
Preprocessing danych OpenAQ — Globalna jakość powietrza (Europa)
=================================================================
Wejście:  openaq_data/stations.parquet
          openaq_data/measurements_daily.parquet
          openaq_data/station_profiles.parquet

Wyjście:  openaq_clean_v2/
    ── parquet ──────────────────────────────────────────────────
    stations_clean.parquet             metadane stacji
    measurements_clean.parquet         pomiary dzienne (+ year/month/season)
    station_profiles_clean.parquet     uproszczony profil (4 średnie)
    station_profiles_extended.parquet  rozszerzony profil pod PCA/UMAP (16 cech)
    monthly_aggregates.parquet         agregaty kraj×miesiąc×parametr
    ── csv (Tableau) ────────────────────────────────────────────
    stations_clean.csv                 → mapa stacji (Osoba A)
    measurements_daily.csv             → trendy, epizody (Osoba B)
    monthly_aggregates.csv             → sezonowość, heatmapy (Osoba B)
    station_profiles_extended.csv      → PCA/UMAP (Osoba A)
    ── txt ──────────────────────────────────────────────────────
    preprocessing_report.txt

Zależności: pip install pandas numpy pyarrow
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────

INPUT_DIR  = "openaq_data"
OUTPUT_DIR = "openaq_clean_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Normy WHO 2021 (µg/m³, wartości dobowe) — używane w epizodach przez Osobę B
WHO_LIMITS = {
    "pm25": 15.0,
    "pm10": 45.0,
    "no2":  25.0,
    "o3":   100.0,
}

# Progi fizyczne — wartości niemożliwe fizycznie = błąd czujnika
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

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ═════════════════════════════════════════════
# 1. WCZYTANIE
# ═════════════════════════════════════════════

log("=" * 60)
log("  PREPROCESSING — OpenAQ Europa")
log("=" * 60)

df_stations = pd.read_parquet(os.path.join(INPUT_DIR, "stations.parquet"))
df_meas     = pd.read_parquet(os.path.join(INPUT_DIR, "measurements_daily.parquet"))
df_profiles = pd.read_parquet(os.path.join(INPUT_DIR, "station_profiles.parquet"))

log(f"\n[1] Wczytano dane:")
log(f"    stations:      {len(df_stations):>7,} wierszy")
log(f"    measurements:  {len(df_meas):>7,} wierszy")
log(f"    profiles:      {len(df_profiles):>7,} wierszy")


# ═════════════════════════════════════════════
# 2. CZYSZCZENIE STACJI
# ═════════════════════════════════════════════

log(f"\n[2] Czyszczenie stacji...")

n_before = len(df_stations)
df_stations = df_stations.dropna(subset=["latitude", "longitude"])
df_stations = df_stations.drop_duplicates(subset=["location_id"])

# Europa: lat 34–72, lon -25–45
mask_eu = (
    df_stations["latitude"].between(34, 72) &
    df_stations["longitude"].between(-25, 45)
)
n_bad = (~mask_eu).sum()
df_stations = df_stations[mask_eu].reset_index(drop=True)

log(f"    Przed:         {n_before:,} stacji")
log(f"    Złe współrz.:  {n_bad} usuniętych (poza Europą)")
log(f"    Po:            {len(df_stations):,} stacji")

valid_ids = set(df_stations["location_id"])


# ═════════════════════════════════════════════
# 3. CZYSZCZENIE POMIARÓW
# ═════════════════════════════════════════════

log(f"\n[3] Czyszczenie pomiarów...")

n_before = len(df_meas)

df_meas["date"] = pd.to_datetime(df_meas["date"], errors="coerce")
df_meas = df_meas.dropna(subset=["date", "value"])

# Zostaw tylko znane parametry
known_params = list(PHYSICAL_LIMITS.keys())
df_meas["parameter"] = df_meas["parameter"].str.lower().str.strip()
df_meas = df_meas[df_meas["parameter"].isin(known_params)]

# Przytnij zakres dat do 2022–2025
n_before_date = len(df_meas)
df_meas = df_meas[df_meas["date"].between(DATE_FROM, DATE_TO)]
n_date_cut = n_before_date - len(df_meas)
if n_date_cut:
    log(f"    Poza zakresem dat (< 2022):  {n_date_cut:,} usuniętych")

# Usuń ujemne
n_negative = (df_meas["value"] < 0).sum()
df_meas = df_meas[df_meas["value"] >= 0]

# Filtry fizyczne
n_physical = 0
for param, (lo, hi) in PHYSICAL_LIMITS.items():
    mask = (df_meas["parameter"] == param) & ~df_meas["value"].between(lo, hi)
    n_physical += mask.sum()
    df_meas = df_meas[~mask]

log(f"    Ujemne wartości:           {n_negative:,} usuniętych")
log(f"    Poza limitami fizycznymi:  {n_physical:,} usuniętych")

# Outlier IQR per (location_id, parameter) — poprawka pandas 2.x
log(f"    Usuwanie outlierów (IQR×{IQR_MULTIPLIER})...")

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

# Wyrównanie indeksów — naprawia warning "Boolean Series key will be reindexed"
mask_keep = mask_keep.reindex(df_meas.index, fill_value=False)
df_meas = df_meas[mask_keep].reset_index(drop=True)

# Zostaw tylko stacje z poprawnych współrzędnych
df_meas = df_meas[df_meas["location_id"].isin(valid_ids)]

# Kolumny pomocnicze
df_meas["year"]   = df_meas["date"].dt.year
df_meas["month"]  = df_meas["date"].dt.month
df_meas["season"] = df_meas["month"].map(SEASON_MAP)

# Flaga przekroczenia normy WHO (przydatna dla Osoby B — epizody)
df_meas["above_who"] = df_meas.apply(
    lambda r: int(r["value"] > WHO_LIMITS.get(r["parameter"], np.inf)),
    axis=1
)

n_after = len(df_meas)
log(f"    Przed:         {n_before:,} rekordów")
log(f"    Po:            {n_after:,} rekordów")
log(f"    Usunięto:      {n_before - n_after:,} ({(n_before - n_after) / n_before * 100:.1f}%)")


# ═════════════════════════════════════════════
# 4. PROFILE STACJI — wersja prosta (4 cechy)
# ═════════════════════════════════════════════

log(f"\n[4] Profile stacji — wersja prosta (średnie globalne)...")

df_profiles.columns = [c.lower().strip() for c in df_profiles.columns]
df_profiles = df_profiles[df_profiles["location_id"].isin(valid_ids)]

param_cols = [c for c in known_params if c in df_profiles.columns]
log(f"    Parametry w profilu:  {param_cols}")

df_profiles = df_profiles.dropna(subset=param_cols, thresh=2)

for col in param_cols:
    n_filled = df_profiles[col].isna().sum()
    if n_filled:
        med = df_profiles[col].median()
        df_profiles[col] = df_profiles[col].fillna(med)
        log(f"    Uzupełniono {col}: {n_filled} braków → mediana ({med:.2f})")

for col in param_cols:
    cap = df_profiles[col].quantile(0.995)
    n_capped = (df_profiles[col] > cap).sum()
    df_profiles[col] = df_profiles[col].clip(upper=cap)
    if n_capped:
        log(f"    Przycięto {col} do {cap:.2f}: {n_capped} stacji")

log(f"    Profili: {len(df_profiles)}")


# ═════════════════════════════════════════════
# 5. PROFILE ROZSZERZONE pod PCA/UMAP (16 cech)
# ═════════════════════════════════════════════
#
# Dla każdej stacji budujemy wektor 16 cech:
#   4 × mean_<param>          — globalna średnia
#   4 × std_<param>           — zmienność (jak bardzo stacja "skacze")
#   4 × mean_winter_<param>   — średnia zimowa  (XII, I, II)
#   4 × mean_summer_<param>   — średnia letnia  (VI, VII, VIII)
#
# Sezonowość pozwala UMAP oddzielić np. stacje smogowe (wysoki PM zimą)
# od stacji fotochemicznych (wysoki O3 latem).

log(f"\n[5] Profile rozszerzone pod PCA/UMAP (16 cech)...")

# Baza: pomiary dzienne z kolumnami pomocniczymi
df_base = df_meas[["location_id", "parameter", "value", "month"]].copy()

# --- cechy globalne ---
feat_mean = (
    df_base.groupby(["location_id", "parameter"])["value"]
    .mean()
    .unstack("parameter")
    .add_prefix("mean_")
)
feat_std = (
    df_base.groupby(["location_id", "parameter"])["value"]
    .std()
    .unstack("parameter")
    .add_prefix("std_")
)

# --- cechy sezonowe ---
df_winter = df_base[df_base["month"].isin([12, 1, 2])]
df_summer = df_base[df_base["month"].isin([6, 7, 8])]

feat_winter = (
    df_winter.groupby(["location_id", "parameter"])["value"]
    .mean()
    .unstack("parameter")
    .add_prefix("winter_")
)
feat_summer = (
    df_summer.groupby(["location_id", "parameter"])["value"]
    .mean()
    .unstack("parameter")
    .add_prefix("summer_")
)

# Złącz wszystkie cechy
df_ext = feat_mean.join([feat_std, feat_winter, feat_summer], how="outer")
df_ext = df_ext.reset_index()

# Dołącz metadane stacji
meta_cols = ["location_id", "station_name", "city",
             "country_code", "country_name", "latitude", "longitude"]
df_ext = df_ext.merge(
    df_stations[meta_cols],
    on="location_id", how="left"
)

# Zachowaj stacje z przynajmniej połową cech
feature_cols = [c for c in df_ext.columns if c.startswith(("mean_", "std_", "winter_", "summer_"))]
df_ext = df_ext.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)

# Uzupełnij pozostałe braki medianą kolumny
for col in feature_cols:
    n_na = df_ext[col].isna().sum()
    if n_na:
        df_ext[col] = df_ext[col].fillna(df_ext[col].median())

# Przytnij outlierów do 99.5 percentyla
for col in feature_cols:
    cap = df_ext[col].quantile(0.995)
    df_ext[col] = df_ext[col].clip(upper=cap)

log(f"    Stacji z pełnym profilem rozszerzonym: {len(df_ext)}")
log(f"    Liczba cech: {len(feature_cols)}")
log(f"    Cechy: {feature_cols}")


# ═════════════════════════════════════════════
# 6. AGREGATY MIESIĘCZNE
# ═════════════════════════════════════════════

log(f"\n[6] Tworzenie agregatów miesięcznych...")

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

log(f"    Agregaty miesięczne: {len(df_monthly):,} wierszy")


# ═════════════════════════════════════════════
# 7. ZAPIS PARQUET
# ═════════════════════════════════════════════

log(f"\n[7] Zapisywanie plików PARQUET...")

df_stations.to_parquet(
    os.path.join(OUTPUT_DIR, "stations_clean.parquet"), index=False)
log(f"    stations_clean.parquet              ({len(df_stations):,} stacji)")

df_meas.to_parquet(
    os.path.join(OUTPUT_DIR, "measurements_clean.parquet"), index=False)
log(f"    measurements_clean.parquet          ({len(df_meas):,} rekordów)")

df_profiles.to_parquet(
    os.path.join(OUTPUT_DIR, "station_profiles_clean.parquet"), index=False)
log(f"    station_profiles_clean.parquet      ({len(df_profiles):,} stacji)")

df_ext.to_parquet(
    os.path.join(OUTPUT_DIR, "station_profiles_extended.parquet"), index=False)
log(f"    station_profiles_extended.parquet   ({len(df_ext):,} stacji, {len(feature_cols)} cech)")

df_monthly.to_parquet(
    os.path.join(OUTPUT_DIR, "monthly_aggregates.parquet"), index=False)
log(f"    monthly_aggregates.parquet          ({len(df_monthly):,} wierszy)")


# ═════════════════════════════════════════════
# 8. EKSPORT CSV DLA TABLEAU
# ═════════════════════════════════════════════

log(f"\n[8] Eksport CSV dla Tableau...")

# Osoba A — mapa stacji
df_stations.to_csv(
    os.path.join(OUTPUT_DIR, "stations_clean.csv"), index=False, encoding="utf-8-sig")
log(f"    stations_clean.csv              → Osoba A: mapa stacji")

# Osoba A — PCA/UMAP (profil rozszerzony)
df_ext.to_csv(
    os.path.join(OUTPUT_DIR, "station_profiles_extended.csv"), index=False, encoding="utf-8-sig")
log(f"    station_profiles_extended.csv   → Osoba A: PCA/UMAP scatter plot")

# Osoba B — pomiary dzienne (trendy, epizody)
# Ograniczamy kolumny żeby CSV nie był za duży
cols_b_daily = [
    "date", "location_id", "station_name", "city",
    "country_code", "country_name", "latitude", "longitude",
    "parameter", "value", "unit", "year", "month", "season", "above_who"
]
df_meas[cols_b_daily].to_csv(
    os.path.join(OUTPUT_DIR, "measurements_daily.csv"), index=False, encoding="utf-8-sig")
log(f"    measurements_daily.csv          → Osoba B: trendy, epizody")

# Osoba B — agregaty miesięczne (sezonowość, heatmapy)
df_monthly.to_csv(
    os.path.join(OUTPUT_DIR, "monthly_aggregates.csv"), index=False, encoding="utf-8-sig")
log(f"    monthly_aggregates.csv          → Osoba B: sezonowość, heatmapy")


# ═════════════════════════════════════════════
# 9. PODSUMOWANIE
# ═════════════════════════════════════════════

log(f"\n{'='*60}")
log("  PODSUMOWANIE")
log(f"{'='*60}")

log(f"\nKraje:       {df_meas['country_code'].nunique()}")
log(f"Stacje:      {df_meas['location_id'].nunique()}")
log(f"Zakres dat:  {df_meas['date'].min().date()} → {df_meas['date'].max().date()}")

log(f"\nRekordy per parametr:")
for p, cnt in df_meas.groupby("parameter")["value"].count().sort_values(ascending=False).items():
    who = WHO_LIMITS.get(p, "–")
    log(f"    {p:<8}: {cnt:>10,}   (norma WHO: {who} µg/m³)")

log(f"\nMediany globalne:")
for p, val in df_meas.groupby("parameter")["value"].median().items():
    log(f"    {p:<8}: {val:.2f} µg/m³")

log(f"\nTop 10 krajów (liczba rekordów):")
for country, cnt in (
    df_meas.groupby("country_name")["value"]
    .count().sort_values(ascending=False).head(10).items()
):
    log(f"    {country:<30}: {cnt:>8,}")

log(f"\nPliki w {OUTPUT_DIR}/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
    log(f"    {f:<45} {size_kb:>6} KB")

# Zapis raportu
report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log(f"\n  Raport → {report_path}")
log("  Preprocessing zakończony!\n")
