import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
INPUT_DIR = "openaq_clean_v2"
INPUT_FILE = os.path.join(INPUT_DIR, "measurements_clean.parquet")
OUTPUT_FILE = os.path.join(INPUT_DIR, "measurements_episodes.csv")
OUTPUT_SUMMARY = os.path.join(INPUT_DIR, "measurements_episodes_summary.csv")
LIMITS_FILE = "who_limits.csv" 

def main():
    df_meas = pd.read_parquet(INPUT_FILE)
    df_limits = pd.read_csv(LIMITS_FILE)
    
    df_limits["parameter"] = df_limits["parameter"].str.lower().str.strip()
    df_meas = df_meas.merge(df_limits, on="parameter", how="left")
    df_meas["above_who"] = (df_meas["value"] > df_meas["who_limit"]).astype(int)
    df_meas["date"] = pd.to_datetime(df_meas["date"])
    
# 1. SORTOWANIE (Kluczowe dla poprawnego działania)
    df_meas = df_meas.sort_values(["location_id", "parameter", "date"]).reset_index(drop=True)

    # 2. BEZPIECZNE OKNO ROZWIJALNE (Omijamy błąd z "cannot reindex...")
    # Tymczasowo ustawiamy datę jako indeks i wyliczamy sumę z 3 dni (3D)
    rolled = (
        df_meas.set_index("date")
        .groupby(["location_id", "parameter"])["above_who"]
        .rolling("3D").sum()
        .reset_index(name="rolling_3d_sum")  # Zamieniamy wynik na zwykłą tabelę z kolumną "rolling_3d_sum"
    )

    # Zabezpieczenie: usuwamy potencjalne duplikaty w dacie (gdyby jakieś czujniki wysłały dwa pomiary na dzień)
    rolled = rolled.drop_duplicates(subset=["location_id", "parameter", "date"])

    # Bezpiecznie doklejamy wynik do głównej tabeli bazując na wartościach, a nie na indeksach (jak WYSZUKAJ.PIONOWO)
    df_meas = df_meas.merge(rolled, on=["location_id", "parameter", "date"], how="left")

    # 3. FLAGOWANIE EPOZODÓW
    # Najpierw tworzymy Series z wartościami True/False
    is_ep_bool = df_meas["rolling_3d_sum"] >= 3
    
    # Gdzie False, tam wstawiamy NaN (aby bfill mogło zadziałać), gdzie True tam 1
    df_meas["is_episode"] = is_ep_bool.map({True: 1, False: np.nan})

    # Teraz cofamy jedynki o dwa dni wstecz (bfill zadziała, bo są wartości NaN)
    df_meas["is_episode"] = (
        df_meas.groupby(["location_id", "parameter"])["is_episode"]
        .bfill(limit=2)
        .fillna(0) # Całą resztę NaN zamieniamy bezpiecznie na 0
    ).astype(int)

    # 4. AUTOMATYCZNE GENEROWANIE EPISODE_ID I DLUGOSCI (Nowość)
    # Wykrywamy moment, w którym zmienia się status epizodu (np. z 0 na 1 lub z 1 na 0)
    change_cond = df_meas["is_episode"] != df_meas.groupby(["location_id", "parameter"])["is_episode"].shift()
    df_meas["block_id"] = change_cond.groupby([df_meas["location_id"], df_meas["parameter"]]).cumsum()

    # Filtrujemy tylko te wiersze, które są faktycznym epizodem i nadajemy im unikalne ID
    df_ep_only = df_meas[df_meas["is_episode"] == 1]
    
    if not df_ep_only.empty:
        df_meas.loc[df_ep_only.index, "episode_id"] = df_ep_only.groupby(["location_id", "parameter", "block_id"]).ngroup() + 1
        df_meas["episode_length"] = df_meas.groupby("episode_id")["date"].transform("count")
    else:
        df_meas["episode_id"] = np.nan
        df_meas["episode_length"] = np.nan

    # Czyszczenie kolumn roboczych
    df_meas = df_meas.drop(columns=["rolling_3d_sum", "block_id", "who_limit"], errors="ignore")

    # 5. ZAPIS PLIKU GŁÓWNEGO (Dziennego)
    cols_b_daily = [
        "date", "location_id", "station_name", "city",
        "country_code", "country_name", "latitude", "longitude",
        "parameter", "value", "unit", "year", "month", "season", 
        "above_who", "is_episode", "episode_id", "episode_length"
    ]
    df_meas[cols_b_daily].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 6. GENEROWANIE PLIKU AGREGATU (Specjalnie pod wykres Gantta w Tableau)
    df_summary = (
        df_meas[df_meas["episode_id"].notna()]
        .groupby(["episode_id", "location_id", "station_name", "city", "country_code", "country_name", "parameter"])
        .agg(
            episode_start=("date", "min"),
            episode_end=("date", "max"),
            episode_length=("episode_length", "first"),
            max_value=("value", "max")
        )
        .reset_index()
    )
    df_summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()