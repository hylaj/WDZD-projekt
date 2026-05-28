"""
OpenAQ Data Downloader - Projekt: Globalna jakość powietrza (CAŁA EUROPA)
==========================================================================
Skrypt pobiera dane z OpenAQ API v3 dla krajów Europy i zapisuje
gotowe pliki PARQUET do analizy (mapa stacji, szeregi czasowe, PCA/UMAP).

PRZED URUCHOMIENIEM:
1. Zarejestruj się na https://explore.openaq.org i pobierz klucz API
2. Wklej klucz w zmienną API_KEY poniżej
3. Zainstaluj zależności: pip install requests pandas tqdm pyarrow
"""

import requests
import pandas as pd
import time
import os
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURACJA — zmień według potrzeb
# ─────────────────────────────────────────────

API_KEY = "ff97a634265ffa3f56a933b562283ab0858b7c953d436f9af36ef5e4a6c32ede"   # ← wklej swój klucz!

# Kraje do pobrania (Cała Europa - kody ISO 2)
COUNTRIES = {
    "AL": "Albania", "AD": "Andora", "AT": "Austria", "BE": "Belgia",
    "BA": "Bośnia i Hercegowina", "BG": "Bułgaria", "HR": "Chorwacja",
    "CY": "Cypr", "CZ": "Czechy", "DK": "Dania", "EE": "Estonia",
    "FI": "Finlandia", "FR": "Francja", "DE": "Niemcy", "GR": "Grecja",
    "HU": "Węgry", "IS": "Islandia", "IE": "Irlandia", "IT": "Włochy",
    "XK": "Kosowo", "LV": "Łotwa", "LI": "Liechtenstein", "LT": "Litwa",
    "LU": "Luksemburg", "MT": "Malta", "MD": "Mołdawia", "MC": "Monako",
    "ME": "Czarnogóra", "NL": "Holandia", "MK": "Macedonia Północna",
    "NO": "Norwegia", "PL": "Polska", "PT": "Portugalia", "RO": "Rumunia",
    "RS": "Serbia", "SK": "Słowacja", "SI": "Słowenia", "ES": "Hiszpania",
    "SE": "Szwecja", "CH": "Szwajcaria", "GB": "Wielka Brytania",
    "UA": "Ukraina",
}

# Parametry jakości powietrza (ID w OpenAQ v3)
# POPRAWKA: NO₂=5 (µg/m³), O₃=3 (µg/m³) — poprzednie ID 7 i 10 to były wersje w ppm
PARAMETERS = {
    1: "pm10",
    2: "pm25",
    5: "no2",   # ← POPRAWIONE (było: 7)
    3: "o3",    # ← POPRAWIONE (było: 10)
}

# Zakres dat (4 pełne lata)
DATE_FROM = "2022-01-01"
DATE_TO   = "2025-12-31"

# Ze względu na całą Europę, limit ustawiamy na 50 oficjalnych stacji per kraj
# (to i tak da nam ponad 2000 stacji do analizy)
MAX_STATIONS_PER_COUNTRY = 50

# Folder zapisu
OUTPUT_DIR = "openaq_data_v2"

# ─────────────────────────────────────────────
# USTAWIENIA API
# ─────────────────────────────────────────────

BASE_URL = "https://api.openaq.org/v3"
HEADERS  = {
    "X-API-Key": API_KEY,
    "Accept": "application/json"
}

RATE_LIMIT_DELAY = 0.5


# ─────────────────────────────────────────────
# FUNKCJE POMOCNICZE
# ─────────────────────────────────────────────

def api_get(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            print("  ⚠️  Limit zapytań — czekam 60 sekund...")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"   Błąd API: {e}")
        return None

def get_country_id(country_code):
    data = api_get("countries", params={"limit": 200})
    if not data:
        return None
    for c in data.get("results", []):
        if c.get("code") == country_code:
            return c.get("id")
    return None

def get_stations(country_code, parameter_ids, limit=50):
    country_id = get_country_id(country_code)
    if not country_id:
        return []

    stations = []
    page = 1

    while len(stations) < limit:
        params = {
            "countries_id": country_id,
            "limit": 100,
            "page": page,
            "monitor": True,
        }
        data = api_get("locations", params=params)
        if not data or not data.get("results"):
            break

        for loc in data["results"]:
            coords = loc.get("coordinates", {})
            if not coords or not coords.get("latitude"):
                continue

            sensors = loc.get("sensors", [])
            loc_param_ids = {s.get("parameter", {}).get("id") for s in sensors}

            if not loc_param_ids.intersection(parameter_ids):
                continue

            station = {
                "location_id":   loc["id"],
                "station_name":  loc.get("name", ""),
                "city":          loc.get("locality", ""),
                "country_code":  country_code,
                "country_name":  COUNTRIES[country_code],
                "latitude":      coords["latitude"],
                "longitude":     coords["longitude"],
                "sensors":       sensors,
            }
            stations.append(station)

            if len(stations) >= limit:
                break

        meta = data.get("meta", {})
        found_raw = meta.get("found", 0)
        try:
            found = int(str(found_raw).replace(">", "").replace("<", "").strip())
        except (ValueError, TypeError):
            found = 9999

        if len(stations) >= limit or page * 100 >= found:
            break
        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    return stations

def get_daily_measurements(sensor_id, date_from, date_to):
    all_results = []
    page = 1

    while True:
        params = {
            "date_from": f"{date_from}T00:00:00Z",
            "date_to":   f"{date_to}T23:59:59Z",
            "limit":     1000,
            "page":      page,
        }
        data = api_get(f"sensors/{sensor_id}/days", params=params)
        if not data or not data.get("results"):
            break

        all_results.extend(data["results"])

        meta = data.get("meta", {})
        found_raw = meta.get("found", 0)

        # Bezpieczne parsowanie (API OpenAQ czasem zwraca np. ">10000" jako string)
        try:
            found = int(str(found_raw).replace(">", "").replace("<", "").replace("+", "").strip())
        except (ValueError, TypeError):
            found = 9999999  # Bezpieczny zapas, jeśli API rzuci czymś zupełnie niespodziewanym

        if page * 1000 >= found:
            break
        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    return all_results


# ─────────────────────────────────────────────
# GŁÓWNA LOGIKA POBIERANIA
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("  OpenAQ Data Downloader — CAŁA EUROPA")
    print(f"{'='*60}\n")

    if API_KEY == "TUTAJ_WKLEJ_SWOJ_KLUCZ_API" or API_KEY == "TWÓJ_KLUCZ_API" or API_KEY == "X":
        print("  Uzupełnij zmienną API_KEY przed uruchomieniem!")
        return

    all_stations     = []
    all_measurements = []

    # ── KROK 1: Pobierz stacje dla każdego kraju ──────────────────
    print(" KROK 1: Pobieranie stacji pomiarowych...")
    print("-" * 40)

    for country_code, country_name in COUNTRIES.items():
        print(f"   {country_name} ({country_code})...", end="")
        stations = get_stations(
            country_code,
            parameter_ids=set(PARAMETERS.keys()),
            limit=MAX_STATIONS_PER_COUNTRY
        )
        print(f" {len(stations)} stacji.")
        all_stations.extend(stations)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_stations:
        print("\n  Nie pobrano żadnych stacji. Sprawdź klucz API.")
        return

    # ── KROK 2: Pobierz pomiary dla każdej stacji ─────────────────
    print(f"\n\n KROK 2: Pobieranie pomiarów ({DATE_FROM} → {DATE_TO})")
    print("-" * 40)
    print(f"  Łącznie stacji: {len(all_stations)}\n")

    station_rows = []

    for station in tqdm(all_stations, desc="  Pobieranie"):
        station_id   = station["location_id"]
        station_name = station["station_name"]

        station_row = {
            "location_id":  station_id,
            "station_name": station_name,
            "city":         station["city"],
            "country_code": station["country_code"],
            "country_name": station["country_name"],
            "latitude":     station["latitude"],
            "longitude":    station["longitude"],
        }

        for sensor in station.get("sensors", []):
            sensor_id  = sensor.get("id")
            param_id   = sensor.get("parameter", {}).get("id")
            param_name = sensor.get("parameter", {}).get("name", "")

            if param_id not in PARAMETERS:
                continue

            measurements = get_daily_measurements(sensor_id, DATE_FROM, DATE_TO)

            for m in measurements:
                period  = m.get("period", {})
                dt_from = period.get("datetimeFrom", {}).get("utc", "")
                date    = dt_from[:10] if dt_from else ""
                value   = m.get("value")

                if value is None or value < 0:
                    continue

                all_measurements.append({
                    "date":         date,
                    "location_id":  station_id,
                    "station_name": station_name,
                    "city":         station["city"],
                    "country_code": station["country_code"],
                    "country_name": station["country_name"],
                    "latitude":     station["latitude"],
                    "longitude":    station["longitude"],
                    "parameter":    param_name,
                    "value":        round(value, 4),
                    "unit":         sensor.get("parameter", {}).get("units", "µg/m³"),
                })

            time.sleep(RATE_LIMIT_DELAY)

        station_rows.append(station_row)

    # ── KROK 3: Zapis plików (PARQUET) ────────────────────────────
    print("\n\n💾 KROK 3: Zapisywanie plików (.parquet)...")
    print("-" * 40)

    df_stations = pd.DataFrame(station_rows)
    stations_path = os.path.join(OUTPUT_DIR, "stations.parquet")
    df_stations.to_parquet(stations_path, index=False)
    print(f"   stations.parquet          ({len(df_stations)} stacji)")

    if not all_measurements:
        print("   Brak pomiarów — sprawdź daty lub dostępność danych")
        return

    df_meas = pd.DataFrame(all_measurements)
    df_meas["date"] = pd.to_datetime(df_meas["date"])
    df_meas = df_meas.sort_values(["country_code", "location_id", "date"])

    meas_path = os.path.join(OUTPUT_DIR, "measurements_daily.parquet")
    df_meas.to_parquet(meas_path, index=False)
    print(f"   measurements_daily.parquet ({len(df_meas):,} rekordów)")

    df_pivot = df_meas.groupby(
        ["location_id", "station_name", "city", "country_code", "country_name",
         "latitude", "longitude", "parameter"]
    )["value"].mean().reset_index()

    df_profiles = df_pivot.pivot_table(
        index=["location_id", "station_name", "city",
               "country_code", "country_name", "latitude", "longitude"],
        columns="parameter",
        values="value"
    ).reset_index()
    df_profiles.columns.name = None

    param_cols = [c for c in df_profiles.columns if c in PARAMETERS.values()]
    df_profiles = df_profiles.dropna(subset=param_cols, thresh=len(param_cols) // 2 + 1)

    profiles_path = os.path.join(OUTPUT_DIR, "station_profiles.parquet")
    df_profiles.to_parquet(profiles_path, index=False)
    print(f"   station_profiles.parquet  ({len(df_profiles)} stacji z profilami)")

    print(f"\n   Parametry w danych:")
    for p, cnt in df_meas["parameter"].value_counts().items():
        print(f"      {p:<8}: {cnt:>10,}")

if __name__ == "__main__":
    main()