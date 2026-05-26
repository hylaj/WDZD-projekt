"""
OpenAQ Data Downloader - Projekt: Globalna jakość powietrza
============================================================
Skrypt pobiera dane z OpenAQ API v3 dla wybranych krajów i zapisuje
gotowe pliki CSV do analizy (mapa stacji, szeregi czasowe, PCA/UMAP).

PRZED URUCHOMIENIEM:
1. Zarejestruj się na https://explore.openaq.org i pobierz klucz API
2. Wklej klucz w zmienną API_KEY poniżej
3. Zainstaluj zależności: pip install requests pandas tqdm

STRUKTURA PLIKÓW WYJŚCIOWYCH:
- stations.csv          → stacje z GPS (do mapy)
- measurements_daily.csv → dzienne średnie (do szeregów czasowych)
- station_profiles.csv  → profile stacji gotowe do PCA/UMAP
"""

import requests
import pandas as pd
import time
import os
from tqdm import tqdm

# ─────────────────────────────────────────────
# KONFIGURACJA — zmień według potrzeb
# ─────────────────────────────────────────────

API_KEY = "2d4a8773ff03d7782b5e6987cd01f1ad2abf0e764d6afbe466a57ad901a13fc2"   # ← wklej klucz z https://explore.openaq.org

# Kraje do pobrania (kody ISO) — wybierz 2-3 kontrastujące regiony
COUNTRIES = {
    "IN": "Indie",       # Azja rozwijająca się — wysokie zanieczyszczenie
    "DE": "Niemcy",      # Europa zachodnia    — niskie zanieczyszczenie
    "US": "USA",         # Ameryka Północna    — średnie zanieczyszczenie
}

# Parametry jakości powietrza (ID w OpenAQ v3)
PARAMETERS = {
    2:  "pm25",   # PM2.5 — najważniejszy wskaźnik zdrowotny
    1:  "pm10",   # PM10
    7:  "no2",    # NO₂
    10: "o3",     # O₃
}

# Zakres dat
DATE_FROM = "2022-01-01"
DATE_TO   = "2023-12-31"

# Maksymalna liczba stacji na kraj (zwiększ jeśli chcesz więcej danych)
MAX_STATIONS_PER_COUNTRY = 50

# Folder zapisu
OUTPUT_DIR = "openaq_data"

# ─────────────────────────────────────────────
# USTAWIENIA API
# ─────────────────────────────────────────────

BASE_URL = "https://api.openaq.org/v3"
HEADERS  = {
    "X-API-Key": API_KEY,
    "Accept": "application/json"
}

# Przerwa między zapytaniami (sekundy) — nie przekraczaj limitu API
RATE_LIMIT_DELAY = 0.5


# ─────────────────────────────────────────────
# FUNKCJE POMOCNICZE
# ─────────────────────────────────────────────

def api_get(endpoint, params=None):
    """Wykonuje zapytanie GET do OpenAQ API z obsługą błędów."""
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
        print(f"  ❌ Błąd API: {e}")
        return None


def get_country_id(country_code):
    """Pobiera numeryczne ID kraju z OpenAQ na podstawie kodu ISO."""
    data = api_get("countries", params={"limit": 200})
    if not data:
        return None
    for c in data.get("results", []):
        if c.get("code") == country_code:
            return c.get("id")
    return None


def get_stations(country_code, parameter_ids, limit=50):
    """
    Pobiera listę stacji pomiarowych dla danego kraju
    które mierzą co najmniej jeden z wybranych parametrów.
    """
    country_id = get_country_id(country_code)
    if not country_id:
        print(f"  ❌ Nie znaleziono kraju: {country_code}")
        return []

    stations = []
    page = 1

    while len(stations) < limit:
        params = {
            "countries_id": country_id,
            "limit": 100,
            "page": page,
            "monitor": True,          # tylko oficjalne stacje rządowe
        }
        data = api_get("locations", params=params)
        if not data or not data.get("results"):
            break

        for loc in data["results"]:
            # Filtruj stacje które mają GPS i mierzą potrzebne parametry
            coords = loc.get("coordinates", {})
            if not coords or not coords.get("latitude"):
                continue

            sensors = loc.get("sensors", [])
            loc_param_ids = {s.get("parameter", {}).get("id") for s in sensors}

            # Stacja musi mierzyć co najmniej jeden z naszych parametrów
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

        # Sprawdź czy są kolejne strony
        meta = data.get("meta", {})
        found_raw = meta.get("found", 0)
        try:
            found = int(str(found_raw).replace(">", "").replace("<", "").strip())
        except (ValueError, TypeError):
            found = 9999  # jeśli nie można sparsować, zakładamy że jest więcej
 
        if len(stations) >= limit or page * 100 >= found:
            break
        page += 1
        time.sleep(RATE_LIMIT_DELAY)
 

    return stations


def get_daily_measurements(sensor_id, date_from, date_to):
    """
    Pobiera dzienne średnie wartości dla konkretnego sensora.
    Używa endpointu /days który jest najbardziej niezawodny.
    """
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
        found = meta.get("found", 0)
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
    print("  OpenAQ Data Downloader — Jakość Powietrza")
    print(f"{'='*60}\n")

    if API_KEY == "TWÓJ_KLUCZ_API":
        print("❌ Uzupełnij zmienną API_KEY przed uruchomieniem!")
        print("   Klucz pobierz na: https://explore.openaq.org")
        return

    all_stations   = []
    all_measurements = []

    # ── KROK 1: Pobierz stacje dla każdego kraju ──────────────────
    print("📍 KROK 1: Pobieranie stacji pomiarowych...")
    print("-" * 40)

    for country_code, country_name in COUNTRIES.items():
        print(f"\n  🌍 {country_name} ({country_code})")
        stations = get_stations(
            country_code,
            parameter_ids=set(PARAMETERS.keys()),
            limit=MAX_STATIONS_PER_COUNTRY
        )
        print(f"     Znaleziono: {len(stations)} stacji")
        all_stations.extend(stations)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_stations:
        print("\n❌ Nie pobrano żadnych stacji. Sprawdź klucz API.")
        return

    # ── KROK 2: Pobierz pomiary dla każdej stacji ─────────────────
    print(f"\n\n📊 KROK 2: Pobieranie pomiarów ({DATE_FROM} → {DATE_TO})")
    print("-" * 40)
    print(f"  Łącznie stacji: {len(all_stations)}")
    print(f"  Parametry: {', '.join(PARAMETERS.values())}\n")

    station_rows = []   # do stations.csv
    param_id_map = {v: k for k, v in PARAMETERS.items()}

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

        # Pobierz dane dla każdego sensora (sensor = jeden parametr na stacji)
        for sensor in station.get("sensors", []):
            sensor_id    = sensor.get("id")
            param_id     = sensor.get("parameter", {}).get("id")
            param_name   = sensor.get("parameter", {}).get("name", "")

            # Pobierz tylko interesujące nas parametry
            if param_id not in PARAMETERS:
                continue

            measurements = get_daily_measurements(sensor_id, DATE_FROM, DATE_TO)

            for m in measurements:
                period = m.get("period", {})
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

    # ── KROK 3: Zapis plików ──────────────────────────────────────
    print("\n\n💾 KROK 3: Zapisywanie plików...")
    print("-" * 40)

    # stations.csv — do mapy stacji
    df_stations = pd.DataFrame(station_rows)
    stations_path = os.path.join(OUTPUT_DIR, "stations.csv")
    df_stations.to_csv(stations_path, index=False)
    print(f"  ✅ stations.csv          ({len(df_stations)} stacji)")

    if not all_measurements:
        print("  ❌ Brak pomiarów — sprawdź daty lub dostępność danych")
        return

    # measurements_daily.csv — szeregi czasowe
    df_meas = pd.DataFrame(all_measurements)
    df_meas["date"] = pd.to_datetime(df_meas["date"])
    df_meas = df_meas.sort_values(["country_code", "location_id", "date"])

    meas_path = os.path.join(OUTPUT_DIR, "measurements_daily.csv")
    df_meas.to_csv(meas_path, index=False)
    print(f"  ✅ measurements_daily.csv ({len(df_meas):,} rekordów)")

    # station_profiles.csv — gotowe do PCA/UMAP
    # Każda stacja = jeden wiersz, kolumny = średnie wartości parametrów
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

    # Usuń stacje z za dużą liczbą braków (potrzebne do PCA)
    param_cols = [c for c in df_profiles.columns if c in PARAMETERS.values()]
    df_profiles = df_profiles.dropna(subset=param_cols, thresh=len(param_cols)//2 + 1)

    profiles_path = os.path.join(OUTPUT_DIR, "station_profiles.csv")
    df_profiles.to_csv(profiles_path, index=False)
    print(f"  ✅ station_profiles.csv  ({len(df_profiles)} stacji z profilami)")

    # ── KROK 4: Podsumowanie ──────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PODSUMOWANIE")
    print(f"{'='*60}")
    print(f"\n  Pliki zapisano w folderze: ./{OUTPUT_DIR}/\n")

    print(f"  {'Kraj':<20} {'Stacje':>8} {'Pomiary':>10}")
    print(f"  {'-'*40}")
    for cc, cn in COUNTRIES.items():
        n_st = len(df_stations[df_stations.country_code == cc])
        n_m  = len(df_meas[df_meas.country_code == cc])
        print(f"  {cn:<20} {n_st:>8} {n_m:>10,}")

    print(f"\n  {'ŁĄCZNIE':<20} {len(df_stations):>8} {len(df_meas):>10,}")
    print(f"\n  Zakres dat:   {df_meas.date.min().date()} → {df_meas.date.max().date()}")
    print(f"  Parametry:    {', '.join(df_meas.parameter.unique())}")
    print(f"\n  ✅ Gotowe do analizy!\n")
    print("  Następne kroki:")
    print("  1. stations.csv          → folium/plotly mapa stacji")
    print("  2. measurements_daily.csv → analiza szeregów, sezonowość")
    print("  3. station_profiles.csv  → PCA, UMAP, PaCMAP\n")


if __name__ == "__main__":
    main()