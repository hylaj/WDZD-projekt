import requests
import pandas as pd
import time
import os
from tqdm import tqdm

API_KEY = "X"

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

PARAMETERS = {
    1: "pm10",
    2: "pm25",
    5: "no2",
    3: "o3",
}

DATE_FROM = "2022-01-01"
DATE_TO   = "2025-12-31"
MAX_STATIONS_PER_COUNTRY = 50
OUTPUT_DIR = "openaq_data_v2"

BASE_URL = "https://api.openaq.org/v3"
HEADERS  = {"X-API-Key": API_KEY, "Accept": "application/json"}
RATE_LIMIT_DELAY = 0.5


def api_get(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            print("Limit zapytań — czekam 60 sekund...")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd API: {e}")
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

            stations.append({
                "location_id":  loc["id"],
                "station_name": loc.get("name", ""),
                "city":         loc.get("locality", ""),
                "country_code": country_code,
                "country_name": COUNTRIES[country_code],
                "latitude":     coords["latitude"],
                "longitude":    coords["longitude"],
                "sensors":      sensors,
            })

            if len(stations) >= limit:
                break

        meta = data.get("meta", {})
        try:
            found = int(str(meta.get("found", 0)).replace(">", "").replace("<", "").strip())
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
        try:
            found = int(str(meta.get("found", 0))
                        .replace(">", "").replace("<", "").replace("+", "").strip())
        except (ValueError, TypeError):
            found = 9999999

        if page * 1000 >= found:
            break
        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    return all_results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if API_KEY == "X":
        print("Uzupełnij zmienną API_KEY przed uruchomieniem!")
        return

    all_stations     = []
    all_measurements = []

    for country_code, country_name in COUNTRIES.items():
        print(f"{country_name} ({country_code})...", end=" ")
        stations = get_stations(
            country_code,
            parameter_ids=set(PARAMETERS.keys()),
            limit=MAX_STATIONS_PER_COUNTRY
        )
        print(f"{len(stations)} stacji")
        all_stations.extend(stations)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_stations:
        print("Nie pobrano żadnych stacji.")
        return

    print(f"\nPobieranie pomiarów ({DATE_FROM} → {DATE_TO})")
    print(f"Łącznie stacji: {len(all_stations)}\n")

    station_rows = []

    for station in tqdm(all_stations, desc="Pobieranie"):
        station_id   = station["location_id"]
        station_name = station["station_name"]

        station_rows.append({
            "location_id":  station_id,
            "station_name": station_name,
            "city":         station["city"],
            "country_code": station["country_code"],
            "country_name": station["country_name"],
            "latitude":     station["latitude"],
            "longitude":    station["longitude"],
        })

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

    df_stations = pd.DataFrame(station_rows)
    df_stations.to_parquet(os.path.join(OUTPUT_DIR, "stations.parquet"), index=False)
    print(f"stations.parquet          ({len(df_stations)} stacji)")

    if not all_measurements:
        print("Brak pomiarów.")
        return

    df_meas = pd.DataFrame(all_measurements)
    df_meas["date"] = pd.to_datetime(df_meas["date"])
    df_meas = df_meas.sort_values(["country_code", "location_id", "date"])
    df_meas.to_parquet(os.path.join(OUTPUT_DIR, "measurements_daily.parquet"), index=False)
    print(f"measurements_daily.parquet ({len(df_meas):,} rekordów)")

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
    df_profiles.to_parquet(os.path.join(OUTPUT_DIR, "station_profiles.parquet"), index=False)
    print(f"station_profiles.parquet  ({len(df_profiles)} stacji z profilami)")

    print(f"\nParametry w danych:")
    for p, cnt in df_meas["parameter"].value_counts().items():
        print(f"  {p:<8}: {cnt:>10,}")

if __name__ == "__main__":
    main()
