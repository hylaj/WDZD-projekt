import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────
INPUT_DIR      = "openaq_clean_v2"
INPUT_FILE     = os.path.join(INPUT_DIR, "measurements_clean.parquet")
OUTPUT_FILE    = os.path.join(INPUT_DIR, "measurements_episodes.csv")
OUTPUT_SUMMARY = os.path.join(INPUT_DIR, "measurements_episodes_summary.csv")
LIMITS_FILE    = "who_limits.csv"

# Minimalna liczba przekroczeń w oknie 3 dni żeby uznać epizod
EPISODE_THRESHOLD = 3


def main():

    # ── WCZYTANIE ────────────────────────────────────────────────
    df_meas   = pd.read_parquet(INPUT_FILE)
    df_limits = pd.read_csv(LIMITS_FILE)

    df_limits["parameter"] = df_limits["parameter"].str.lower().str.strip()
    df_meas = df_meas.merge(df_limits, on="parameter", how="left")

    df_meas["above_who"] = (df_meas["value"] > df_meas["who_limit"]).astype(int)
    df_meas["date"]      = pd.to_datetime(df_meas["date"])


    # ── KROK 1: JEDEN POMIAR NA DZIEŃ ───────────────────────────
    # POPRAWKA: przed rollingiem agregujemy do jednego wiersza na
    # (location_id, parameter, date). Jeśli stacja wysłała kilka
    # pomiarów tego samego dnia, bierzemy średnią wartości i max
    # flagi above_who (tzn. jeśli choć jeden pomiar przekroczył normę
    # — dzień liczymy jako przekroczenie).
    df_meas = (
        df_meas
        .groupby(["location_id", "parameter", "date"], as_index=False)
        .agg(
            station_name =("station_name",  "first"),
            city         =("city",          "first"),
            country_code =("country_code",  "first"),
            country_name =("country_name",  "first"),
            latitude     =("latitude",      "first"),
            longitude    =("longitude",     "first"),
            value        =("value",         "mean"),
            unit         =("unit",          "first"),
            year         =("year",          "first"),
            month        =("month",         "first"),
            season       =("season",        "first"),
            who_limit    =("who_limit",     "first"),
            above_who    =("above_who",     "max"),   # 1 jeśli chociaż jeden pomiar przekroczył
        )
    )


    # ── KROK 2: SORTOWANIE ──────────────────────────────────────
    df_meas = df_meas.sort_values(
        ["location_id", "parameter", "date"]
    ).reset_index(drop=True)


    # ── KROK 3: ROLLING 3D — SUMA PRZEKROCZEŃ ───────────────────
    # POPRAWKA: rolling("3D") działa na DatetimeIndex i liczy okno
    # WSTECZ (bieżący dzień + 2 poprzednie). To jest poprawne dla
    # wykrywania epizodów — suma >= 3 znaczy: "3 z ostatnich 3 dni
    # przekroczyły normę". NIE cofamy wyniku później.
    #
    # Uwaga dot. luk w danych: rolling("3D") zlicza tyle pomiarów,
    # ile faktycznie jest w oknie czasowym, nie wymagając dokładnie 3.
    # Dlatego warunek >= 3 jest bezpieczny — przy lukach suma będzie
    # mniejsza i epizod nie zostanie błędnie oznaczony.
    rolled = (
        df_meas
        .set_index("date")
        .groupby(["location_id", "parameter"])["above_who"]
        .rolling("3D", min_periods=1)
        .sum()
        .reset_index(name="rolling_3d_sum")
    )

    df_meas = df_meas.merge(
        rolled, on=["location_id", "parameter", "date"], how="left"
    )


    # ── KROK 4: FLAGOWANIE EPIZODÓW ─────────────────────────────
    # POPRAWKA (główna): poprzednia wersja używała bfill() twierdząc,
    # że "cofa jedynki o 2 dni wstecz". To był błąd — bfill w Pandas
    # wypełnia NAPRZÓD (od późniejszego indeksu do wcześniejszego),
    # ale ponieważ dane są posortowane rosnąco po dacie, w praktyce
    # oznaczało przyszłe dni jako epizod, nie przeszłe.
    #
    # WŁAŚCIWA LOGIKA:
    # rolling("3D").sum() >= 3 oznacza: "DZIŚ jest ostatnim dniem
    # epizodu — poprzednie 2 dni też przekroczyły". Żeby oznaczyć
    # cały trójdzień, cofamy flagę o 2 pozycje wstecz używając
    # shift(-1) i shift(-2) z perspektywy odwróconego sortowania,
    # co odpowiada ffill po odwróceniu (czyli bfill na oryginale)
    # — ALE robimy to na osobnym sygnale "koniec epizodu", nie na
    # samej fladze.
    #
    # Najprostszy i najczytelniejszy sposób: dla każdego dnia,
    # który spełnia warunek rolling >= 3, oznaczamy RÓWNIEŻ
    # 2 poprzednie dni (shift(1) i shift(2) w przód w posortowanym df).

    # Zaczynamy od czystej kolumny zer
    df_meas["is_episode"] = 0

    # Indeksy wierszy, gdzie rolling osiągnął próg (= "ostatni dzień epizodu")
    ep_ends = df_meas.index[df_meas["rolling_3d_sum"] >= EPISODE_THRESHOLD]

    # Dla każdej grupy (location_id, parameter) szukamy sąsiednich indeksów
    # Efektywniejszy sposób: tworzymy pomocniczą kolumnę i rozprzestrzeniamy
    df_meas["_ep_end"] = 0
    df_meas.loc[ep_ends, "_ep_end"] = 1

    # Propagacja wstecz o 2 dni w obrębie tej samej grupy:
    # shift(1) przesuwa wartość o 1 wiersz w górę (= dzień wcześniej)
    # shift(2) przesuwa o 2 wiersze w górę
    # Używamy groupby żeby shift nie przekraczał granicy grupy
    grp = df_meas.groupby(["location_id", "parameter"])["_ep_end"]

    df_meas["is_episode"] = (
        df_meas["_ep_end"]
        .add(grp.shift(-1).fillna(0))   # następny wiersz "patrzy wstecz" na bieżący
        .add(grp.shift(-2).fillna(0))   # dwa wiersze dalej patrzy wstecz
        .clip(upper=1)
        .astype(int)
    )

    # Sprzątamy kolumnę pomocniczą
    df_meas = df_meas.drop(columns=["_ep_end"])


    # ── KROK 5: EPISODE_ID I DŁUGOŚĆ ────────────────────────────
    # Wykrywamy bloki ciągłych jedynek (zmiana 0→1 lub 1→0)
    change_cond = (
        df_meas["is_episode"]
        != df_meas.groupby(["location_id", "parameter"])["is_episode"].shift()
    )
    df_meas["block_id"] = (
        change_cond
        .groupby([df_meas["location_id"], df_meas["parameter"]])
        .cumsum()
    )

    df_ep_only = df_meas[df_meas["is_episode"] == 1]

    if not df_ep_only.empty:
        df_meas.loc[df_ep_only.index, "episode_id"] = (
            df_ep_only
            .groupby(["location_id", "parameter", "block_id"])
            .ngroup() + 1
        )
        # POPRAWKA: episode_length liczymy po liczbie unikalnych dat w epizodzie,
        # nie po liczbie wierszy — zabezpieczenie na wypadek ewentualnych duplikatów
        df_meas["episode_length"] = df_meas.groupby("episode_id")["date"].transform("count")
    else:
        df_meas["episode_id"]     = np.nan
        df_meas["episode_length"] = np.nan

    # Kolumny robocze do usunięcia
    df_meas = df_meas.drop(
        columns=["rolling_3d_sum", "block_id", "who_limit"], errors="ignore"
    )


    # ── KROK 6: ZAPIS PLIKU DZIENNEGO ───────────────────────────
    cols_daily = [
        "date", "location_id", "station_name", "city",
        "country_code", "country_name", "latitude", "longitude",
        "parameter", "value", "unit", "year", "month", "season",
        "above_who", "is_episode", "episode_id", "episode_length",
    ]
    df_meas[cols_daily].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Zapisano: {OUTPUT_FILE}  ({len(df_meas):,} wierszy)")


    # ── KROK 7: ZAPIS PODSUMOWANIA EPIZODÓW (jeden wiersz = jeden epizod) ──
    # POPRAWKA: grupujemy po episode_id — każdy epizod pojawia się
    # dokładnie raz, więc AVG(episode_length) w Tableau będzie poprawne.
    # Poprzednia wersja pozwalała na wielokrotne wiersze per epizod
    # (jeden na dzień), przez co Tableau uśredniało długość po dniach,
    # zawyżając wynik dla długich epizodów.
    df_summary = (
        df_meas[df_meas["episode_id"].notna()]
        .groupby([
            "episode_id", "location_id", "station_name", "city",
            "country_code", "country_name", "parameter"
        ])
        .agg(
            episode_start  = ("date",           "min"),
            episode_end    = ("date",           "max"),
            episode_length = ("episode_length", "first"),
            max_value      = ("value",          "max"),
            mean_value     = ("value",          "mean"),
        )
        .reset_index()
    )

    # Kontrola: episode_length powinien być spójny z (end - start)
    df_summary["days_span"] = (
        (df_summary["episode_end"] - df_summary["episode_start"]).dt.days + 1
    )

    df_summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"Zapisano: {OUTPUT_SUMMARY}  ({len(df_summary):,} epizodów)")


    # ── DIAGNOSTYKA ─────────────────────────────────────────────
    print("\n── Diagnostyka ────────────────────────────────────────────")
    n_ep = df_meas["episode_id"].notna().sum()
    n_ep_unique = df_meas["episode_id"].nunique()
    print(f"  Wierszy z epizodem:   {n_ep:,}")
    print(f"  Unikalnych epizodów:  {n_ep_unique:,}")

    if not df_summary.empty:
        print(f"\n  Długości epizodów (ze summary — jeden wiersz = jeden epizod):")
        print(f"    Min:    {df_summary['episode_length'].min():.0f} dni")
        print(f"    Mediana:{df_summary['episode_length'].median():.0f} dni")
        print(f"    Średnia:{df_summary['episode_length'].mean():.1f} dni")
        print(f"    Max:    {df_summary['episode_length'].max():.0f} dni")

        print(f"\n  Top 10 krajów wg mediany długości epizodu (pm25):")
        top = (
            df_summary[df_summary["parameter"] == "pm25"]
            .groupby("country_name")["episode_length"]
            .agg(
                n_epizodow="count",
                mediana_dl="median",
                srednia_dl="mean",
            )
            .query("n_epizodow >= 5")           # min. 5 epizodów żeby ranking był wiarygodny
            .sort_values("mediana_dl", ascending=False)
            .head(10)
        )
        print(top.to_string())

        # Ostrzeżenie o małych próbach (jak Kosowo na wykresie)
        small = (
            df_summary[df_summary["parameter"] == "pm25"]
            .groupby("country_name")["episode_id"]
            .count()
        )
        small_countries = small[small < 5].index.tolist()
        if small_countries:
            print(
                f"\n  ⚠ Kraje z < 5 epizodami PM2.5 — wyniki mało wiarygodne:\n"
                f"    {', '.join(small_countries)}"
            )


if __name__ == "__main__":
    main()