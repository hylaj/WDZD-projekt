import pandas as pd
import numpy as np
import os


INPUT_DIR      = "openaq_clean_v2"
INPUT_FILE     = os.path.join(INPUT_DIR, "measurements_clean.parquet")
OUTPUT_FILE    = os.path.join(INPUT_DIR, "measurements_episodes.csv")
OUTPUT_SUMMARY = os.path.join(INPUT_DIR, "measurements_episodes_summary.csv")
LIMITS_FILE    = "who_limits.csv"

EPISODE_THRESHOLD = 3


def main():

    df_meas   = pd.read_parquet(INPUT_FILE)
    df_limits = pd.read_csv(LIMITS_FILE)

    df_limits["parameter"] = df_limits["parameter"].str.lower().str.strip()
    df_meas = df_meas.merge(df_limits, on="parameter", how="left")

    df_meas["above_who"] = (df_meas["value"] > df_meas["who_limit"]).astype(int)
    df_meas["date"]      = pd.to_datetime(df_meas["date"])



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

    df_meas = df_meas.sort_values(
        ["location_id", "parameter", "date"]
    ).reset_index(drop=True)

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

    df_meas["is_episode"] = 0

    ep_ends = df_meas.index[df_meas["rolling_3d_sum"] >= EPISODE_THRESHOLD]

    df_meas["_ep_end"] = 0
    df_meas.loc[ep_ends, "_ep_end"] = 1

    grp = df_meas.groupby(["location_id", "parameter"])["_ep_end"]

    df_meas["is_episode"] = (
        df_meas["_ep_end"]
        .add(grp.shift(-1).fillna(0))   
        .add(grp.shift(-2).fillna(0))   
        .clip(upper=1)
        .astype(int)
    )

    df_meas = df_meas.drop(columns=["_ep_end"])


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
        df_meas["episode_length"] = df_meas.groupby("episode_id")["date"].transform("count")
    else:
        df_meas["episode_id"]     = np.nan
        df_meas["episode_length"] = np.nan

    df_meas = df_meas.drop(
        columns=["rolling_3d_sum", "block_id", "who_limit"], errors="ignore"
    )


    cols_daily = [
        "date", "location_id", "station_name", "city",
        "country_code", "country_name", "latitude", "longitude",
        "parameter", "value", "unit", "year", "month", "season",
        "above_who", "is_episode", "episode_id", "episode_length",
    ]
    df_meas[cols_daily].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Zapisano: {OUTPUT_FILE}  ({len(df_meas):,} wierszy)")



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

    df_summary["days_span"] = (
        (df_summary["episode_end"] - df_summary["episode_start"]).dt.days + 1
    )

    df_summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"Zapisano: {OUTPUT_SUMMARY}  ({len(df_summary):,} epizodów)")


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
