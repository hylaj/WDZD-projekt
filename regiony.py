import pandas as pd

df_monthly  = pd.read_parquet("openaq_clean_v2/monthly_aggregates.parquet")
df_pca      = pd.read_parquet("openaq_clean_v2/pca_umap_results.parquet")

region_map = (
    df_pca.groupby("country_code")["region"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)

df_monthly = df_monthly.merge(region_map, on="country_code", how="left")

df_monthly.to_csv(
    "openaq_clean_v2/monthly_aggregates.csv",
    index=False, encoding="utf-8-sig"
)

print(df_monthly[["country_name", "region", "parameter", "mean"]].head(10))
print(f"\nRegiony: {sorted(df_monthly['region'].dropna().unique())}")
print(f"Kolumny: {list(df_monthly.columns)}")
print(f"Wierszy: {len(df_monthly):,}")
