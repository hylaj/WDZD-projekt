import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

INPUT_DIR  = "openaq_clean_v2"
OUTPUT_DIR = "openaq_clean_v2"

FEATURE_COLS = [
    "mean_pm25", "mean_pm10", "mean_no2", "mean_o3",
    "std_pm25",  "std_pm10",  "std_no2",  "std_o3",
    "winter_pm25", "winter_pm10", "winter_no2", "winter_o3",
    "summer_pm25", "summer_pm10", "summer_no2", "summer_o3",
]

META_COLS = [
    "location_id", "station_name", "city",
    "country_code", "country_name", "latitude", "longitude",
]

UMAP_PARAMS = {
    "n_neighbors":  15,
    "min_dist":     0.1,
    "n_components": 2,
    "metric":       "euclidean",
    "random_state": 42,
}

N_PCA_COMPONENTS = 10

df = pd.read_parquet(os.path.join(INPUT_DIR, "station_profiles_extended.parquet"))

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

reducer = umap.UMAP(**UMAP_PARAMS)
X_umap = reducer.fit_transform(X_pca)

df_results = df[META_COLS].copy()

for i in range(min(4, N_PCA_COMPONENTS)):
    df_results[f"PC{i+1}"] = X_pca[:, i]

df_results["UMAP1"] = X_umap[:, 0]
df_results["UMAP2"] = X_umap[:, 1]

for col in ["mean_pm25", "mean_pm10", "mean_no2", "mean_o3",
            "winter_pm25", "summer_pm25", "winter_o3", "summer_o3"]:
    if col in df.columns:
        df_results[col] = df[col].values

def classify_region(lat, lon):
    if lat > 55:
        return "Europa Północna"
    elif lat < 42:
        if lon < 0:
            return "Europa Południowo-Zachodnia"
        else:
            return "Europa Południowa"
    elif lon < 5:
        return "Europa Zachodnia"
    elif lon > 20:
        return "Europa Wschodnia"
    else:
        return "Europa Środkowa"

df_results["region"] = df_results.apply(
    lambda r: classify_region(r["latitude"], r["longitude"]), axis=1
)

df_results.to_parquet(os.path.join(OUTPUT_DIR, "pca_umap_results.parquet"), index=False)
df_results.to_csv(os.path.join(OUTPUT_DIR, "pca_umap_results.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame({
    "component":      [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)],
    "explained":      explained,
    "explained_pct":  explained * 100,
    "cumulative":     cumulative,
    "cumulative_pct": cumulative * 100,
}).to_csv(os.path.join(OUTPUT_DIR, "pca_variance.csv"), index=False, encoding="utf-8-sig")

df_loadings = pd.DataFrame(
    pca.components_[:4].T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(4)]
).reset_index().rename(columns={"index": "feature"})
df_loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"), index=False, encoding="utf-8-sig")

print(f"Gotowe!")
print(f"Stacji: {len(df_results)}")
print(f"PC1+PC2: {cumulative[1]*100:.1f}% wariancji")
print(f"UMAP1: [{X_umap[:,0].min():.2f}, {X_umap[:,0].max():.2f}]")
print(f"UMAP2: [{X_umap[:,1].min():.2f}, {X_umap[:,1].max():.2f}]")
