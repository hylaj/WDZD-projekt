"""
PCA + UMAP na profilach stacji — Globalna jakość powietrza (Europa)
====================================================================
Wejście:  openaq_clean_v2/station_profiles_extended.parquet  (982 stacji, 16 cech)

Wyjście:  openaq_clean_v2/
    pca_umap_results.parquet   — wyniki PCA + UMAP z metadanymi (pod Tableau)
    pca_umap_results.csv       — to samo w CSV (Tableau)
    pca_variance.csv           — wyjaśniona wariancja PCA (pod wykres scree plot)
    pca_loadings.csv           — ładunki PCA (które cechy wpływają na PC1/PC2)

Zależności: pip install pandas numpy scikit-learn umap-learn pyarrow
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────

INPUT_DIR  = "openaq_clean_v2"
OUTPUT_DIR = "openaq_clean_v2"

# Kolumny z cechami (16 cech zbudowanych w preprocessing)
FEATURE_COLS = [
    "mean_pm25", "mean_pm10", "mean_no2", "mean_o3",
    "std_pm25",  "std_pm10",  "std_no2",  "std_o3",
    "winter_pm25", "winter_pm10", "winter_no2", "winter_o3",
    "summer_pm25", "summer_pm10", "summer_no2", "summer_o3",
]

# Metadane stacji — zachowujemy do wyników
META_COLS = [
    "location_id", "station_name", "city",
    "country_code", "country_name", "latitude", "longitude",
]

# Parametry UMAP
UMAP_PARAMS = {
    "n_neighbors":   15,    # lokalność sąsiedztwa (5–50); więcej = bardziej globalna struktura
    "min_dist":      0.1,   # minimalna odległość punktów w 2D (0.0–0.5)
    "n_components":  2,     # wymiar wyjściowy
    "metric":        "euclidean",
    "random_state":  42,    # reprodukowalność
}

# Liczba składowych PCA do zachowania
N_PCA_COMPONENTS = 10

# ─────────────────────────────────────────────
# 1. WCZYTANIE
# ─────────────────────────────────────────────

print("=" * 60)
print("  PCA + UMAP — Profile stacji pomiarowych")
print("=" * 60)

df = pd.read_parquet(os.path.join(INPUT_DIR, "station_profiles_extended.parquet"))
print(f"\n[1] Wczytano: {len(df)} stacji, {len(df.columns)} kolumn")

# Sprawdź czy wszystkie kolumny cech są dostępne
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"    UWAGA: brakuje kolumn: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

print(f"    Cechy: {len(FEATURE_COLS)}")
print(f"    Brakujące wartości: {df[FEATURE_COLS].isna().sum().sum()}")

# ─────────────────────────────────────────────
# 2. PRZYGOTOWANIE MACIERZY CECH
# ─────────────────────────────────────────────

print(f"\n[2] Przygotowanie danych...")

X = df[FEATURE_COLS].values

# Standaryzacja — konieczna przed PCA i UMAP
# (cechy mają różne skale: O3 ~50 µg/m³, PM2.5 ~8 µg/m³)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"    Macierz cech: {X_scaled.shape}")
print(f"    Średnia po standaryzacji: {X_scaled.mean():.4f} (powinno być ~0)")
print(f"    Std po standaryzacji:     {X_scaled.std():.4f}  (powinno być ~1)")

# ─────────────────────────────────────────────
# 3. PCA
# ─────────────────────────────────────────────

print(f"\n[3] PCA ({N_PCA_COMPONENTS} składowych)...")

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"    Wyjaśniona wariancja:")
for i, (ev, cum) in enumerate(zip(explained, cumulative)):
    bar = "█" * int(ev * 50)
    print(f"    PC{i+1:>2}: {ev*100:5.1f}%  (łącznie: {cum*100:5.1f}%)  {bar}")

print(f"\n    PC1+PC2 wyjaśniają: {cumulative[1]*100:.1f}% wariancji")
print(f"    PC1–PC4 wyjaśniają: {cumulative[3]*100:.1f}% wariancji")

# Ładunki PCA — które cechy dominują w PC1 i PC2
df_loadings = pd.DataFrame(
    pca.components_[:4].T,
    index=FEATURE_COLS,
    columns=[f"PC{i+1}" for i in range(4)]
)
df_loadings["feature"] = df_loadings.index
df_loadings = df_loadings.reset_index(drop=True)

print(f"\n    Top 5 cech wg wpływu na PC1:")
top_pc1 = df_loadings[["feature", "PC1"]].reindex(
    df_loadings["PC1"].abs().sort_values(ascending=False).index
).head(5)
for _, row in top_pc1.iterrows():
    print(f"      {row['feature']:<20}: {row['PC1']:+.3f}")

print(f"\n    Top 5 cech wg wpływu na PC2:")
top_pc2 = df_loadings[["feature", "PC2"]].reindex(
    df_loadings["PC2"].abs().sort_values(ascending=False).index
).head(5)
for _, row in top_pc2.iterrows():
    print(f"      {row['feature']:<20}: {row['PC2']:+.3f}")

# ─────────────────────────────────────────────
# 4. UMAP
# ─────────────────────────────────────────────

print(f"\n[4] UMAP...")
print(f"    Parametry: n_neighbors={UMAP_PARAMS['n_neighbors']}, "
      f"min_dist={UMAP_PARAMS['min_dist']}, metric={UMAP_PARAMS['metric']}")
print(f"    Uruchamianie... (kilka sekund)")

reducer = umap.UMAP(**UMAP_PARAMS)

# UMAP uruchamiamy na wynikach PCA (pierwsze 10 składowych)
# zamiast na surowych 16 cechach — szybciej i stabilniej
X_umap = reducer.fit_transform(X_pca)

print(f"    Gotowe. Zakres UMAP1: [{X_umap[:,0].min():.2f}, {X_umap[:,0].max():.2f}]")
print(f"            Zakres UMAP2: [{X_umap[:,1].min():.2f}, {X_umap[:,1].max():.2f}]")

# ─────────────────────────────────────────────
# 5. BUDOWA TABELI WYNIKÓW
# ─────────────────────────────────────────────

print(f"\n[5] Budowa tabeli wyników...")

df_results = df[META_COLS].copy()

# Współrzędne PCA (pierwsze 4 składowe)
for i in range(min(4, N_PCA_COMPONENTS)):
    df_results[f"PC{i+1}"] = X_pca[:, i]

# Współrzędne UMAP
df_results["UMAP1"] = X_umap[:, 0]
df_results["UMAP2"] = X_umap[:, 1]

# Dołącz oryginalne cechy (przydatne do hover w Tableau)
for col in ["mean_pm25", "mean_pm10", "mean_no2", "mean_o3",
            "winter_pm25", "summer_pm25", "winter_o3", "summer_o3"]:
    if col in df.columns:
        df_results[col] = df[col].values

# Prosta klasyfikacja regionu geograficznego na podstawie współrzędnych stacji
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

print(f"    Wierszy: {len(df_results)}")
print(f"    Kolumny: {list(df_results.columns)}")
print(f"\n    Rozkład regionów:")
for region, cnt in df_results["region"].value_counts().items():
    print(f"      {region:<35}: {cnt:>4} stacji")

# ─────────────────────────────────────────────
# 6. ZAPIS
# ─────────────────────────────────────────────

print(f"\n[6] Zapisywanie wyników...")

# Główny plik wyników
df_results.to_parquet(
    os.path.join(OUTPUT_DIR, "pca_umap_results.parquet"), index=False)
df_results.to_csv(
    os.path.join(OUTPUT_DIR, "pca_umap_results.csv"), index=False, encoding="utf-8-sig")
print(f"    pca_umap_results.parquet/csv    ({len(df_results)} stacji)")

# Wyjaśniona wariancja PCA (pod scree plot)
df_variance = pd.DataFrame({
    "component":   [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)],
    "explained":   explained,
    "explained_pct": explained * 100,
    "cumulative":  cumulative,
    "cumulative_pct": cumulative * 100,
})
df_variance.to_csv(
    os.path.join(OUTPUT_DIR, "pca_variance.csv"), index=False, encoding="utf-8-sig")
print(f"    pca_variance.csv                ({len(df_variance)} składowych)")

# Ładunki PCA (pod biplot lub tabelę w sprawozdaniu)
df_loadings.to_csv(
    os.path.join(OUTPUT_DIR, "pca_loadings.csv"), index=False, encoding="utf-8-sig")
print(f"    pca_loadings.csv                ({len(df_loadings)} cech × 4 składowe)")

print(f"\n{'='*60}")
print("  GOTOWE")
print(f"{'='*60}")
print(f"""
Pliki gotowe do Tableau:
  pca_umap_results.csv    → scatter plot PCA i UMAP (Osoba A)
  pca_variance.csv        → scree plot PCA         (Osoba A)

Kolumny w pca_umap_results.csv:
  PC1, PC2            — osie wykresu PCA
  UMAP1, UMAP2        — osie wykresu UMAP
  country_name        — kolor punktów
  region              — alternatywny kolor punktów
  mean_pm25 itd.      — rozmiar punktów / tooltip
  latitude, longitude — do połączenia z mapą przez location_id
""")
