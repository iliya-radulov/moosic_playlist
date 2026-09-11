import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("../outputs/songs_final_clusters.csv")
features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence']

# Same global scaling philosophy as the main pipeline — refit on the FULL
# dataset, not just this subset, so this cluster's numbers stay comparable
# to everything else you've already looked at
scaler = MinMaxScaler().set_output(transform="pandas")
scaled_all = scaler.fit_transform(df[features])

TARGET_CLUSTER = 99  # the 97-song cluster from before — adjust if it's now labeled differently
subset = df[df['final_cluster'] == TARGET_CLUSTER]
subset_scaled = scaled_all.loc[subset.index]

print(f"Re-splitting cluster {TARGET_CLUSTER} ({len(subset)} songs)\n")

# Step 1: try a small handful of k, just to see if there's a clean natural split
print("=== Exploring k=2 to 4 ===")
for k in range(2, 5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(subset_scaled)
    sil = silhouette_score(subset_scaled, labels)
    sizes = pd.Series(labels).value_counts().sort_index().tolist()
    print(f"k={k}: silhouette={sil:.3f}, sizes={sizes}")

# Step 2: commit to k=2 first (simplest, matches the language/genre split you
# spotted by eye) and look at what actually separated
print("\n=== k=4 breakdown ===")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
subset = subset.copy()
subset['sub_cluster'] = kmeans.fit_predict(subset_scaled)

for sub in sorted(subset['sub_cluster'].unique()):
    group = subset[subset['sub_cluster'] == sub]
    print(f"\n--- Sub-cluster {sub} ({len(group)} songs) ---")
    print("Feature averages:")
    print(group[features].mean().round(3))
    print("\nSample songs:")
    print(group['name'].sample(min(10, len(group)), random_state=1).to_string(index=False))