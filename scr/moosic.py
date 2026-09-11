# Step 1: Create the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# put all outputs in one folder so nothing scatters across your project
os.makedirs("../outputs", exist_ok=True)

np.random.seed(42)
df = pd.read_csv("../data/5000_songs.csv")
df.columns = df.columns.str.strip()

print("=== Moosic Dataset (5000 Songs) ===")
print(df)
print(f"\nShape: {df.shape}")
print(f"Features: {df.columns.tolist()}")

# Step 2: Scale the data
features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence']
X = df[features]

my_min_max = MinMaxScaler().set_output(transform="pandas")
scaled_features_df = my_min_max.fit_transform(X)

print("\n=== Scaled Data ===")
print(scaled_features_df.round(3))

# Step 3: K-Means Clustering — exploring k
# (dropped the per-k song-list printout: at 40 values of k that's an enormous
#  wall of output and you don't need membership detail until you've picked k)
k_values = range(2, 25)
inertias = []
silhouettes = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features_df)

    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(scaled_features_df, kmeans.labels_)
    silhouettes.append(sil_score)

    print(f"K = {k:>2}  |  Inertia: {kmeans.inertia_:9.2f}  |  Silhouette: {sil_score:.3f}")

# Step 4: elbow / silhouette plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(k_values, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

axes[1].plot(k_values, silhouettes, 'ro-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (higher = better)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig("../outputs/elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()

# Step 5: Radar chart — FIXED
best_k = 42

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features_df)

# FIX 1a: for plotting, use the centroids KMeans actually fit on — already 0-1,
# no inverse_transform. This is the piece that was breaking the chart.
scaled_centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=features)

# Keep the original-unit version separately — only for the printed table
original_centroids = my_min_max.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(original_centroids, columns=features)
centroid_df['cluster'] = range(best_k)

print("\n=== Cluster Centroids (original scale, for reading) ===")
print(centroid_df.round(3))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
categories = features
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i in range(best_k):
    values = scaled_centroid_df.iloc[i][features].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)  # valid now — these really are 0-1 values
ax.set_title(f'Cluster Profiles (K={best_k})', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig(f"../outputs/cluster_profiles_k{best_k}.png", dpi=150, bbox_inches="tight")
plt.show()


# every song with its final cluster assignment — this is your real deliverable
df.to_csv(f"../outputs/songs_clustered_k{best_k}.csv", index=False)

# the elbow/silhouette numbers, so you don't have to re-run 40 KMeans fits
# every time you want to re-check where the knee was
metrics_df = pd.DataFrame({"k": list(k_values), "inertia": inertias, "silhouette": silhouettes})
metrics_df.to_csv("../outputs/k_selection_metrics.csv", index=False)

# centroid table
centroid_df.to_csv(f"../outputs/cluster_centroids_k{best_k}.csv", index=False)

# FIX 2: both loops below now correctly nested — previously "for song in ..."
# sat outside "for cluster in ...", so it only ran once, using leftover data
# from whichever cluster the loop last touched.
print("\n=== Final Playlists (Clusters) ===")
for cluster in range(best_k):
    songs_in_cluster = df[df['cluster'] == cluster]['name'].tolist()
    print(f"\n🎵 Playlist {cluster + 1} ({len(songs_in_cluster)} songs):")
    for song in songs_in_cluster:
        print(f"  - {song}")

print("\n=== Cluster Counts ===")
print(df['cluster'].value_counts())

report_lines = []
report_lines.append(f"# Moosic K-Means Report — K={best_k}\n")
report_lines.append(f"Dataset: {df.shape[0]} songs, features: {features}\n")
report_lines.append("## Cluster Sizes\n")
report_lines.append(df['cluster'].value_counts().sort_index().to_markdown())
report_lines.append("\n\n## Cluster Centroids (original scale)\n")
report_lines.append(centroid_df.round(3).to_markdown(index=False))

report_lines.append("\n\n## Sample Songs per Cluster\n")
for cluster in range(best_k):
    cluster_songs = df[df['cluster'] == cluster]
    sample = cluster_songs['name'].head(10).tolist()
    report_lines.append(f"\n### Cluster {cluster} ({len(cluster_songs)} songs)\n")
    for song in sample:
        report_lines.append(f"- {song}")
    if len(cluster_songs) > 10:
        report_lines.append(f"- ...and {len(cluster_songs) - 10} more")

with open(f"../outputs/moosic_report_{best_k}.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Report saved to ../outputs/moosic_report_{best_k}.md")