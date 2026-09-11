import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import os

np.random.seed(42)
RANDOM_STATE = 42
TARGET_MIN, TARGET_MAX = 30, 40
MIN_SONGS_TO_SPLIT = 2
MAX_K_PER_SPLIT = 5


os.makedirs("../outputs", exist_ok=True)

df = pd.read_csv("../data/5000_songs.csv")
df.columns = df.columns.str.strip()
features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence']
#features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence',
#            'speechiness', 'instrumentalness']
# Scale ONCE, globally — every branch shares this space so leaf centroids
# stay comparable during the final reassignment pass
scaler = MinMaxScaler().set_output(transform="pandas")
scaled_all = scaler.fit_transform(df[features])

leaves = []  # each entry: list of df row-indices belonging to that leaf

def recursive_split(indices):
    n = len(indices)
    subset_scaled = scaled_all.loc[indices]

    if n <= TARGET_MAX or n < MIN_SONGS_TO_SPLIT:
        leaves.append(indices)
        return

    k = min(MAX_K_PER_SPLIT, max(2, n // TARGET_MAX))
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(subset_scaled)

    for c in range(k):
        child_indices = [idx for idx, lab in zip(indices, labels) if lab == c]
        if child_indices:
            recursive_split(child_indices)

recursive_split(list(df.index))
print(f"Recursive split produced {len(leaves)} leaf clusters before reassignment.")

def merge_undersized_leaves(leaves, leaf_centroids, min_size, max_size):
    leaves = [list(l) for l in leaves]
    leaf_centroids = leaf_centroids.copy()
    changed = True
    while changed:
        changed = False
        sizes = [len(l) for l in leaves]
        for i, size in enumerate(sizes):
            if size < min_size and len(leaves) > 1:
                candidates = [j for j in range(len(leaves)) if j != i]

                # real distances from leaf i's centroid to every other centroid
                dists = np.linalg.norm(leaf_centroids[candidates] - leaf_centroids[i], axis=1)
                order = np.argsort(dists)
                sorted_candidates = [candidates[o] for o in order]  # nearest → farthest

                # walk nearest-first, take the first one that still fits under max_size
                nearest = next(
                    (j for j in sorted_candidates if len(leaves[j]) + size <= max_size),
                    sorted_candidates[0]  # nothing fits — fall back to truly nearest anyway
                )

                leaves[nearest].extend(leaves[i])
                del leaves[i]
                leaf_centroids = np.delete(leaf_centroids, i, axis=0)
                changed = True
                break
    return leaves, leaf_centroids
# Leaf centroids, in the shared scaled space
leaf_centroids = np.array([scaled_all.loc[leaf].mean(axis=0).values for leaf in leaves])

# Reassignment FIRST — fixes cross-branch drift
final_labels = pairwise_distances_argmin(scaled_all.values, leaf_centroids)
df['final_cluster'] = final_labels

# Rebuild cluster membership from what reassignment actually produced
reassigned = [df[df['final_cluster'] == c].index.tolist() for c in sorted(df['final_cluster'].unique())]
reassigned_centroids = np.array([scaled_all.loc[c].mean(axis=0).values for c in reassigned])

# NOW merge — operating on the real final groups, not the pre-reassignment leaves
final_clusters, final_centroids = merge_undersized_leaves(reassigned, reassigned_centroids, TARGET_MIN, TARGET_MAX)

# Write the merge result directly into df — no further reassignment,
# so the size guarantee actually survives to the output
df['final_cluster'] = -1
for new_label, members in enumerate(final_clusters):
    df.loc[members, 'final_cluster'] = new_label

sizes = df['final_cluster'].value_counts().sort_index()

in_range = sizes[(sizes >= TARGET_MIN) & (sizes <= TARGET_MAX)]
print(f"\n{len(in_range)} / {len(sizes)} playlists within [{TARGET_MIN}, {TARGET_MAX}] "
      f"({100 * len(in_range) / len(sizes):.0f}%)")

# THE primary "are we there" diagnostic at this scale — not a radar chart
plt.figure(figsize=(10, 5))
plt.hist(sizes, bins=30, color='steelblue', edgecolor='black')
plt.axvline(TARGET_MIN, color='red', linestyle='--', label=f'target min ({TARGET_MIN})')
plt.axvline(TARGET_MAX, color='red', linestyle='--', label=f'target max ({TARGET_MAX})')
plt.xlabel('Playlist size')
plt.ylabel('Number of playlists')
plt.title(f'Final playlist size distribution ({len(sizes)} playlists)')
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/final_size_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

df.to_csv("../outputs/songs_final_clusters.csv", index=False)

# Spot-check radar — a small, readable SAMPLE, not all of them
sample_ids = sorted(sizes.index)[::max(1, len(sizes) // 6)][:6]
sample_centroids = pd.DataFrame(
    {c: scaled_all.loc[df[df['final_cluster'] == c].index].mean() for c in sample_ids}
).T

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]
for c in sample_ids:
    values = sample_centroids.loc[c].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {c} (n={sizes[c]})')
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)
ax.set_ylim(0, 1)
ax.set_title('Spot-check: sample of final playlists', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig("../outputs/spotcheck_radar.png", dpi=150, bbox_inches="tight")
plt.show()


# Report
lines = [
    "# Moosic Recursive Clustering Report\n",
    f"Target playlist size: {TARGET_MIN}-{TARGET_MAX} songs\n",
    f"Total songs: {df.shape[0]}\n",
    f"Final number of playlists: {len(sizes)}\n",
    f"Within target range: {len(in_range)} / {len(sizes)} ({100*len(in_range)/len(sizes):.0f}%)\n",
    "\n## Size distribution\n", sizes.describe().to_markdown(),
    "\n\n## All final cluster sizes\n", sizes.to_markdown(),
]
with open("../outputs/recursive_clustering_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Report saved to ../outputs/recursive_clustering_report.md")

def run_pipeline(target_min, target_max, min_songs_to_split=10, max_k_per_split=5, label=None):
    # ... same body as before, but using these args instead of the hardcoded constants ...
    # at the end, instead of/alongside the full report, append one row:
    result = {
        "label": label or f"min{target_min}_max{target_max}_split{min_songs_to_split}_k{max_k_per_split}",
        "target_min": target_min, "target_max": target_max,
        "min_songs_to_split": min_songs_to_split, "max_k_per_split": max_k_per_split,
        "num_playlists": len(sizes),
        "pct_in_range": 100 * len(in_range) / len(sizes),
        "mean_size": sizes.mean(), "median_size": sizes.median(),
        "min_size": sizes.min(), "max_size": sizes.max(),
        "min_leaf_size_before_merge": min(len(l) for l in leaves),          # pre-merge, still valid as a "before" baseline
        "min_leaf_size_after_merge": min(len(c) for c in final_clusters),   # ← use final_clusters, and use min (not max) to check the floor held
        "max_leaf_size_after_merge": max(len(c) for c in final_clusters),   # ← separate field to check the ceiling held too
    }
    log_path = "../outputs/experiment_log.csv"
    log_df = pd.DataFrame([result])
    log_df.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    return sizes  # or df, whatever you want to inspect further per-run


# from sklearn.manifold import TSNE
# from sklearn import set_config
# import seaborn as sns

# set_config(transform_output="pandas")

# # reuse the scaled features and final cluster labels you already have
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(scaled_all)  # scaled_all from your existing pipeline
# tsne_results['Cluster'] = df['final_cluster']

# # sns.relplot(
# #     data=tsne_results,
# #     x=tsne_results.columns[0],
# #     y=tsne_results.columns[1],
# #     hue='Cluster',
# #     palette='viridis',
# #     s=40
# # ).set(title="t-SNE Visualisation of Final Playlists")
# # plt.savefig("../outputs/tsne_visualisation.png")
# # plt.show()


# tsne_results['BroadCluster'] = df['cluster']  # your original k=8 labels, not the ~130 leaf ones

# sns.relplot(
#     data=tsne_results,
#     x=tsne_results.columns[0],
#     y=tsne_results.columns[1],
#     hue='BroadCluster',
#     palette='tab10',   # qualitative palette — built for a handful of unordered categories
#     s=40
# ).set(title="t-SNE colored by broad (k=8) clusters")
# plt.savefig("../outputs/tsne_visualisation_broad.png")
# plt.show()


# highlight = 99  # or whatever cluster ID you want to check
# tsne_results['highlight'] = (df['final_cluster'] == highlight).map({True: 'This playlist', False: 'Everything else'})

# sns.relplot(
#     data=tsne_results,
#     x=tsne_results.columns[0],
#     y=tsne_results.columns[1],
#     hue='highlight',
#     palette={'This playlist': 'crimson', 'Everything else': 'lightgray'},
#     s=40
# ).set(title=f"Where cluster {highlight} sits in the full song space")
# plt.savefig("../outputs/tsne_visualisation_highlight.png")
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

candidate_features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence',
                       'loudness', 'speechiness', 'instrumentalness', 'liveness']

corr = df[candidate_features].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.show()


run_pipeline(TARGET_MIN, TARGET_MAX, MIN_SONGS_TO_SPLIT, MAX_K_PER_SPLIT, label="default_run")