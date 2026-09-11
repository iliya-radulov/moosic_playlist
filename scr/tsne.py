import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import os
from sklearn.manifold import TSNE
from sklearn import set_config
import seaborn as sns

set_config(transform_output="pandas")

np.random.seed(42)
RANDOM_STATE = 42
TARGET_MIN, TARGET_MAX = 30, 40
MIN_SONGS_TO_SPLIT = 2
MAX_K_PER_SPLIT = 5

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


# reuse the scaled features and final cluster labels you already have
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_all)  # scaled_all from your existing pipeline
tsne_results['Cluster'] = df['final_cluster']

sns.relplot(
    data=tsne_results,
    x=tsne_results.columns[0],
    y=tsne_results.columns[1],
    hue='Cluster',
    palette='viridis',
    s=40
).set(title="t-SNE Visualisation of Final Playlists")
plt.savefig("../outputs/tsne_visualisation.png")