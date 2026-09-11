import pandas as pd

df = pd.read_csv("../outputs/songs_final_clusters.csv")
features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence']

# find the cluster(s) around size 144
sizes = df['final_cluster'].value_counts()
target_cluster = sizes[sizes == 144].index[0]  # adjust if there's more than one match
print(f"Cluster {target_cluster}: {sizes[target_cluster]} songs")

cluster_songs = df[df['final_cluster'] == target_cluster]

# 1. Just look at some song names — the real "does this feel like a playlist" check
print("\nSample songs:")
print(cluster_songs['name'].sample(20, random_state=1).to_string(index=False))

# 2. Compare this cluster's average feature profile against the WHOLE dataset's average —
#    this tells you whether it's "generic" (close to overall average on everything)
#    or "distinct but still oversized" (far from average, just genuinely big)
overall_mean = df[features].mean()
cluster_mean = cluster_songs[features].mean()

comparison = pd.DataFrame({
    "overall_avg": overall_mean,
    "cluster_avg": cluster_mean,
    "difference": cluster_mean - overall_mean
}).round(3)
print("\nFeature comparison vs. dataset average:")
print(comparison)

# 3. How spread out is it internally? A tight cluster that's just big is a different
#    story than a loose one — std dev per feature tells you that
print("\nWithin-cluster std dev (lower = tighter/more coherent):")
print(cluster_songs[features].std().round(3))