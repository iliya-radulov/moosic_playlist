# Step 1: Create the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create synthetic data for 10 songs (mimicking Spotify features)
np.random.seed(42)

# songs = [
#     "Bohemian Rhapsody - Queen",
#     "Stairway to Heaven - Led Zeppelin", 
#     "Billie Jean - Michael Jackson",
#     "Shape of You - Ed Sheeran",
#     "Blinding Lights - The Weeknd",
#     "Clocks - Coldplay",
#     "Hotel California - Eagles",
#     "Smells Like Teen Spirit - Nirvana",
#     "Uptown Funk - Mark Ronson",
#     "Viva La Vida - Coldplay"
# ]

# # Create audio features (each row = one song)
# # Features: [danceability, energy, acoustiness, tempo, valence]
# data = np.array([
#     [0.35, 0.55, 0.80, 120, 0.40],  # Bohemian Rhapsody (rock ballad)
#     [0.30, 0.50, 0.75, 115, 0.35],  # Stairway to Heaven (rock ballad)
#     [0.80, 0.75, 0.10, 118, 0.65],  # Billie Jean (pop/dance)
#     [0.85, 0.80, 0.15, 96,  0.75],  # Shape of You (pop/dance)
#     [0.75, 0.70, 0.05, 115, 0.60],  # Blinding Lights (pop/dance)
#     [0.40, 0.60, 0.70, 130, 0.50],  # Clocks (alternative rock)
#     [0.50, 0.45, 0.65, 120, 0.45],  # Hotel California (classic rock)
#     [0.60, 0.85, 0.20, 117, 0.30],  # Smells Like Teen Spirit (grunge)
#     [0.70, 0.65, 0.25, 115, 0.70],  # Uptown Funk (funk/pop)
#     [0.45, 0.55, 0.60, 128, 0.55]   # Viva La Vida (alternative)
# ])

df = pd.read_csv("../data/5000_songs.csv")

# Create DataFrame
# df = pd.DataFrame(data, columns=['danceability', 'energy', 'acoustiness', 'tempo', 'valence'])
# df['song'] = songs

print("=== Moosic Dataset (10 Songs) ===")
print(df)
print(f"\nShape: {df.shape}")
print(f"Features: {df.columns.tolist()}")

# Step 2: Scale the data
# Select features for clustering
df.columns = df.columns.str.strip()
features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence']
X = df[features]

# Scale the data (this is crucial!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== Scaled Data ===")
print(pd.DataFrame(X_scaled, columns=features).round(3))


# Step 3: K-Means Clustering
# Try different numbers of clusters
k_values = range(2, 100)  # 2 to 99 clusters
inertias = []
silhouettes = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouettes.append(sil_score)
    
    print(f"\n=== K = {k} ===")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {sil_score:.3f}")
    
    # Show which songs went to which cluster
    df[f'cluster_{k}'] = kmeans.labels_
    print("\nCluster assignments:")
    for cluster in range(k):
        songs_in_cluster = df[df[f'cluster_{k}'] == cluster]['name'].tolist()
        print(f"  Cluster {cluster}: {songs_in_cluster}")


# Step 4: Visualize the elbow curve
# Plot inertia (elbow method)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_values, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

# Plot silhouette scores
axes[1].plot(k_values, silhouettes, 'ro-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (higher = better)')
axes[1].grid(True)

plt.tight_layout()
plt.show()

#Step 5: Radar charts 
# Pick K=3 (or whatever looked best from the plots)
best_k = 8

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate cluster centroids (average features per cluster)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features)
centroid_df['cluster'] = range(best_k)

print("\n=== Cluster Centroids (original scale) ===")
print(centroid_df.round(3))

# Radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Prepare data for radar chart
categories = features
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Plot each cluster
for i in range(best_k):
    values = centroid_df.iloc[i][features].tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)

# Format the radar chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)  # All features are 0-1 after scaling
ax.set_title(f'Cluster Profiles (K={best_k})', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.show()

# Show final cluster assignments with song names
print("\n=== Final Playlists (Clusters) ===")
for cluster in range(best_k):
    songs_in_cluster = df[df['cluster'] == cluster]['name'].tolist()
    print(f"\n🎵 Playlist {cluster + 1} ({len(songs_in_cluster)} songs):")
    for song in songs_in_cluster:
        print(f"  - {song}")