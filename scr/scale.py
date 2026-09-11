import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your actual dataset
df = pd.read_csv("songs.csv")

# Show what we're working with
print("=== Dataset Info ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# Select audio features for clustering (exclude song_name, artist)
# These are the numeric features Spotify provides
features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Check if all features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"\n⚠️ Missing features: {missing_features}")
    print("Using available features instead...")
    # Use only features that exist
    features = [f for f in features if f in df.columns]

# Extract the features
X = df[features]

print(f"\nUsing features: {features}")
print(f"Data shape: {X.shape}")

# Handle any missing values (if any)
if X.isnull().any().any():
    print("\n⚠️ Found missing values - filling with mean...")
    X = X.fillna(X.mean())

# Scale the data (important for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nScaled data shape: {X_scaled.shape}")
print(f"Mean of scaled features: {X_scaled.mean(axis=0).round(3)}")
print(f"Std of scaled features: {X_scaled.std(axis=0).round(3)}")

# Try different numbers of clusters
k_values = range(2, min(11, len(df) + 1))  # Don't try more clusters than songs
inertias = []
silhouettes = []

print("\n=== Testing different K values ===")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouettes.append(sil_score)
    
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

# Plot elbow curve
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
plt.show()

# Pick the best K (you can change this)
# Let's use silhouette score to pick the best K
best_k = k_values[np.argmax(silhouettes)]
print(f"\n🏆 Best K based on silhouette score: {best_k}")

# Run K-Means with the best K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Show the playlists
print(f"\n=== Playlists (K={best_k}) ===")
for cluster in range(best_k):
    cluster_songs = df[df['cluster'] == cluster]
    print(f"\n🎵 Playlist {cluster + 1} ({len(cluster_songs)} songs):")
    for idx, row in cluster_songs.iterrows():
        print(f"  - {row['song_name']} by {row['artist']}")

# Show cluster profiles (average features)
print("\n=== Cluster Profiles (average values) ===")
for cluster in range(best_k):
    cluster_data = df[df['cluster'] == cluster]
    print(f"\n🎵 Playlist {cluster + 1} ({len(cluster_data)} songs):")
    avg_features = cluster_data[features].mean()
    for feature in features:
        print(f"  {feature}: {avg_features[feature]:.3f}")

# Save the results
df.to_csv('songs_with_clusters.csv', index=False)
print(f"\n✅ Saved clustered data to 'songs_with_clusters.csv'")