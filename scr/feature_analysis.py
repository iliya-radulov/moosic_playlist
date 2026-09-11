import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import set_config
import os

set_config(transform_output="pandas")
os.makedirs("../outputs", exist_ok=True)

# Load fresh — same source data, but this script doesn't care about
# clustering, splitting, or merging at all
df = pd.read_csv("../data/5000_songs.csv")
df.columns = df.columns.str.strip()

features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence',
            'speechiness', 'instrumentalness']

print(f"Dataset shape: {df.shape}")
print(f"Features: {features}")

# --- Correlation matrix ---
corr = df[features].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Feature correlation matrix (trimmed 7-feature set)")
plt.tight_layout()
plt.savefig("../outputs/correlation_matrix_7features.png", dpi=150, bbox_inches="tight")
plt.show()

# --- PCA ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

pca = PCA()
pca.fit(scaled)

explained = pca.explained_variance_ratio_
variance_df = pd.DataFrame(explained, columns=["Variance explained"])
print("\n=== Explained variance per component ===")
print(variance_df.round(3))
variance_df.to_csv("../outputs/pca_explained_variance.csv", index=False)

sns.relplot(
    kind='line', x=range(len(explained)), y=explained,
    marker='o', aspect=1.3
).set(title="Proportion of variance explained by each principal component") \
 .set_axis_labels("Principal component number", "Proportion of variance")
plt.savefig("../outputs/pca_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

cumulative = np.cumsum(explained)
n_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
print(f"\nComponents needed for 95% variance: {n_for_95}")