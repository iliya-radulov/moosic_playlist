import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("../data/songs.csv").head(10)
import seaborn as sns
import matplotlib.pyplot as plt

 

for col in df.columns:
    col_lower = col.lower().strip()
    if 'song' in col_lower or 'track' in col_lower or 'title' in col_lower:
        song_col = col
    if 'artist' in col_lower:
        artist_col = col

# If we found them, use them
if song_col and artist_col:
    print(f"\n✅ Found: song='{song_col}', artist='{artist_col}'")
    print(f"\n=== First 10 Songs ===")
    print(df[[song_col, artist_col]].head(10))
else:
    print("\n⚠️ Could not find song_name and artist columns")
    print("Using index as song identifier instead...")
    df['song_identifier'] = df.index.astype(str) + ": " + df.iloc[:, 0].astype(str)

 
available_features = [f for f in features if f in df.columns]
print(f"\n✅ Available features: {available_features}")

if not available_features:
    print("❌ No audio features found! Using all numeric columns instead...")
    # Use all numeric columns except duration_ms and time_signature if they exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [c for c in numeric_cols if c not in ['duration_ms', 'time_signature', 'key', 'mode']]

print(f"\nUsing features: {available_features}")

# Prepare data
X = df[available_features]
Z = df[features]  # For correlation matrix later
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config
import numpy as np
import pandas as pd
import seaborn as sns

set_config(transform_output="pandas")

features = ['danceability', 'energy', 'acousticness', 'tempo', 'valence',
            'speechiness', 'instrumentalness']

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

pca = PCA()
pca.fit(scaled)

explained = pca.explained_variance_ratio_
print(pd.DataFrame(explained, columns=["Variance explained"]).round(3))

# elbow plot, same convention as your PCA notebook
sns.relplot(
    kind='line',
    x=range(len(explained)),
    y=explained,
    marker='o',
    aspect=1.3
).set(title="Proportion of variance explained by each principal component") \
 .set_axis_labels("Principal component number", "Proportion of variance")

# cumulative view — same as the notebook's second method
cumulative = np.cumsum(explained)
print(f"\nComponents needed for 95% variance: {np.searchsorted(cumulative, 0.95) + 1}")