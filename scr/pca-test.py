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