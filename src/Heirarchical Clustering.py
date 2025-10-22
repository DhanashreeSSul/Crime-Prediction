import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')
# -------------------------------------------------------------
# STEP 4A: Agglomerative (Hierarchical) Clustering
# -------------------------------------------------------------

# Select clustering features
agg_features = df[['Count', 'Year', 'Latitude', 'Longitude', 'Month', 'Time_of_Day', 'Crime_Head']]

# Apply Agglomerative Clustering
# agg_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Agglomerative_Cluster'] = agg_cluster.fit_predict(agg_features)

# Evaluate with silhouette score
agg_score = silhouette_score(agg_features, df['Agglomerative_Cluster'])
print(f"Silhouette Score (Agglomerative): {agg_score:.3f}")

# Visualization: Dendrogram
plt.figure(figsize=(10, 5))
Z = linkage(agg_features, method='ward')
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Cluster visualization (Latitude vs Longitude)
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Agglomerative_Cluster'], cmap='tab10', s=50)
plt.title("Agglomerative Clustering on Crime Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
