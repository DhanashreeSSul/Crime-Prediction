import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df =  pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# STEP 3: Apply K-Means Clustering
# -------------------------------------------------------------

# Select relevant columns
kmeans_features = df[['Count', 'Year', 'Latitude', 'Longitude', 'Time_of_Day', 'Month', 'Crime_Head']]

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(kmeans_features)

# Evaluate using silhouette score
score = silhouette_score(kmeans_features, df['KMeans_Cluster'])
print(f"Silhouette Score (K-Means): {score:.3f}")

# Visualize clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['KMeans_Cluster'], cmap='viridis', s=50)
plt.title("K-Means Crime Clusters (Geographical View)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label='Cluster')
plt.show()
