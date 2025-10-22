import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
df =  pd.read_csv('Crime-Prediction/preprocessed_data.csv')
# -------------------------------------------------------------
# STEP 4: Apply DBSCAN
# -------------------------------------------------------------

db_features = df[['Count', 'Latitude', 'Longitude', 'Month', 'Time_of_Day']]
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(db_features)

# Count clusters and outliers
unique_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'] else 0)
outliers = list(df['DBSCAN_Cluster']).count(-1)
print(f"DBSCAN found {unique_clusters} clusters and {outliers} outliers")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['DBSCAN_Cluster'], cmap='rainbow', s=50)
plt.title("DBSCAN Crime Density Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
