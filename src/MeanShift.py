import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
df =  pd.read_csv('Crime-Prediction/preprocessed_data.csv')
# -------------------------------------------------------------
# STEP 4B: MeanShift Clustering
# -------------------------------------------------------------

ms_features = df[['Count', 'Latitude', 'Longitude', 'Month', 'Time_of_Day']]
ms = MeanShift()
df['MeanShift_Cluster'] = ms.fit_predict(ms_features)

unique_ms = df['MeanShift_Cluster'].nunique()
print(f"MeanShift found {unique_ms} clusters.")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['MeanShift_Cluster'], cmap='rainbow', s=50)
plt.title("MeanShift Clustering: Crime Density Hotspots")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
