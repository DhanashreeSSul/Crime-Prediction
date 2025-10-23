# DONE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# Features requested by user
feature_columns = [
    'Main_Category',
    'Month',
    'Victim_Age',
    'Victim_Age_Group',
    'Victim_Caste_Category',
    'Marital_Status',
    'Victim_Occupation',
    'Victim_Education_Level',
]

data = df[feature_columns].copy()

# Treat Victim_Age as numeric, other columns as categorical
categorical_cols = [
    'Main_Category',
    'Month',
    'Victim_Age_Group',
    'Victim_Caste_Category',
    'Marital_Status',
    'Victim_Occupation',
    'Victim_Education_Level',
]

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_encoded)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Evaluate cluster quality
silhouette = silhouette_score(scaled_features, cluster_labels)
print(f'Silhouette Score: {silhouette:.3f}')

# Examine average feature contribution per cluster
cluster_profile = data_encoded.copy()
cluster_profile['cluster'] = cluster_labels
cluster_summary = cluster_profile.groupby('cluster').mean()

top_attributes = {}
for cluster_id, row in cluster_summary.iterrows():
    top_cols = row.sort_values(ascending=False).head(5)
    top_attributes[cluster_id] = list(top_cols.index)
    print(f"Cluster {cluster_id} top attributes:")
    for col, value in top_cols.items():
        print(f"  {col}: {value:.2f}")

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(scaled_features)

plt.figure(figsize=(11, 8))
scatter = plt.scatter(
    pca_coords[:, 0],
    pca_coords[:, 1],
    c=cluster_labels,
    cmap='viridis',
    alpha=0.7,
    s=40,
    edgecolors='none'
)

# Annotate cluster centroids in PCA space
centers_2d = pca.transform(kmeans.cluster_centers_)
for cluster_id, (cx, cy) in enumerate(centers_2d):
    plt.scatter(cx, cy, marker='X', s=200, c='white', edgecolors='black', linewidths=1.2)
    label_lines = '\n'.join(top_attributes.get(cluster_id, []))
    plt.text(
        cx,
        cy,
        f'C{cluster_id}\n{label_lines}',
        fontsize=9,
        ha='center',
        va='center',
        color='black',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.4')
    )

plt.title('K-Means clusters: Main category vs victim demographics (PCA scatter)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster id')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()