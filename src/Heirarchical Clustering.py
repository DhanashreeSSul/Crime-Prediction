import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# Hierarchical clustering: Main category vs victim age profile
# -------------------------------------------------------------

# Build a main_category × victim_age_group frequency matrix
main_age_matrix = (
	df.groupby(['Main_Category', 'Victim_Age_Group'])
	  .size()
	  .unstack(fill_value=0)
)

# Add summary age statistics per main category
age_summary = df.groupby('Main_Category').agg(
	avg_victim_age=('Victim_Age', 'mean'),
	median_victim_age=('Victim_Age', 'median'),
	case_count=('Victim_Age', 'size'),
)

main_age_profile = main_age_matrix.join(age_summary)

# Scale features so clustering reflects relative age mix, not raw counts
scaler = StandardScaler()
main_scaled = scaler.fit_transform(main_age_profile)

# Agglomerative clustering (Ward linkage) on main categories
clusterer = AgglomerativeClustering(n_clusters=4, linkage='ward')
main_labels = clusterer.fit_predict(main_scaled)
main_age_profile['MainCategory_Cluster'] = main_labels

# Identify dominant victim age group per main category for context
dominant_age_group = main_age_matrix.idxmax(axis=1)
main_age_profile['Dominant_Age_Group'] = dominant_age_group

# Preview cluster membership (encoded ids) and dominant age group
print('Main category clusters (encoded ids):')
for cluster_id in sorted(main_age_profile['MainCategory_Cluster'].unique()):
	members = main_age_profile.index[main_age_profile['MainCategory_Cluster'] == cluster_id]
	preview = ', '.join(f"{int(m)}({main_age_profile.loc[m, 'Dominant_Age_Group']})" for m in members[:10])
	if len(members) > 10:
		preview = f"{preview}, ..."
	print(f"  Cluster {cluster_id} ({len(members)} main categories): {preview}")

# Silhouette score measures how distinctly clusters separate
silhouette = silhouette_score(main_scaled, main_labels)
print(f'Silhouette Score (Main_Category vs Victim_Age profile): {silhouette:.3f}')

# Dendrogram shows merge sequence and distance jumps
plt.figure(figsize=(10, 5))
main_Z = linkage(main_scaled, method='ward')
dendrogram(
	main_Z,
	labels=[str(int(idx)) for idx in main_age_profile.index],
	truncate_mode='lastp',
	p=10,
	leaf_rotation=45,
	leaf_font_size=9,
)
plt.title('Main category dendrogram (victim age profile)')
plt.xlabel('Encoded main category ids')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# PCA projection provides a 2D view of cluster separation
pca = PCA(n_components=2, random_state=42)
main_pca_coords = pca.fit_transform(main_scaled)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
	main_pca_coords[:, 0],
	main_pca_coords[:, 1],
	c=main_labels,
	cmap='tab10',
	s=80,
	alpha=0.85,
)
plt.title('Main category clusters by victim-age mix (PCA projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster id')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
