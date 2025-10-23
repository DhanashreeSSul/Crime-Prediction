import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# DBSCAN clustering on victim/context attributes requested by user
# -------------------------------------------------------------

feature_columns = [
	'States/UTs',
	'Count',
	'Year',
	'Main_Category',
	'Month',
	'Victim_Age',
	'Victim_Age_Group',
	'Victim_Caste_Category',
	'Marital_Status',
	'Victim_Occupation',
	'Victim_Education_Level',
]

dbscan_source = df[feature_columns].copy()

# One-hot encode categorical columns (object dtype) and scale for DBSCAN
categorical_cols = dbscan_source.select_dtypes(include='object').columns.tolist()
encoded_features = pd.get_dummies(dbscan_source, columns=categorical_cols, drop_first=False)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(encoded_features)

# Run DBSCAN (adjust eps/min_samples as needed for your data density)
dbscan = DBSCAN(eps=1.8, min_samples=10)
cluster_labels = dbscan.fit_predict(scaled_features)
df['DBSCAN_Cluster'] = cluster_labels

# Cluster diagnostics
unique_clusters = sorted(set(cluster_labels) - {-1})
noise_count = np.sum(cluster_labels == -1)
print(f"DBSCAN identified {len(unique_clusters)} clusters and {noise_count} noise points")

mask_core = cluster_labels != -1
if mask_core.sum() > len(unique_clusters) and len(unique_clusters) >= 2:
	sil_score = silhouette_score(scaled_features[mask_core], cluster_labels[mask_core])
	print(f"Silhouette Score (excluding noise): {sil_score:.3f}")
else:
	print("Silhouette Score: not enough core clusters to compute")

# -------------------------------------------------------------
# Helper encodings for plotting categorical axes with readable ticks
# -------------------------------------------------------------

def encode_for_axis(series):
	unique_vals = sorted(series.astype(str).unique())
	mapping = {val: idx for idx, val in enumerate(unique_vals)}
	codes = series.astype(str).map(mapping)
	return codes, mapping

plot_df = df[['Main_Category', 'Victim_Caste_Category',
			  'Victim_Age', 'DBSCAN_Cluster']].copy()

main_codes, main_labels = encode_for_axis(plot_df['Main_Category'])
caste_codes, caste_labels = encode_for_axis(plot_df['Victim_Caste_Category'])

plot_df['main_code'] = main_codes
plot_df['caste_code'] = caste_codes

# Color mapping for clusters (-1 reserved for noise)
unique_labels = sorted(np.unique(cluster_labels))
cmap = plt.cm.tab10
colors = [cmap(label % 10) if label != -1 else (0.5, 0.5, 0.5, 1.0) for label in cluster_labels]

# -------------------------------------------------------------
# Plot 1: Victim Caste vs Victim Age
# -------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(
	plot_df['caste_code'],
	plot_df['Victim_Age'],
	c=colors,
	s=40,
	alpha=0.7,
	edgecolor='black',
	linewidth=0.3
)

ax1.set_title('DBSCAN clusters: Victim Caste vs Victim Age', fontsize=11)
ax1.set_xlabel('Victim Caste Category', fontsize=9)
ax1.set_ylabel('Victim Age', fontsize=9)
ax1.set_xticks(list(caste_labels.values()))
ax1.set_xticklabels(list(caste_labels.keys()), rotation=45, fontsize=7, ha='right')
ax1.grid(True, alpha=0.3, linestyle='--')

# -------------------------------------------------------------
# Plot 2: Victim Caste vs Main Category
# -------------------------------------------------------------

ax2.scatter(
	plot_df['caste_code'],
	plot_df['main_code'],
	c=colors,
	s=40,
	alpha=0.7,
	edgecolor='black',
	linewidth=0.3
)

ax2.set_title('DBSCAN clusters: Victim Caste vs Main Category', fontsize=11)
ax2.set_xlabel('Victim Caste Category', fontsize=9)
ax2.set_ylabel('Main Category', fontsize=9)
ax2.set_xticks(list(caste_labels.values()))
ax2.set_xticklabels(list(caste_labels.keys()), rotation=45, fontsize=7, ha='right')
ax2.set_yticks(list(main_labels.values()))
ax2.set_yticklabels(list(main_labels.keys()), fontsize=7)
ax2.grid(True, alpha=0.3, linestyle='--')

# Legend for clusters
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}' if label != -1 else 'Noise',
						  markerfacecolor=(0.5, 0.5, 0.5, 1.0) if label == -1 else cmap(label % 10), markersize=8)
				   for label in unique_labels]
fig.legend(handles=legend_elements, loc='upper center', ncol=min(len(legend_elements), 6), frameon=True, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
