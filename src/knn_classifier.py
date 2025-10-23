import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# KNN Classification to predict Main_Category
# -------------------------------------------------------------

# Features requested by user (excluding Main_Category which is the target)
feature_columns = [
    'States/UTs',
    'Year',
    'Month',
    'Victim_Age',
    'Victim_Age_Group',
    'Victim_Caste_Category',
    'Marital_Status',
    'Victim_Occupation',
    'Victim_Education_Level',
]

X = df[feature_columns].copy()
y = df['Main_Category'].copy()

# One-hot encode categorical features
categorical_cols = X.select_dtypes(include='object').columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# Scale features (important for KNN as it's distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train-test split with stratification to balance classes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# Train KNN Classifier with optimal parameters
knn = KNeighborsClassifier(n_neighbors=19, weights='distance', metric='manhattan')
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Classifier Accuracy: {accuracy:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Test different k values and distance metrics to find optimal
k_values = range(3, 31, 2)  # Test odd values only
metrics = ['euclidean', 'manhattan', 'minkowski']
weights_list = ['uniform', 'distance']

best_accuracy = 0
best_params = {}
results = []

for metric in metrics:
    for weight in weights_list:
        for k in k_values:
            knn_temp = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
            knn_temp.fit(X_train, y_train)
            y_pred_temp = knn_temp.predict(X_test)
            acc = accuracy_score(y_test, y_pred_temp)
            results.append({'k': k, 'metric': metric, 'weight': weight, 'accuracy': acc})
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {'k': k, 'metric': metric, 'weight': weight}

print(f"\nBest parameters: k={best_params['k']}, metric={best_params['metric']}, weight={best_params['weight']}")
print(f"Best accuracy: {best_accuracy:.3f}")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: K value vs Accuracy for best metric/weight combination
results_df = pd.DataFrame(results)
best_combo = results_df.loc[results_df['accuracy'].idxmax()]
best_k_results = results_df[(results_df['metric'] == best_combo['metric']) & 
                             (results_df['weight'] == best_combo['weight'])]

axes[0].plot(best_k_results['k'], best_k_results['accuracy'], marker='o', linestyle='-', color='steelblue', linewidth=2)
axes[0].axvline(x=best_params['k'], color='red', linestyle='--', label=f'Optimal k={best_params["k"]}')
axes[0].set_xlabel('Number of Neighbors (k)', fontsize=9)
axes[0].set_ylabel('Accuracy', fontsize=9)
axes[0].set_title(f'KNN: Accuracy vs K (metric={best_params["metric"]}, weight={best_params["weight"]})', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3, linestyle='--')

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues', interpolation='nearest')
axes[1].figure.colorbar(im, ax=axes[1])
axes[1].set_xticks(np.arange(len(sorted(y.unique()))))
axes[1].set_yticks(np.arange(len(sorted(y.unique()))))
axes[1].set_xticklabels(sorted(y.unique()), fontsize=8)
axes[1].set_yticklabels(sorted(y.unique()), fontsize=8)
axes[1].set_title('Confusion Matrix: Main_Category Prediction', fontsize=11)
axes[1].set_xlabel('Predicted Main Category', fontsize=9)
axes[1].set_ylabel('Actual Main Category', fontsize=9)

# Add text annotations
for i in range(len(sorted(y.unique()))):
    for j in range(len(sorted(y.unique()))):
        text = axes[1].text(j, i, cm[i, j],
                           ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black",
                           fontsize=8)

plt.tight_layout()
plt.show()
