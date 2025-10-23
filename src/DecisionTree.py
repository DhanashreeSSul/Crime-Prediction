import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# Decision Tree Classification to predict Main_Category
# -------------------------------------------------------------

# Features requested by user (excluding Main_Category which is the target)
feature_columns = [
    'States/UTs',
    'Count',
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=42)
dtree.fit(X_train, y_train)

# Predictions
y_pred = dtree.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': dtree.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\nTop 10 Most Important Features:")
print(feature_importance.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Decision Tree (simplified view)
plot_tree(
    dtree,
    feature_names=X_encoded.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    fontsize=7,
    max_depth=3,
    ax=axes[0]
)
axes[0].set_title("Decision Tree for Main_Category Prediction (depth limited to 3 for visualization)", fontsize=11)

# Plot 2: Feature Importance
axes[1].barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
axes[1].set_xlabel('Importance', fontsize=9)
axes[1].set_ylabel('Feature', fontsize=9)
axes[1].set_title('Top 10 Feature Importances', fontsize=11)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix: Main_Category Prediction', fontsize=11)
plt.xlabel('Predicted Main Category', fontsize=9)
plt.ylabel('Actual Main Category', fontsize=9)
plt.tight_layout()
plt.show()
