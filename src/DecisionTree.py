import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')
# -------------------------------------------------------------
# STEP 6: Decision Tree Classification
# -------------------------------------------------------------

X = df[['Victim_Education_Level', 'Victim_Occupation', 'Marital_Status', 
        'Month', 'Time_of_Day', 'States/UTs']]
y = df['Crime_Head']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
dtree.fit(X_train, y_train)

# Predictions
y_pred = dtree.predict(X_test)

# Evaluation
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred))

# Visualize Tree
plt.figure(figsize=(18,10))
plot_tree(dtree, feature_names=X.columns, class_names=True, filled=True, fontsize=8)
plt.title("Decision Tree for Crime Classification")
plt.show()
