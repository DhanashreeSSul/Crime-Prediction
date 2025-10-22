import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix 
df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')
# -------------------------------------------------------------
# STEP 5: K-Nearest Neighbors Classification
# -------------------------------------------------------------

# Features and Target
X = df[['Month', 'Time_of_Day', 'Victim_Education_Level', 
        'Victim_Occupation', 'Marital_Status', 'States/UTs']]
y = df['Main_Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print("KNN Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
