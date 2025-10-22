# -------------------------------------------------------------
# STEP 1: Import Libraries and Load the Dataset
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv("Crime-Prediction/final data.csv")  # use your cleaned file
print("Shape:", df.shape)
print(df.head())

# -------------------------------------------------------------
# STEP 2: Clean Data and Encode Categorical Columns
# -------------------------------------------------------------

# Drop duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna(subset=['Crime_Head', 'Main_Category', 'Count'])

# Encode categorical columns for numerical algorithms
label_cols = ['States/UTs', 'District', 'Crime_Head', 'Main_Category',
              'Month', 'Time_of_Day', 'Victim_Education_Level',
              'Victim_Occupation', 'Marital_Status']

le = LabelEncoder()
for col in label_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Normalize numerical columns for clustering
scaler = StandardScaler()
num_cols = ['Count', 'Year', 'Latitude', 'Longitude']
df[num_cols] = scaler.fit_transform(df[num_cols])

print("Data after encoding and scaling:")
print(df.head())

# At the end of preprocessing.py
df.to_csv('Crime-Prediction/preprocessed_data.csv', index=False)
print("✓ Preprocessed data saved to preprocessed_data.csv")