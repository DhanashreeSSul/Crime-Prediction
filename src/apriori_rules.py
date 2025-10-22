import pandas as pd     
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# STEP 7: Association Rule Mining with Apriori
# -------------------------------------------------------------

# Convert categorical columns into one-hot encoded format
categorical_cols = ['Crime_Head', 'Month', 'Time_of_Day', 
                    'Victim_Education_Level', 'Victim_Occupation', 
                    'Marital_Status', 'States/UTs']

df_apriori = pd.get_dummies(df[categorical_cols])

# Apply Apriori
frequent_itemsets = apriori(df_apriori, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Sort and display strongest rules
rules = rules.sort_values(by='lift', ascending=False)
print("Top 10 Association Rules:\n", rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
