import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('Crime-Prediction/preprocessed_data.csv')

# -------------------------------------------------------------
# Association Rule Mining with Apriori Algorithm
# -------------------------------------------------------------

# Select categorical features for pattern discovery
categorical_cols = [
    'Main_Category',
    'Month',
    'Victim_Age_Group',
    'Victim_Caste_Category',
    'Marital_Status',
    'Victim_Occupation',
    'Victim_Education_Level',
    'States/UTs'
]

# Create binary transaction matrix (one-hot encoding)
df_transactions = pd.get_dummies(df[categorical_cols], prefix_sep='_')

# Convert to boolean for apriori algorithm
df_transactions = df_transactions.astype(bool)

print(f"Transaction matrix shape: {df_transactions.shape}")
print(f"Total transactions: {len(df_transactions)}")

# Apply Apriori to find frequent itemsets
# min_support: minimum proportion of transactions containing the itemset
frequent_itemsets = apriori(df_transactions, min_support=0.1, use_colnames=True, max_len=3)

print(f"\nFound {len(frequent_itemsets)} frequent itemsets")

if len(frequent_itemsets) > 0:
    # Generate association rules
    # metric options: 'support', 'confidence', 'lift', 'leverage', 'conviction'
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))
    
    # Add additional metrics
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    
    # Filter for meaningful rules (lift > 1 means positive correlation)
    rules = rules[rules['lift'] > 1.0]
    rules = rules.sort_values(by='lift', ascending=False)
    
    print(f"Generated {len(rules)} association rules with lift > 1.0\n")
    
    if len(rules) > 0:
        print("Top 15 Association Rules by Lift:")
        print("="*100)
        for idx, row in rules.head(15).iterrows():
            antecedent = ', '.join(list(row['antecedents']))
            consequent = ', '.join(list(row['consequents']))
            print(f"\nRule {idx + 1}:")
            print(f"  IF {antecedent}")
            print(f"  THEN {consequent}")
            print(f"  Support: {row['support']:.3f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Support vs Confidence scatter
        scatter1 = axes[0, 0].scatter(rules['support'], rules['confidence'], 
                                      c=rules['lift'], cmap='viridis', s=50, alpha=0.7)
        axes[0, 0].set_xlabel('Support', fontsize=9)
        axes[0, 0].set_ylabel('Confidence', fontsize=9)
        axes[0, 0].set_title('Association Rules: Support vs Confidence (colored by Lift)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('Lift', fontsize=9)
        
        # Plot 2: Top rules by lift
        top_rules = rules.head(10).copy()
        top_rules['rule_id'] = [f'R{i+1}' for i in range(len(top_rules))]
        axes[0, 1].barh(top_rules['rule_id'], top_rules['lift'], color='steelblue')
        axes[0, 1].set_xlabel('Lift', fontsize=9)
        axes[0, 1].set_ylabel('Rule ID', fontsize=9)
        axes[0, 1].set_title('Top 10 Rules by Lift', fontsize=11)
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Plot 3: Lift distribution
        axes[1, 0].hist(rules['lift'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(rules['lift'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {rules["lift"].mean():.2f}')
        axes[1, 0].set_xlabel('Lift', fontsize=9)
        axes[1, 0].set_ylabel('Frequency', fontsize=9)
        axes[1, 0].set_title('Distribution of Lift Values', fontsize=11)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Plot 4: Confidence vs Lift scatter
        scatter2 = axes[1, 1].scatter(rules['confidence'], rules['lift'], 
                                      c=rules['support'], cmap='plasma', s=50, alpha=0.7)
        axes[1, 1].set_xlabel('Confidence', fontsize=9)
        axes[1, 1].set_ylabel('Lift', fontsize=9)
        axes[1, 1].set_title('Association Rules: Confidence vs Lift (colored by Support)', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        cbar2 = plt.colorbar(scatter2, ax=axes[1, 1])
        cbar2.set_label('Support', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Export rules to CSV
        rules_export = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_export.to_csv('Crime-Prediction/association_rules.csv', index=False)
        print(f"\n✓ Association rules exported to Crime-Prediction/association_rules.csv")
    else:
        print("No rules found with lift > 1.0. Try lowering min_support or min_threshold.")
else:
    print("No frequent itemsets found. Try lowering min_support parameter.")
