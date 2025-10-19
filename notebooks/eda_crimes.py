# notebooks/eda_crimes.py  (or run cells in Jupyter)
# notebooks/eda_crimes.py  (or run cells in Jupyter)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Correctly define base path to be relative to the script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "crimes_cleaned.csv")

df = pd.read_csv(DATA_PATH)

# The data is in long format, so we analyze it differently.
# 'Crime_Head' contains the type of crime, and 'Count' has the value.

# 1. Basic stats on the 'Count' column
print(df['Count'].describe())

# 2. Top 10 states by total crimes (latest year)
latest_year = df['Year'].max()
latest_df = df[df['Year'] == latest_year]
top_states = latest_df.groupby('State')['Count'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 states by total crimes in", latest_year)
print(top_states)

# 3. Time series: national total per year
ts = df.groupby('Year')['Count'].sum().reset_index()
plt.figure(figsize=(8,4))
plt.plot(ts['Year'], ts['Count'], marker='o')
plt.title("National Total Crimes vs Year")
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.grid(True)
# Ensure the report/plots directory exists
os.makedirs(os.path.join(BASE_DIR, "report", "plots"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "report", "plots", "national_trend.png"), bbox_inches='tight')
plt.show()

# 4. Correlation heatmap between crime types (requires pivoting the data)

# To make the heatmap readable, let's select a smaller subset of major crime categories.
# You can customize this list to explore different crime types.
major_crimes = [
    'Rape',
    'Kidnapping & Abduction',
    'Dowry Deaths',
    'Assault on women with intent to outrage her modesty',
    'Insult to the modesty of Women',
    'Cruelty by Husband or his Relatives',
    'Importation of Girls from Foreign Country'
]

# Filter the dataframe to only include the major crimes
df_major_crimes = df[df['Crime_Head'].isin(major_crimes)]

# Pivot the filtered data
df_pivot = df_major_crimes.pivot_table(index=['Year', 'State'], columns='Crime_Head', values='Count', aggfunc='sum')
# Fill NaNs that may result from pivoting
df_pivot = df_pivot.fillna(0)

# Get the list of crime types from the pivoted columns
crime_cols = df_pivot.columns

corr = df_pivot[crime_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Between Major Crime Types")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout() # Adjust layout to make room for labels
plt.savefig(os.path.join(BASE_DIR, "report", "plots", "corr_crimes.png"), bbox_inches='tight')
plt.show()

