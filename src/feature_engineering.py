# src/feature_engineering.py
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(__file__))
IN = os.path.join(BASE, "data", "processed", "crimes_on_women_long.csv")
OUT = os.path.join(BASE, "data", "processed", "crimes_features.csv")

# Load preprocessed long-format data
df = pd.read_csv(IN)
print("🔹 Loaded:", IN)
print(df.head())

# Clean column names
df.columns = df.columns.str.strip()

# Pivot to get each crime as separate column again
pivot_df = df.pivot_table(
    index=["State", "Year"],
    columns="Crime_Head",
    values="Count",
    aggfunc="sum",
    fill_value=0
).reset_index()

# Rename columns for clarity (replace special chars)
pivot_df.columns = [col.replace(" ", "_").replace("&", "and") for col in pivot_df.columns]

# Create total crimes column
crime_cols = [c for c in pivot_df.columns if c not in ["State", "Year"]]
pivot_df['Total'] = pivot_df[crime_cols].sum(axis=1)

# Compute each crime’s share
for c in crime_cols:
    pivot_df[f'share_{c}'] = pivot_df[c] / pivot_df['Total'].replace(0, np.nan)

# Sort and compute Year-over-Year growth per state
pivot_df = pivot_df.sort_values(['State', 'Year'])
pivot_df['prev_total'] = pivot_df.groupby('State')['Total'].shift(1)
pivot_df['yoy_growth'] = (
    (pivot_df['Total'] - pivot_df['prev_total'])
    / pivot_df['prev_total'].replace(0, np.nan)
).fillna(0)

# OPTIONAL — population merge if available
pop_path = os.path.join(BASE, "data", "raw", "state_female_population.csv")
if os.path.exists(pop_path):
    pop = pd.read_csv(pop_path)
    pivot_df = pivot_df.merge(pop, on=['State', 'Year'], how='left')
    pivot_df['inc_per_100k'] = pivot_df['Total'] / pivot_df['female_population'] * 100000
else:
    print("⚠️ Population data file not found — skipping per-100k rate calculation.")

# Save final engineered dataset
pivot_df.to_csv(OUT, index=False)
print(f"✅ Saved engineered dataset → {OUT}")
print("\nColumns:", pivot_df.columns.tolist())
