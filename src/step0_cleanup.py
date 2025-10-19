# src/step0_cleanup.py
import os, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Use the combined file which has all data in a consistent long format
IN = os.path.join(BASE_DIR, "data", "processed", "crc_combined_long.csv") 
OUT = os.path.join(BASE_DIR, "data", "processed", "crimes_cleaned.csv")

df = pd.read_csv(IN, low_memory=False)
# drop unnamed index column if exists
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
# standardize state names and year type
df['State'] = df['State'].str.strip().str.title()
df['Year'] = df['Year'].astype(int)

# The data is already in a long format, so we don't need to sum columns.
# The 'Total' can be calculated by grouping, but for a simple cleanup,
# we will just use the data as is. If a total is needed later, it can be
# calculated from the 'Count' column.

df.to_csv(OUT, index=False)
print("Saved cleaned file:", OUT)
