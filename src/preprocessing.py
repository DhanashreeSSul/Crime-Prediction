# Preprocessing script
import pandas as pd
import glob, os

# Define base path to navigate to the project root from src
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_raw_path = os.path.join(base_path, 'data', 'raw')
# Automatically find all CSV files, excluding the one processed separately
all_files = glob.glob(os.path.join(data_raw_path, '*.csv'))
crimes_on_women_data_path = os.path.join(data_raw_path, 'CrimesOnWomenData.csv')
files = [f for f in all_files if f != crimes_on_women_data_path]


frames = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    # normalize column names
    df = df.rename(columns=lambda c: c.strip())

    # Special handling for the wide-format file
    if 'crcCAW_r1.csv' in f:
        # This file is wide, needs to be melted
        id_vars = [c for c in df.columns if not c.isdigit()]
        value_vars = [c for c in df.columns if c.isdigit()]
        
        df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Count')
        
        # Rename columns to standard names
        state_col = next((c for c in df_long.columns if c.strip().lower() in ['state/ut', 'state/uts', 'states/uts']), None)
        crime_head_col = next((c for c in df_long.columns if c.strip().lower() in ['crime head']), None)
        
        rename_dict = {}
        if state_col:
            rename_dict[state_col] = "State"
        if crime_head_col:
            rename_dict[crime_head_col] = "Crime_Head"
            
        df_long = df_long.rename(columns=rename_dict)
        
        # Ensure year is integer
        df_long['Year'] = pd.to_numeric(df_long['Year'])
        
        # Select and append
        required_cols = ["State", "Crime_Head", "Year", "Count"]
        if all(col in df_long.columns for col in required_cols):
            frames.append(df_long[required_cols])
        else:
            print(f"Skipping melted file {f} due to missing columns. Found: {df_long.columns.tolist()}")
        continue # Move to next file

    # find the year column (it might be named '2013' etc)
    year_cols = [c for c in df.columns if c.isdigit()]
    if year_cols:
        year = year_cols[0]
        # handle both 'State/UTs' and 'States/UTs' and rename to 'State'
        state_col = next((c for c in df.columns if c.strip().lower() in ['state/uts', 'states/uts']), None)
        
        rename_dict = {year: "Count", "Crime Head": "Crime_Head"}
        if state_col:
            rename_dict[state_col] = "State"
            
        df_long = df.rename(columns=rename_dict)
        df_long["Year"] = int(year)
        
        # Ensure required columns exist before selecting
        required_cols = ["State", "Crime_Head", "Year", "Count"]
        if all(col in df_long.columns for col in required_cols):
            df_long = df_long[required_cols]
            frames.append(df_long)
        else:
            print(f"Skipping file {f} due to missing columns. Found: {df_long.columns.tolist()}")
    else:
        print("No year column found in", f)

combined_years = pd.concat(frames, ignore_index=True)
data_processed_path = os.path.join(base_path, 'data', 'processed')
os.makedirs(data_processed_path, exist_ok=True) # Ensure the directory exists
combined_years.to_csv(os.path.join(data_processed_path, "crc_combined_long.csv"), index=False)

df = pd.read_csv(crimes_on_women_data_path, low_memory=False)
df.info()
df.head()


df_long = df.melt(id_vars=['State','Year'], value_vars=['Rape','K&A','DD','AoW','AoM','DV','WT'],
                  var_name='Crime_Head', value_name='Count')
df_long.to_csv(os.path.join(data_processed_path, "crimes_on_women_long.csv"), index=False)


df['State'] = df['State'].str.strip().str.title()
df = df.drop_duplicates()
# The df from this point is not the same as combined_years, it's from CrimesOnWomenData.csv
# The following operations are on a different dataframe. Let's assume 'df' is what we want to process.
# Also, 'Crime_Head' is not in df's columns after reading CrimesOnWomenData.csv, it's created in df_long.
# Let's assume the user wants to process the `combined_years` dataframe from this point forward.
# To avoid confusion, let's rename combined_years to df.
df = combined_years
df['State'] = df['State'].str.strip().str.title()
df = df.drop_duplicates()

df['Count'] = df.groupby(['State','Crime_Head'])['Count'].transform(lambda x: x.fillna(x.median()))
q1 = df['Count'].quantile(0.25); q3 = df['Count'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['Count'] < q1 - 1.5*iqr) | (df['Count'] > q3 + 1.5*iqr)]

# Safely handle population data
try:
    pop_path = os.path.join(data_raw_path, "state_female_population_2011_2018.csv")
    pop = pd.read_csv(pop_path)
    df = df.merge(pop, on=['State','Year'], how='left')
    df['incidents_per_100k_females'] = df['Count'] / df['female_population'] * 100000
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}


    # Example: scale the incidents_per_100k_females
    X = df[['incidents_per_100k_females']].fillna(0)
    df['inc_minmax'] = scalers['minmax'].fit_transform(X)
    df['inc_std'] = scalers['standard'].fit_transform(X)
    df['inc_robust'] = scalers['robust'].fit_transform(X)

except FileNotFoundError:
    print("Population data file not found. Skipping population-based calculations.")


stats = df.groupby(['State','Year'])['Count'].agg(['mean','median','std','var']).reset_index()

