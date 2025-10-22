import pandas as pd

# Load the datasets with their corresponding years
df_0 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet0.csv')
df_0['year'] = 2001
df_0.rename(columns={'2001': 'Count'}, inplace=True)
df_1 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet1.csv')
df_1['year'] = 2002
df_1.rename(columns={'2002': 'Count'}, inplace=True)

df_2 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet2.csv')
df_2['year'] = 2003
df_2.rename(columns={'2003': 'Count'}, inplace=True)

df_3 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet3.csv')
df_3['year'] = 2004
df_3.rename(columns={'2004': 'Count'}, inplace=True)

df_4 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet4.csv')
df_4['year'] = 2005
df_4.rename(columns={'2005': 'Count'}, inplace=True)

df_5 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet5.csv')
df_5['year'] = 2006
df_5.rename(columns={'2006': 'Count'}, inplace=True)

df_6 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet6.csv')
df_6['year'] = 2007
df_6.rename(columns={'2007': 'Count'}, inplace=True)

df_7 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet7.csv')
df_7['year'] = 2008
df_7.rename(columns={'2008': 'Count'}, inplace=True)

df_8 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet8.csv')
df_8['year'] = 2009
df_8.rename(columns={'2009': 'Count'}, inplace=True)

df_9 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet9.csv')
df_9['year'] = 2010
df_9.rename(columns={'2010': 'Count'}, inplace=True)

df_10 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet10.csv')
df_10['year'] = 2011
df_10.rename(columns={'2011': 'Count'}, inplace=True)

df_11 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet11.csv')
df_11['year'] = 2012
df_11.rename(columns={'2012': 'Count'}, inplace=True)

df_12 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet12.csv')
df_12['year'] = 2013
df_12.rename(columns={'2013': 'Count'}, inplace=True)

df_13 = pd.read_csv('Crime-Prediction/data/raw/crcCAW_r1 - Sheet13.csv')
df_13['year'] = 2014
df_13.rename(columns={'2014': 'Count'}, inplace=True)

# Merge all dataframes
merged_df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, 
                       df_8, df_9, df_10, df_11, df_12, df_13], 
                      ignore_index=True)

# Save the merged dataset
merged_df.to_csv('Crime-Prediction/merged_crime_data.csv', index=False)

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Years included: {sorted(merged_df['year'].unique())}")


# Simplifying all categories into common categories