import pandas as pd
import numpy as np

dataset = pd.read_csv('datasets\netflix1.csv')
print(dataset)

missing_values = dataset.isnull().sum()
print(missing_values)

df_dropped_any_na = dataset.dropna()
print(df_dropped_any_na)

df_dropped_all_na = dataset.dropna(how='all')
print(df_dropped_all_na)

df_dropped_more_than_2_na = dataset.dropna(thresh=len(dataset.columns) - 2)
print(df_dropped_more_than_2_na)

# Assume the column name is 'column_name'
df_dropped_specific_na = dataset.dropna(subset=['listed_in'])
print(df_dropped_specific_na)
  

# df = pd.DataFrame(dataset)
#
# df_ignore = df.dropna()
# print("\nDataFrame after dropping rows with any missing values:")
# print(df_ignore)
#
# df_defaults = df.fillna(0)
# print("\nDataFrame after filling missing values with default value (0):")
# print(df_defaults)
#
#
# # Handling Duplicates
# duplicates = df.duplicated()
# print("\nDuplicate rows in DataFrame:")
# print(duplicates)
#
# df_no_duplicates = df.drop_duplicates()
# print("\nDataFrame after removing duplicate rows:")
# print(df_no_duplicates)