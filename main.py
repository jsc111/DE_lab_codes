import pandas as pd
import numpy as np

dataset = pd.read_csv('datasets\healthcare_dataset.csv')
print(dataset)

print("*****First 5 rows*****")
print(dataset.head(5), end="\n\n")

print("*****Last 5 rows*****")
print(dataset.tail(5), end="\n\n")

print(dataset.info())

print("*****Columns*****")
print(dataset.columns)

print("*****Describe*****")
print(dataset.describe())

print("*****DataSet*****")
print(dataset.dtypes)

print("*****Size*****")
print(dataset.size)

print("*****Shape*****")
print(dataset.shape)

print("*****Dimension*****")
print(dataset.ndim)

print("*****Check if null*****")
print(dataset.isnull())

print("I*****s null summ*****")
print(dataset.isnull().sum())

print("*****Checks null values loc*****")
print(dataset.isnull().values.any())

print("*****Checks null values in a particular column*****")
print(dataset['Blood Type'].isnull().values.any())

print("*****Prints given number of rows of a particular column*****")
print(dataset[0:5]['Gender'])

print("*****Reading Specific Columns*****")
print(dataset.loc[0:5, ['Gender', 'Age']])



print ( "fill the missing values.")
median = dataset["Age"].median()
print(median)
# dataset["Age"].fillna(median, inplace=True)
print(dataset.describe())
print("\n")
print(dataset)

print("fill the missing values.")
mean = dataset["Billing Amount"].mean()
print(mean)
# dataset["Billing Amount"].fillna(mean, inplace=True)
print(dataset.describe())
print("\n")
print(dataset)

