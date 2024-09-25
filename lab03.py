# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Load/Read and Display the Dataset (with Missing Values)
df = pd.read_csv('synthetic_dataset.csv')

# Display the dataset (first few rows)
print("Dataset Preview:")
print(df.head())

# Display missing values count per column
print("\nMissing values count per column:")
print(df.isnull().sum())

# 2. Identify Duplicates
duplicates = df.duplicated()
print(f"\nNumber of duplicate rows: {duplicates.sum()}")

# 3. Handle Data Redundancy (Drop the Records with Duplicates)
df = df.drop_duplicates()
print(f"\nShape after removing duplicates: {df.shape}")

# 4. Perform Correlation Analysis (Pearson's Ratio)
correlation_matrix = df.corr(method='pearson')
print("\nCorrelation matrix:")
print(correlation_matrix)

# 5. Display Heatmap of the Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Pearson Correlation')
plt.show()

# 6. Data Visualization (at least 4 different graphs)

# Histogram
df.hist(figsize=(10, 8))
plt.suptitle('Histogram of Features')
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('Boxplot of Features')
plt.show()

# Density Plot
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(12, 8))
plt.suptitle('Density Plot of Features')
plt.show()

# Pairplot
sns.pairplot(df)
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

# 7. Perform Min/Max Scaling
scaler = MinMaxScaler()
df_min_max_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("\nMin/Max Scaled Data Preview:")
print(df_min_max_scaled.head())

# 8. Perform ZScore Scaling
z_scaler = StandardScaler()
df_zscore_scaled = pd.DataFrame(z_scaler.fit_transform(df), columns=df.columns)
print("\nZ-Score Scaled Data Preview:")
print(df_zscore_scaled.head())

# 9. Perform Data Smoothing using Binning Method
# We'll bin one of the numeric columns, assuming 'col1' is a numeric feature
df['Binned_col1'] = pd.cut(df['col1'], bins=3, labels=["Low", "Medium", "High"])
print("\nBinned data for 'col1':")
print(df[['col1', 'Binned_col1']].head())

# 10. Perform Feature Selection (SelectKBest)
# Assume the last column is the target variable (classification)
X = df.drop('target_column', axis=1)  # Features
y = df['target_column']  # Target variable

# Select the top 5 features
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Display selected features
selected_features = X.columns[selector.get_support()]
print("\nSelected Features after Feature Selection:")
print(selected_features)

# Visualize the most significant features
plt.figure(figsize=(8, 6))
sns.barplot(x=selected_features, y=selector.scores_[selector.get_support()])
plt.title('Feature Importance (Top 5)')
plt.show()
