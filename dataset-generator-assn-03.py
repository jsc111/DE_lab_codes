# Import required libraries
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples and features
num_samples = 100
num_features = 5

# Generate random data
data = {
    'Feature1': np.random.randint(1, 100, num_samples),
    'Feature2': np.random.normal(loc=50, scale=15, size=num_samples),  # Normally distributed
    'Feature3': np.random.uniform(10, 60, num_samples),  # Uniformly distributed
    'Feature4': np.random.choice([1, 0], size=num_samples, p=[0.3, 0.7]),  # Binary categorical feature
    'Feature5': np.random.exponential(scale=10, size=num_samples),  # Exponentially distributed
}

# Create a DataFrame
df = pd.DataFrame(data)

# Introduce some missing values in Feature2 and Feature5
df.loc[np.random.choice(df.index, size=10, replace=False), 'Feature2'] = np.nan
df.loc[np.random.choice(df.index, size=5, replace=False), 'Feature5'] = np.nan

# Introduce some duplicate rows
df = pd.concat([df, df.iloc[:3]], ignore_index=True)

# Add a target column for classification (binary)
df['Target'] = np.random.choice([0, 1], size=len(df))

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to CSV (if needed)
df.to_csv('synthetic_dataset.csv', index=False)
