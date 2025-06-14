# Notebook 1: Data Processing & EDA

This document explains the first notebook in the OEE-Forecasting project: `OEE-Insight_1.ipynb`. This notebook focuses on data processing, exploratory data analysis, and the calculation of OOE metrics.

## Overview

The purpose of this notebook is to:
1. Load and clean the raw production and operational datasets
2. Perform exploratory data analysis to understand patterns and relationships
3. Calculate the OOE (Overall Operations Effectiveness) metrics
4. Create and export the processed OOE dataset for forecasting

## Prerequisites

Before running this notebook, ensure you have:
- Installed all required dependencies (see [Installation Guide](../installation.html))
- Placed the raw data files in the `data/raw/` directory (or can use the synthetic data generator)

## Key Sections

### 1. Data Loading

```python
import pandas as pd
import glob
import os

# Load production data
production_data = pd.read_csv('data/raw/production_data.csv', parse_dates=['Timestamp'])

# Load operational data
operational_data = pd.read_csv('data/raw/operational_data.csv', parse_dates=['Timestamp'])

# Display basic information
print(f"Production data shape: {production_data.shape}")
print(f"Operational data shape: {operational_data.shape}")
```

This section loads the raw datasets and displays their basic information.

### 2. Data Cleaning

The notebook performs several data cleaning steps:
- Handling missing values using appropriate methods (imputation, removal)
- Removing outliers using statistical methods
- Correcting data types and formats
- Merging datasets based on common keys

Key code example:

```python
# Check for missing values
missing_values = production_data.isnull().sum()
print("Missing values in production data:")
print(missing_values)

# Handle missing values
production_data = production_data.fillna({
    'Planned Production Time': production_data['Planned Production Time'].mean(),
    'Units Produced': 0,
    'Good Units': 0
})

# Check for outliers using IQR method
Q1 = production_data['Units Produced'].quantile(0.25)
Q3 = production_data['Units Produced'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((production_data['Units Produced'] < lower_bound) | 
            (production_data['Units Produced'] > upper_bound))
print(f"Number of outliers in Units Produced: {outliers.sum()}")
```

### 3. Exploratory Data Analysis

This section includes various visualizations and statistical analyses to understand the data better:
- Time series plots of production metrics
- Correlation analysis between different variables
- Distribution of key performance indicators
- Seasonal patterns and trends

Example visualization:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
plt.figure(figsize=(14, 8))

# Plot production over time
plt.plot(production_data['Timestamp'], production_data['Units Produced'])
plt.title('Units Produced Over Time')
plt.xlabel('Date')
plt.ylabel('Units')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation heatmap
correlation_matrix = production_data.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Production Metrics')
plt.tight_layout()
plt.show()
```

### 4. OOE Calculation

This section calculates the three components of OOE and the final OOE metric:

```python
# Calculate Availability
availability = production_data['Actual Production Time'] / production_data['Planned Production Time']

# Calculate Performance
performance = ((production_data['Units Produced'] / production_data['Actual Production Time']) / 
               production_data['Standard Units per Hour'])

# Calculate Quality
quality = production_data['Good Units'] / production_data['Units Produced']
quality = quality.fillna(1)  # Handle division by zero

# Calculate OOE
ooe = availability * performance * quality

# Add to dataframe
production_data['Availability'] = availability
production_data['Performance'] = performance
production_data['Quality'] = quality
production_data['OOE'] = ooe
```

### 5. Feature Engineering

Additional features are created to enhance the forecasting models:

```python
# Add time-based features
production_data['Date'] = production_data['Timestamp'].dt.date
production_data['DayOfWeek'] = production_data['Timestamp'].dt.dayofweek
production_data['Month'] = production_data['Timestamp'].dt.month
production_data['Quarter'] = production_data['Timestamp'].dt.quarter

# Aggregate to daily level
daily_ooe = production_data.groupby('Date').agg({
    'OOE': 'mean',
    'Availability': 'mean',
    'Performance': 'mean',
    'Quality': 'mean',
    'DayOfWeek': 'first',
    'Month': 'first',
    'Quarter': 'first'
}).reset_index()

# Create lag features
for lag in [1, 2, 7]:
    daily_ooe[f'OOE_lag{lag}'] = daily_ooe['OOE'].shift(lag)

# Create rolling statistics
daily_ooe['OOE_rolling_mean_7'] = daily_ooe['OOE'].rolling(window=7).mean()
daily_ooe['OOE_rolling_std_7'] = daily_ooe['OOE'].rolling(window=7).std()
```

### 6. Data Export

The final processed dataset is exported for use in subsequent notebooks:

```python
# Drop rows with NaN values created by lags
daily_ooe = daily_ooe.dropna()

# Export to CSV
daily_ooe.to_csv('data/processed/daily_ooe.csv', index=False)
print(f"Exported processed OOE dataset with {daily_ooe.shape[0]} records.")
```

## Results and Insights

Key findings from the exploratory data analysis:
- Average OOE across the entire dataset is X%
- Identified a weekly seasonality pattern with lower OOE on weekends
- The Quality component is generally high (>95%), while Availability shows the most variation
- Strong correlation observed between maintenance events and drops in Availability
- Gradual improvement trend in overall OOE over the analyzed period

## Next Steps

After running this notebook, you will have a processed OOE dataset (`daily_ooe.csv`) that is ready for forecasting. Proceed to [Notebook 2](notebook2.html) to apply statistical forecasting models to this dataset.

## Troubleshooting

Common issues that might occur in this notebook:
- **File not found errors**: Ensure the raw data files are in the correct directory
- **Memory errors**: For very large datasets, consider sampling or chunking
- **Division by zero warnings**: These are handled in the OOE calculation section