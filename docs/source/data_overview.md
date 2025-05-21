# Data Overview

This section provides detailed information about the datasets used in the OEE-Forecasting project for Mitsui Morocco's production line.

## Dataset Sources

The project uses two primary datasets:

1. **Production Line Data**: Contains metrics related to manufacturing operations
2. **Operational Data**: Contains details about machine performance and downtime

These datasets are processed in Notebook 1 (OEE-Insight_1) to calculate the Overall Operations Effectiveness (OOE) metrics.

## Data Structure

### Production Line Dataset

The production line dataset includes the following key fields:

| Field | Description | Data Type |
|-------|-------------|-----------|
| Timestamp | Date and time of the record | datetime |
| Line ID | Identifier for the production line | string |
| Planned Production Time | Scheduled runtime in minutes | float |
| Actual Production Time | Actual runtime in minutes | float |
| Units Produced | Number of units manufactured | integer |
| Good Units | Number of units passing quality control | integer |
| Standard Units per Hour | Expected production rate | float |

### Operational Data

The operational dataset contains:

| Field | Description | Data Type |
|-------|-------------|-----------|
| Timestamp | Date and time of the record | datetime |
| Machine ID | Identifier for the specific machine | string |
| Status Code | Operating status code | integer |
| Downtime Reason | Explanation for machine stoppage | string |
| Maintenance Type | Type of maintenance if applicable | string |
| Temperature | Operating temperature | float |
| Vibration | Vibration measurements | float |

## OOE Calculation

The OOE (Overall Operations Effectiveness) is calculated using three components:

1. **Availability**:
   ```
   Availability = Actual Production Time / Planned Production Time
   ```

2. **Performance**:
   ```
   Performance = (Units Produced / Actual Production Time) / Standard Units per Hour
   ```

3. **Quality**:
   ```
   Quality = Good Units / Units Produced
   ```

4. **OOE**:
   ```
   OOE = Availability × Performance × Quality
   ```

The calculated OOE data is saved as a time series dataset for forecasting in subsequent notebooks.

## Data Preprocessing

The raw datasets undergo several preprocessing steps in Notebook 1:

1. **Data Cleaning**:
   - Handling missing values
   - Removing outliers and anomalies
   - Standardizing date/time formats

2. **Feature Engineering**:
   - Calculating the OOE components (Availability, Performance, Quality)
   - Creating additional time-based features (day of week, month, etc.)
   - Deriving statistical features (rolling means, variances, etc.)

3. **Data Transformation**:
   - Aggregating data to daily intervals
   - Normalization/scaling for deep learning models
   - Creating lag features for time series models

## Final Dataset Structure

After preprocessing, the resulting OOE dataset used for forecasting has the following structure:

| Column | Description | Data Type |
|--------|-------------|-----------|
| Date | Date of the record | date |
| OOE | Overall Operations Effectiveness | float |
| Availability | Availability component | float |
| Performance | Performance component | float |
| Quality | Quality component | float |
| DayOfWeek | Day of the week (0-6) | integer |
| Month | Month (1-12) | integer |
| IsHoliday | Holiday flag (0/1) | boolean |
| MaintenanceDay | Scheduled maintenance flag (0/1) | boolean |
| OOE_lag1 | OOE value from previous day | float |
| OOE_lag2 | OOE value from 2 days ago | float |
| OOE_lag7 | OOE value from 7 days ago | float |
| OOE_rolling_mean_7 | 7-day rolling average of OOE | float |

## Dataset Statistics

Key statistics for the final OOE dataset:

- **Time Period**: Approximately 2 years of daily data
- **Number of Records**: ~730 daily records
- **Average OOE**: Typically ranges from 0.65 to 0.85
- **Seasonality**: Weekly patterns observed, with lower OOE on weekends
- **Trend**: Gradual improvement in OOE over time due to operational enhancements

## Data Access

The raw datasets are not provided in the repository due to confidentiality reasons. However, Notebook 1 includes code for generating synthetic datasets with similar statistical properties for testing and development purposes.

Users with access to the actual production data should place the CSV files in the `data/raw/` directory before running the notebooks.

## Next Steps

After understanding the data structure, proceed to [Notebook 1](notebooks/notebook1.html) to see how the data is processed and transformed into the OOE dataset used for forecasting.