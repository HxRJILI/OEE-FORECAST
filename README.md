# OOE-Forecasting Documentation

## Overview

This documentation guides you through the OOE-Forecasting project for Mitsui Morocco's production line. The project analyzes and forecasts daily Overall Operations Effectiveness (OOE) using statistical models and deep learning techniques to enhance decision-making and improve operational efficiency.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Data Overview](#data-overview)
3. [Notebook 1: OEE-Insight_1 - Data Processing & EDA](#notebook-1-oee-insight_1---data-processing--eda)
4. [Notebook 2: OEE-Insight_2 - Statistical Forecasting](#notebook-2-oee-insight_2---statistical-forecasting)
5. [Notebook 3: OEE-Insight_3 - Deep Learning Models](#notebook-3-oee-insight_3---deep-learning-models)
6. [Model Comparison & Results](#model-comparison--results)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)

## Project Setup

### Prerequisites

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - pmdarima
  - tensorflow/keras
  - statsmodels

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/OOE-Forecasting.git
   cd OOE-Forecasting
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv ooe-env
   source ooe-env/bin/activate  # On Windows: ooe-env\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Overview

The project utilizes two primary datasets from Mitsui Morocco's production line:
1. Production data - Contains metrics related to manufacturing operations
2. Operational data - Contains details about machine performance and downtime

These datasets are processed to calculate the Overall Operations Effectiveness (OOE), which is a key performance indicator for production efficiency.

## Notebook 1: OEE-Insight_1 - Data Processing & EDA

This notebook performs initial data exploration and transforms raw production data into OOE metrics.

### Running the Notebook

1. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

2. Navigate to and open `OEE-Insight_1.ipynb`.

3. Run all cells sequentially to:
   - Load and clean the datasets
   - Perform exploratory data analysis
   - Calculate OOE metrics
   - Save processed OOE dataset to your working directory

### Key Components

- **Data Loading**: Import raw production and operational datasets
- **Data Cleaning**: Handle missing values, outliers, and inconsistencies
- **Feature Engineering**: Calculate OOE from its components (Availability, Performance, Quality)
- **Exploratory Analysis**: Visualize trends, patterns, and correlations in production data
- **Data Export**: Save the calculated OOE dataset for use in subsequent notebooks

### Expected Output

After running this notebook, you should have:
- Visualizations of production data trends
- Correlation analyses between different operational metrics
- A processed OOE dataset saved in your working directory

## Notebook 2: OEE-Insight_2 - Statistical Forecasting

This notebook applies statistical time series forecasting models to the OOE data.

### Running the Notebook

1. Ensure you've successfully run Notebook 1 first.
2. Open and run `OEE-Insight_2.ipynb`.

### Key Components

- **Data Preparation**: Time series preprocessing (stationarity checks, differencing)
- **Model Selection**: Automated selection and optimization using pmdarima
- **Models Implemented**:
  - Auto ARIMA: Automatically determines optimal ARIMA parameters
  - SARIMA: Handles seasonality in the OOE data
  - Exponential Smoothing (ETS): Captures trends and seasonality

### Statistical Model Parameters

Each statistical model is optimized with parameters specific to the OOE time series:

- **ARIMA Parameters**:
  - p: AutoRegressive order
  - d: Differencing order
  - q: Moving Average order
  
- **SARIMA Additional Parameters**:
  - P: Seasonal AutoRegressive order
  - D: Seasonal Differencing order
  - Q: Seasonal Moving Average order
  - s: Seasonality period

- **ETS Parameters**:
  - Alpha: Smoothing parameter for level
  - Beta: Smoothing parameter for trend
  - Gamma: Smoothing parameter for seasonality
  - Trend type: Additive or multiplicative
  - Seasonal type: Additive or multiplicative

### Expected Output

After running this notebook, you should have:
- Time series decomposition plots
- Model diagnostics (ACF/PACF plots, residual analysis)
- Forecast plots for each statistical model
- Performance metrics (MSE, RMSE, R²) for comparison

## Notebook 3: OEE-Insight_3 - Deep Learning Models

This notebook implements various deep learning architectures to forecast OOE.

### Running the Notebook

1. Ensure you've successfully run Notebook 1 first.
2. Open and run `OEE-Insight_3.ipynb`.

### Key Components

- **Data Preparation**: Sequence creation, normalization, train-test splitting
- **Deep Learning Models**:
  - Recurrent Models: 
    - LSTM (Long Short-Term Memory)
    - GRU (Gated Recurrent Unit)
    - Vanilla RNN
  - Hybrid Models:
    - CNN + LSTM
    - LSTM Autoencoder + Dense layers
  - Advanced Architectures:
    - TCN (Temporal Convolutional Network)
- **Hyperparameter Tuning**: Optimization of model architectures

### Neural Network Architectures

Details of the implemented neural network structures:

- **LSTM Model**:
  ```
  LSTM Layer(units=64, return_sequences=True)
  Dropout(0.2)
  LSTM Layer(units=32)
  Dense Layer(units=1)
  ```

- **GRU Model**:
  ```
  GRU Layer(units=64, return_sequences=True)
  Dropout(0.2)
  GRU Layer(units=32)
  Dense Layer(units=1)
  ```

- **CNN+LSTM Model**:
  ```
  Conv1D Layer(filters=64, kernel_size=2)
  MaxPooling1D(pool_size=2)
  LSTM Layer(units=50)
  Dense Layer(units=1)
  ```

- **TCN Model**:
  ```
  TCN Layer(nb_filters=64, kernel_size=2, dilations=[1, 2, 4, 8])
  Dense Layer(units=1)
  ```

### Expected Output

After running this notebook, you should have:
- Training and validation loss curves
- Forecast plots for each deep learning model
- Performance metrics (MSE, RMSE, R²) for comparison

## Model Comparison & Results

The project compares various forecasting approaches:

### Statistical Models
- **Auto ARIMA**: Good for short-term forecasting with limited seasonality
- **SARIMA**: Effective when strong seasonal patterns exist in OOE data

### Deep Learning Models
- **LSTM/GRU**: Captures complex temporal dependencies and long-term patterns
- **Hybrid Models**: Combines strengths of different architectures
- **Stacked RNN**: Multiple recurrent layers on top of one layer.

### Selecting the Best Model

The optimal model depends on:
- Forecast horizon (short vs. long-term predictions)
- Pattern complexity in your OOE data
- Computational resources available
- Required prediction accuracy

Review the performance metrics and visualizations to determine which model best suits your specific forecasting needs.

## Troubleshooting

### Common Issues

1. **Missing OOE dataset after running Notebook 1**:
   - Ensure you have write permissions in the working directory
   - Check for error messages related to file operations
   - Verify that the calculation steps executed correctly

2. **Statistical Models Errors**:
   - Non-stationary data: Apply additional differencing
   - Convergence issues: Adjust model parameters or training settings
   - Install pmdarima: `pip install pmdarima`

3. **Deep Learning Model Errors**:
   - GPU/memory issues: Reduce batch size or model complexity
   - Overfitting: Increase dropout rate or add regularization
   - Unstable training: Adjust learning rate or use gradient clipping

### Getting Help

If you encounter issues not covered in this documentation:
1. Check the error messages carefully
2. Refer to library documentation (TensorFlow, pmdarima, etc.)
3. Create an issue in the project repository with details about your problem

## References

- OOE Calculation Methodology: [OEE Industry Standard](https://www.oee.com/)
- Statistical Forecasting: [pmdarima Documentation](https://alkaline-ml.com/pmdarima/)
- Deep Learning for Time Series: [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)