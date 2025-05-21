# Project Overview

## Introduction

The OEE-Forecasting project analyzes and forecasts daily Overall Operations Effectiveness (OOE) for Mitsui Morocco's production line. By leveraging advanced statistical models and deep learning techniques, we aim to enhance decision-making and improve operational efficiency through accurate prediction of future OOE values.

This project provides manufacturing managers and engineers with powerful tools to anticipate production efficiency trends, enabling proactive maintenance scheduling and resource optimization.

## Understanding OEE

Overall Equipment Effectiveness (OEE) is the gold standard for measuring manufacturing productivity. It identifies the percentage of manufacturing time that is truly productive. An OEE score of 100% represents perfect production: manufacturing only good parts, as fast as possible, with no stop time.

OEE is calculated using three critical components:

### Availability (A)

Availability measures how much of the scheduled production time the equipment is actually running, taking into account both unplanned and planned stops.

```
Availability = Run Time / Planned Production Time
```

A score of 100% means the process is always running during planned production time with no stops.

### Performance (P)

Performance accounts for anything that causes the manufacturing process to run at less than its maximum possible speed, including slow cycles and small stops.

```
Performance = (Ideal Cycle Time × Total Count) / Run Time
```

A performance score of 100% indicates that when the process is running, it's operating at its theoretical maximum speed.

### Quality (Q)

Quality takes into account defects, including parts that require rework or are scrapped.

```
Quality = Good Count / Total Count
```

A quality score of 100% means there are no defects—only good parts are being produced.

### OEE Calculation

These three factors multiply together to determine the overall OEE:

```
OEE = Availability × Performance × Quality
```

For example, if:
- Availability = 85%
- Performance = 90%
- Quality = 98%

Then:
OEE = 0.85 × 0.90 × 0.98 = 0.7497 or approximately 75%

This means that only about 75% of the planned production time is truly productive.

## Project Methodology

Our approach to forecasting OOE follows a structured four-stage methodology:

### 1. Exploratory Data Analysis (EDA)

We begin with comprehensive data exploration to understand the underlying patterns:

- Temporal analysis of OEE trends, identifying weekly, monthly, and seasonal patterns
- Correlation analysis between OEE components and influencing factors
- Statistical tests for stationarity and seasonality
- Visualization of OEE fluctuations through time series plots, heatmaps, and distribution analysis
- Outlier detection and treatment to ensure model robustness

### 2. Statistical Forecasting Models

We implemented advanced time series forecasting techniques, using the pmdarima library to automate model selection and parameter tuning:

- **Auto ARIMA**: Automatically determines optimal parameters for ARIMA models
- **SARIMA (Seasonal ARIMA)**: Extends ARIMA to incorporate seasonal patterns prevalent in manufacturing data
- **Exponential Smoothing (ETS)**: Applies weighted averages with exponentially decreasing weights for older observations

These statistical models serve as our baseline forecasting approach, capturing linear relationships and seasonal patterns in the OEE data.

### 3. Deep Learning Models

To capture more complex non-linear relationships and temporal dependencies, we experimented with various neural network architectures:

- **Recurrent Neural Networks**:
  - LSTM (Long Short-Term Memory): Specialized for long-term dependency capture
  - GRU (Gated Recurrent Unit): Simplified LSTM variant with comparable performance
  - Vanilla RNN: Basic recurrent architecture for baseline comparison

- **Hybrid Models**:
  - CNN + LSTM: Convolutional layers for feature extraction followed by LSTM for temporal modeling
  - LSTM Autoencoder + Dense: Unsupervised feature learning combined with supervised prediction
  - Stacked RNN: Multiple recurrent layers for hierarchical feature extraction

- **Advanced Architectures**:
  - TCN (Temporal Convolutional Network): Dilated convolutions for efficient long-range dependency modeling
  - Dilated CNN: Expanded receptive fields to capture broader temporal patterns

### 4. Evaluation & Optimization

We rigorously evaluated all models using multiple performance metrics:

- **Mean Squared Error (MSE)**: Measures average squared difference between predictions and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, providing error measure in original units
- **R-squared (R²)**: Indicates proportion of variance in OEE explained by the model
- **Mean Absolute Percentage Error (MAPE)**: Accuracy measure as a percentage

Hyperparameter optimization was performed using:
- Grid search for statistical models
- Bayesian optimization for deep learning models
- Early stopping and learning rate scheduling to prevent overfitting

The best models were selected based on their forecasting accuracy across different time horizons, from short-term operational planning to long-term strategic decision-making.

## Project Structure

The project is organized into three comprehensive notebooks, creating a complete forecasting pipeline:

1. **OEE-Insight_1**: Data processing and exploratory data analysis
   - Data cleaning and preprocessing
   - OEE component calculation
   - Visualization and statistical analysis
   - Feature engineering and data preparation

2. **OEE-Insight_2**: Statistical forecasting models
   - Stationarity testing and time series decomposition
   - ARIMA, SARIMA, and ETS model implementation
   - Parameter optimization and model selection
   - Forecast visualization and error analysis

3. **OEE-Insight_3**: Deep learning models implementation
   - Sequence data preparation for neural networks
   - Implementation of RNN, LSTM, GRU, and hybrid architectures
   - Advanced model architectures including TCN and dilated CNN
   - Comparative analysis of model performance
   - Ensemble methods for improved forecasting accuracy

Each notebook builds upon the results of the previous one, providing a systematic approach to forecasting OEE that progresses from basic understanding and preparation to advanced modeling techniques.