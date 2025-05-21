# Project Overview

## Introduction

The OEE-Forecasting project analyzes and forecasts daily Overall Operations Effectiveness (OOE) for Mitsui Morocco's production line using statistical models and deep learning techniques. The primary goal is to enhance decision-making and improve operational efficiency by accurately predicting future OOE values.

## What is OOE?

Overall Operations Effectiveness (OOE) is a key performance indicator that measures the efficiency of a manufacturing process. It is calculated using three components:

- **Availability**: The ratio of actual running time to planned production time
- **Performance**: The ratio of actual output to standard output capacity
- **Quality**: The ratio of good units to total units produced

The formula is:
```
OOE = Availability × Performance × Quality
```

OOE provides a comprehensive view of production efficiency, taking into account downtime, speed losses, and quality defects.

## Project Methodology

Our approach to forecasting OOE follows a structured methodology:

### 1. Exploratory Data Analysis (EDA)

- Analyzed trends, correlations, and patterns in production data
- Visualized OOE fluctuations over time
- Identified key factors influencing OOE values

### 2. Statistical Forecasting Models

To automate model selection and tuning, we used pmdarima, which optimized:

- Auto ARIMA
- SARIMA (Seasonal ARIMA)
- Exponential Smoothing (ETS)

### 3. Deep Learning Models

We experimented with various architectures to model temporal dependencies in OOE data:

- Recurrent models: LSTM, GRU, Vanilla RNN
- Hybrid models: CNN + LSTM, LSTM Autoencoder + Dense
- Advanced architectures: TCN (Temporal Convolutional Network)

### 4. Evaluation & Optimization

- Compared model performance using MSE, RMSE, and R²
- Applied hyperparameter tuning and optimization strategies
- Selected the best models for different forecasting horizons

## Project Structure

The project is organized into three main notebooks:

1. **OEE-Insight_1**: Data processing and exploratory data analysis
2. **OEE-Insight_2**: Statistical forecasting models
3. **OEE-Insight_3**: Deep learning models implementation

Each notebook builds upon the results of the previous one, creating a comprehensive forecasting pipeline.