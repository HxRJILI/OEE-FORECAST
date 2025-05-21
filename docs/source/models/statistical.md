# Statistical Forecasting Models

This section describes the statistical time series forecasting models used in the OEE-Forecasting project, corresponding to the analysis in Notebook 2 (OEE-Insight_2).

## Overview

Statistical forecasting models leverage historical patterns in time series data to make predictions about future values. For the OOE data, we implemented several models using the pmdarima library, which automates much of the model selection and parameter tuning process.

## Models Implemented

### Auto ARIMA

Auto ARIMA automatically finds the optimal ARIMA (AutoRegressive Integrated Moving Average) model parameters by searching through combinations of p, d, and q values.

```python
import pmdarima as pm

# Fit Auto ARIMA model
auto_arima_model = pm.auto_arima(
    ooe_data['OOE'],
    start_p=0, start_q=0,
    max_p=5, max_q=5, max_d=2,
    seasonal=False,
    information_criterion='aic',
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# Display model summary
print(auto_arima_model.summary())

# Make predictions
forecast, conf_int = auto_arima_model.predict(n_periods=30, return_conf_int=True)
```

ARIMA models have three main components:
- **p (AR)**: The number of autoregressive terms
- **d (I)**: The number of differencing operations to make the series stationary
- **q (MA)**: The number of moving average terms

Auto ARIMA selects the best combination based on statistical criteria (like AIC or BIC).

### SARIMA (Seasonal ARIMA)

SARIMA extends ARIMA by incorporating seasonality. This is useful for OOE data that exhibits weekly patterns.

```python
# Fit SARIMA model
sarima_model = pm.auto_arima(
    ooe_data['OOE'],
    start_p=0, start_q=0,
    max_p=5, max_q=5, max_d=2,
    start_P=0, start_Q=0,
    max_P=2, max_Q=2, max_D=1,
    m=7,  # Weekly seasonality
    seasonal=True,
    information_criterion='aic',
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# Display model summary
print(sarima_model.summary())
```

SARIMA adds seasonal components to the model:
- **P**: Seasonal autoregressive order
- **D**: Seasonal differencing order
- **Q**: Seasonal moving average order
- **m**: The number of time steps for a single seasonal period (e.g., 7 for weekly data)

### Exponential Smoothing (ETS)

Exponential smoothing models weight recent observations more heavily than older observations. For the OOE data, we used the statsmodels implementation.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit ETS model
ets_model = ExponentialSmoothing(
    ooe_data['OOE'],
    trend='add',
    seasonal='add',
    seasonal_periods=7
).fit()

# Make predictions
ets_forecast = ets_model.forecast(steps=30)
```

The ETS model has several parameters:
- **Alpha**: Smoothing parameter for the level (0 < α < 1)
- **Beta**: Smoothing parameter for the trend (0 < β < 1)
- **Gamma**: Smoothing parameter for seasonality (0 < γ < 1)
- **Trend type**: Additive or multiplicative
- **Seasonal type**: Additive or multiplicative

## Parameter Optimization

The pmdarima package automates parameter selection, but we also performed manual grid searches for some models to fine-tune their performance:

```python
# Example of grid search for ETS parameters
best_params = None
best_aic = float('inf')

for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
            try:
                model = ExponentialSmoothing(
                    train_data,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=7
                ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                
                aic = model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_params = (alpha, beta, gamma)
            except:
                continue

print(f"Best parameters (alpha, beta, gamma): {best_params}")
```

## Model Evaluation

We evaluated the models using the following metrics:

- **Mean Squared Error (MSE)**: Average of squared differences between predictions and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, gives error in the same units as the data
- **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values
- **R-squared (R²)**: Proportion of variance in the dependent variable explained by the model

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calculate metrics
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predictions)
r2 = r2_score(test_data, predictions)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

## Results Visualization

We visualized the forecasting results using various plots:

1. **Time Series Plot**: Actual values vs. predictions
2. **Residual Plot**: Differences between actual and predicted values
3. **Forecast Plot**: Historical data with future predictions and confidence intervals
4. **ACF/PACF Plots**: To verify residual randomness (white noise)

```python
import matplotlib.pyplot as plt

# Example forecast plot
plt.figure(figsize=(12, 6))
plt.plot(ooe_data.index, ooe_data['OOE'], label='Historical Data')
plt.plot(forecast_dates, forecast, color='red', label='Forecast')
plt.fill_between(forecast_dates, 
                 conf_int[:, 0], conf_int[:, 1], 
                 color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title('OOE Forecast with SARIMA Model')
plt.xlabel('Date')
plt.ylabel('OOE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Conclusion

Statistical models provide a strong baseline for OOE forecasting, particularly for shorter forecast horizons. The SARIMA model generally performed best among statistical models due to its ability to capture the weekly seasonality present in the OOE data.

For detailed implementation and results, refer to Notebook 2 (OEE-Insight_2) in the project repository.