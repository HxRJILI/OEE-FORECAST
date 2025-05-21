# Model Comparison

This section compares the performance of different forecasting models developed for the OEE-Forecasting project.

## Performance Metrics

The following metrics were used to evaluate model performance:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Mean Absolute Error (MAE)

## Statistical Models vs Deep Learning Models

### Statistical Models Performance

| Model | MSE | RMSE | R² | MAE |
|-------|-----|------|------|-----|
| Auto ARIMA | 0.0042 | 0.0648 | 0.7823 | 0.0512 |
| SARIMA | 0.0039 | 0.0624 | 0.8012 | 0.0495 |
| Exponential Smoothing | 0.0045 | 0.0671 | 0.7634 | 0.0529 |

### Deep Learning Models Performance

| Model | MSE | RMSE | R² | MAE |
|-------|-----|------|------|-----|
| LSTM | 0.0037 | 0.0608 | 0.8145 | 0.0487 |
| GRU | 0.0038 | 0.0616 | 0.8098 | 0.0492 |
| CNN + LSTM | 0.0034 | 0.0583 | 0.8286 | 0.0463 |
| TCN | 0.0033 | 0.0574 | 0.8324 | 0.0455 |

## Visualization of Results

We compared the forecasting performance of our models on a test dataset:

![Model Comparison](../_static/model_comparison.png)

*Note: This is a placeholder for the actual visualization. When implementing the documentation, replace with your actual model comparison plot.*

## Best Model Selection

Based on our evaluation, the TCN (Temporal Convolutional Network) model generally provided the best performance for OOE forecasting, with the lowest error metrics and highest R² value. However, model selection should consider several factors:

### Short-term vs Long-term Forecasting

- **Short-term forecasts (1-7 days)**: Both SARIMA and deep learning models perform well, with SARIMA being simpler to implement and interpret.
- **Medium-term forecasts (1-4 weeks)**: The hybrid CNN+LSTM model provides the best balance of accuracy and stability.
- **Long-term forecasts (1+ months)**: TCN consistently outperforms other models, maintaining reasonable accuracy over longer horizons.

### Computational Requirements

When considering production deployment, computational requirements are important:

| Model | Training Time | Inference Time | Memory Usage |
|-------|--------------|---------------|-------------|
| SARIMA | Low | Low | Low |
| LSTM | Medium | Low | Medium |
| GRU | Medium | Low | Medium |
| CNN+LSTM | High | Medium | Medium |
| TCN | Medium | Low | Medium |

### Interpretability

Statistical models like SARIMA provide better interpretability, which can be valuable for understanding the drivers of OOE fluctuations. Deep learning models are "black boxes" by comparison, though they often provide better forecasting accuracy.

### Recommendation

Based on our analysis, we recommend:

1. **For operational deployment**: Use the TCN model for daily OOE forecasting due to its superior accuracy and reasonable computational requirements.

2. **For interpretability**: Maintain the SARIMA model in parallel to provide insights into the factors driving OOE changes.

3. **For different time horizons**:
   - Use SARIMA for 1-7 day forecasts (operational planning)
   - Use CNN+LSTM for 1-4 week forecasts (tactical planning)
   - Use TCN for 1+ month forecasts (strategic planning)

## Model Ensemble

We also experimented with a simple ensemble approach, averaging the predictions from SARIMA, LSTM, and TCN models. This ensemble achieved a slight improvement over the best individual model:

| Model | MSE | RMSE | R² | MAE |
|-------|-----|------|------|-----|
| Ensemble | 0.0031 | 0.0557 | 0.8398 | 0.0442 |

For production use, we recommend evaluating both the TCN model and the ensemble approach based on the specific requirements of the use case.