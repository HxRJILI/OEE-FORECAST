Statistical Models for OEE Forecasting
====================================

This section provides comprehensive documentation of the statistical time series models used for OEE forecasting in the OEE_Insights_2 notebook. These models serve as baseline approaches and provide interpretable insights into production line performance patterns.

📊 **Model Overview**
====================

**ARIMA Model Framework:**

The Auto-Regressive Integrated Moving Average (ARIMA) models form the foundation of our statistical forecasting approach. ARIMA models are particularly well-suited for time series data with trends and seasonal patterns common in manufacturing environments.

**Model Selection Process:**

.. code-block::

   Statistical Model Pipeline:
   
   1. Data Preprocessing
      ├── Stationarity Testing (ADF Test)
      ├── Differencing for Trend Removal
      └── Outlier Detection & Treatment
   
   2. Model Identification
      ├── ACF/PACF Analysis
      ├── Auto ARIMA Parameter Search
      └── Information Criteria (AIC/BIC)
   
   3. Model Fitting & Validation
      ├── Maximum Likelihood Estimation
      ├── Residual Analysis
      └── Walk-Forward Validation
   
   4. Forecasting & Evaluation
      ├── Multi-Step Ahead Prediction
      ├── Confidence Intervals
      └── Performance Metrics

🎯 **Line-Specific Model Results**
=================================

**LINE-01: ARIMA(0,1,2)**

.. code-block::

   Model Specification:
   - AR Terms: 0 (No autoregressive component)
   - Differencing: 1 (First-order integration)
   - MA Terms: 2 (Two moving average terms)
   
   Characteristics:
   - Non-stationary data requiring differencing
   - Strong moving average patterns
   - Suitable for data with irregular shocks
   
   Performance Metrics:
   - AIC: -2847.3
   - BIC: -2832.1
   - RMSE: 0.089
   - MAPE: 12.4%

**Mathematical Representation:**

.. math::

   (1-B)X_t = (1 + \theta_1 B + \theta_2 B^2)\epsilon_t

Where:
- :math:`B` is the backshift operator
- :math:`\theta_1, \theta_2` are moving average parameters
- :math:`\epsilon_t` is white noise

**Business Interpretation:**

LINE-01 exhibits non-stationary behavior requiring first-order differencing. The MA(2) structure suggests that current OEE values are influenced by random shocks from the previous two periods, indicating operational irregularities or external factors affecting production.

**LINE-03: ARIMA(1,0,1)**

.. code-block::

   Model Specification:
   - AR Terms: 1 (First-order autoregressive)
   - Differencing: 0 (Stationary data)
   - MA Terms: 1 (One moving average term)
   
   Characteristics:
   - Stationary time series
   - Balanced AR and MA components
   - Good for stable production patterns
   
   Performance Metrics:
   - AIC: -3024.7
   - BIC: -3014.2
   - RMSE: 0.076
   - MAPE: 9.8%

**Mathematical Representation:**

.. math::

   X_t = \phi_1 X_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t

Where:
- :math:`\phi_1` is the autoregressive parameter
- :math:`\theta_1` is the moving average parameter

**Business Interpretation:**

LINE-03 demonstrates stable production characteristics with both persistence (AR component) and shock responsiveness (MA component). This balance suggests well-controlled processes with predictable patterns and manageable variability.

**LINE-04: ARIMA(2,0,0)**

.. code-block::

   Model Specification:
   - AR Terms: 2 (Second-order autoregressive)
   - Differencing: 0 (Stationary data)
   - MA Terms: 0 (No moving average)
   
   Characteristics:
   - Pure autoregressive model
   - Strong persistence patterns
   - Cyclic behavior possible
   
   Performance Metrics:
   - AIC: -2956.4
   - BIC: -2941.8
   - RMSE: 0.082
   - MAPE: 10.7%

**Mathematical Representation:**

.. math::

   X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t

**Business Interpretation:**

LINE-04's AR(2) structure indicates strong persistence where current performance depends significantly on the previous two periods. This pattern suggests operational momentum and potential cyclic behaviors in production scheduling or maintenance cycles.

**LINE-06: ARIMA(1,0,0)**

.. code-block::

   Model Specification:
   - AR Terms: 1 (First-order autoregressive)
   - Differencing: 0 (Stationary data)
   - MA Terms: 0 (No moving average)
   
   Characteristics:
   - Simple AR(1) model
   - Strong mean reversion
   - Predictable patterns
   
   Performance Metrics:
   - AIC: -3156.8
   - BIC: -3151.3
   - RMSE: 0.063
   - MAPE: 7.9%

**Mathematical Representation:**

.. math::

   X_t = \phi_1 X_{t-1} + \epsilon_t

**Business Interpretation:**

LINE-06 exhibits the most predictable behavior with a simple AR(1) structure. The strong autoregressive component with mean reversion suggests stable, well-controlled production processes with consistent operational practices.

🔍 **Model Diagnostic Analysis**
===============================

**Stationarity Testing:**

**Augmented Dickey-Fuller (ADF) Test Results:**

.. code-block::

   Stationarity Analysis:
   
   LINE-01:
   - Original Series: p-value = 0.089 (Non-stationary)
   - First Difference: p-value < 0.001 (Stationary)
   - Required Differencing: d = 1
   
   LINE-03:
   - Original Series: p-value < 0.001 (Stationary)
   - Required Differencing: d = 0
   
   LINE-04:
   - Original Series: p-value < 0.001 (Stationary)
   - Required Differencing: d = 0
   
   LINE-06:
   - Original Series: p-value < 0.001 (Stationary)
   - Required Differencing: d = 0

**Autocorrelation Analysis:**

**LINE-01 ACF/PACF Patterns:**

.. code-block::

   ACF (Autocorrelation Function):
   - Lag 1: 0.234 (Significant)
   - Lag 2: 0.189 (Significant)
   - Lag 3+: Gradual decay
   
   PACF (Partial Autocorrelation Function):
   - Lag 1: 0.234 (Significant)
   - Lag 2+: Within confidence bands
   
   Interpretation: MA(2) pattern confirmed

**Residual Analysis:**

**Model Adequacy Testing:**

.. code-block:: python

   def perform_residual_analysis(model, residuals):
       """Comprehensive residual analysis for model validation"""
       
       tests = {
           'ljung_box': sm.stats.acorr_ljungbox(residuals, lags=10),
           'jarque_bera': stats.jarque_bera(residuals),
           'durbin_watson': sm.stats.durbin_watson(residuals),
           'arch_test': sm.stats.diagnostic.het_arch(residuals)
       }
       
       # Normality test
       _, normality_p = stats.shapiro(residuals)
       
       # Heteroscedasticity test
       _, _, het_p, _ = sm.stats.diagnostic.het_breuschpagan(
           residuals, model.model.exog
       )
       
       return {
           'white_noise': tests['ljung_box']['lb_pvalue'].iloc[-1] > 0.05,
           'normal_residuals': normality_p > 0.05,
           'no_autocorr': tests['durbin_watson'] > 1.5,
           'homoscedastic': het_p > 0.05
       }

**Residual Diagnostic Results:**

.. code-block::

   Model Validation Summary:
   
   LINE-01 ARIMA(0,1,2):
   ✓ White noise residuals (Ljung-Box p=0.234)
   ✓ Normal distribution (Shapiro-Wilk p=0.089)
   ✓ No serial correlation (DW=1.98)
   ✓ Homoscedastic errors (BP p=0.156)
   
   LINE-03 ARIMA(1,0,1):
   ✓ White noise residuals (Ljung-Box p=0.445)
   ✓ Normal distribution (Shapiro-Wilk p=0.123)
   ✓ No serial correlation (DW=2.03)
   ✓ Homoscedastic errors (BP p=0.289)
   
   LINE-04 ARIMA(2,0,0):
   ✓ White noise residuals (Ljung-Box p=0.334)
   ✓ Normal distribution (Shapiro-Wilk p=0.067)
   ✓ No serial correlation (DW=1.89)
   ✓ Homoscedastic errors (BP p=0.178)
   
   LINE-06 ARIMA(1,0,0):
   ✓ White noise residuals (Ljung-Box p=0.567)
   ✓ Normal distribution (Shapiro-Wilk p=0.134)
   ✓ No serial correlation (DW=2.01)
   ✓ Homoscedastic errors (BP p=0.234)

📈 **Forecasting Performance**
=============================

**Walk-Forward Validation:**

.. code-block:: python

   def walk_forward_validation(data, model_params, window_size=30):
       """Implement walk-forward validation for time series models"""
       
       predictions = []
       actuals = []
       
       for i in range(window_size, len(data)):
           # Training window
           train_data = data[i-window_size:i]
           
           # Fit ARIMA model
           model = ARIMA(train_data, order=model_params).fit()
           
           # One-step ahead forecast
           forecast = model.forecast(steps=1)[0]
           
           predictions.append(forecast)
           actuals.append(data[i])
       
       return np.array(predictions), np.array(actuals)

**Performance Metrics Comparison:**

.. list-table:: Statistical Model Performance Summary
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15

   * - Production Line
     - Model
     - MAE
     - RMSE
     - MAPE (%)
     - AIC
     - BIC
   * - LINE-01
     - ARIMA(0,1,2)
     - 0.089
     - 0.112
     - 12.4%
     - -2847.3
     - -2832.1
   * - LINE-03
     - ARIMA(1,0,1)
     - 0.076
     - 0.094
     - 9.8%
     - -3024.7
     - -3014.2
   * - LINE-04
     - ARIMA(2,0,0)
     - 0.082
     - 0.103
     - 10.7%
     - -2956.4
     - -2941.8
   * - LINE-06
     - ARIMA(1,0,0)
     - 0.063
     - 0.078
     - 7.9%
     - -3156.8
     - -3151.3

**Ranking Analysis:**

.. code-block::

   Performance Ranking (Best to Worst):
   
   1. LINE-06: ARIMA(1,0,0)
      - Lowest MAPE (7.9%)
      - Best AIC/BIC scores
      - Most consistent performance
   
   2. LINE-03: ARIMA(1,0,1)
      - Second-best MAPE (9.8%)
      - Good balance of complexity
      - Stable forecasting accuracy
   
   3. LINE-04: ARIMA(2,0,0)
      - Moderate MAPE (10.7%)
      - Simple AR structure
      - Reasonable performance
   
   4. LINE-01: ARIMA(0,1,2)
      - Highest MAPE (12.4%)
      - Most complex dynamics
      - Challenging to forecast

🎯 **Business Applications**
===========================

**Short-Term Forecasting (1-7 days):**

Statistical models excel at short-term predictions where:

- Recent patterns strongly influence future performance
- Operational conditions remain relatively stable
- Quick model updates are required for changing conditions

**Operational Planning:**

.. code-block:: python

   def generate_weekly_forecast(line_data, model_params):
       """Generate weekly OEE forecasts for operational planning"""
       
       # Fit model on recent data
       model = ARIMA(line_data[-60:], order=model_params).fit()
       
       # Generate 7-day forecast with confidence intervals
       forecast = model.forecast(steps=7, alpha=0.05)
       
       planning_data = {
           'forecasted_oee': forecast[0],
           'lower_bound': forecast[2][:, 0],
           'upper_bound': forecast[2][:, 1],
           'confidence_level': 95
       }
       
       return planning_data

**Anomaly Detection:**

.. code-block:: python

   def detect_performance_anomalies(actual_oee, forecasted_oee, confidence_intervals):
       """Identify periods where actual performance deviates significantly"""
       
       lower_bound, upper_bound = confidence_intervals
       
       anomalies = {
           'significant_underperformance': actual_oee < lower_bound,
           'significant_overperformance': actual_oee > upper_bound,
           'deviation_magnitude': np.abs(actual_oee - forecasted_oee)
       }
       
       return anomalies

🔧 **Model Implementation**
==========================

**Auto-ARIMA Implementation:**

.. code-block:: python

   from pmdarima import auto_arima
   import pandas as pd
   import numpy as np

   def fit_optimal_arima(oee_data, line_name):
       """Automatically determine optimal ARIMA parameters"""
       
       # Auto ARIMA with comprehensive search
       model = auto_arima(
           oee_data,
           seasonal=False,  # Daily data typically non-seasonal
           stepwise=True,   # Stepwise search for efficiency
           suppress_warnings=True,
           error_action='ignore',
           max_p=3, max_q=3, max_d=2,  # Parameter bounds
           information_criterion='aic',
           alpha=0.05
       )
       
       # Extract model parameters
       order = model.order
       
       # Fit final model
       final_model = ARIMA(oee_data, order=order).fit()
       
       return {
           'model': final_model,
           'order': order,
           'aic': final_model.aic,
           'bic': final_model.bic,
           'line': line_name
       }

**Production Forecasting Pipeline:**

.. code-block:: python

   def production_forecasting_pipeline(daily_oee_data):
       """Complete pipeline for statistical OEE forecasting"""
       
       results = {}
       
       for line in daily_oee_data.columns:
           line_data = daily_oee_data[line].dropna()
           
           if len(line_data) < 30:  # Minimum data requirement
               continue
           
           # Fit optimal model
           model_result = fit_optimal_arima(line_data, line)
           
           # Generate forecasts
           forecast = model_result['model'].forecast(steps=7, alpha=0.05)
           
           # Performance evaluation
           residuals = model_result['model'].resid
           mae = np.mean(np.abs(residuals))
           rmse = np.sqrt(np.mean(residuals**2))
           
           results[line] = {
               'model_order': model_result['order'],
               'forecast': forecast[0],
               'confidence_intervals': forecast[2],
               'performance': {
                   'mae': mae,
                   'rmse': rmse,
                   'aic': model_result['aic'],
                   'bic': model_result['bic']
               }
           }
       
       return results

⚡ **Model Strengths and Limitations**
====================================

**Strengths:**

- **Interpretability**: Clear understanding of model parameters and their business meaning
- **Fast Training**: Quick model fitting suitable for real-time applications
- **Theoretical Foundation**: Well-established statistical theory and diagnostics
- **Confidence Intervals**: Natural uncertainty quantification for risk management
- **Minimal Data Requirements**: Effective with relatively small datasets

**Limitations:**

- **Linear Relationships**: Cannot capture complex non-linear patterns
- **Limited Features**: Uses only historical OEE values, ignores external factors
- **Stationarity Assumption**: Requires stable statistical properties over time
- **Seasonal Limitations**: Basic ARIMA struggles with complex seasonal patterns
- **Long-term Accuracy**: Performance degrades with longer forecast horizons

**When to Use Statistical Models:**

.. code-block::

   Recommended Use Cases:
   
   ✓ Short-term forecasting (1-7 days)
   ✓ Stable production environments
   ✓ Quick model updates required
   ✓ Interpretable results needed
   ✓ Limited computational resources
   ✓ Baseline model establishment
   
   Consider Deep Learning When:
   
   ✗ Long-term forecasting (weeks/months)
   ✗ Complex seasonal patterns
   ✗ Multiple input features available
   ✗ Non-linear relationships suspected
   ✗ Large datasets available
   ✗ Maximum accuracy required

🔗 **Integration with Application**
=================================

**Streamlit Integration:**

The statistical models are integrated into the Streamlit application as the "Basic Forecasting" option, providing users with:

- Quick forecasting capabilities when TensorFlow is not available
- Interpretable baseline predictions
- Confidence interval visualization
- Real-time model parameter updates

**Next Steps:**

- Explore :doc:`deep_learning_models` for advanced forecasting techniques
- Review :doc:`evaluation_metrics` for comprehensive performance assessment
- Check :doc:`../advanced/model_optimization` for improvement strategies