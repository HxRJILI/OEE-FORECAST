OEE Insights 2: Statistical Analysis and ARIMA Modeling
=======================================================

.. note::
   **Notebook**: ``OEE_Insights_2.ipynb``
   
   **Prerequisites**: Completion of :doc:`oee_insights_1`
   
   **Purpose**: Statistical time series analysis and ARIMA-based forecasting

This notebook builds upon the preprocessed data from OEE_Insights_1 to conduct comprehensive statistical analysis and implement ARIMA (AutoRegressive Integrated Moving Average) models for OEE forecasting.

 **Overview**
===============

The second notebook focuses on:

- **Stationarity Analysis**: Testing time series properties using ADF and KPSS tests
- **Statistical Modeling**: ARIMA model identification and parameter estimation
- **Model Validation**: Walk-forward validation for realistic performance assessment
- **Forecasting**: Multi-step ahead predictions with confidence intervals
- **Comparative Analysis**: Performance evaluation across production lines

 **Objectives**
================

1. **Time Series Characterization**: Identify statistical properties of OEE data
2. **Model Selection**: Determine optimal ARIMA parameters for each production line
3. **Forecasting Framework**: Establish baseline statistical forecasting capability
4. **Performance Benchmarking**: Create statistical baselines for deep learning comparison

 **Statistical Framework**
===========================

**Time Series Analysis Components:**

1. **Stationarity Testing**
   - Augmented Dickey-Fuller (ADF) test
   - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
   - Visual inspection with ACF/PACF plots

2. **Model Identification**
   - Autocorrelation Function (ACF) analysis
   - Partial Autocorrelation Function (PACF) analysis
   - Information criteria (AIC, BIC) optimization

3. **Parameter Estimation**
   - Maximum Likelihood Estimation (MLE)
   - Automated model selection with pmdarima
   - Cross-validation for model robustness

 **Stationarity Analysis**
============================

**Testing Methodology:**

For each production line, the notebook conducts comprehensive stationarity tests:

.. code-block:: python

   from statsmodels.tsa.stattools import adfuller, kpss
   
   def analyze_stationarity(oee_data, line_name):
       """Comprehensive stationarity analysis"""
       
       # ADF Test (Null hypothesis: series has unit root - non-stationary)
       result_adf = adfuller(oee_data)
       
       # KPSS Test (Null hypothesis: series is stationary)
       result_kpss = kpss(oee_data, regression='c', nlags='auto')
       
       return result_adf, result_kpss

**Results by Production Line:**

**LINE-01 Analysis:**

.. code-block::

   ADF Test Results:
   - ADF Statistic: -2.145
   - p-value: 0.230
   - Critical Values: {'1%': -3.689, '5%': -2.971, '10%': -2.625}
   
   KPSS Test Results:
   - KPSS Statistic: 0.485
   - p-value: 0.073
   - Critical Values: {'1%': 0.739, '5%': 0.463, '10%': 0.347}

**Interpretation**: Both tests indicate non-stationarity, requiring differencing.

**LINE-03 Analysis:**

.. code-block::

   ADF Test Results:
   - ADF Statistic: -4.512
   - p-value: 0.0002
   - Critical Values: {'1%': -3.689, '5%': -2.971, '10%': -2.625}
   
   KPSS Test Results:
   - KPSS Statistic: 0.285
   - p-value: 0.216
   - Critical Values: {'1%': 0.739, '5%': 0.463, '10%': 0.347}

**Interpretation**: Both tests agree the series is stationary, suitable for ARMA modeling.

**LINE-04 Analysis:**

.. code-block::

   ADF Test Results:
   - ADF Statistic: -3.845
   - p-value: 0.003
   - Critical Values: {'1%': -3.689, '5%': -2.971, '10%': -2.625}
   
   KPSS Test Results:
   - KPSS Statistic: 0.312
   - p-value: 0.184
   - Critical Values: {'1%': 0.739, '5%': 0.463, '10%': 0.347}

**Interpretation**: Stationary series, suitable for ARMA modeling.

**LINE-06 Analysis:**

.. code-block::

   ADF Test Results:
   - ADF Statistic: -3.956
   - p-value: 0.002
   - Critical Values: {'1%': -3.689, '5%': -2.971, '10%': -2.625}
   
   KPSS Test Results:
   - KPSS Statistic: 0.298
   - p-value: 0.195
   - Critical Values: {'1%': 0.739, '5%': -2.971, '10%': -2.625}

**Interpretation**: Stationary series, suitable for ARMA modeling.

**Overall Daily OEE:**

.. code-block::

   ADF Test Results:
   - ADF Statistic: -3.124
   - p-value: 0.025
   - Critical Values: {'1%': -3.689, '5%': -2.971, '10%': -2.625}
   
   KPSS Test Results:
   - KPSS Statistic: 0.398
   - p-value: 0.121
   - Critical Values: {'1%': 0.739, '5%': 0.463, '10%': 0.347}

**Interpretation**: Stationary series, suitable for ARMA modeling.

 **ACF/PACF Analysis**
========================

**Pattern Recognition for Model Selection:**

The notebook generates ACF and PACF plots to identify appropriate ARIMA parameters:

.. code-block:: python

   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   
   def plot_acf_pacf(oee_data, line_name, max_lags=20):
       """Generate ACF and PACF plots for model identification"""
       
       fig, axes = plt.subplots(1, 2, figsize=(15, 5))
       
       plot_acf(oee_data, lags=max_lags, ax=axes[0])
       axes[0].set_title(f'Autocorrelation (ACF) for {line_name}')
       
       plot_pacf(oee_data, lags=max_lags, ax=axes[1], method='ols')
       axes[1].set_title(f'Partial Autocorrelation (PACF) for {line_name}')

**Model Identification Patterns:**

**LINE-01 (After Differencing):**
   - **ACF**: Sharp cutoff after lag 1
   - **PACF**: Gradual decay
   - **Suggested Model**: ARIMA(0,1,1) or ARIMA(0,1,2)

**LINE-03:**
   - **ACF**: Exponential decay with some oscillation
   - **PACF**: Cutoff after lag 1 with potential lag 2 significance
   - **Suggested Model**: ARIMA(1,0,1) or ARIMA(2,0,1)

**LINE-04:**
   - **ACF**: Gradual decay
   - **PACF**: Sharp cutoff after lag 2
   - **Suggested Model**: ARIMA(2,0,0) - Pure AR(2)

**LINE-06:**
   - **ACF**: Gradual decay
   - **PACF**: Cutoff after lag 1
   - **Suggested Model**: ARIMA(1,0,0) - Pure AR(1)

**Overall Daily OEE:**
   - **ACF**: Gradual decay pattern
   - **PACF**: Cutoff after lag 2-3
   - **Suggested Model**: ARIMA(2,0,0) or ARIMA(3,0,0)

 **Automated Model Selection**
===============================

**pmdarima Auto-ARIMA Implementation:**

.. code-block:: python

   import pmdarima as pm
   
   def find_optimal_arima(oee_data, line_name, seasonal=False):
       """Automated ARIMA model selection using pmdarima"""
       
       model = pm.auto_arima(
           oee_data,
           start_p=0, max_p=5,      # AR order search space
           start_q=0, max_q=5,      # MA order search space  
           d=None,                  # Let auto_arima determine d
           max_d=2,                 # Maximum differencing order
           seasonal=seasonal,        # Enable/disable seasonal components
           stepwise=True,           # Stepwise search for efficiency
           suppress_warnings=True,   # Clean output
           trace=True,              # Show search progress
           error_action='ignore'    # Handle problematic models gracefully
       )
       
       return model

**Selected Models by Production Line:**

.. list-table:: Optimal ARIMA Models
   :header-rows: 1
   :widths: 25 25 25 25

   * - Production Line
     - Selected Model
     - AIC Score
     - Interpretation
   * - LINE-01
     - ARIMA(0,1,2)
     - -125.43
     - Non-stationary with MA(2) component
   * - LINE-03  
     - ARIMA(1,0,1)
     - -98.76
     - Stationary with AR(1) and MA(1)
   * - LINE-04
     - ARIMA(2,0,0)
     - -87.92
     - Pure autoregressive AR(2) model
   * - LINE-06
     - ARIMA(1,0,0)
     - -102.15
     - Simple autoregressive AR(1) model
   * - Overall OEE
     - ARIMA(1,0,1)
     - -156.89
     - Mixed ARMA(1,1) model

 **Model Diagnostics**
========================

**Residual Analysis:**

For each fitted model, comprehensive diagnostic testing:

.. code-block:: python

   def model_diagnostics(fitted_model, line_name):
       """Comprehensive model diagnostic plots and tests"""
       
       # Generate diagnostic plots
       fitted_model.plot_diagnostics(figsize=(15, 10))
       plt.suptitle(f'Diagnostic Plots for {line_name}')
       plt.tight_layout()
       plt.show()
       
       # Ljung-Box test for residual autocorrelation
       residuals = fitted_model.resid()
       ljung_box_result = sm.stats.acorr_ljungbox(residuals, lags=10)
       
       return ljung_box_result

**Key Diagnostic Results:**

- **Residual Normality**: Most models show approximately normal residuals
- **Autocorrelation**: Residuals generally show no significant autocorrelation
- **Heteroscedasticity**: Some models show mild heteroscedasticity
- **Model Adequacy**: All selected models pass basic adequacy tests

 **Walk-Forward Validation**
=============================

**Methodology:**

Implements realistic validation using expanding window approach:

.. code-block:: python

   def walk_forward_validation(original_data, model_params, n_test_periods=30):
       """
       Walk-forward validation for ARIMA models
       
       Args:
           original_data: Full time series
           model_params: ARIMA order (p,d,q)
           n_test_periods: Number of periods for testing
       """
       
       train_data = original_data[:-n_test_periods]
       test_data = original_data[-n_test_periods:]
       
       history = list(train_data.copy())
       predictions = []
       actuals = []
       
       for t in range(len(test_data)):
           # Fit model on current history
           current_model = pm.ARIMA(order=model_params['order'],
                                   seasonal_order=model_params.get('seasonal_order', (0,0,0,0)))
           current_model.fit(history)
           
           # Forecast one step ahead
           yhat = current_model.predict(n_periods=1)[0]
           predictions.append(yhat)
           
           # Get actual value
           obs = test_data.iloc[t]
           actuals.append(obs)
           
           # Update history
           history.append(obs)
       
       return actuals, predictions

**Validation Results:**

.. list-table:: Walk-Forward Validation Performance
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Production Line
     - Model
     - MAE
     - RMSE  
     - MAPE (%)
   * - LINE-01
     - ARIMA(0,1,2)
     - 0.0847
     - 0.1203
     - 15.2%
   * - LINE-03
     - ARIMA(1,0,1)
     - 0.0523
     - 0.0697
     - 7.8%
   * - LINE-04
     - ARIMA(2,0,0)
     - 0.0634
     - 0.0889
     - 10.1%
   * - LINE-06
     - ARIMA(1,0,0)
     - 0.0456
     - 0.0612
     - 6.9%
   * - Overall OEE
     - ARIMA(1,0,1)
     - 0.0612
     - 0.0834
     - 9.3%

 **Forecasting Results**
==========================

**Multi-Step Forecasting:**

Each model generates forecasts with confidence intervals:

.. code-block:: python

   def generate_forecast(fitted_model, n_periods=7, alpha=0.05):
       """Generate point forecasts with confidence intervals"""
       
       forecast, conf_int = fitted_model.predict(
           n_periods=n_periods, 
           return_conf_int=True,
           alpha=alpha
       )
       
       return forecast, conf_int

**Forecast Performance by Line:**

**LINE-01 (ARIMA(0,1,2)):**
   - **7-day forecast accuracy**: 82.3%
   - **Trend detection**: Good at capturing directional changes
   - **Confidence intervals**: Wider due to higher volatility
   - **Best for**: Short-term operational planning

**LINE-03 (ARIMA(1,0,1)):**
   - **7-day forecast accuracy**: 89.1%
   - **Trend detection**: Excellent stability prediction
   - **Confidence intervals**: Narrow, high confidence
   - **Best for**: Maintenance scheduling, resource planning

**LINE-04 (ARIMA(2,0,0)):**
   - **7-day forecast accuracy**: 85.7%
   - **Trend detection**: Good momentum prediction
   - **Confidence intervals**: Moderate width
   - **Best for**: Production capacity planning

**LINE-06 (ARIMA(1,0,0)):**
   - **7-day forecast accuracy**: 91.2%
   - **Trend detection**: Excellent predictability
   - **Confidence intervals**: Very narrow
   - **Best for**: Performance benchmarking, target setting

**Overall OEE (ARIMA(1,0,1)):**
   - **7-day forecast accuracy**: 87.4%
   - **Trend detection**: Good aggregate trend capture
   - **Confidence intervals**: Moderate width
   - **Best for**: Strategic planning, corporate reporting

 **Visual Analysis Results**
=============================

**Forecast Visualization Example (LINE-06):**

.. code-block:: python

   # Generate and plot forecasts
   forecast, conf_int = model_line06.predict(n_periods=14, return_conf_int=True)
   
   plt.figure(figsize=(12, 6))
   
   # Historical data
   plt.plot(historical_dates, historical_values, label='Historical OEE', color='blue')
   
   # Forecasts
   plt.plot(forecast_dates, forecast, label='ARIMA Forecast', color='red', linestyle='--')
   
   # Confidence intervals
   plt.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], 
                   color='red', alpha=0.2, label='95% Confidence Interval')
   
   plt.title('LINE-06 OEE Forecast - ARIMA(1,0,0)')
   plt.xlabel('Date')
   plt.ylabel('OEE')
   plt.legend()
   plt.grid(True)
   plt.show()

 **Model Interpretation**
===========================

**Statistical Insights by Model Type:**

**ARIMA(0,1,2) - LINE-01:**
   - **Characteristic**: Non-stationary with strong moving average component
   - **Behavior**: Responds to recent shocks with gradual decay
   - **Implication**: Performance influenced by recent operational disruptions
   - **Management Focus**: Implement shock absorption strategies

**ARIMA(1,0,1) - LINE-03, Overall:**
   - **Characteristic**: Mixed autoregressive and moving average
   - **Behavior**: Balances trend following with shock adjustment
   - **Implication**: Stable baseline with responsive adjustment capability
   - **Management Focus**: Maintain current operational standards

**ARIMA(2,0,0) - LINE-04:**
   - **Characteristic**: Pure autoregressive with 2-period memory
   - **Behavior**: Strong momentum-based patterns
   - **Implication**: Performance highly dependent on recent history
   - **Management Focus**: Leverage momentum for sustained improvement

**ARIMA(1,0,0) - LINE-06:**
   - **Characteristic**: Simple autoregressive model
   - **Behavior**: Predictable mean-reverting tendencies  
   - **Implication**: Highly stable and predictable performance
   - **Management Focus**: Use as benchmark for other lines

 **Model Comparison Framework**
=================================

**Selection Criteria:**

1. **Statistical Fit**: AIC/BIC information criteria
2. **Predictive Accuracy**: Walk-forward validation metrics
3. **Interpretability**: Model complexity and parameter significance
4. **Robustness**: Performance across different time periods

**Ranking by Overall Performance:**

.. list-table:: Model Performance Ranking
   :header-rows: 1
   :widths: 15 20 15 15 15 20

   * - Rank
     - Production Line
     - Model
     - MAE
     - MAPE (%)
     - Key Strength
   * - 1
     - LINE-06
     - ARIMA(1,0,0)
     - 0.0456
     - 6.9%
     - Highest predictability
   * - 2
     - LINE-03
     - ARIMA(1,0,1)
     - 0.0523
     - 7.8%
     - Best balance of accuracy/stability
   * - 3
     - Overall OEE
     - ARIMA(1,0,1)
     - 0.0612
     - 9.3%
     - Good aggregate forecasting
   * - 4
     - LINE-04
     - ARIMA(2,0,0)
     - 0.0634
     - 10.1%
     - Strong momentum capture
   * - 5
     - LINE-01
     - ARIMA(0,1,2)
     - 0.0847
     - 15.2%
     - Handles non-stationarity well

 **Business Impact Analysis**
==============================

**Operational Implications:**

**High-Performance Lines (LINE-06, LINE-03):**
   - **Forecast Reliability**: 87-91% accuracy enables confident planning
   - **Resource Allocation**: Predictable patterns support optimal staffing
   - **Maintenance Scheduling**: Stable performance windows identified
   - **Benchmark Setting**: Use as performance targets for other lines

**Improvement Opportunity Lines (LINE-01, LINE-04):**
   - **Variability Management**: Higher forecast uncertainty requires buffer planning
   - **Root Cause Focus**: Non-stationary patterns indicate systemic issues
   - **Intervention Timing**: Model signals optimal timing for improvements
   - **Risk Mitigation**: Wider confidence intervals guide contingency planning

**Strategic Planning Applications:**

1. **Capacity Planning**: 7-14 day forecasts support production scheduling
2. **Quality Assurance**: Performance predictions enable proactive quality management
3. **Maintenance Optimization**: Trend analysis informs preventive maintenance timing
4. **Investment Decisions**: Model stability indicates equipment replacement priorities

 **Integration Pathway**
=========================

**Connection to Deep Learning (OEE_Insights_3):**

The ARIMA models serve as statistical baselines for evaluating deep learning performance:

- **Benchmark Establishment**: ARIMA results provide minimum acceptable accuracy
- **Feature Engineering**: Statistical patterns inform neural network architecture
- **Ensemble Opportunities**: Statistical and deep learning forecasts can be combined
- **Model Selection**: Comparative performance guides production deployment decisions

**Streamlit Integration:**

ARIMA models are integrated into the forecasting application:

- **Basic Forecasting**: Simple statistical methods for quick predictions
- **Baseline Comparison**: Deep learning models compared against ARIMA performance
- **Fallback Option**: Statistical models used when deep learning is unavailable
- **Ensemble Forecasting**: Combined predictions for improved accuracy

 **Technical Implementation Notes**
====================================

**Libraries and Versions:**

.. code-block:: python

   # Key dependencies
   import pmdarima as pm          # version 2.0.3
   import statsmodels as sm       # version 0.14.0
   import pandas as pd            # version 1.5.0
   import numpy as np             # version 1.24.3
   import matplotlib.pyplot as plt

**Performance Optimization:**

- **Parallel Processing**: Auto-ARIMA uses multiple cores for parameter search
- **Memory Management**: Large datasets processed in chunks for efficiency
- **Caching**: Model parameters cached to avoid recomputation
- **Vectorization**: NumPy operations used for forecast generation

**Error Handling:**

.. code-block:: python

   try:
       model = pm.auto_arima(data, **params)
   except Exception as e:
       # Fallback to simpler model
       model = pm.ARIMA(order=(1,0,1)).fit(data)
       warnings.warn(f"Auto-ARIMA failed, using fallback: {e}")

 **Limitations and Assumptions**
=================================

**Model Limitations:**

1. **Linear Relationships**: ARIMA assumes linear dependencies
2. **Gaussian Errors**: Assumes normally distributed residuals
3. **Constant Parameters**: Model parameters assumed stable over time
4. **Limited Seasonality**: Simple seasonal patterns only

**Data Assumptions:**

1. **Regular Intervals**: Daily observations assumed evenly spaced
2. **Missing Data**: Gaps in data handled by interpolation
3. **Outlier Sensitivity**: Extreme values can affect model performance
4. **Structural Breaks**: Major operational changes not automatically detected

**Forecast Limitations:**

1. **Forecast Horizon**: Accuracy degrades beyond 7-14 days
2. **Uncertainty Quantification**: Confidence intervals may be underestimated
3. **Regime Changes**: Models may not capture sudden operational shifts
4. **External Factors**: Economic, seasonal, or policy changes not modeled

 **Next Steps**
================

**Immediate Actions:**

1. **Model Monitoring**: Implement automated model performance tracking
2. **Parameter Updates**: Regular retraining as new data becomes available
3. **Ensemble Development**: Combine multiple model forecasts for robustness
4. **Business Integration**: Deploy forecasts into production planning systems

**Advanced Development:**

1. **Seasonal Analysis**: Extend models to capture weekly/monthly patterns
2. **External Variables**: Incorporate leading indicators (orders, maintenance schedules)
3. **Regime Detection**: Implement structural break detection algorithms
4. **Deep Learning Comparison**: Evaluate against neural network models in OEE_Insights_3

**Research Opportunities:**

1. **Multivariate Models**: Vector autoregression (VAR) across production lines
2. **State Space Models**: Kalman filtering for time-varying parameters
3. **Machine Learning Hybrid**: ARIMA-ML ensemble approaches
4. **Real-time Adaptation**: Online learning for parameter updating

**Continue to**: :doc:`oee_insights_3` for deep learning model implementation and comparison.