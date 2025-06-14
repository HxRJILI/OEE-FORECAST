Model Evaluation Metrics
========================

This section provides comprehensive documentation of the evaluation methodologies and metrics used to assess forecasting model performance in the OEE analytics system. Understanding these metrics is crucial for selecting appropriate models and interpreting prediction quality.

 **Evaluation Framework**
===========================

**Multi-Dimensional Assessment:**

.. code-block::

   Evaluation Framework:
   
   ├── Accuracy Metrics
   │   ├── Mean Absolute Error (MAE)
   │   ├── Root Mean Square Error (RMSE)
   │   ├── Mean Absolute Percentage Error (MAPE)
   │   └── Symmetric MAPE (sMAPE)
   │
   ├── Validation Strategies
   │   ├── Walk-Forward Validation
   │   ├── Time Series Cross-Validation
   │   └── Hold-Out Test Sets
   │
   ├── Business Impact Metrics
   │   ├── Forecast Bias Analysis
   │   ├── Directional Accuracy
   │   └── Confidence Interval Coverage
   │
   └── Model Comparison
       ├── Statistical Significance Tests
       ├── Relative Performance Rankings
       └── Robustness Analysis

**Evaluation Philosophy:**

Our evaluation approach prioritizes:

- **Business Relevance**: Metrics that matter for manufacturing operations
- **Temporal Robustness**: Performance consistency over time
- **Practical Applicability**: Real-world deployment considerations
- **Statistical Rigor**: Proper validation to prevent overfitting

 **Core Accuracy Metrics**
============================

**1. Mean Absolute Error (MAE)**

**Definition and Calculation:**

.. math::

   MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|

Where:
- :math:`y_i` = actual OEE value
- :math:`\hat{y}_i` = predicted OEE value
- :math:`n` = number of predictions

**Implementation:**

.. code-block:: python

   def calculate_mae(y_true, y_pred):
       """Calculate Mean Absolute Error"""
       return np.mean(np.abs(y_true - y_pred))

**Business Interpretation:**

MAE represents the average absolute deviation of predictions from actual values. For OEE forecasting:

.. code-block::

   MAE Interpretation for OEE:
   
   MAE = 0.05 → Average error of 5 percentage points
   MAE = 0.10 → Average error of 10 percentage points
   
   Business Impact:
   - MAE < 0.05: Excellent accuracy for operational planning
   - MAE 0.05-0.10: Good accuracy, suitable for strategic planning
   - MAE 0.10-0.15: Fair accuracy, useful for trend analysis
   - MAE > 0.15: Poor accuracy, model needs improvement

**Project Results:**

.. list-table:: MAE Performance by Model and Production Line
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Model
     - LINE-01
     - LINE-03
     - LINE-04
     - LINE-06
     - Average
   * - Multi-Kernel CNN
     - 0.0756
     - 0.0698
     - 0.0723
     - **0.0591**
     - 0.0692
   * - Stacked RNN + Masking
     - 0.0821
     - 0.0743
     - 0.0789
     - 0.0634
     - 0.0747
   * - WaveNet-Style CNN
     - 0.0734
     - 0.0712
     - 0.0701
     - 0.0645
     - 0.0698
   * - LSTM
     - 0.0798
     - 0.0721
     - 0.0756
     - 0.0667
     - 0.0736
   * - ARIMA (Best Statistical)
     - 0.0890
     - 0.0760
     - 0.0820
     - 0.0630
     - 0.0775

**2. Root Mean Square Error (RMSE)**

**Definition and Calculation:**

.. math::

   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

**Implementation:**

.. code-block:: python

   def calculate_rmse(y_true, y_pred):
       """Calculate Root Mean Square Error"""
       return np.sqrt(np.mean((y_true - y_pred)**2))

**Business Interpretation:**

RMSE penalizes larger errors more heavily than MAE, making it sensitive to outliers:

.. code-block::

   RMSE vs MAE Comparison:
   
   If RMSE >> MAE:
   - Large occasional errors (equipment failures, shutdowns)
   - Model struggles with extreme events
   - Need better outlier handling
   
   If RMSE ≈ MAE:
   - Consistent error distribution
   - Model performs uniformly
   - Good general-purpose forecasting

**3. Mean Absolute Percentage Error (MAPE)**

**Definition and Calculation:**

.. math::

   MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|

**Implementation:**

.. code-block:: python

   def calculate_mape(y_true, y_pred, epsilon=1e-8):
       """Calculate MAPE with protection against division by zero"""
       return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

**Business Interpretation:**

MAPE provides intuitive percentage-based error measurement:

.. code-block::

   MAPE Interpretation Guidelines:
   
   MAPE < 5%:    Excellent forecasting accuracy
   MAPE 5-10%:   Good accuracy for most business purposes
   MAPE 10-15%:  Reasonable accuracy for strategic planning
   MAPE 15-25%:  Fair accuracy, suitable for trend analysis
   MAPE > 25%:   Poor accuracy, model revision needed

**Project Results - MAPE Analysis:**

.. code-block::

   Best MAPE Results:
   
   ★ LINE-06 with Multi-Kernel CNN: 8.63%
   - Excellent operational accuracy
   - Suitable for daily production planning
   - Minimal business impact from forecast errors
   
   Overall Performance Range:
   - Best models: 8.6% - 12.4% MAPE
   - Statistical baselines: 7.9% - 12.4% MAPE
   - Deep learning advantage: 15-20% improvement on complex lines

**4. Symmetric Mean Absolute Percentage Error (sMAPE)**

**Definition and Calculation:**

.. math::

   sMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}

**Implementation:**

.. code-block:: python

   def calculate_smape(y_true, y_pred):
       """Calculate Symmetric MAPE"""
       numerator = np.abs(y_true - y_pred)
       denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
       return 100 * np.mean(numerator / denominator)

**Advantages over MAPE:**

- **Symmetry**: Equal penalty for over- and under-forecasting
- **Bounded**: Limited to 0-200% range
- **Robust**: Less sensitive to very small actual values

 **Validation Strategies**
===========================

**1. Walk-Forward Validation**

**Methodology:**

.. code-block:: python

   def walk_forward_validation(data, model_class, window_size=30, test_size=7):
       """Implement walk-forward validation for time series"""
       
       results = []
       
       for i in range(window_size, len(data) - test_size + 1, test_size):
           # Training window
           train_data = data[i-window_size:i]
           
           # Test window
           test_data = data[i:i+test_size]
           
           # Train model
           model = model_class()
           model.fit(train_data)
           
           # Generate predictions
           predictions = model.predict(len(test_data))
           
           # Calculate metrics
           mae = calculate_mae(test_data, predictions)
           rmse = calculate_rmse(test_data, predictions)
           mape = calculate_mape(test_data, predictions)
           
           results.append({
               'period': i,
               'mae': mae,
               'rmse': rmse,
               'mape': mape,
               'predictions': predictions,
               'actual': test_data
           })
       
       return results

**Advantages:**

- **Realistic Simulation**: Mimics real-world deployment scenario
- **Temporal Integrity**: Respects time series ordering
- **Robustness**: Tests model performance across different periods
- **Practical Relevance**: Directly applicable to production use

**2. Time Series Cross-Validation**

**Implementation:**

.. code-block:: python

   def time_series_cv(data, n_splits=5, train_size=60):
       """Time series cross-validation with expanding window"""
       
       fold_results = []
       total_size = len(data)
       
       for i in range(n_splits):
           # Expanding training window
           train_end = train_size + i * (total_size - train_size) // n_splits
           test_start = train_end
           test_end = min(test_start + 30, total_size)
           
           train_data = data[:train_end]
           test_data = data[test_start:test_end]
           
           fold_results.append({
               'fold': i,
               'train_indices': (0, train_end),
               'test_indices': (test_start, test_end),
               'train_data': train_data,
               'test_data': test_data
           })
       
       return fold_results

 **Business Impact Metrics**
=============================

**1. Forecast Bias Analysis**

**Calculation:**

.. code-block:: python

   def calculate_forecast_bias(y_true, y_pred):
       """Analyze systematic bias in forecasts"""
       
       bias = np.mean(y_pred - y_true)
       bias_percentage = 100 * bias / np.mean(y_true)
       
       return {
           'absolute_bias': bias,
           'percentage_bias': bias_percentage,
           'bias_direction': 'over' if bias > 0 else 'under'
       }

**Business Interpretation:**

.. code-block::

   Forecast Bias Impact:
   
   Positive Bias (Over-forecasting):
   - Conservative resource planning
   - Potential over-staffing
   - Higher operational costs
   
   Negative Bias (Under-forecasting):
   - Optimistic planning
   - Risk of under-resourcing
   - Potential production shortfalls
   
   Ideal Bias: Close to 0%
   Acceptable Range: ±2% for OEE forecasting

**2. Directional Accuracy**

**Implementation:**

.. code-block:: python

   def calculate_directional_accuracy(y_true, y_pred):
       """Calculate percentage of correct directional predictions"""
       
       # Calculate period-over-period changes
       true_direction = np.diff(y_true)
       pred_direction = np.diff(y_pred)
       
       # Compare signs (direction)
       correct_directions = np.sign(true_direction) == np.sign(pred_direction)
       
       directional_accuracy = np.mean(correct_directions) * 100
       
       return {
           'directional_accuracy': directional_accuracy,
           'total_periods': len(correct_directions),
           'correct_predictions': np.sum(correct_directions)
       }

**Business Value:**

Directional accuracy is often more important than absolute accuracy for:
- Trend identification
- Resource allocation decisions
- Maintenance scheduling
- Capacity planning

**3. Confidence Interval Coverage**

**Implementation:**

.. code-block:: python

   def evaluate_confidence_intervals(y_true, confidence_intervals, alpha=0.05):
       """Evaluate confidence interval coverage"""
       
       lower_bounds, upper_bounds = confidence_intervals
       
       # Check if actual values fall within intervals
       within_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
       
       # Calculate coverage percentage
       coverage = np.mean(within_interval) * 100
       expected_coverage = (1 - alpha) * 100
       
       return {
           'actual_coverage': coverage,
           'expected_coverage': expected_coverage,
           'coverage_difference': coverage - expected_coverage,
           'well_calibrated': abs(coverage - expected_coverage) < 5
       }

 **Model Comparison Framework**
===============================

**1. Statistical Significance Testing**

**Diebold-Mariano Test:**

.. code-block:: python

   def diebold_mariano_test(errors1, errors2, h=1):
       """Test for significant difference in forecast accuracy"""
       
       from scipy import stats
       
       # Calculate loss differential
       d = errors1**2 - errors2**2
       
       # Calculate test statistic
       d_mean = np.mean(d)
       d_var = np.var(d, ddof=1)
       
       # Adjust for forecast horizon
       d_var_adj = d_var * (1 + 2*sum([i/len(d) for i in range(1, h)]))
       
       # Test statistic
       dm_stat = d_mean / np.sqrt(d_var_adj / len(d))
       
       # P-value (two-tailed test)
       p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
       
       return {
           'dm_statistic': dm_stat,
           'p_value': p_value,
           'significant': p_value < 0.05,
           'better_model': 1 if dm_stat < 0 else 2
       }

**2. Relative Performance Rankings**

**Implementation:**

.. code-block:: python

   def calculate_model_rankings(evaluation_results):
       """Rank models based on multiple metrics"""
       
       metrics = ['mae', 'rmse', 'mape', 'directional_accuracy']
       models = list(evaluation_results.keys())
       
       rankings = {}
       
       for metric in metrics:
           # Get metric values for all models
           values = [evaluation_results[model][metric] for model in models]
           
           # Rank (lower is better for error metrics, higher for directional accuracy)
           if metric == 'directional_accuracy':
               sorted_indices = np.argsort(values)[::-1]  # Descending
           else:
               sorted_indices = np.argsort(values)  # Ascending
           
           # Assign ranks
           for rank, idx in enumerate(sorted_indices):
               model = models[idx]
               if model not in rankings:
                   rankings[model] = {}
               rankings[model][metric] = rank + 1
       
       # Calculate average rank
       for model in models:
           avg_rank = np.mean(list(rankings[model].values()))
           rankings[model]['average_rank'] = avg_rank
       
       return rankings

 **Comprehensive Performance Report**
=====================================

**Model Performance Summary:**

.. code-block::

    CHAMPION MODELS BY PRODUCTION LINE:
   
   LINE-01 (Most Challenging):
   ├── Champion: WaveNet-Style CNN
   │   ├── MAE: 0.0734 (7.3% error)
   │   ├── MAPE: 10.9%
   │   └── Directional Accuracy: 73%
   └── Runner-up: Multi-Kernel CNN
   
   LINE-03 (Balanced Performance):
   ├── Champion: Multi-Kernel CNN
   │   ├── MAE: 0.0698 (7.0% error)
   │   ├── MAPE: 9.7%
   │   └── Directional Accuracy: 76%
   └── Runner-up: WaveNet-Style CNN
   
   LINE-04 (Trend-Following):
   ├── Champion: WaveNet-Style CNN
   │   ├── MAE: 0.0701 (7.0% error)
   │   ├── MAPE: 10.2%
   │   └── Directional Accuracy: 74%
   └── Runner-up: Multi-Kernel CNN
   
   LINE-06 (Most Predictable):
   ├── Champion: Multi-Kernel CNN ★★★
   │   ├── MAE: 0.0591 (5.9% error)
   │   ├── MAPE: 8.63% (BEST OVERALL)
   │   └── Directional Accuracy: 81%
   └── Statistical Baseline: ARIMA(1,0,0) also performs well

**Cross-Model Analysis:**

.. code-block::

   CONSISTENCY RANKINGS:
   
   1. Multi-Kernel CNN
      - Most consistent performer across all lines
      - Best overall average performance
      - Excellent pattern recognition capabilities
   
   2. WaveNet-Style CNN
      - Superior on complex, irregular patterns
      - Best for challenging production lines
      - Good long-range dependency modeling
   
   3. Stacked RNN with Masking
      - Most robust to missing data
      - Consistent baseline performance
      - Good stability over time
   
   4. LSTM
      - Solid sequential pattern modeling
      - Good memory for long-term dependencies
      - Reliable performance across scenarios

**Error Distribution Analysis:**

.. code-block:: python

   def analyze_error_distribution(errors):
       """Comprehensive error distribution analysis"""
       
       from scipy import stats
       
       analysis = {
           'mean_error': np.mean(errors),
           'std_error': np.std(errors),
           'skewness': stats.skew(errors),
           'kurtosis': stats.kurtosis(errors),
           'min_error': np.min(errors),
           'max_error': np.max(errors),
           'percentiles': {
               '5th': np.percentile(errors, 5),
               '25th': np.percentile(errors, 25),
               '50th': np.percentile(errors, 50),
               '75th': np.percentile(errors, 75),
               '95th': np.percentile(errors, 95)
           }
       }
       
       # Normality test
       _, normality_p = stats.shapiro(errors)
       analysis['normal_distribution'] = normality_p > 0.05
       
       return analysis

 **Production Deployment Metrics**
===================================

**Real-Time Performance Monitoring:**

.. code-block:: python

   def create_performance_monitor():
       """Real-time model performance monitoring system"""
       
       class ModelMonitor:
           def __init__(self, alert_thresholds):
               self.thresholds = alert_thresholds
               self.performance_history = []
           
           def update_performance(self, actual, predicted):
               """Update performance metrics with new data point"""
               
               error = abs(actual - predicted)
               percentage_error = 100 * error / actual
               
               self.performance_history.append({
                   'timestamp': datetime.now(),
                   'actual': actual,
                   'predicted': predicted,
                   'error': error,
                   'percentage_error': percentage_error
               })
               
               # Check for performance degradation
               self.check_alerts()
           
           def check_alerts(self):
               """Monitor for performance degradation"""
               
               recent_errors = [p['percentage_error'] 
                              for p in self.performance_history[-10:]]
               
               if len(recent_errors) >= 10:
                   avg_recent_error = np.mean(recent_errors)
                   
                   if avg_recent_error > self.thresholds['warning']:
                       return self.trigger_alert('warning', avg_recent_error)
                   elif avg_recent_error > self.thresholds['critical']:
                       return self.trigger_alert('critical', avg_recent_error)
       
       return ModelMonitor

**Business KPI Integration:**

.. code-block::

   Business Impact Assessment:
   
   Forecast Accuracy Impact on:
   ├── Production Planning Efficiency
   │   ├── Resource allocation optimization
   │   ├── Capacity planning accuracy
   │   └── Inventory management
   ├── Maintenance Scheduling
   │   ├── Predictive maintenance timing
   │   ├── Spare parts management
   │   └── Downtime minimization
   └── Financial Performance
       ├── Cost reduction through optimization
       ├── Revenue protection via availability
       └── ROI from analytics investment

🔗 **Integration Examples**
==========================

**Streamlit Dashboard Integration:**

.. code-block:: python

   def display_model_performance(evaluation_results):
       """Display model performance metrics in Streamlit"""
       
       st.subheader("📊 Model Performance Comparison")
       
       # Create performance DataFrame
       perf_df = pd.DataFrame(evaluation_results).T
       
       # Display metrics table
       st.dataframe(
           perf_df.style.highlight_min(subset=['mae', 'rmse', 'mape'])
                        .highlight_max(subset=['directional_accuracy'])
                        .format({'mae': '{:.4f}', 'rmse': '{:.4f}', 
                               'mape': '{:.2f}%', 'directional_accuracy': '{:.1f}%'})
       )
       
       # Create visualization
       fig = create_performance_comparison_chart(perf_df)
       st.plotly_chart(fig, use_container_width=True)

**Automated Reporting:**

.. code-block:: python

   def generate_performance_report(model_results, production_line):
       """Generate automated performance report"""
       
       report = f"""
       Model Performance Report - {production_line}
       ============================================
       
       Best Performing Model: {model_results['best_model']}
       
       Key Metrics:
       - MAE: {model_results['mae']:.4f} ({model_results['mae']*100:.1f}% error)
       - MAPE: {model_results['mape']:.2f}%
       - Directional Accuracy: {model_results['directional_accuracy']:.1f}%
       
       Business Impact:
       - Forecast accuracy suitable for: {assess_business_suitability(model_results)}
       - Recommended use case: {recommend_use_case(model_results)}
       
       Model Confidence: {assess_confidence_level(model_results)}
       
       Next Review Date: {calculate_next_review_date()}
       """
       
       return report

**Next Steps:**

- Explore :doc:`../advanced/model_optimization` for performance improvement techniques
- Review :doc:`../api/forecasting` for programmatic access to evaluation metrics
- Check :doc:`../troubleshooting` for common evaluation issues and solutions