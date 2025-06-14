Forecasting API Reference
=========================

This section provides comprehensive documentation for the forecasting components of the OEE analytics system. These APIs handle both statistical and deep learning-based time series forecasting capabilities.

🔮 **Core Forecasting Classes**
===============================

.. py:class:: OEEForecaster

   Main forecasting class that orchestrates different prediction models and methodologies.

   .. py:method:: __init__(model_type='auto', look_back=30, forecast_horizon=7)

      Initialize the OEE forecasting system.

      :param str model_type: Type of forecasting model ('auto', 'statistical', 'deep_learning')
      :param int look_back: Number of historical days to use for prediction
      :param int forecast_horizon: Number of days to forecast into the future
      :raises ValueError: If parameters are invalid

      **Model Type Options:**

      .. code-block::

         Model Types:
         
         'auto':           Automatically select best model based on data
         'statistical':    Use ARIMA-based statistical models
         'deep_learning':  Use neural network models
         'ensemble':       Combine multiple model predictions
         'multi_kernel_cnn': Use Multi-Kernel CNN (often best performer)
         'stacked_rnn':    Use RNN with masking
         'wavenet_cnn':    Use WaveNet-style CNN
         'lstm':           Use LSTM model

      **Example:**

      .. code-block:: python

         # Initialize with automatic model selection
         forecaster = OEEForecaster(
             model_type='auto',
             look_back=30,
             forecast_horizon=7
         )

         # Initialize with specific deep learning model
         forecaster = OEEForecaster(
             model_type='multi_kernel_cnn',
             look_back=30,
             forecast_horizon=14
         )

   .. py:method:: fit(data, production_line=None, validation_split=0.2)

      Train the forecasting model on historical OEE data.

      :param pd.DataFrame data: Historical OEE data with date index
      :param str production_line: Specific production line (None for overall OEE)
      :param float validation_split: Fraction of data to use for validation
      :returns: Training results and model performance metrics
      :rtype: dict

      **Data Requirements:**

      .. code-block::

         Input Data Format:
         
         Required Columns:
         - Date index (datetime)
         - OEE values (float, 0-1 range)
         
         Optional Columns:
         - Availability
         - Performance  
         - Quality
         - Production_Line (for multi-line data)

      **Implementation:**

      .. code-block:: python

         def fit(self, data, production_line=None, validation_split=0.2):
             """
             Train forecasting model with comprehensive validation
             
             Training Process:
             1. Data preprocessing and validation
             2. Model selection (if auto mode)
             3. Feature engineering and scaling
             4. Model training with early stopping
             5. Performance evaluation
             6. Model persistence
             """
             
             # Validate input data
             self._validate_input_data(data)
             
             # Filter for specific production line if specified
             if production_line:
                 if 'Production_Line' in data.columns:
                     data = data[data['Production_Line'] == production_line]
                 self.production_line = production_line
             
             # Prepare data for training
             X, y = self._prepare_training_data(data)
             
             # Split data for validation
             split_idx = int(len(X) * (1 - validation_split))
             X_train, X_val = X[:split_idx], X[split_idx:]
             y_train, y_val = y[:split_idx], y[split_idx:]
             
             # Select and train model
             if self.model_type == 'auto':
                 self.model_type = self._select_optimal_model(X_train, y_train)
             
             self.model = self._create_model()
             
             # Train model
             training_history = self._train_model(X_train, y_train, X_val, y_val)
             
             # Evaluate performance
             performance_metrics = self._evaluate_model(X_val, y_val)
             
             # Store training results
             self.training_results = {
                 'model_type': self.model_type,
                 'training_history': training_history,
                 'performance_metrics': performance_metrics,
                 'data_shape': X.shape,
                 'training_date': datetime.now().isoformat()
             }
             
             return self.training_results

   .. py:method:: predict(steps=None, confidence_level=0.95)

      Generate OEE forecasts for specified number of steps.

      :param int steps: Number of time steps to forecast (default: forecast_horizon)
      :param float confidence_level: Confidence level for prediction intervals
      :returns: Forecast results with predictions and confidence intervals
      :rtype: dict

      **Return Structure:**

      .. code-block:: python

         {
             'forecasts': [0.842, 0.851, 0.838, ...],  # Point predictions
             'dates': ['2024-01-16', '2024-01-17', ...],  # Forecast dates
             'confidence_intervals': {
                 'lower': [0.801, 0.809, 0.796, ...],
                 'upper': [0.883, 0.893, 0.880, ...]
             },
             'model_confidence': 0.85,  # Overall model confidence
             'forecast_metadata': {
                 'model_type': 'multi_kernel_cnn',
                 'look_back_period': 30,
                 'generation_time': '2024-01-15T10:30:00Z'
             }
         }

   .. py:method:: predict_with_scenarios(scenarios=None, steps=None)

      Generate forecasts under different operational scenarios.

      :param dict scenarios: Scenario parameters for conditional forecasting
      :param int steps: Number of steps to forecast
      :returns: Scenario-based forecast results
      :rtype: dict

      **Scenario Examples:**

      .. code-block:: python

         scenarios = {
             'baseline': {'maintenance_days': 0, 'production_rate': 1.0},
             'maintenance': {'maintenance_days': 2, 'production_rate': 0.7},
             'optimization': {'maintenance_days': 1, 'production_rate': 1.1}
         }
         
         scenario_forecasts = forecaster.predict_with_scenarios(scenarios, steps=14)

📊 **Statistical Forecasting Models**
====================================

.. py:class:: ARIMAForecaster

   ARIMA-based statistical forecasting for OEE time series.

   .. py:method:: __init__(order=None, seasonal_order=None, auto_arima=True)

      Initialize ARIMA forecasting model.

      :param tuple order: ARIMA order (p, d, q) - None for auto selection
      :param tuple seasonal_order: Seasonal ARIMA parameters
      :param bool auto_arima: Use automatic parameter selection

      **Automatic Parameter Selection:**

      .. code-block:: python

         def auto_select_arima_parameters(self, data):
             """
             Automatic ARIMA parameter selection using pmdarima
             
             Selection Process:
             1. Stationarity testing (ADF test)
             2. ACF/PACF analysis  
             3. Information criteria comparison (AIC/BIC)
             4. Cross-validation performance
             5. Residual analysis
             """
             
             from pmdarima import auto_arima
             
             # Auto ARIMA with comprehensive search
             model = auto_arima(
                 data,
                 seasonal=False,  # Daily OEE typically non-seasonal
                 stepwise=True,
                 suppress_warnings=True,
                 error_action='ignore',
                 max_p=3, max_q=3, max_d=2,
                 information_criterion='aic',
                 alpha=0.05
             )
             
             return model.order

   .. py:method:: fit(data)

      Fit ARIMA model to historical OEE data.

      **Model Diagnostics:**

      .. code-block:: python

         def perform_model_diagnostics(self, fitted_model):
             """
             Comprehensive ARIMA model diagnostics
             
             Diagnostic Tests:
             - Ljung-Box test for residual autocorrelation
             - Jarque-Bera test for normality
             - Heteroscedasticity tests
             - Stability tests
             """
             
             residuals = fitted_model.resid
             
             diagnostics = {
                 'ljung_box': sm.stats.acorr_ljungbox(residuals, lags=10),
                 'jarque_bera': stats.jarque_bera(residuals),
                 'normality_p_value': stats.shapiro(residuals)[1],
                 'residual_autocorr': self._check_residual_autocorr(residuals),
                 'model_stability': self._check_model_stability(fitted_model)
             }
             
             return diagnostics

🧠 **Deep Learning Forecasting Models**
======================================

.. py:class:: DeepLearningForecaster

   Neural network-based forecasting with multiple architecture options.

   .. py:method:: __init__(architecture='multi_kernel_cnn', **kwargs)

      Initialize deep learning forecaster.

      :param str architecture: Neural network architecture to use
      :param kwargs: Architecture-specific parameters

      **Available Architectures:**

      .. code-block::

         Supported Architectures:
         
         'multi_kernel_cnn':     Multi-scale pattern recognition (best overall)
         'stacked_rnn_masking':  RNN with missing data handling
         'wavenet_cnn':          Dilated convolutions for long dependencies
         'lstm':                 Long Short-Term Memory networks
         'transformer':          Attention-based architecture (experimental)

.. py:class:: MultiKernelCNN

   Multi-Kernel CNN implementation - often the best performing model.

   .. py:method:: create_model(look_back, n_features=1)

      Create Multi-Kernel CNN architecture.

      **Architecture Details:**

      .. code-block:: python

         def create_model(self, look_back, n_features=1):
             """
             Multi-Kernel CNN with parallel processing paths
             
             Architecture Features:
             - Multiple kernel sizes (3, 5, 7) for multi-scale patterns
             - Parallel CNN branches with feature fusion
             - Global pooling for translation invariance
             - Dropout regularization for generalization
             """
             
             input_layer = Input(shape=(look_back, n_features))
             
             # Parallel CNN branches
             branches = []
             kernel_sizes = [3, 5, 7]
             
             for kernel_size in kernel_sizes:
                 branch = Conv1D(
                     filters=32, 
                     kernel_size=kernel_size,
                     activation='relu', 
                     padding='same'
                 )(input_layer)
                 branch = MaxPooling1D(pool_size=2)(branch)
                 branches.append(branch)
             
             # Merge branches
             merged = concatenate(branches)
             
             # Additional processing
             x = Conv1D(filters=64, kernel_size=3, activation='relu')(merged)
             x = GlobalMaxPooling1D()(x)
             x = Dense(50, activation='relu')(x)
             x = Dropout(0.3)(x)
             output = Dense(1, activation='sigmoid')(x)
             
             model = Model(inputs=input_layer, outputs=output)
             
             return model

⚡ **Performance Optimization**
=============================

.. py:function:: optimize_model_performance(model, data, optimization_config)

   Optimize model performance through hyperparameter tuning and architecture search.

   :param model: Model to optimize
   :param data: Training data
   :param dict optimization_config: Optimization configuration
   :returns: Optimized model and performance metrics
   :rtype: tuple

   **Optimization Strategies:**

   .. code-block:: python

      def optimize_model_performance(model, data, optimization_config):
          """
          Comprehensive model optimization
          
          Optimization Techniques:
          1. Hyperparameter tuning (learning rate, batch size, epochs)
          2. Architecture optimization (layer sizes, dropout rates)
          3. Data augmentation strategies
          4. Ensemble method configuration
          5. Early stopping and learning rate scheduling
          """
          
          optimization_results = {}
          
          if optimization_config.get('hyperparameter_tuning', False):
              # Grid search or Bayesian optimization
              best_params = tune_hyperparameters(model, data)
              optimization_results['best_hyperparameters'] = best_params
          
          if optimization_config.get('architecture_search', False):
              # Neural Architecture Search (NAS)
              optimal_architecture = search_architecture(data)
              optimization_results['optimal_architecture'] = optimal_architecture
          
          if optimization_config.get('ensemble_optimization', False):
              # Ensemble method selection
              ensemble_config = optimize_ensemble(model, data)
              optimization_results['ensemble_config'] = ensemble_config
          
          return optimized_model, optimization_results

📈 **Model Evaluation and Validation**
=====================================

.. py:function:: evaluate_forecast_performance(y_true, y_pred, metrics=['mae', 'rmse', 'mape'])

   Comprehensive evaluation of forecast performance.

   :param array_like y_true: True OEE values
   :param array_like y_pred: Predicted OEE values  
   :param list metrics: List of metrics to calculate
   :returns: Dictionary of performance metrics
   :rtype: dict

   **Available Metrics:**

   .. code-block:: python

      def evaluate_forecast_performance(y_true, y_pred, metrics=['mae', 'rmse', 'mape']):
          """
          Calculate comprehensive forecast performance metrics
          
          Available Metrics:
          - mae: Mean Absolute Error
          - rmse: Root Mean Square Error
          - mape: Mean Absolute Percentage Error
          - smape: Symmetric Mean Absolute Percentage Error
          - directional_accuracy: Percentage of correct trend predictions
          - bias: Systematic over/under forecasting
          - coverage: Confidence interval coverage (if available)
          """
          
          results = {}
          
          if 'mae' in metrics:
              results['mae'] = np.mean(np.abs(y_true - y_pred))
          
          if 'rmse' in metrics:
              results['rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))
          
          if 'mape' in metrics:
              results['mape'] = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
          
          if 'directional_accuracy' in metrics:
              true_direction = np.diff(y_true)
              pred_direction = np.diff(y_pred)
              correct_directions = np.sign(true_direction) == np.sign(pred_direction)
              results['directional_accuracy'] = np.mean(correct_directions) * 100
          
          if 'bias' in metrics:
              results['bias'] = np.mean(y_pred - y_true)
              results['bias_percentage'] = 100 * results['bias'] / np.mean(y_true)
          
          return results

.. py:function:: walk_forward_validation(forecaster, data, window_size=30, step_size=7)

   Perform walk-forward validation for realistic performance assessment.

   :param forecaster: Trained forecasting model
   :param pd.DataFrame data: Historical data for validation
   :param int window_size: Size of training window
   :param int step_size: Steps to move forward each iteration
   :returns: Validation results across multiple periods
   :rtype: dict

   **Implementation:**

   .. code-block:: python

      def walk_forward_validation(forecaster, data, window_size=30, step_size=7):
          """
          Walk-forward validation for time series forecasting
          
          Validation Process:
          1. Split data into overlapping train/test windows
          2. Train model on each training window
          3. Generate forecasts for test period
          4. Calculate performance metrics
          5. Aggregate results across all windows
          """
          
          validation_results = []
          
          for i in range(window_size, len(data) - step_size + 1, step_size):
              # Define training and test windows
              train_window = data.iloc[i-window_size:i]
              test_window = data.iloc[i:i+step_size]
              
              # Clone and train forecaster
              temp_forecaster = clone_forecaster(forecaster)
              temp_forecaster.fit(train_window)
              
              # Generate forecasts
              forecast_result = temp_forecaster.predict(steps=len(test_window))
              
              # Evaluate performance
              performance = evaluate_forecast_performance(
                  test_window.values,
                  forecast_result['forecasts']
              )
              
              validation_results.append({
                  'window_start': train_window.index[0],
                  'window_end': train_window.index[-1],
                  'test_start': test_window.index[0],
                  'test_end': test_window.index[-1],
                  'performance': performance,
                  'forecasts': forecast_result['forecasts'],
                  'actuals': test_window.values.tolist()
              })
          
          # Aggregate results
          aggregated_performance = aggregate_validation_results(validation_results)
          
          return {
              'individual_windows': validation_results,
              'aggregated_performance': aggregated_performance,
              'validation_summary': create_validation_summary(validation_results)
          }

🎯 **Specialized Forecasting Functions**
=======================================

.. py:function:: forecast_production_line_oee(line_data, production_line, forecast_days=7)

   Generate OEE forecasts for a specific production line.

   :param pd.DataFrame line_data: Historical OEE data for the production line
   :param str production_line: Production line identifier
   :param int forecast_days: Number of days to forecast
   :returns: Line-specific forecast results
   :rtype: dict

   **Line-Specific Optimization:**

   .. code-block:: python

      def forecast_production_line_oee(line_data, production_line, forecast_days=7):
          """
          Production line-specific OEE forecasting
          
          Line-Specific Features:
          - Automatic model selection based on line characteristics
          - Custom preprocessing for each line's patterns
          - Performance benchmarking against line history
          - Alert generation for unusual forecasts
          """
          
          # Analyze line characteristics
          line_characteristics = analyze_line_patterns(line_data, production_line)
          
          # Select optimal model for this line
          optimal_model = select_line_specific_model(line_characteristics)
          
          # Initialize and train forecaster
          forecaster = OEEForecaster(
              model_type=optimal_model,
              look_back=determine_optimal_lookback(line_characteristics),
              forecast_horizon=forecast_days
          )
          
          forecaster.fit(line_data, production_line=production_line)
          
          # Generate forecasts
          forecast_result = forecaster.predict(steps=forecast_days)
          
          # Add line-specific context
          forecast_result.update({
              'production_line': production_line,
              'line_characteristics': line_characteristics,
              'model_recommendation_reason': explain_model_choice(
                  optimal_model, line_characteristics
              ),
              'performance_alerts': check_forecast_alerts(
                  forecast_result, line_data
              )
          })
          
          return forecast_result

.. py:function:: forecast_overall_plant_oee(all_lines_data, forecast_days=7, aggregation_method='weighted')

   Generate overall plant OEE forecasts by aggregating line-level predictions.

   :param dict all_lines_data: OEE data for all production lines
   :param int forecast_days: Number of days to forecast
   :param str aggregation_method: Method for combining line forecasts
   :returns: Plant-level forecast results
   :rtype: dict

   **Aggregation Methods:**

   .. code-block::

      Aggregation Methods:
      
      'simple':     Simple average of line forecasts
      'weighted':   Production volume-weighted average
      'capacity':   Production capacity-weighted average
      'historical': Historical contribution-based weighting

🔧 **Real-Time Forecasting**
===========================

.. py:class:: RealTimeForecaster

   Real-time forecasting system for production environments.

   .. py:method:: __init__(update_frequency='daily', auto_retrain=True)

      Initialize real-time forecasting system.

      :param str update_frequency: How often to update forecasts
      :param bool auto_retrain: Automatically retrain models with new data

   .. py:method:: update_with_new_data(new_data, retrain_threshold=0.1)

      Update forecasts with newly available data.

      :param pd.DataFrame new_data: New OEE data
      :param float retrain_threshold: Performance degradation threshold for retraining
      :returns: Updated forecast results
      :rtype: dict

      **Real-Time Update Process:**

      .. code-block:: python

         def update_with_new_data(self, new_data, retrain_threshold=0.1):
             """
             Real-time forecast updates with adaptive retraining
             
             Update Process:
             1. Validate new data quality
             2. Append to historical data
             3. Check model performance degradation
             4. Retrain model if necessary
             5. Generate updated forecasts
             6. Update confidence metrics
             """
             
             # Validate new data
             self._validate_new_data(new_data)
             
             # Update historical data
             self.historical_data = pd.concat([self.historical_data, new_data])
             
             # Check if retraining is needed
             if self._should_retrain(retrain_threshold):
                 self._retrain_model()
                 self.last_retrain_date = datetime.now()
             
             # Generate updated forecasts
             updated_forecasts = self.predict()
             
             # Update performance tracking
             self._update_performance_tracking(new_data)
             
             return updated_forecasts

🔄 **Model Management**
======================

.. py:function:: save_forecast_model(model, model_metadata, save_path)

   Save trained forecasting model with metadata.

   :param model: Trained forecasting model
   :param dict model_metadata: Model configuration and performance data
   :param str save_path: Path to save model files
   :returns: Success status and file paths
   :rtype: dict

   **Model Persistence:**

   .. code-block:: python

      def save_forecast_model(model, model_metadata, save_path):
          """
          Comprehensive model saving with versioning
          
          Saved Components:
          - Model weights and architecture
          - Training configuration
          - Performance metrics
          - Data preprocessing parameters
          - Version information
          """
          
          import joblib
          import json
          from pathlib import Path
          
          save_dir = Path(save_path)
          save_dir.mkdir(parents=True, exist_ok=True)
          
          # Save model
          if hasattr(model, 'save'):  # TensorFlow/Keras model
              model.save(save_dir / 'model.h5')
          else:  # Scikit-learn or other
              joblib.dump(model, save_dir / 'model.pkl')
          
          # Save metadata
          metadata = {
              'model_type': model_metadata.get('model_type'),
              'training_date': datetime.now().isoformat(),
              'performance_metrics': model_metadata.get('performance_metrics'),
              'model_config': model_metadata.get('config'),
              'version': model_metadata.get('version', '1.0')
          }
          
          with open(save_dir / 'metadata.json', 'w') as f:
              json.dump(metadata, f, indent=2)
          
          return {
              'success': True,
              'model_path': str(save_dir / 'model.h5'),
              'metadata_path': str(save_dir / 'metadata.json')
          }

.. py:function:: load_forecast_model(model_path)

   Load saved forecasting model with metadata.

   :param str model_path: Path to saved model directory
   :returns: Loaded model and metadata
   :rtype: tuple

📊 **Usage Examples**
====================

**Basic Forecasting**

.. code-block:: python

   # Initialize forecaster with automatic model selection
   forecaster = OEEForecaster(model_type='auto', look_back=30, forecast_horizon=7)

   # Train on historical data
   training_results = forecaster.fit(historical_oee_data, production_line='LINE-01')

   # Generate forecasts
   forecast_results = forecaster.predict(steps=7, confidence_level=0.95)

   print(f"7-day OEE forecast: {forecast_results['forecasts']}")
   print(f"Model used: {forecast_results['forecast_metadata']['model_type']}")

**Advanced Multi-Line Forecasting**

.. code-block:: python

   # Forecast all production lines
   all_forecasts = {}

   for line in ['LINE-01', 'LINE-03', 'LINE-04', 'LINE-06']:
       line_forecaster = OEEForecaster(model_type='multi_kernel_cnn')
       line_data = daily_oee_data[daily_oee_data['Production_Line'] == line]
       
       line_forecaster.fit(line_data, production_line=line)
       all_forecasts[line] = line_forecaster.predict(steps=14)

   # Generate plant-wide forecast
   plant_forecast = forecast_overall_plant_oee(all_forecasts, forecast_days=14)

**Real-Time Updates**

.. code-block:: python

   # Initialize real-time forecaster
   rt_forecaster = RealTimeForecaster(update_frequency='daily', auto_retrain=True)

   # Initial training
   rt_forecaster.fit(historical_data)

   # Simulate real-time updates
   for new_day_data in daily_data_stream:
       updated_forecasts = rt_forecaster.update_with_new_data(new_day_data)
       
       if updated_forecasts.get('model_retrained'):
           print("Model retrained due to performance degradation")
       
       print(f"Updated forecast: {updated_forecasts['forecasts'][:3]}")

**Model Comparison**

.. code-block:: python

   # Compare different models
   models_to_test = ['multi_kernel_cnn', 'stacked_rnn_masking', 'wavenet_cnn', 'lstm']
   
   comparison_results = {}
   
   for model_type in models_to_test:
       forecaster = OEEForecaster(model_type=model_type)
       forecaster.fit(training_data)
       
       # Perform walk-forward validation
       validation_results = walk_forward_validation(forecaster, validation_data)
       
       comparison_results[model_type] = validation_results['aggregated_performance']

   # Select best model
   best_model = min(comparison_results.keys(), 
                   key=lambda x: comparison_results[x]['mae'])
   
   print(f"Best performing model: {best_model}")

🚨 **Error Handling and Diagnostics**
====================================

**Common Exceptions**

.. py:exception:: ForecastingError

   Base exception for forecasting-related errors.

.. py:exception:: ModelTrainingError

   Raised when model training fails.

.. py:exception:: PredictionError

   Raised when prediction generation fails.

**Diagnostic Functions**

.. code-block:: python

   def diagnose_forecasting_issues(forecaster, data):
       """
       Comprehensive diagnostic system for forecasting problems
       
       Diagnostic Areas:
       - Data quality and completeness
       - Model performance degradation
       - Prediction consistency
       - Memory and computational issues
       """
       
       diagnostics = {
           'data_quality': assess_data_quality_for_forecasting(data),
           'model_health': check_model_health(forecaster),
           'prediction_consistency': check_prediction_consistency(forecaster),
           'computational_performance': assess_computational_performance(forecaster)
       }
       
       # Generate recommendations
       recommendations = generate_forecasting_recommendations(diagnostics)
       
       return {
           'diagnostics': diagnostics,
           'recommendations': recommendations,
           'overall_health': assess_overall_health(diagnostics)
       }

**Next Steps:**

- Explore :doc:`advisory_system` for AI-powered forecasting insights
- Review :doc:`../models/evaluation_metrics` for detailed performance analysis
- Check :doc:`../advanced/model_optimization` for improving forecast accuracy