OEE Forecasting Interface
=========================

The OEE Forecasting page provides advanced predictive analytics capabilities, enabling manufacturing teams to forecast production line performance using both statistical and deep learning models.

 **Forecasting Overview**
===========================

**Purpose and Capabilities:**

The forecasting interface serves multiple business purposes:

- **Production Planning**: Predict OEE performance for upcoming periods
- **Resource Allocation**: Optimize staffing and material planning based on forecasts
- **Maintenance Scheduling**: Time maintenance activities during predicted low-performance periods
- **Risk Management**: Identify potential performance issues before they occur
- **Continuous Improvement**: Track forecast accuracy to improve operational planning

**Forecasting Methodologies:**

The system provides two complementary approaches:

1. **Basic Statistical Methods**: Quick, interpretable forecasts using traditional time series techniques
2. **Advanced Deep Learning**: Sophisticated neural network models for complex pattern recognition

 **Interface Architecture**
============================

**Page Structure:**

.. code-block::

   Forecasting Page Layout:
   ├── System Status & Diagnostics
   ├── Configuration Section
   │   ├── Target Selection (Overall OEE / Specific Line)
   │   ├── Forecast Horizon (1-30 days)
   │   └── Model Parameters
   ├── Model Selection & Recommendation
   │   ├── Available Models Display
   │   ├── Model Performance Comparison
   │   └── Automatic Recommendation System
   ├── Forecast Generation
   │   ├── Training Progress Indicators
   │   ├── Real-time Results Display
   │   └── Interactive Visualizations
   └── Results Analysis
       ├── Forecast Accuracy Metrics
       ├── Confidence Intervals
       └── Business Impact Assessment

**Dynamic Feature Detection:**

The interface automatically adapts based on system capabilities:

.. code-block:: python

   def show_forecasting_page(daily_oee_data, overall_daily_oee):
       """Main forecasting page with adaptive feature detection"""
       
       st.header("🔮 OEE Forecasting with Deep Learning")
       
       # System diagnostics and capability detection
       show_system_diagnostics()
       
       if not DEEP_LEARNING_AVAILABLE:
           # Fallback to statistical methods only
           show_tensorflow_troubleshooting()
           show_basic_forecasting(daily_oee_data, overall_daily_oee)
           return
       
       # Full deep learning interface
       show_advanced_forecasting_interface(daily_oee_data, overall_daily_oee)

 **System Diagnostics and Troubleshooting**
==============================================

**Comprehensive System Status:**

The page begins with detailed system diagnostics to help users understand available capabilities:

.. code-block:: python

   def show_system_diagnostics():
       """Display comprehensive system diagnostics"""
       
       with st.expander("🔍 Deep Learning Libraries Diagnostic", expanded=not DEEP_LEARNING_AVAILABLE):
           st.markdown("**Import Status:**")
           
           # Show detailed import status
           for detail in IMPORT_ERROR_DETAILS:
               if "✅" in detail:
                   st.success(detail)
               else:
                   st.error(detail)
           
           # System information
           import sys
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("**System Information:**")
               st.code(f"""
Python: {sys.version.split()[0]}
Platform: {sys.platform}
Executable: {sys.executable}
               """)
           
           with col2:
               st.markdown("**Available Models:**")
               available_models = detect_available_models()
               for model, status in available_models.items():
                   status_icon = "✅" if status else "❌"
                   st.markdown(f"- {status_icon} {model}")

**Windows-Specific TensorFlow Troubleshooting:**

For Windows users experiencing TensorFlow issues, detailed troubleshooting guidance is provided:

.. code-block:: python

   def show_tensorflow_troubleshooting():
       """Windows-specific TensorFlow troubleshooting guide"""
       
       if TENSORFLOW_DLL_ERROR:
           st.error("❌ TensorFlow DLL loading failed. This is a common Windows issue.")
           
           tab1, tab2, tab3 = st.tabs(["🔧 Solution 1", "🔧 Solution 2", "🔧 Solution 3"])
           
           with tab1:
               st.markdown("### Microsoft Visual C++ Redistributable")
               st.markdown("""
               This is the most common fix for Windows TensorFlow DLL issues:
               
               1. Download [Microsoft Visual C++ Redistributable](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
               2. Install the redistributable package
               3. Restart your computer
               4. Restart Streamlit
               """)
           
           with tab2:
               st.markdown("### Use CPU-Only TensorFlow")
               st.code("""
   # Uninstall current TensorFlow
   pip uninstall tensorflow
   
   # Install CPU-only version
   pip install tensorflow-cpu
               """, language="bash")
           
           with tab3:
               st.markdown("### Conda Alternative")
               st.code("""
   # Create new conda environment
   conda create -n tf_env python=3.9 tensorflow scikit-learn streamlit pandas plotly
   conda activate tf_env
               """, language="bash")

 **Basic Statistical Forecasting**
====================================

**Always-Available Fallback Methods:**

Even when deep learning is unavailable, users can access statistical forecasting methods:

.. code-block:: python

   def show_basic_forecasting(daily_oee_data, overall_daily_oee):
       """Statistical forecasting interface"""
       
       st.subheader("📈 Basic Statistical Forecasting")
       st.info("These methods use simple statistical techniques and don't require TensorFlow.")
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           forecast_target = st.selectbox(
               "Select Target:",
               options=['Overall Daily OEE'] + [f'Line: {line}' for line in sorted(daily_oee_data['PRODUCTION_LINE'].unique())],
               key="basic_forecast_target"
           )
       
       with col2:
           forecast_days = st.slider("Forecast Days:", min_value=1, max_value=30, value=7, key="basic_forecast_days")
       
       with col3:
           method = st.selectbox(
               "Method:", 
               options=['Moving Average', 'Linear Trend', 'Exponential Smoothing'], 
               key="basic_method"
           )
       
       if st.button("📊 Generate Basic Forecast", use_container_width=True):
           generate_statistical_forecast(forecast_target, forecast_days, method, daily_oee_data, overall_daily_oee)

**Statistical Method Implementations:**

.. code-block:: python

   def create_basic_forecast(data_1d, forecast_steps, method):
       """Implementation of basic statistical forecasting methods"""
       
       try:
           if method == 'Moving Average':
               # Simple moving average forecast
               window_size = min(7, len(data_1d) // 2)
               if window_size < 1:
                   window_size = 1
               avg_value = np.mean(data_1d[-window_size:])
               forecasts = np.full(forecast_steps, avg_value)
               
           elif method == 'Linear Trend':
               # Linear regression forecast
               x = np.arange(len(data_1d))
               coeffs = np.polyfit(x, data_1d, 1)
               future_x = np.arange(len(data_1d), len(data_1d) + forecast_steps)
               forecasts = np.polyval(coeffs, future_x)
               
           elif method == 'Exponential Smoothing':
               # Simple exponential smoothing
               alpha = 0.3
               s = data_1d[0]
               for i in range(1, len(data_1d)):
                   s = alpha * data_1d[i] + (1 - alpha) * s
               forecasts = np.full(forecast_steps, s)
           
           # Ensure realistic bounds (0% to 120% OEE)
           forecasts = np.clip(forecasts, 0.0, 1.2)
           return forecasts
           
       except Exception as e:
           st.error(f"Error creating forecast: {str(e)}")
           return None

**Statistical Forecast Visualization:**

.. code-block:: python

   def display_statistical_forecast(historical_data, forecasts, method, target):
       """Display statistical forecast results with confidence indicators"""
       
       fig = go.Figure()
       
       # Historical data
       historical_show = min(30, len(historical_data))
       fig.add_trace(go.Scatter(
           x=dates[-historical_show:], 
           y=historical_data[-historical_show:],
           mode='lines+markers', 
           name='Historical OEE',
           line=dict(color='blue', width=2), 
           marker=dict(size=4)
       ))
       
       # Forecast data
       fig.add_trace(go.Scatter(
           x=future_dates, 
           y=forecasts,
           mode='lines+markers', 
           name=f'Forecast ({method})',
           line=dict(color='red', width=2, dash='dash'), 
           marker=dict(size=6, symbol='diamond')
       ))
       
       # Add forecast start indicator
       fig.add_vline(
           x=dates[-1], 
           line_dash="dash", 
           line_color="gray", 
           annotation_text="Forecast Start"
       )
       
       fig.update_layout(
           title=f'Basic OEE Forecast for {target} ({method})',
           xaxis_title='Date', 
           yaxis_title='OEE',
           yaxis=dict(tickformat=',.0%'), 
           hovermode='x unified', 
           height=500
       )
       
       st.plotly_chart(fig, use_container_width=True)

 **Advanced Deep Learning Interface**
=======================================

**Configuration Section:**

The advanced interface provides comprehensive model configuration options:

.. code-block:: python

   def show_advanced_forecasting_interface(daily_oee_data, overall_daily_oee):
       """Main deep learning forecasting interface"""
       
       # Initialize session state for forecasting
       if 'forecasting_results' not in st.session_state:
           st.session_state.forecasting_results = {}
       if 'model_recommendations' not in st.session_state:
           st.session_state.model_recommendations = {}
       
       # Configuration section
       st.subheader("🎯 Forecasting Configuration")
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           forecast_target = st.selectbox(
               "Select Forecast Target:",
               options=['Overall Daily OEE'] + [f'Line: {line}' for line in sorted(daily_oee_data['PRODUCTION_LINE'].unique())],
               key="forecast_target"
           )
       
       with col2:
           forecast_days = st.slider(
               "Forecast Horizon (Days):",
               min_value=1, max_value=30, value=7,
               key="forecast_days"
           )
       
       with col3:
           training_epochs = st.slider(
               "Training Epochs:",
               min_value=20, max_value=100, value=50,
               key="training_epochs"
           )

**Model Selection Framework:**

.. code-block:: python

   def show_model_selection_interface():
       """Advanced model selection with detailed descriptions"""
       
       st.subheader("🤖 Model Selection")
       
       model_options = {
           "Stacked RNN with Masking": {
               'builder': lambda input_shape: build_stacked_simplernn_with_masking(input_shape, [64, 32], 0.25),
               'look_back': 14,
               'use_padding': True,
               'target_padded_length': 20,
               'description': "RNN with masking layer, LB=14, Padded to 20. Good for sequences with missing data.",
               'best_for': "Irregular data patterns, missing data handling",
               'training_time': "Medium (30-60 seconds)",
               'accuracy': "High for complex patterns"
           },
           "Multi-Kernel CNN": {
               'builder': build_multi_kernel_cnn,
               'look_back': 30,
               'use_padding': False,
               'target_padded_length': None,
               'description': "CNN with multiple kernel sizes, LB=30. Captures different time patterns.",
               'best_for': "Stable patterns, fast inference",
               'training_time': "Fast (15-30 seconds)",
               'accuracy': "Excellent for regular patterns"
           },
           "WaveNet-style CNN": {
               'builder': lambda input_shape: build_wavenet_style_cnn(input_shape, 2, 32, 2, 16, 0.178),
               'look_back': 14,
               'use_padding': False,
               'target_padded_length': None,
               'description': "Dilated CNN, LB=14. Advanced architecture for complex patterns.",
               'best_for': "Complex temporal dependencies",
               'training_time': "Medium (45-90 seconds)",
               'accuracy': "High for complex data"
           }
       }
       
       col1, col2 = st.columns([2, 3])
       
       with col1:
           selected_model = st.selectbox(
               "Choose Model:",
               options=list(model_options.keys()),
               key="selected_model"
           )
           
           model_config = model_options[selected_model]
           
           # Model information card
           st.info(f"""
           **{selected_model}**
           
           {model_config['description']}
           
           **Best for:** {model_config['best_for']}
           **Training time:** {model_config['training_time']}
           **Accuracy:** {model_config['accuracy']}
           """)
       
       with col2:
           show_model_recommendation_system(forecast_target, model_options)

**Automated Model Recommendation System:**

.. code-block:: python

   def show_model_recommendation_system(forecast_target, model_options):
       """Intelligent model recommendation based on data characteristics"""
       
       st.markdown("### 🎯 Get Model Recommendation")
       
       if st.button("🔍 Find Best Model for This Data", use_container_width=True):
           with st.spinner("Testing all models to find the best one..."):
               
               # Prepare data based on target
               data_source, source_name = prepare_forecast_data(forecast_target)
               
               if data_source.empty:
                   st.error("No data available for the selected target.")
                   return
               
               # Analyze data characteristics
               data_characteristics = analyze_data_characteristics(data_source)
               
               # Test all models
               best_model, all_results = recommend_best_model(
                   data_source['OEE'].values, 
                   fit_scaler(data_source['OEE'].values), 
                   source_name
               )
               
               if best_model:
                   # Store recommendations in session state
                   st.session_state.model_recommendations[forecast_target] = {
                       'best_model': best_model,
                       'all_results': all_results,
                       'data_characteristics': data_characteristics
                   }
                   
                   st.success(f"✅ Best model found: **{best_model['name']}** (MAE: {best_model['mae']:.4f})")
                   
                   # Show performance comparison
                   show_model_performance_comparison(all_results)

**Model Performance Comparison:**

.. code-block:: python

   def show_model_performance_comparison(all_results):
       """Display comprehensive model performance comparison"""
       
       st.markdown("### 🏆 Model Performance Comparison")
       
       # Create comparison DataFrame
       comparison_data = []
       for model_name, results in all_results.items():
           if results:
               comparison_data.append({
                   'Model': model_name,
                   'MAE': f"{results['mae']:.4f}",
                   'RMSE': f"{results['rmse']:.4f}",
                   'MAPE': f"{results['mape']:.2f}%",
                   'Training Time': estimate_training_time(model_name),
                   'Complexity': get_model_complexity(model_name)
               })
       
       if comparison_data:
           comparison_df = pd.DataFrame(comparison_data)
           
           # Highlight best model
           def highlight_best_model(row):
               if row['MAE'] == min(comparison_df['MAE']):
                   return ['background-color: lightgreen'] * len(row)
               return [''] * len(row)
           
           styled_df = comparison_df.style.apply(highlight_best_model, axis=1)
           st.dataframe(styled_df, hide_index=True, use_container_width=True)
           
           # Performance insights
           best_model_name = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
           st.info(f"💡 **Recommendation**: {best_model_name} shows the best performance for your data pattern.")

 **Forecast Generation Process**
=================================

**Interactive Forecast Creation:**

.. code-block:: python

   def show_forecast_generation_interface():
       """Main forecast generation interface with progress tracking"""
       
       st.subheader("🚀 Generate Forecast")
       
       col1, col2 = st.columns([1, 3])
       
       with col1:
           if st.button("📈 Create Forecast", use_container_width=True, type="primary"):
               generate_advanced_forecast()
       
       with col2:
           # Display any existing forecast results
           display_existing_forecast_results()

   def generate_advanced_forecast():
       """Advanced forecast generation with progress tracking"""
       
       progress_container = st.container()
       
       with progress_container:
           progress_bar = st.progress(0)
           status_text = st.empty()
           
           try:
               # Step 1: Data preparation
               status_text.text("🔄 Preparing data for modeling...")
               progress_bar.progress(10)
               
               data_source, source_name = prepare_forecast_data(st.session_state.forecast_target)
               oee_values = data_source['OEE'].values
               dates = data_source['Date'].values
               
               # Step 2: Data validation
               status_text.text("✅ Validating data quality...")
               progress_bar.progress(20)
               
               if len(oee_values) < 20:
                   st.error(f"❌ Insufficient data. Need at least 20 data points, but only have {len(oee_values)}.")
                   return
               
               # Step 3: Model preparation
               status_text.text("🏗️ Building model architecture...")
               progress_bar.progress(30)
               
               model_config = get_selected_model_config()
               scaler = RobustScaler()
               scaler.fit(oee_values.reshape(-1, 1))
               
               # Step 4: Model training
               status_text.text("🧠 Training neural network...")
               progress_bar.progress(50)
               
               forecast_values = create_forecast(
                   model_builder_func=model_config['builder'],
                   data_1d=oee_values,
                   scaler_obj=scaler,
                   look_back=model_config['look_back'],
                   forecast_steps=st.session_state.forecast_days,
                   use_padding=model_config['use_padding'],
                   target_padded_length=model_config['target_padded_length'],
                   epochs=st.session_state.training_epochs
               )
               
               # Step 5: Results preparation
               status_text.text("📊 Preparing results visualization...")
               progress_bar.progress(80)
               
               if forecast_values is not None:
                   # Generate future dates
                   last_date = pd.to_datetime(dates[-1])
                   future_dates = [last_date + timedelta(days=i+1) for i in range(st.session_state.forecast_days)]
                   
                   # Store results
                   st.session_state.forecasting_results[st.session_state.forecast_target] = {
                       'model_name': st.session_state.selected_model,
                       'forecast_values': forecast_values,
                       'future_dates': future_dates,
                       'historical_data': oee_values,
                       'historical_dates': dates,
                       'forecast_days': st.session_state.forecast_days,
                       'training_epochs': st.session_state.training_epochs,
                       'timestamp': datetime.now()
                   }
                   
                   # Step 6: Complete
                   progress_bar.progress(100)
                   status_text.text("✅ Forecast generation complete!")
                   
                   st.success(f"✅ Forecast generated successfully using {st.session_state.selected_model}!")
                   time.sleep(1)  # Brief pause to show completion
                   
               else:
                   st.error("❌ Failed to generate forecast. Please try a different model or check your data.")
                   
           except Exception as e:
               st.error(f"❌ Error during forecast generation: {str(e)}")
           
           finally:
               # Clean up progress indicators
               progress_bar.empty()
               status_text.empty()

 **Results Visualization and Analysis**
=========================================

**Interactive Forecast Display:**

.. code-block:: python

   def display_forecast_results(forecast_data):
       """Comprehensive forecast results display"""
       
       # Main forecast visualization
       fig = create_forecast_visualization(forecast_data)
       st.plotly_chart(fig, use_container_width=True)
       
       # Summary metrics
       col1, col2, col3 = st.columns(3)
       
       with col1:
           avg_forecast = np.mean(forecast_data['forecast_values'])
           st.metric("Average Forecast OEE", f"{avg_forecast:.1%}")
       
       with col2:
           last_historical = forecast_data['historical_data'][-1]
           change = avg_forecast - last_historical
           st.metric("Change from Current", f"{change:+.1%}")
       
       with col3:
           forecast_range = np.max(forecast_data['forecast_values']) - np.min(forecast_data['forecast_values'])
           st.metric("Forecast Range", f"{forecast_range:.1%}")

   def create_forecast_visualization(forecast_data):
       """Create comprehensive forecast visualization"""
       
       fig = go.Figure()
       
       # Historical data (last 30 days for context)
       historical_show = min(30, len(forecast_data['historical_data']))
       historical_dates = pd.to_datetime(forecast_data['historical_dates'][-historical_show:])
       historical_values = forecast_data['historical_data'][-historical_show:]
       
       fig.add_trace(go.Scatter(
           x=historical_dates,
           y=historical_values,
           mode='lines+markers',
           name='Historical OEE',
           line=dict(color='blue', width=2),
           marker=dict(size=4),
           hovertemplate='<b>Historical</b><br>' +
                        'Date: %{x|%Y-%m-%d}<br>' +
                        'OEE: %{y:.1%}<extra></extra>'
       ))
       
       # Forecast data
       fig.add_trace(go.Scatter(
           x=forecast_data['future_dates'],
           y=forecast_data['forecast_values'],
           mode='lines+markers',
           name=f'Forecast ({forecast_data["model_name"]})',
           line=dict(color='red', width=2, dash='dash'),
           marker=dict(size=6, symbol='diamond'),
           hovertemplate='<b>Forecast</b><br>' +
                        'Date: %{x|%Y-%m-%d}<br>' +
                        'Predicted OEE: %{y:.1%}<extra></extra>'
       ))
       
       # Add forecast start line
       forecast_start = pd.to_datetime(forecast_data['historical_dates'][-1])
       fig.add_vline(
           x=forecast_start,
           line_dash="dot",
           line_color="gray",
           annotation_text="Forecast Start",
           annotation_position="top"
       )
       
       # Add performance reference lines
       fig.add_hline(y=0.85, line_dash="dot", line_color="green", 
                    annotation_text="World Class (85%)", annotation_position="right")
       fig.add_hline(y=0.70, line_dash="dot", line_color="orange", 
                    annotation_text="Good Performance (70%)", annotation_position="right")
       
       # Layout configuration
       fig.update_layout(
           title=f'OEE Forecast for {forecast_data.get("target", "Selected Target")}',
           xaxis_title='Date',
           yaxis_title='OEE',
           yaxis=dict(tickformat=',.0%', range=[0, 1.1]),
           hovermode='x unified',
           height=600,
           showlegend=True,
           legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
       )
       
       return fig

**Detailed Forecast Table:**

.. code-block:: python

   def show_detailed_forecast_table(forecast_data):
       """Display detailed forecast data in tabular format"""
       
       st.markdown("### 📊 Detailed Forecast")
       
       # Create comprehensive forecast DataFrame
       forecast_df = pd.DataFrame({
           'Date': [d.strftime('%Y-%m-%d') for d in forecast_data['future_dates']],
           'Day of Week': [d.strftime('%A') for d in forecast_data['future_dates']],
           'Forecasted OEE': [f"{v:.1%}" for v in forecast_data['forecast_values']],
           'Performance Level': [classify_performance_level(v) for v in forecast_data['forecast_values']],
           'Day': [f"Day +{i+1}" for i in range(len(forecast_data['forecast_values']))],
           'Business Impact': [assess_business_impact(v) for v in forecast_data['forecast_values']]
       })
       
       # Style the dataframe based on performance levels
       def style_performance_rows(row):
           oee_value = float(row['Forecasted OEE'].strip('%')) / 100
           if oee_value >= 0.85:
               return ['background-color: #90EE90'] * len(row)  # Light green
           elif oee_value >= 0.70:
               return ['background-color: #FFE4B5'] * len(row)  # Light yellow
           elif oee_value >= 0.50:
               return ['background-color: #F0E68C'] * len(row)  # Light orange
           else:
               return ['background-color: #FFA07A'] * len(row)  # Light red
       
       styled_df = forecast_df.style.apply(style_performance_rows, axis=1)
       st.dataframe(styled_df, hide_index=True, use_container_width=True)

   def classify_performance_level(oee_value):
       """Classify OEE performance level"""
       if oee_value >= 0.85:
           return "🟢 Excellent"
       elif oee_value >= 0.70:
           return "🟡 Good"
       elif oee_value >= 0.50:
           return "🟠 Fair"
       else:
           return "🔴 Poor"

   def assess_business_impact(oee_value):
       """Assess business impact of forecasted performance"""
       if oee_value >= 0.85:
           return "Optimal production efficiency"
       elif oee_value >= 0.70:
           return "Good performance, minor optimization opportunities"
       elif oee_value >= 0.50:
           return "Below target, intervention recommended"
       else:
           return "Critical performance, immediate action required"

 **Model Information and Education**
=====================================

**Model Architecture Explanations:**

.. code-block:: python

   def show_model_information_section():
       """Educational content about model architectures"""
       
       st.divider()
       st.markdown("### 📚 Model Information")
       
       with st.expander("🔍 How the Models Work"):
           
           tab1, tab2, tab3 = st.tabs(["🧠 RNN Models", "🔬 CNN Models", "⚙️ Technical Details"])
           
           with tab1:
               st.markdown("""
               **Stacked RNN with Masking:**
               - Uses recurrent neural networks with masking to handle variable-length sequences
               - Good for data with missing values or irregular intervals
               - Look-back window: 14 days, padded to 20
               - Best for: Irregular data patterns, handling missing data
               
               **Stacked RNN without Masking:**
               - Standard RNN approach, faster training
               - Good baseline performance for regular time series
               - Look-back window: 7 days, padded to 35
               - Best for: Regular patterns, faster inference
               """)
           
           with tab2:
               st.markdown("""
               **Multi-Kernel CNN:**
               - Uses multiple convolutional filters with different kernel sizes
               - Captures patterns at different time scales (3, 5, 7 day patterns)
               - Look-back window: 30 days, no padding
               - Best for: Stable patterns, fast training and inference
               
               **WaveNet-style CNN:**
               - Advanced dilated convolutional architecture
               - Can capture long-range dependencies efficiently
               - Look-back window: 14 days, no padding
               - Best for: Complex temporal dependencies, advanced pattern recognition
               """)
           
           with tab3:
               st.markdown("""
               **Training Process:**
               1. Data preprocessing and normalization
               2. Sequence generation with look-back windows
               3. Model architecture construction
               4. Training with early stopping and learning rate reduction
               5. Forecast generation and post-processing
               
               **Performance Metrics:**
               - **MAE (Mean Absolute Error)**: Average forecast error magnitude
               - **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
               - **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric
               
               **Validation Method:**
               - Walk-forward validation simulates real-world forecasting conditions
               - Models retrained with each new data point
               - Provides realistic performance estimates
               """)

**Important Notes and Limitations:**

.. code-block:: python

   def show_important_notes():
       """Display important usage notes and limitations"""
       
       with st.expander("⚠️ Important Notes"):
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("""
               **Data Requirements:**
               - Models need sufficient historical data to train effectively
               - Minimum 20 data points, recommended 50+
               - More data generally leads to better forecasts
               
               **Forecast Accuracy:**
               - Longer forecast horizons typically have lower accuracy
               - 1-7 day forecasts: High accuracy
               - 8-14 day forecasts: Good accuracy
               - 15+ day forecasts: Use with caution
               """)
           
           with col2:
               st.markdown("""
               **Model Selection:**
               - Different models may perform better for different production lines
               - Use the recommendation system for guidance
               - Consider both accuracy and training time
               
               **Validation:**
               - Always validate forecasts against actual outcomes
               - Monitor forecast accuracy over time
               - Retrain models when accuracy degrades
               """)

 **Export and Integration Features**
=====================================

**Forecast Export Options:**

.. code-block:: python

   def show_export_options(forecast_data):
       """Provide comprehensive export options for forecast results"""
       
       st.markdown("### 📤 Export and Integration")
       
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           if st.button("📊 Export Forecast Data", use_container_width=True):
               forecast_csv = create_forecast_export(forecast_data)
               st.download_button(
                   label="Download CSV",
                   data=forecast_csv,
                   file_name=f"oee_forecast_{forecast_data['target']}_{datetime.now().strftime('%Y%m%d')}.csv",
                   mime="text/csv"
               )
       
       with col2:
           if st.button("📈 Export Chart", use_container_width=True):
               forecast_chart = create_forecast_visualization(forecast_data)
               chart_html = forecast_chart.to_html()
               st.download_button(
                   label="Download HTML",
                   data=chart_html,
                   file_name=f"oee_forecast_chart_{datetime.now().strftime('%Y%m%d')}.html",
                   mime="text/html"
               )
       
       with col3:
           if st.button("📋 Export Report", use_container_width=True):
               forecast_report = generate_forecast_report(forecast_data)
               st.download_button(
                   label="Download PDF",
                   data=forecast_report,
                   file_name=f"oee_forecast_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                   mime="application/pdf"
               )
       
       with col4:
           if st.button("🔗 API Endpoint", use_container_width=True):
               api_info = generate_api_information(forecast_data)
               st.code(api_info, language="json")

   def create_forecast_export(forecast_data):
       """Create exportable CSV data"""
       
       export_df = pd.DataFrame({
           'Date': [d.strftime('%Y-%m-%d') for d in forecast_data['future_dates']],
           'Forecasted_OEE': forecast_data['forecast_values'],
           'Model_Used': [forecast_data['model_name']] * len(forecast_data['forecast_values']),
           'Forecast_Horizon_Days': [i+1 for i in range(len(forecast_data['forecast_values']))],
           'Generated_Timestamp': [datetime.now().isoformat()] * len(forecast_data['forecast_values']),
           'Performance_Level': [classify_performance_level(v) for v in forecast_data['forecast_values']]
       })
       
       return export_df.to_csv(index=False)

 **Continuous Improvement Features**
=====================================

**Forecast Accuracy Tracking:**

.. code-block:: python

   def show_accuracy_tracking():
       """Display forecast accuracy tracking over time"""
       
       st.markdown("### 📈 Forecast Accuracy Tracking")
       
       # Load historical forecast accuracy data
       accuracy_history = load_forecast_accuracy_history()
       
       if accuracy_history:
           # Create accuracy trend chart
           accuracy_fig = create_accuracy_trend_chart(accuracy_history)
           st.plotly_chart(accuracy_fig, use_container_width=True)
           
           # Accuracy metrics summary
           col1, col2, col3 = st.columns(3)
           
           with col1:
               avg_accuracy = np.mean([a['accuracy'] for a in accuracy_history])
               st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
           
           with col2:
               accuracy_trend = calculate_accuracy_trend(accuracy_history)
               st.metric("Accuracy Trend", f"{accuracy_trend:+.1%}")
           
           with col3:
               best_model = find_best_performing_model(accuracy_history)
               st.metric("Best Model", best_model)
       
       else:
           st.info("💡 Forecast accuracy tracking will be available after generating forecasts and comparing with actual results.")

**Model Performance Monitoring:**

.. code-block:: python

   def show_model_performance_monitoring():
       """Monitor model performance and suggest improvements"""
       
       st.markdown("### 🔍 Model Performance Monitoring")
       
       # Check for model drift
       model_drift = detect_model_drift()
       
       if model_drift['detected']:
           st.warning(f"⚠️ Model drift detected: {model_drift['description']}")
           
           col1, col2 = st.columns(2)
           
           with col1:
               if st.button("🔄 Retrain Models"):
                   retrain_all_models()
                   st.success("Models retrained successfully!")
           
           with col2:
               if st.button("📊 Analyze Drift"):
                   show_drift_analysis(model_drift)
       
       else:
           st.success("✅ All models performing within expected parameters")

 **Future Enhancements**
=========================

**Planned Features:**

.. code-block:: python

   def show_future_enhancements():
       """Display planned future enhancements"""
       
       with st.expander("🚀 Coming Soon"):
           
           st.markdown("""
           **Short-term Enhancements:**
           - Uncertainty quantification with confidence intervals
           - Ensemble forecasting combining multiple models
           - Automated model selection based on data characteristics
           - Real-time model performance monitoring
           
           **Medium-term Enhancements:**
           - Seasonal pattern detection and modeling
           - External factor integration (weather, demand, etc.)
           - Multi-variate forecasting (multiple production lines)
           - Advanced visualization with interactive controls
           
           **Long-term Vision:**
           - Transformer-based models for improved accuracy
           - Causal inference for what-if scenario analysis
           - Integration with maintenance scheduling systems
           - Automated optimization recommendations
           """)

 **User Guide and Best Practices**
===================================

**Forecasting Workflow:**

.. code-block:: python

   def show_forecasting_workflow():
       """Display recommended forecasting workflow"""
       
       with st.expander("📚 Forecasting Best Practices"):
           
           st.markdown("""
           **Recommended Workflow:**
           
           1. **Data Assessment**
              - Ensure at least 30 days of historical data
              - Check for data quality issues
              - Verify production line is actively operating
           
           2. **Model Selection**
              - Use the recommendation system for guidance
              - Consider data characteristics and business needs
              - Test multiple models for comparison
           
           3. **Forecast Generation**
              - Start with shorter horizons (1-7 days)
              - Gradually extend to longer periods as confidence builds
              - Document forecast assumptions and limitations
           
           4. **Results Analysis**
              - Review forecast patterns for reasonableness
              - Compare with historical performance
              - Identify potential business impacts
           
           5. **Action Planning**
              - Use forecasts for resource planning
              - Schedule maintenance during low-performance periods
              - Prepare contingency plans for poor forecasts
           
           6. **Continuous Improvement**
              - Track actual vs. forecasted performance
              - Retrain models with new data
              - Adjust forecasting approach based on accuracy
           """)

**Next Steps:**

Continue exploring the application features:

- :doc:`advisory_system` - AI-powered advisory system
- :doc:`../models/deep_learning_models` - Technical model details
- :doc:`../advanced/model_optimization` - Model optimization techniques
- :doc:`../troubleshooting` - Common issues and solutions