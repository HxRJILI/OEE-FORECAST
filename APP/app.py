import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime, date, timedelta
import matplotlib.ticker as mtick

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="OEE Manufacturing Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Learning imports with detailed error reporting
DEEP_LEARNING_AVAILABLE = False
IMPORT_ERROR_DETAILS = []
TENSORFLOW_DLL_ERROR = False

try:
    import tensorflow as tf
    try:
        tf_version = tf.__version__
    except AttributeError:
        try:
            tf_version = tf.version.VERSION
        except:
            tf_version = "Unknown (but imported successfully)"
    IMPORT_ERROR_DETAILS.append(f"✅ TensorFlow imported successfully (version: {tf_version})")
except ImportError as e:
    error_msg = str(e)
    if "DLL load failed" in error_msg or "_pywrap_tensorflow_internal" in error_msg:
        TENSORFLOW_DLL_ERROR = True
        IMPORT_ERROR_DETAILS.append("❌ TensorFlow DLL loading failed (Windows compatibility issue)")
    else:
        IMPORT_ERROR_DETAILS.append(f"❌ TensorFlow import failed: {error_msg}")

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (SimpleRNN, Dense, Dropout, Masking, 
                                       Conv1D, Input, Concatenate, GlobalAveragePooling1D)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam
    IMPORT_ERROR_DETAILS.append("✅ TensorFlow.Keras modules imported successfully")
except ImportError as e:
    if not TENSORFLOW_DLL_ERROR:
        IMPORT_ERROR_DETAILS.append(f"❌ TensorFlow.Keras import failed: {str(e)}")

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import sklearn
    sklearn_version = sklearn.__version__
    IMPORT_ERROR_DETAILS.append(f"✅ Scikit-learn imported successfully (version: {sklearn_version})")
except ImportError as e:
    IMPORT_ERROR_DETAILS.append(f"❌ Scikit-learn import failed: {str(e)}")

# Check if all imports succeeded
if not TENSORFLOW_DLL_ERROR:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from sklearn.preprocessing import RobustScaler
        
        test_model = Sequential([Dense(1, input_shape=(1,))])
        test_scaler = RobustScaler()
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        DEEP_LEARNING_AVAILABLE = True
        IMPORT_ERROR_DETAILS.append("✅ All deep learning libraries are working correctly")
        IMPORT_ERROR_DETAILS.append("✅ Test model creation successful")
    except Exception as e:
        IMPORT_ERROR_DETAILS.append(f"❌ Deep learning functionality test failed: {str(e)}")
        DEEP_LEARNING_AVAILABLE = False
else:
    IMPORT_ERROR_DETAILS.append("❌ Skipping functionality test due to TensorFlow DLL issue")

# RAG System Integration - SIMPLIFIED AND SAFE
ADVISORY_AVAILABLE = False
try:
    # Only try to import if the user has set up the RAG system
    if os.path.exists('advisory_integration.py'):
        from advisory_integration import (
            add_advisory_system_to_sidebar, 
            handle_advisory_pages, 
            integrate_advisory_system,
            check_advisory_system_status
        )
        ADVISORY_AVAILABLE = True
except Exception as e:
    # Don't crash if RAG system has issues
    ADVISORY_AVAILABLE = False

warnings.filterwarnings('ignore')

# Suppress PyTorch warnings specifically
import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTab > div:first-child > div:first-child {
        padding-top: 0;
    }
    .line-status-excellent {
        background-color: #90EE90;
        border: 2px solid #2E8B57;
    }
    .line-status-good {
        background-color: #FFE4B5;
        border: 2px solid #FFD700;
    }
    .line-status-fair {
        background-color: #F0E68C;
        border: 2px solid #FF8C00;
    }
    .line-status-poor {
        background-color: #FFA07A;
        border: 2px solid #DC143C;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f8ff;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .advisory-chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .knowledge-source {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Status Categories and Cycle Times
SCHEDULED_STOPS = ["Meeting", "Cleaning(5S)", "Break Time", "Lunch Break", "Other"]
PRODUCTION_STATUS = ["Production"]
UNEXPECTED_STOPS = [
    "FF Check", "Awaiting Instruction", "Awaiting Materials", "Change Over",
    "Machine Failure", "Quality Check", "Machine Inspection", "Awaiting Box"
]
NO_PLAN = ["No Plan"]
END_OPS = ["End Of Operations"]

CYCLE_TIMES = {
    'LINE-01': 11.0,
    'LINE-02': 11.0,
    'LINE-03': 5.5,
    'LINE-04': 11.0,
    'LINE-05': 11.0,
    'LINE-06': 11.0
}

def create_status_mapping():
    """Create status to category mapping"""
    status_to_category_map = {}
    for status in SCHEDULED_STOPS:
        status_to_category_map[status] = "Scheduled Stop"
    for status in PRODUCTION_STATUS:
        status_to_category_map[status] = "Production"
    for status in UNEXPECTED_STOPS:
        status_to_category_map[status] = "Unexpected Stop"
    for status in NO_PLAN:
        status_to_category_map[status] = "No Plan"
    for status in END_OPS:
        status_to_category_map[status] = "End Of Operations"
    return status_to_category_map

# Deep Learning Model Building Functions
import hashlib
import json
from pathlib import Path

def get_data_fingerprint(data_1d):
    """Create a hash fingerprint of the data to detect changes"""
    data_str = str(data_1d.tolist())
    return hashlib.md5(data_str.encode()).hexdigest()[:8]

def get_model_filename(model_name, source_name, look_back, use_padding, target_padded_length, data_fingerprint):
    """Generate standardized model filename"""
    padding_info = f"_pad{target_padded_length}" if use_padding else "_nopad"
    return f"model_{model_name}_{source_name}_lb{look_back}{padding_info}_{data_fingerprint}.h5"

def save_model_with_metadata(model, model_path, metadata):
    """Save model and its metadata"""
    try:
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        full_model_path = models_dir / model_path
        metadata_path = models_dir / f"{model_path.replace('.h5', '_metadata.json')}"
        
        # Save model
        model.save(str(full_model_path))
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        st.warning(f"Could not save model: {e}")
        return False

def load_model_with_metadata(model_path):
    """Load model and its metadata"""
    try:
        models_dir = Path("models")
        full_model_path = models_dir / model_path
        metadata_path = models_dir / f"{model_path.replace('.h5', '_metadata.json')}"
        
        if not full_model_path.exists() or not metadata_path.exists():
            return None, None
        
        # Load model
        from tensorflow.keras.models import load_model
        model = load_model(str(full_model_path))
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None, None

def is_model_current(metadata, data_fingerprint, max_age_hours=24):
    """Check if saved model is still current"""
    try:
        # Check data fingerprint
        if metadata.get('data_fingerprint') != data_fingerprint:
            return False
        
        # Check age
        from datetime import datetime, timedelta
        saved_time = datetime.fromisoformat(metadata.get('timestamp', '2020-01-01'))
        if datetime.now() - saved_time > timedelta(hours=max_age_hours):
            return False
        
        return True
    except:
        return False

def mean_absolute_percentage_error_safe(y_true, y_pred, epsilon=1e-8):
    """Calculate MAPE with safe division to avoid division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), epsilon)))) * 100

# Configuration constants (matching notebook)
FORECAST_HORIZON = 3  # Multi-step prediction as in notebook

def create_sequences(data_1d, look_back, forecast_horizon=1):
    """
    Create sequences for time series forecasting.
    MATCHES NOTEBOOK: Supports multi-step forecasting
    
    Args:
        data_1d: 1D array of time series data
        look_back: Number of previous time steps to use as input
        forecast_horizon: Number of steps to predict ahead
    
    Returns:
        X: Input sequences [samples, look_back, 1]
        y: Target values [samples] or [samples, forecast_horizon]
    """
    X, y = [], []
    if len(data_1d) <= look_back + forecast_horizon - 1:
        return np.array(X), np.array(y)
    
    for i in range(len(data_1d) - look_back - forecast_horizon + 1):
        input_seq = data_1d[i:(i + look_back)]
        output_val = data_1d[i + look_back : i + look_back + forecast_horizon]
        X.append(input_seq)
        y.append(output_val)
    
    X = np.array(X)
    if X.ndim == 2 and X.size > 0:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
    y = np.array(y)
    # For multi-step prediction, keep y as 2D [samples, forecast_horizon]
    # For single-step, flatten to 1D [samples]
    if forecast_horizon == 1 and y.ndim > 1 and y.size > 0:
        y = y.ravel()
    
    return X, y

def build_stacked_simplernn_with_masking(input_shape_padded, units_list=[64, 32], dropout_rate=0.25):
    """
    Builds a stacked SimpleRNN model WITH a Masking layer for padded sequences.
    MATCHES NOTEBOOK: Multi-step prediction with FORECAST_HORIZON=3
    """
    try:
        # Validate input shape
        if not isinstance(input_shape_padded, (tuple, list)) or len(input_shape_padded) != 2:
            st.error(f"Expected 2D input shape (timesteps, features), got {input_shape_padded}")
            return None
        
        # Validate units list
        if not units_list or not all(isinstance(u, int) and u > 0 for u in units_list):
            st.error(f"Invalid units_list: {units_list}")
            return None
        
        model = Sequential(name=f"StackedSimpleRNN_Masking_Units{'_'.join(map(str, units_list))}")
        
        # Masking layer to handle padded sequences (ignores timesteps where all features are 0)
        model.add(Masking(mask_value=0., input_shape=input_shape_padded))
        
        # Stacked SimpleRNN layers
        for i, units in enumerate(units_list):
            is_last_rnn = (i == len(units_list) - 1)
            model.add(SimpleRNN(
                units,
                activation='tanh',
                return_sequences=not is_last_rnn
            ))
            model.add(Dropout(dropout_rate))
        
        # Output layer - MATCHES NOTEBOOK: Multi-step prediction
        model.add(Dense(FORECAST_HORIZON, activation='linear'))
        
        # Test that the model can be built properly
        try:
            model.build(input_shape=(None,) + input_shape_padded)
            return model
        except Exception as build_error:
            st.error(f"Error building model structure: {build_error}")
            return None
            
    except Exception as e:
        st.error(f"Error creating stacked RNN with masking: {e}")
        return None

def build_stacked_simplernn_no_masking(input_shape_padded, units_list=[64, 32], dropout_rate=0.2):
    """
    Builds a stacked SimpleRNN model WITHOUT a Masking layer.
    MATCHES NOTEBOOK: Multi-step prediction with FORECAST_HORIZON=3
    """
    model = Sequential(name=f"StackedSimpleRNN_NoMask_Units{'_'.join(map(str, units_list))}")
    
    # First SimpleRNN layer
    model.add(SimpleRNN(
        units_list[0],
        activation='tanh',
        input_shape=input_shape_padded,
        return_sequences=True if len(units_list) > 1 else False
    ))
    model.add(Dropout(dropout_rate))
    
    # Additional RNN layers
    for i in range(1, len(units_list)):
        is_last_rnn = (i == len(units_list) - 1)
        model.add(SimpleRNN(
            units_list[i],
            activation='tanh',
            return_sequences=not is_last_rnn
        ))
        model.add(Dropout(dropout_rate))
    
    # Output layer - MATCHES NOTEBOOK: Multi-step prediction
    model.add(Dense(FORECAST_HORIZON, activation='linear'))
    return model

def build_multi_kernel_cnn(input_shape):
    """
    Builds a CNN with multiple parallel convolutional towers using different kernel sizes.
    MATCHES NOTEBOOK: Single-step prediction
    """
    inputs = Input(shape=input_shape, name="MultiKernel_Input")
    towers_outputs = []
    
    # Tower 1: Small kernel (size 3)
    if input_shape[0] >= 3:
        tower_1 = Conv1D(filters=16, kernel_size=3, activation='relu', 
                        padding='causal', name="MK_Tower1_Conv")(inputs)
        tower_1 = GlobalAveragePooling1D(name="MK_Tower1_GAP")(tower_1)
        towers_outputs.append(tower_1)
    
    # Tower 2: Medium kernel (size 5)
    if input_shape[0] >= 5:
        tower_2 = Conv1D(filters=16, kernel_size=5, activation='relu', 
                        padding='causal', name="MK_Tower2_Conv")(inputs)
        tower_2 = GlobalAveragePooling1D(name="MK_Tower2_GAP")(tower_2)
        towers_outputs.append(tower_2)
    
    # Tower 3: Large kernel (size 7)
    if input_shape[0] >= 7:
        tower_3 = Conv1D(filters=16, kernel_size=7, activation='relu', 
                        padding='causal', name="MK_Tower3_Conv")(inputs)
        tower_3 = GlobalAveragePooling1D(name="MK_Tower3_GAP")(tower_3)
        towers_outputs.append(tower_3)
    
    # Fallback for very short sequences
    if not towers_outputs:
        kernel_s = min(3, input_shape[0])
        if kernel_s < 1:
            kernel_s = 1
        fallback = Conv1D(filters=16, kernel_size=kernel_s, activation='relu', 
                         padding='causal')(inputs)
        merged = GlobalAveragePooling1D()(fallback)
    elif len(towers_outputs) == 1:
        merged = towers_outputs[0]
    else:
        merged = Concatenate(name="MK_Concatenate_Towers")(towers_outputs)
    
    # Dense layers
    merged_dropout = Dropout(0.3, name="MK_Merged_Drop")(merged)
    dense_output = Dense(32, activation='relu', name="MK_Dense1")(merged_dropout)
    final_dropout = Dropout(0.3, name="MK_Final_Drop")(dense_output)
    # MATCHES NOTEBOOK: Single-step prediction for CNN
    outputs = Dense(1, name="MK_FinalOutput")(final_dropout)
    
    model = Model(inputs=inputs, outputs=outputs, name="MultiKernelCNN")
    return model

def build_wavenet_style_cnn(input_shape, 
                           n_conv_layers=6, 
                           base_filters=32, 
                           kernel_size=2, 
                           dense_units=16, 
                           dropout_rate=0.2):
    """
    Builds a WaveNet-style model with dilated convolutions.
    MATCHES NOTEBOOK: Multi-step prediction with FORECAST_HORIZON=3
    """
    model = Sequential(name="WaveNetStyle_DilatedCNN")
    
    # First layer
    model.add(Conv1D(filters=base_filters, 
                     kernel_size=kernel_size, 
                     dilation_rate=1, 
                     activation='relu', 
                     padding='causal',
                     input_shape=input_shape,
                     name="WN_Conv1_Dil1"))
    
    # Subsequent dilated layers
    for i in range(1, n_conv_layers):
        dilation_rate = 2 ** i  # 2, 4, 8, 16, 32
        model.add(Conv1D(filters=base_filters, 
                         kernel_size=kernel_size, 
                         dilation_rate=dilation_rate, 
                         activation='relu', 
                         padding='causal',
                         name=f"WN_Conv{i+1}_Dil{dilation_rate}"))
    
    # Final 1D convolution with kernel size 1
    model.add(Conv1D(filters=1, kernel_size=1, activation='linear', 
                     padding='causal', name="WN_Final_Conv1x1"))
    
    # Global pooling and dense layers
    model.add(GlobalAveragePooling1D(name="WN_GlobalAvgPool"))
    model.add(Dense(dense_units, activation='relu', name="WN_Dense"))
    model.add(Dropout(dropout_rate, name="WN_Dropout"))
    # MATCHES NOTEBOOK: Multi-step prediction
    model.add(Dense(FORECAST_HORIZON, activation='linear', name="WN_Output"))
    
    return model

def create_basic_forecast(data_1d, forecast_steps, method):
    """Create forecast using basic statistical methods (no TensorFlow required)."""
    try:
        if method == 'Moving Average':
            window_size = min(7, len(data_1d) // 2)
            if window_size < 1:
                window_size = 1
            avg_value = np.mean(data_1d[-window_size:])
            forecasts = np.full(forecast_steps, avg_value)
            
        elif method == 'Linear Trend':
            x = np.arange(len(data_1d))
            coeffs = np.polyfit(x, data_1d, 1)
            future_x = np.arange(len(data_1d), len(data_1d) + forecast_steps)
            forecasts = np.polyval(coeffs, future_x)
            
        elif method == 'Exponential Smoothing':
            alpha = 0.3
            s = data_1d[0]
            for i in range(1, len(data_1d)):
                s = alpha * data_1d[i] + (1 - alpha) * s
            forecasts = np.full(forecast_steps, s)
            
        else:
            return None
        
        forecasts = np.clip(forecasts, 0.0, 1.2)
        return forecasts
        
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None

def evaluate_model_performance(model_builder_func, model_name, data_1d, look_back, scaler_obj, 
                               epochs=30, batch_size=32, use_padding=False, target_padded_length=None, forecast_horizon=1):
    """
    Evaluate a model's performance using walk-forward validation.
    MATCHES NOTEBOOK: Uses multi-step training but evaluates first step only
    
    Returns:
        dict: Contains MAE, RMSE, MAPE metrics
    """
    if not DEEP_LEARNING_AVAILABLE:
        return None
        
    try:
        # Data preparation
        train_size = int(len(data_1d) * 0.7)
        val_size = int(len(data_1d) * 0.15)
        
        train_data = data_1d[:train_size]
        val_data = data_1d[train_size:train_size + val_size]
        test_data = data_1d[train_size + val_size:]
        
        if len(test_data) < 5:  # Need minimum test data
            return None
        
        # Scale data
        train_scaled = scaler_obj.transform(train_data.reshape(-1, 1)).flatten()
        val_scaled = scaler_obj.transform(val_data.reshape(-1, 1)).flatten()
        test_scaled = scaler_obj.transform(test_data.reshape(-1, 1)).flatten()
        
        # Walk-forward validation
        initial_history = np.concatenate([train_scaled, val_scaled])
        predictions_scaled = []
        actuals_scaled = []
        
        # Determine forecast horizon based on padding usage (better detection method)
        # RNN models use padding, CNN models don't
        model_forecast_horizon = forecast_horizon
        
        for i in range(min(len(test_scaled), 10)):  # Limit to 10 steps for performance
            current_train_window = np.concatenate([initial_history, test_scaled[:i]])
            
            # Create sequences with appropriate forecast horizon
            # Create sequences with appropriate forecast horizon for training
            X_train, y_train = create_sequences(current_train_window, look_back, model_forecast_horizon)
            
            if X_train.shape[0] < 5:  # Need minimum training samples
                continue
                
            # Handle padding if required
            if use_padding and target_padded_length:
                X_train_2d = np.squeeze(X_train, axis=-1) if X_train.ndim == 3 else X_train
                X_train_padded = pad_sequences(X_train_2d, maxlen=target_padded_length, 
                                             padding='pre', truncating='pre', dtype='float32', value=0.0)
                X_train = X_train_padded.reshape((X_train_padded.shape[0], target_padded_length, 1))
                input_shape = (target_padded_length, 1)
            else:
                input_shape = (look_back, 1)
            
            # Build and train model
            tf.keras.backend.clear_session()
            try:
                model = model_builder_func(input_shape)
                if model is None:
                    st.warning(f"Model builder returned None for {model_name}")
                    return None
                
                # Validate model structure
                model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])
                
                # Validate that model can handle the data
                test_pred = model.predict(X_train[:1], verbose=0)
                if test_pred is None:
                    st.warning(f"Model prediction test failed for {model_name}")
                    return None
                
                # Train with early stopping
                early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0)
                
                # More robust training with try-catch
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                callbacks=[early_stopping], verbose=0)
                
                if history is None or not hasattr(history, 'history'):
                    st.warning(f"Training failed for {model_name}")
                    return None
                    
            except Exception as e:
                st.warning(f"Error building/training {model_name}: {str(e)}")
                return None
            
            # Prepare prediction input
            if use_padding and target_padded_length:
                last_sequence = current_train_window[-look_back:].reshape(1, -1)
                last_sequence_padded = pad_sequences(last_sequence, maxlen=target_padded_length, 
                                                   padding='pre', truncating='pre', dtype='float32', value=0.0)
                pred_input = last_sequence_padded.reshape(1, target_padded_length, 1)
            else:
                pred_input = current_train_window[-look_back:].reshape(1, look_back, 1)
            
            # Make prediction - MATCHES NOTEBOOK: Take first step of multi-step prediction
            prediction_output = model.predict(pred_input, verbose=0)
            if model_forecast_horizon > 1:
                prediction_scaled = prediction_output[0, 0]  # First step of multi-step prediction
            else:
                prediction_scaled = prediction_output[0, 0]  # Single-step prediction
                
            predictions_scaled.append(prediction_scaled)
            actuals_scaled.append(test_scaled[i])
        
        if len(predictions_scaled) < 3:  # Need minimum predictions
            return None
            
        # Convert back to original scale
        predictions_scaled_arr = np.array(predictions_scaled).reshape(-1, 1)
        actuals_scaled_arr = np.array(actuals_scaled).reshape(-1, 1)
        
        predictions_original = scaler_obj.inverse_transform(predictions_scaled_arr).flatten()
        actuals_original = scaler_obj.inverse_transform(actuals_scaled_arr).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actuals_original, predictions_original)
        rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
        mape = mean_absolute_percentage_error_safe(actuals_original, predictions_original)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': predictions_original,
            'actuals': actuals_original,
            'model_name': model_name
        }
        
    except Exception as e:
        st.warning(f"Error evaluating {model_name}: {str(e)}")
        return None

def recommend_best_model(data_1d, scaler_obj, source_name, force_retrain=False):
    """
    Test all models and recommend the best one based on MAE.
    Now with model caching support.
    
    Returns:
        dict: Best model info and all results
    """
    if not DEEP_LEARNING_AVAILABLE:
        return None, {}
    
    models_to_test = [
        {
            'name': 'Stacked RNN (Masking)',
            'builder': lambda input_shape: build_stacked_simplernn_with_masking(input_shape, [64, 32], 0.25),
            'look_back': 14,
            'use_padding': True,
            'target_padded_length': 20,
            'forecast_horizon': 3
        },
        {
            'name': 'Stacked RNN (No Masking)',
            'builder': lambda input_shape: build_stacked_simplernn_no_masking(input_shape, [64, 32], 0.2),
            'look_back': 7,
            'use_padding': True,
            'target_padded_length': 35,
            'forecast_horizon': 3
        },
        {
            'name': 'Multi-Kernel CNN',
            'builder': build_multi_kernel_cnn,
            'look_back': 30,
            'use_padding': False,
            'target_padded_length': None,
            'forecast_horizon': 1
        },
        {
            'name': 'WaveNet CNN',
            'builder': lambda input_shape: build_wavenet_style_cnn(input_shape, 2, 32, 2, 16, 0.178),
            'look_back': 14,
            'use_padding': False,
            'target_padded_length': None,
            'forecast_horizon': 3
        }
    ]
    
    results = {}
    best_model = None
    best_mae = float('inf')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_config in enumerate(models_to_test):
        status_text.text(f"Testing {model_config['name']} for {source_name}...")
        
        # Check if we have enough data for this look_back
        if len(data_1d) < model_config['look_back'] + 10:
            continue
            
        # Use the original evaluation function for now (simpler approach)
        result = evaluate_model_performance(
            model_builder_func=model_config['builder'],
            model_name=model_config['name'],
            data_1d=data_1d,
            look_back=model_config['look_back'],
            scaler_obj=scaler_obj,
            epochs=20,
            batch_size=16,
            use_padding=model_config['use_padding'],
            target_padded_length=model_config['target_padded_length'],
            forecast_horizon=model_config['forecast_horizon']
        )
        
        if result and result['mae'] < best_mae:
            best_mae = result['mae']
            best_model = model_config.copy()
            best_model.update(result)
        
        if result:
            results[model_config['name']] = result
            
        progress_bar.progress((i + 1) / len(models_to_test))
    
    status_text.text("Model evaluation complete!")
    progress_bar.empty()
    status_text.empty()
    
    return best_model, results

def create_forecast(model_builder_func, data_1d, scaler_obj, look_back, forecast_steps=7,
                   use_padding=False, target_padded_length=None, epochs=50):
    """
    Create forecast using the selected model.
    MATCHES NOTEBOOK: Uses multi-step training but extracts appropriate predictions
    
    Returns:
        numpy.array: Forecasted values in original scale
    """
    if not DEEP_LEARNING_AVAILABLE:
        return None
        
    try:
        # Check if data is constant (would cause scaling issues)
        if np.std(data_1d) < 1e-8:
            st.error("❌ Data has no variation - cannot forecast constant values")
            return None
        
        # Scale all data
        scaled_data = scaler_obj.transform(data_1d.reshape(-1, 1)).flatten()
        
        # Determine forecast horizon based on padding usage (better detection method)
        # RNN models use padding, CNN models don't
        model_forecast_horizon = FORECAST_HORIZON if use_padding else 1
        
        # Create training sequences with appropriate forecast horizon
        X_train, y_train = create_sequences(scaled_data, look_back, model_forecast_horizon)
        
        if X_train.shape[0] < 5:
            st.error("❌ Not enough training sequences created")
            return None
        
        # Handle padding if required
        if use_padding and target_padded_length:
            X_train_2d = np.squeeze(X_train, axis=-1) if X_train.ndim == 3 else X_train
            X_train_padded = pad_sequences(X_train_2d, maxlen=target_padded_length, 
                                         padding='pre', truncating='pre', dtype='float32', value=0.0)
            X_train = X_train_padded.reshape((X_train_padded.shape[0], target_padded_length, 1))
            input_shape = (target_padded_length, 1)
        else:
            input_shape = (look_back, 1)
        
        # Build and train model
        tf.keras.backend.clear_session()
        model = model_builder_func(input_shape)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])
        
        # Train model
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)
        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                           callbacks=[early_stopping, reduce_lr], verbose=0)
        
        # Test model with one prediction
        test_pred = model.predict(X_train[:1], verbose=0)
        
        if np.isnan(test_pred).any():
            st.error("❌ Model produced NaN predictions - training failed")
            return None
        
        # Generate forecasts
        forecasts_scaled = []
        current_sequence = scaled_data[-look_back:].copy()
        
        for step in range(forecast_steps):
            # Prepare input
            if use_padding and target_padded_length:
                input_seq = current_sequence.reshape(1, -1)
                input_padded = pad_sequences(input_seq, maxlen=target_padded_length, 
                                           padding='pre', truncating='pre', dtype='float32', value=0.0)
                pred_input = input_padded.reshape(1, target_padded_length, 1)
            else:
                pred_input = current_sequence.reshape(1, look_back, 1)
            
            # Make prediction
            prediction_output = model.predict(pred_input, verbose=0)
            
            if model_forecast_horizon > 1:
                # For multi-step models, use first prediction for one-step-ahead forecasting
                next_pred_scaled = prediction_output[0, 0]
            else:
                # For single-step models, use the prediction directly
                next_pred_scaled = prediction_output[0, 0]
            
            if np.isnan(next_pred_scaled):
                st.error(f"❌ NaN prediction at step {step}")
                return None
                
            forecasts_scaled.append(next_pred_scaled)
            
            # Update sequence for next prediction (append new prediction, remove oldest)
            current_sequence = np.append(current_sequence[1:], next_pred_scaled)
        
        # Convert back to original scale
        forecasts_scaled_arr = np.array(forecasts_scaled).reshape(-1, 1)
        forecasts_original = scaler_obj.inverse_transform(forecasts_scaled_arr).flatten()
        
        return forecasts_original
        
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None

@st.cache_data
def check_processed_files():
    """Check if processed files exist"""
    required_files = [
        'line_status_cleaned_final.csv',
        'daily_oee_report.csv',
        'overall_daily_oee.csv'
    ]
    
    line_files = []
    for line in ['LINE-01', 'LINE-03', 'LINE-04', 'LINE-06']:
        line_file = f'daily_oee_report_{line}.csv'
        line_files.append(line_file)
    
    all_files = required_files + line_files
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    return len(existing_files) == len(all_files), existing_files

@st.cache_data
def load_raw_data():
    """Load raw data files"""
    try:
        df_ls = pd.read_csv('line_status_notcleaned.csv')
        df_prd = pd.read_csv('production_data.csv')
        return df_ls, df_prd, True
    except FileNotFoundError as e:
        st.error(f"Required input files not found: {e}")
        return None, None, False

def preprocess_line_status(df_ls):
    """Preprocess line status data following notebook logic"""
    with st.spinner("Preprocessing line status data..."):
        df_ls = df_ls.sort_index()
        df_ls['START_DATETIME'] = pd.to_datetime(df_ls['START_DATETIME'])
        df_ls = df_ls.set_index('START_DATETIME')
        
        columns_to_drop = ['STATUS_NM', 'STATUS', 'Unnamed: 8'] if 'Unnamed: 8' in df_ls.columns else ['STATUS_NM', 'STATUS']
        if 'DURATION' in df_ls.columns:
            columns_to_drop.append('DURATION')
        df_ls = df_ls.drop(columns_to_drop, axis=1, errors='ignore')
        
        if 'IS_DELETED' in df_ls.columns:
            df_ls = df_ls[df_ls['IS_DELETED'] != 1]
            df_ls = df_ls.drop(columns=['IS_DELETED'], axis=1)
        
        if 'FINISH_DATETIME' in df_ls.columns:
            df_ls = df_ls.drop(['FINISH_DATETIME'], axis=1)
        
        finish_datetime = []
        for i in range(len(df_ls)):
            if df_ls.iloc[i]['STATUS_NAME'] == 'End Of Operations':
                finish_datetime.append(df_ls.index[i])
            elif i < len(df_ls) - 1:
                finish_datetime.append(df_ls.index[i + 1])
            else:
                finish_datetime.append(pd.NaT)
        
        df_ls['FINISH_DATETIME'] = finish_datetime
        df_ls = df_ls.drop_duplicates()
        
        return df_ls

def preprocess_production_data(df_prd):
    """Preprocess production data"""
    with st.spinner("Preprocessing production data..."):
        df_prd = df_prd.sort_index()
        df_prd['FINISH_DATETIME'] = pd.to_datetime(df_prd['FINISH_DATETIME'])
        df_prd['START_DATETIME'] = pd.to_datetime(df_prd['START_DATETIME'])
        df_prd['month'] = df_prd['FINISH_DATETIME'].dt.month
        df_prd['Date'] = df_prd['FINISH_DATETIME'].dt.date
        return df_prd

def calculate_oee(df_ls, df_prd):
    """Calculate OEE metrics following notebook logic"""
    with st.spinner("Calculating OEE metrics..."):
        df_ls['FINISH_DATETIME'] = pd.to_datetime(df_ls['FINISH_DATETIME'])
        df_ls['DURATION'] = df_ls['FINISH_DATETIME'] - df_ls.index
        df_ls['Duration_Seconds'] = df_ls['DURATION'].dt.total_seconds()
        df_ls['Date'] = df_ls.index.date
        
        planned_statuses = ['Production', 'Scheduled Stop', 'Unexpected Stop']
        df_planned_time = df_ls[df_ls['STATUS_NAME'].isin(planned_statuses)]
        daily_planned_time = df_planned_time.groupby(['PRODUCTION_LINE', 'Date'])['Duration_Seconds'].sum().reset_index()
        daily_planned_time = daily_planned_time.rename(columns={'Duration_Seconds': 'Planned_Production_Time_Seconds'})
        
        df_run_time = df_ls[df_ls['STATUS_NAME'] == 'Production']
        daily_run_time = df_run_time.groupby(['PRODUCTION_LINE', 'Date'])['Duration_Seconds'].sum().reset_index()
        daily_run_time = daily_run_time.rename(columns={'Duration_Seconds': 'Actual_Run_Time_Seconds'})
        
        daily_times = pd.merge(daily_planned_time, daily_run_time, on=['PRODUCTION_LINE', 'Date'], how='outer')
        daily_times['Planned_Production_Time_Seconds'] = daily_times['Planned_Production_Time_Seconds'].fillna(0)
        daily_times['Actual_Run_Time_Seconds'] = daily_times['Actual_Run_Time_Seconds'].fillna(0)
        
        daily_output = df_prd.groupby(['LINE', 'Date']).size().reset_index(name='Total_Actual_Output')
        daily_output = daily_output.rename(columns={'LINE': 'PRODUCTION_LINE'})
        
        daily_oee_data = pd.merge(daily_times, daily_output, on=['PRODUCTION_LINE', 'Date'], how='outer')
        daily_oee_data['Planned_Production_Time_Seconds'] = daily_oee_data['Planned_Production_Time_Seconds'].fillna(0)
        daily_oee_data['Actual_Run_Time_Seconds'] = daily_oee_data['Actual_Run_Time_Seconds'].fillna(0)
        daily_oee_data['Total_Actual_Output'] = daily_oee_data['Total_Actual_Output'].fillna(0).astype(int)
        
        daily_oee_data['Ideal_Cycle_Time_Seconds'] = daily_oee_data['PRODUCTION_LINE'].map(CYCLE_TIMES)
        
        daily_oee_data['Availability'] = np.where(
            daily_oee_data['Planned_Production_Time_Seconds'] > 0,
            daily_oee_data['Actual_Run_Time_Seconds'] / daily_oee_data['Planned_Production_Time_Seconds'],
            0
        )
        
        daily_oee_data['Performance'] = np.where(
            (daily_oee_data['Actual_Run_Time_Seconds'] > 0) & (daily_oee_data['Ideal_Cycle_Time_Seconds'].notna()),
            (daily_oee_data['Total_Actual_Output'] * daily_oee_data['Ideal_Cycle_Time_Seconds']) / daily_oee_data['Actual_Run_Time_Seconds'],
            0
        )
        
        daily_oee_data['Quality'] = np.where(daily_oee_data['Total_Actual_Output'] > 0, 1.0, 0)
        daily_oee_data['OEE'] = daily_oee_data['Availability'] * daily_oee_data['Performance'] * daily_oee_data['Quality']
        
        daily_oee_data['Date'] = pd.to_datetime(daily_oee_data['Date'])
        daily_oee_data = daily_oee_data.sort_values(by=['PRODUCTION_LINE', 'Date']).reset_index(drop=True)
        
        return daily_oee_data

def save_processed_data(df_ls, daily_oee_data):
    """Save processed data to files"""
    with st.spinner("Saving processed data..."):
        df_ls.to_csv('line_status_cleaned_final.csv', index=True)
        daily_oee_data.to_csv('daily_oee_report.csv', index=False)
        
        unique_lines = daily_oee_data['PRODUCTION_LINE'].unique()
        for line in unique_lines:
            df_line = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == line].copy()
            df_line.to_csv(f'daily_oee_report_{line}.csv', index=False)
        
        grouped = daily_oee_data.groupby('Date').agg({
            'Planned_Production_Time_Seconds': 'sum',
            'Actual_Run_Time_Seconds': 'sum',
            'Total_Actual_Output': 'sum',
            'Ideal_Cycle_Time_Seconds': 'mean'
        }).reset_index()
        
        grouped['Availability'] = grouped['Actual_Run_Time_Seconds'] / grouped['Planned_Production_Time_Seconds']
        grouped['Performance'] = (grouped['Ideal_Cycle_Time_Seconds'] * grouped['Total_Actual_Output']) / grouped['Actual_Run_Time_Seconds']
        grouped['Quality'] = 1.0
        grouped['OEE'] = grouped['Availability'] * grouped['Performance'] * grouped['Quality']
        
        overall_daily_oee = grouped[['Date', 'Availability', 'Performance', 'Quality', 'OEE']]
        overall_daily_oee.to_csv("overall_daily_oee.csv", index=False)

@st.cache_data
def load_processed_data():
    """Load processed OEE data"""
    daily_oee_data = pd.read_csv('daily_oee_report.csv')
    daily_oee_data['Date'] = pd.to_datetime(daily_oee_data['Date'])
    
    overall_daily_oee = pd.read_csv('overall_daily_oee.csv')
    overall_daily_oee['Date'] = pd.to_datetime(overall_daily_oee['Date'])
    
    return daily_oee_data, overall_daily_oee

def create_oee_trend_chart(data, line=None, title_suffix=""):
    """Create OEE trend chart using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['OEE'], mode='lines+markers', name='OEE',
        line=dict(color='#1f77b4', width=3), marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Availability'], mode='lines+markers', name='Availability',
        line=dict(color='#ff7f0e', width=2, dash='dash'), marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Performance'], mode='lines+markers', name='Performance',
        line=dict(color='#2ca02c', width=2, dash='dash'), marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f'OEE and Components Trend {title_suffix}',
        xaxis_title='Date', yaxis_title='Percentage',
        yaxis=dict(tickformat=',.0%', range=[0, 1.1]),
        hovermode='x unified', height=500
    )
    
    return fig

def create_avg_oee_chart(daily_oee_data):
    """Create average OEE comparison chart"""
    avg_oee = daily_oee_data.groupby('PRODUCTION_LINE')[['OEE', 'Availability', 'Performance', 'Quality']].mean().reset_index()
    
    fig = go.Figure()
    metrics = ['OEE', 'Availability', 'Performance', 'Quality']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(name=metric, x=avg_oee['PRODUCTION_LINE'], y=avg_oee[metric], marker_color=colors[i]))
    
    fig.update_layout(
        title='Average OEE and Components per Production Line',
        xaxis_title='Production Line', yaxis_title='Average Value',
        yaxis=dict(tickformat=',.0%', range=[0, 1.1]), barmode='group', height=500
    )
    
    return fig

def get_line_current_status(line, daily_oee_data):
    """Get current status and latest OEE for a production line"""
    line_data = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == line]
    if line_data.empty:
        return "No Data", 0.0, "🔴"
    
    latest_data = line_data.loc[line_data['Date'].idxmax()]
    latest_oee = latest_data['OEE']
    
    if latest_oee >= 0.85:
        status, icon = "Excellent", "🟢"
    elif latest_oee >= 0.70:
        status, icon = "Good", "🟡"
    elif latest_oee >= 0.50:
        status, icon = "Fair", "🟠"
    else:
        status, icon = "Poor", "🔴"
    
    return status, latest_oee, icon

def create_comparison_chart(daily_oee_data, metric):
    """Create comparison chart for selected metric"""
    avg_data = daily_oee_data.groupby('PRODUCTION_LINE')[metric].mean().reset_index()
    avg_data = avg_data.sort_values(metric, ascending=False)
    
    colors = []
    for value in avg_data[metric]:
        if value >= 0.85:
            colors.append('#2E8B57')
        elif value >= 0.70:
            colors.append('#FFD700')
        elif value >= 0.50:
            colors.append('#FF8C00')
        else:
            colors.append('#DC143C')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_data['PRODUCTION_LINE'], y=avg_data[metric], marker_color=colors,
        text=[f"{val:.1%}" for val in avg_data[metric]], textposition='auto'
    ))
    
    fig.update_layout(
        title=f'{metric} Comparison Across Production Lines',
        xaxis_title='Production Line', yaxis_title=f'Average {metric}',
        yaxis=dict(tickformat=',.0%', range=[0, max(1.1, avg_data[metric].max() * 1.1)]),
        height=400
    )
    
    return fig

def create_ranking_table(daily_oee_data, metric):
    """Create ranking table for selected metric"""
    avg_data = daily_oee_data.groupby('PRODUCTION_LINE').agg({
        'OEE': 'mean', 'Availability': 'mean', 'Performance': 'mean', 'Quality': 'mean', 'Total_Actual_Output': 'sum'
    }).round(4)
    
    avg_data = avg_data.sort_values(metric, ascending=False)
    avg_data['Rank'] = range(1, len(avg_data) + 1)
    
    for col in ['OEE', 'Availability', 'Performance', 'Quality']:
        avg_data[f'{col}_formatted'] = avg_data[col].apply(lambda x: f"{x:.1%}")
    
    display_cols = ['Rank', 'OEE_formatted', 'Availability_formatted', 'Performance_formatted', 'Quality_formatted', 'Total_Actual_Output']
    display_df = avg_data[display_cols].copy()
    display_df.columns = ['Rank', 'OEE', 'Availability', 'Performance', 'Quality', 'Total Output']
    
    return display_df

def show_basic_forecasting(daily_oee_data, overall_daily_oee):
    """Show basic forecasting using simple statistical methods"""
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
            "Method:", options=['Moving Average', 'Linear Trend', 'Exponential Smoothing'], key="basic_method"
        )
    
    if st.button("📊 Generate Basic Forecast", use_container_width=True):
        if forecast_target == 'Overall Daily OEE':
            data_source = overall_daily_oee.copy()
        else:
            line_name = forecast_target.replace('Line: ', '')
            data_source = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == line_name].copy()
        
        if data_source.empty:
            st.error("No data available for the selected target.")
            return
        
        data_source = data_source.sort_values('Date').reset_index(drop=True)
        oee_values = data_source['OEE'].values
        dates = pd.to_datetime(data_source['Date'].values)
        
        if len(oee_values) < 5:
            st.error("Need at least 5 data points for forecasting.")
            return
        
        forecasts = create_basic_forecast(oee_values, forecast_days, method)
        
        if forecasts is not None:
            last_date = pd.to_datetime(dates[-1])
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=forecast_days, 
                freq='D'
            )
            
            fig = go.Figure()
            
            historical_show = min(30, len(oee_values))
            fig.add_trace(go.Scatter(
                x=dates[-historical_show:], y=oee_values[-historical_show:],
                mode='lines+markers', name='Historical OEE',
                line=dict(color='blue', width=2), marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecasts,
                mode='lines+markers', name=f'Forecast ({method})',
                line=dict(color='red', width=2, dash='dash'), marker=dict(size=6, symbol='diamond')
            ))
            
            fig.add_vline(x=dates[-1], line_dash="dash", line_color="gray", annotation_text="Forecast Start")
            
            fig.update_layout(
                title=f'Basic OEE Forecast for {forecast_target} ({method})',
                xaxis_title='Date', yaxis_title='OEE',
                yaxis=dict(tickformat=',.0%'), hovermode='x unified', height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = np.mean(forecasts)
                st.metric("Average Forecast OEE", f"{avg_forecast:.1%}")
            
            with col2:
                last_historical = oee_values[-1]
                change = avg_forecast - last_historical
                st.metric("Change from Current", f"{change:+.1%}")
            
            with col3:
                forecast_range = np.max(forecasts) - np.min(forecasts)
                st.metric("Forecast Range", f"{forecast_range:.1%}")
            
            st.markdown("### 📊 Forecast Details")
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Forecasted OEE': [f"{v:.1%}" for v in forecasts],
                'Method': [method] * len(forecasts)
            })
            
            st.dataframe(forecast_df, hide_index=True, use_container_width=True)

def show_forecasting_page(daily_oee_data, overall_daily_oee):
    """Show OEE forecasting page with model selection and recommendations"""
    st.header("🔮 OEE Forecasting with Deep Learning")
    
    # Show diagnostic information
    with st.expander("🔍 Deep Learning Libraries Diagnostic", expanded=not DEEP_LEARNING_AVAILABLE):
        st.markdown("**Import Status:**")
        for detail in IMPORT_ERROR_DETAILS:
            st.markdown(f"- {detail}")
        
        # Show Python environment info
        import sys
        st.markdown(f"**Python executable:** `{sys.executable}`")
        st.markdown(f"**Python version:** `{sys.version}`")
        st.markdown(f"**Platform:** `{sys.platform}`")
        
        # Show installed packages
        try:
            import pkg_resources
            installed_packages = [d.project_name for d in pkg_resources.working_set]
            tensorflow_installed = any('tensorflow' in pkg.lower() for pkg in installed_packages)
            sklearn_installed = any('scikit' in pkg.lower() or 'sklearn' in pkg.lower() for pkg in installed_packages)
            
            st.markdown("**Package Detection:**")
            st.markdown(f"- TensorFlow detected: {'✅ Yes' if tensorflow_installed else '❌ No'}")
            st.markdown(f"- Scikit-learn detected: {'✅ Yes' if sklearn_installed else '❌ No'}")
            
        except Exception as e:
            st.error(f"Could not check installed packages: {e}")
    
    if not DEEP_LEARNING_AVAILABLE:
        if TENSORFLOW_DLL_ERROR:
            st.error("❌ TensorFlow DLL loading failed. This is a common Windows issue.")
            
            st.markdown("### 🪟 Windows TensorFlow Fix:")
            
            st.markdown("**🔧 Solution 1: Install Microsoft Visual C++ Redistributable**")
            st.markdown("This is the most common fix for Windows TensorFlow DLL issues:")
            st.markdown("1. Download and install [Microsoft Visual C++ Redistributable](https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)")
            st.markdown("2. Restart your computer")
            st.markdown("3. Restart Streamlit")
            
            st.markdown("**🔧 Solution 2: Use TensorFlow CPU version**")
            st.code("""# Uninstall current TensorFlow
pip uninstall tensorflow
# Install CPU-only version
pip install tensorflow-cpu""", language="bash")
            
            st.markdown("**🔧 Solution 3: Use conda instead of pip**")
            st.code("""# If you have conda installed
conda install tensorflow
# Or create new conda environment
conda create -n tf_env python=3.9 tensorflow scikit-learn streamlit pandas plotly
conda activate tf_env""", language="bash")
            
        else:
            st.error("❌ Deep learning libraries are not working correctly.")
        
        # Offer basic forecasting as alternative
        st.markdown("---")
        st.info("💡 **Alternative:** While we fix TensorFlow, you can still use basic forecasting methods below!")
        
        show_basic_forecasting(daily_oee_data, overall_daily_oee)
        return
    
    # Initialize session state for forecasting
    if 'forecasting_results' not in st.session_state:
        st.session_state.forecasting_results = {}
    if 'model_recommendations' not in st.session_state:
        st.session_state.model_recommendations = {}
    
    # Forecasting configuration
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
    
    st.divider()
    
    # Model Selection Section
    st.subheader("🤖 Model Selection")
    
    model_options = {
        "Stacked RNN with Masking": {
            'builder': lambda input_shape: build_stacked_simplernn_with_masking(input_shape, [64, 32], 0.25),
            'look_back': 14,
            'use_padding': True,
            'target_padded_length': 20,
            'description': "RNN with masking layer, LB=14, Padded to 20. Good for sequences with missing data."
        },
        "Stacked RNN without Masking": {
            'builder': lambda input_shape: build_stacked_simplernn_no_masking(input_shape, [64, 32], 0.2),
            'look_back': 7,
            'use_padding': True,
            'target_padded_length': 35,
            'description': "Standard RNN, LB=7, Padded to 35. Fast training, good baseline performance."
        },
        "Multi-Kernel CNN": {
            'builder': build_multi_kernel_cnn,
            'look_back': 30,
            'use_padding': False,
            'target_padded_length': None,
            'description': "CNN with multiple kernel sizes, LB=30. Captures different time patterns."
        },
        "WaveNet-style CNN": {
            'builder': lambda input_shape: build_wavenet_style_cnn(input_shape, 2, 32, 2, 16, 0.178),
            'look_back': 14,
            'use_padding': False,
            'target_padded_length': None,
            'description': "Dilated CNN, LB=14. Advanced architecture for complex patterns."
        }
    }
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        selected_model = st.selectbox(
            "Choose Model:",
            options=list(model_options.keys()),
            key="selected_model"
        )
        
        st.info(f"📝 **{selected_model}**\n\n{model_options[selected_model]['description']}")
        
        # Model Recommendation Button
        st.markdown("### 🎯 Get Model Recommendation")
        if st.button("🔍 Find Best Model for This Data", use_container_width=True):
            with st.spinner("Testing all models to find the best one..."):
                # Prepare data based on target
                if forecast_target == 'Overall Daily OEE':
                    data_source = overall_daily_oee.copy()
                    source_name = 'Overall'
                else:
                    line_name = forecast_target.replace('Line: ', '')
                    data_source = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == line_name].copy()
                    source_name = line_name
                
                if data_source.empty:
                    st.error("No data available for the selected target.")
                else:
                    # Prepare data for modeling
                    data_source = data_source.sort_values('Date').reset_index(drop=True)
                    oee_values = data_source['OEE'].values
                    
                    if len(oee_values) < 50:
                        st.warning("⚠️ Limited data available. Recommendations may not be reliable.")
                    
                    # Fit scaler
                    scaler = RobustScaler()
                    scaler.fit(oee_values.reshape(-1, 1))
                    
                    # Get recommendations
                    best_model, all_results = recommend_best_model(oee_values, scaler, source_name)
                    
                    if best_model:
                        st.session_state.model_recommendations[forecast_target] = {
                            'best_model': best_model,
                            'all_results': all_results,
                            'scaler': scaler,
                            'data': oee_values
                        }
                        st.success(f"✅ Best model found: **{best_model['name']}** (MAE: {best_model['mae']:.4f})")
                    else:
                        st.error("❌ Could not evaluate models. Please check your data.")
    
    with col2:
        # Show model recommendations if available
        if forecast_target in st.session_state.model_recommendations:
            rec_data = st.session_state.model_recommendations[forecast_target]
            
            st.markdown("### 🏆 Model Performance Comparison")
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, results in rec_data['all_results'].items():
                if results:
                    comparison_data.append({
                        'Model': model_name,
                        'MAE': f"{results['mae']:.4f}",
                        'RMSE': f"{results['rmse']:.4f}",
                        'MAPE': f"{results['mape']:.2f}%"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Highlight best model
                def highlight_best(s):
                    best_model_name = rec_data['best_model']['name']
                    return ['background-color: lightgreen' if s['Model'] == best_model_name else '' for _ in s]
                
                styled_df = comparison_df.style.apply(highlight_best, axis=1)
                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                
                # Quick select best model button
                if st.button("🎯 Use Recommended Model", use_container_width=True):
                    best_model_name = rec_data['best_model']['name']
                    st.success(f"✅ Recommended model: **{best_model_name}**")
                    st.info("💡 Please select the recommended model from the dropdown above.")
            else:
                st.info("No model comparison data available.")
    
    st.divider()
    
    # Forecasting Section
    st.subheader("🚀 Generate Forecast")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("📈 Create Forecast", use_container_width=True, type="primary"):
            with st.spinner("Training model and generating forecast..."):
                # Prepare data
                if forecast_target == 'Overall Daily OEE':
                    data_source = overall_daily_oee.copy()
                    source_name = 'Overall'
                else:
                    line_name = forecast_target.replace('Line: ', '')
                    data_source = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == line_name].copy()
                    source_name = line_name
                
                if data_source.empty:
                    st.error("No data available for the selected target.")
                else:
                    # Prepare data for modeling
                    data_source = data_source.sort_values('Date').reset_index(drop=True)
                    oee_values = data_source['OEE'].values
                    dates = data_source['Date'].values
                    
                    # Fit scaler
                    scaler = RobustScaler()
                    scaler.fit(oee_values.reshape(-1, 1))
                    
                    # Get model configuration
                    model_config = model_options[selected_model]
                    
                    # Check if we have enough data
                    if len(oee_values) < model_config['look_back'] + 10:
                        st.error(f"❌ Not enough data. Need at least {model_config['look_back'] + 10} data points, but only have {len(oee_values)}.")
                    else:
                        # Create forecast
                        forecast_values = create_forecast(
                            model_builder_func=model_config['builder'],
                            data_1d=oee_values,
                            scaler_obj=scaler,
                            look_back=model_config['look_back'],
                            forecast_steps=forecast_days,
                            use_padding=model_config['use_padding'],
                            target_padded_length=model_config['target_padded_length'],
                            epochs=training_epochs
                        )
                        
                        if forecast_values is not None:
                            # Generate future dates
                            last_date = pd.to_datetime(dates[-1])
                            future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                            
                            # Store results
                            st.session_state.forecasting_results[forecast_target] = {
                                'model_name': selected_model,
                                'forecast_values': forecast_values,
                                'future_dates': future_dates,
                                'historical_data': oee_values,
                                'historical_dates': dates,
                                'forecast_days': forecast_days
                            }
                            
                            st.success(f"✅ Forecast generated successfully using {selected_model}!")
                        else:
                            st.error("❌ Failed to generate forecast. Please try a different model or check your data.")
    
    with col2:
        # Display forecast results
        if forecast_target in st.session_state.forecasting_results:
            forecast_data = st.session_state.forecasting_results[forecast_target]
            
            # Create forecast visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(forecast_data['historical_dates'][-30:]),  # Show last 30 days
                y=forecast_data['historical_data'][-30:],
                mode='lines+markers',
                name='Historical OEE',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast_data['future_dates'],
                y=forecast_data['forecast_values'],
                mode='lines+markers',
                name=f'Forecast ({forecast_data["model_name"]})',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            
            fig.update_layout(
                title=f'OEE Forecast for {forecast_target}',
                xaxis_title='Date',
                yaxis_title='OEE',
                yaxis=dict(tickformat=',.0%'),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
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
            
            # Forecast table
            st.markdown("### 📊 Detailed Forecast")
            forecast_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in forecast_data['future_dates']],
                'Forecasted OEE': [f"{v:.1%}" for v in forecast_data['forecast_values']],
                'Day': [f"Day +{i+1}" for i in range(len(forecast_data['forecast_values']))]
            })
            
            st.dataframe(forecast_df, hide_index=True, use_container_width=True)
    
    # Additional Information
    st.divider()
    st.markdown("### 📚 Model Information")
    
    with st.expander("🔍 How the Models Work"):
        st.markdown("""
        **Stacked RNN with Masking:**
        - Uses recurrent neural networks with masking to handle variable-length sequences
        - Good for data with missing values or irregular intervals
        - Look-back window: 14 days, padded to 20
        
        **Stacked RNN without Masking:**
        - Standard RNN approach, faster training
        - Good baseline performance for regular time series
        - Look-back window: 7 days, padded to 35
        
        **Multi-Kernel CNN:**
        - Uses multiple convolutional filters with different kernel sizes
        - Captures patterns at different time scales
        - Look-back window: 30 days, no padding
        
        **WaveNet-style CNN:**
        - Advanced dilated convolutional architecture
        - Can capture long-range dependencies
        - Look-back window: 14 days, no padding
        """)
    
    with st.expander("⚠️ Important Notes"):
        st.markdown("""
        - **Data Requirements:** Models need sufficient historical data to train effectively
        - **Forecast Accuracy:** Longer forecast horizons typically have lower accuracy
        - **Model Selection:** Different models may perform better for different production lines
        - **Training Time:** Model training may take a few minutes depending on your system
        - **Validation:** Always validate forecasts against actual outcomes when available
        """)

def main():
    st.markdown('<h1 class="main-header">🏭 OEE Manufacturing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Main Dashboard"
    
    st.sidebar.title("📊 Navigation")
    
    # Include advisory system pages if available
    if ADVISORY_AVAILABLE:
        page_options = add_advisory_system_to_sidebar()
        integrate_advisory_system()
    else:
        page_options = ["🏠 Main Dashboard", "📈 Line-Specific Analysis", "📊 Overall Daily Analysis", "🔮 OEE Forecasting"]
        # Show advisory system status in sidebar
        with st.sidebar:
            st.markdown("---")
            st.subheader("🤖 Advisory System")
            if os.path.exists('advisory_integration.py'):
                st.warning("⚠️ Setup Required")
                st.caption("Install dependencies to enable")
            else:
                st.info("ℹ️ Optional Feature")
                st.caption("Add RAG system files to enable")
    
    page = st.sidebar.selectbox(
        "Choose a page:", page_options,
        index=page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
    )
    
    if page != st.session_state.page:
        st.session_state.page = page
        if 'selected_line' in st.session_state:
            del st.session_state.selected_line
    
    # Data loading with proper error handling
    files_exist, existing_files = check_processed_files()
    
    if not files_exist:
        st.warning("⚠️ Processed OEE files not found. Starting preprocessing...")
        
        df_ls, df_prd, success = load_raw_data()
        if not success:
            st.error("❌ Cannot proceed without input files: line_status_notcleaned.csv and production_data.csv")
            st.stop()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preprocessing line status data...")
        progress_bar.progress(20)
        df_ls_clean = preprocess_line_status(df_ls)
        
        status_text.text("Preprocessing production data...")
        progress_bar.progress(40)
        df_prd_clean = preprocess_production_data(df_prd)
        
        status_text.text("Calculating OEE metrics...")
        progress_bar.progress(60)
        daily_oee_data = calculate_oee(df_ls_clean, df_prd_clean)
        
        status_text.text("Saving processed data...")
        progress_bar.progress(80)
        save_processed_data(df_ls_clean, daily_oee_data)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success("🎉 Data preprocessing and OEE calculation completed successfully!")
        st.balloons()
    else:
        st.success(f"✅ Found {len(existing_files)} processed files. Loading data...")
    
    daily_oee_data, overall_daily_oee = load_processed_data()
    
    # Handle advisory system pages first
    if ADVISORY_AVAILABLE:
        if handle_advisory_pages(st.session_state.page, daily_oee_data, overall_daily_oee):
            return
    
    if st.session_state.page == "🏠 Main Dashboard":
        show_main_dashboard(daily_oee_data, overall_daily_oee)
    elif st.session_state.page == "📈 Line-Specific Analysis":
        show_line_analysis(daily_oee_data)
    elif st.session_state.page == "📊 Overall Daily Analysis":
        show_overall_analysis(overall_daily_oee)
    elif st.session_state.page == "🔮 OEE Forecasting":
        show_forecasting_page(daily_oee_data, overall_daily_oee)

def show_main_dashboard(daily_oee_data, overall_daily_oee):
    """Show main dashboard page"""
    st.header("📊 Manufacturing Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_oee = daily_oee_data['OEE'].mean()
    avg_availability = daily_oee_data['Availability'].mean()
    avg_performance = daily_oee_data['Performance'].mean()
    total_output = daily_oee_data['Total_Actual_Output'].sum()
    
    with col1:
        st.metric("Average OEE", f"{avg_oee:.1%}")
    with col2:
        st.metric("Average Availability", f"{avg_availability:.1%}")
    with col3:
        st.metric("Average Performance", f"{avg_performance:.1%}")
    with col4:
        st.metric("Total Output", f"{total_output:,}")
    
    # Quick advisory access
    if ADVISORY_AVAILABLE:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🤖 Ask AI Advisor", use_container_width=True, type="primary"):
                st.session_state.page = "🤖 OEE Advisory"
                st.rerun()
        
        with col2:
            if st.button("📚 Manage Documents", use_container_width=True):
                st.session_state.page = "📚 Document Management"
                st.rerun()
        
        with col3:
            if st.button("⚡ Quick Analysis", use_container_width=True):
                st.session_state.page = "🤖 OEE Advisory"
                st.session_state.quick_analysis_requested = True
                st.rerun()
        
        st.markdown("---")
    
    # Production Lines Status Buttons
    st.subheader("🏭 Production Lines Status")
    st.markdown("*Click on any line to view detailed analysis*")
    
    lines = sorted(daily_oee_data['PRODUCTION_LINE'].unique())
    cols = st.columns(len(lines))
    
    for i, line in enumerate(lines):
        with cols[i]:
            status, current_oee, icon = get_line_current_status(line, daily_oee_data)
            
            button_clicked = st.button(
                f"{icon} **{line}**\n\nStatus: {status}\nOEE: {current_oee:.1%}",
                key=f"line_button_{line}", help=f"Click to analyze {line}", use_container_width=True
            )
            
            if button_clicked:
                st.session_state.page = "📈 Line-Specific Analysis"
                st.session_state.selected_line = line
                st.rerun()
    
    st.divider()
    
    # Compare Production Lines Section
    st.subheader("⚖️ Compare Production Lines")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        comparison_metric = st.selectbox(
            "Select Metric to Compare:", options=['OEE', 'Availability', 'Performance', 'Quality'], key="comparison_metric"
        )
    
    with col2:
        fig_comparison = create_comparison_chart(daily_oee_data, comparison_metric)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.divider()
    
    # Performance Ranking Section
    st.subheader("🏆 Performance Ranking")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        ranking_metric = st.selectbox(
            "Rank Lines By:", options=['OEE', 'Availability', 'Performance', 'Quality'], key="ranking_metric"
        )
        
        st.markdown("### 🥇 Top Performers")
        ranking_df = create_ranking_table(daily_oee_data, ranking_metric)
        
        medals = ["🥇", "🥈", "🥉"]
        for i in range(min(3, len(ranking_df))):
            line_name = ranking_df.index[i]
            rank_value = ranking_df.iloc[i][ranking_metric]
            medal = medals[i] if i < 3 else f"{i+1}."
            st.markdown(f"{medal} **{line_name}**: {rank_value}")
    
    with col2:
        st.markdown("### 📊 Complete Rankings")
        styled_df = ranking_df.style.apply(
            lambda x: ['background-color: #90EE90' if x.name == ranking_df.index[0] 
                      else 'background-color: #FFE4B5' if x.name == ranking_df.index[1]
                      else 'background-color: #F0E68C' if x.name == ranking_df.index[2]
                      else '' for _ in x], axis=1
        )
        st.dataframe(styled_df, use_container_width=True)
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Overall OEE Trend")
        fig_trend = create_oee_trend_chart(overall_daily_oee, title_suffix="(All Lines)")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.subheader("📊 Average Performance by Line")
        fig_avg = create_avg_oee_chart(daily_oee_data)
        st.plotly_chart(fig_avg, use_container_width=True)
    
    # Data summary table
    st.subheader("📋 Production Line Summary")
    summary = daily_oee_data.groupby('PRODUCTION_LINE').agg({
        'OEE': ['mean', 'min', 'max'], 'Availability': 'mean', 'Performance': 'mean', 'Total_Actual_Output': 'sum'
    }).round(3)
    
    summary.columns = ['Avg OEE', 'Min OEE', 'Max OEE', 'Avg Availability', 'Avg Performance', 'Total Output']
    st.dataframe(summary, use_container_width=True)

def show_line_analysis(daily_oee_data):
    """Show line-specific analysis page"""
    if 'selected_line' in st.session_state:
        st.info(f"🔗 Analyzing {st.session_state.selected_line} (selected from dashboard)")
    
    st.header("📈 Line-Specific Analysis")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        default_line = st.session_state.get('selected_line', sorted(daily_oee_data['PRODUCTION_LINE'].unique())[0])
        selected_line = st.selectbox(
            "Select Production Line:", options=sorted(daily_oee_data['PRODUCTION_LINE'].unique()),
            index=sorted(daily_oee_data['PRODUCTION_LINE'].unique()).index(default_line) if default_line in daily_oee_data['PRODUCTION_LINE'].unique() else 0
        )
    
    with col2:
        date_range = st.date_input(
            "Select Date Range:",
            value=(daily_oee_data['Date'].min().date(), daily_oee_data['Date'].max().date()),
            min_value=daily_oee_data['Date'].min().date(),
            max_value=daily_oee_data['Date'].max().date()
        )
    
    with col3:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.session_state.page = "🏠 Main Dashboard"
            if 'selected_line' in st.session_state:
                del st.session_state.selected_line
            st.rerun()
    
    if 'selected_line' in st.session_state:
        del st.session_state.selected_line
    
    line_data = daily_oee_data[daily_oee_data['PRODUCTION_LINE'] == selected_line].copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        line_data = line_data[
            (line_data['Date'].dt.date >= start_date) & 
            (line_data['Date'].dt.date <= end_date)
        ]
    
    if line_data.empty:
        st.warning("No data available for selected filters.")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    status, current_oee, icon = get_line_current_status(selected_line, daily_oee_data)
    
    with col1:
        st.markdown(f"### {icon} Status")
        st.markdown(f"**{status}**")
    with col2:
        st.metric("Average OEE", f"{line_data['OEE'].mean():.1%}")
    with col3:
        st.metric("Average Availability", f"{line_data['Availability'].mean():.1%}")
    with col4:
        st.metric("Average Performance", f"{line_data['Performance'].mean():.1%}")
    with col5:
        st.metric("Total Output", f"{line_data['Total_Actual_Output'].sum():,}")
    
    st.subheader(f"📈 OEE Trend for {selected_line}")
    fig_trend = create_oee_trend_chart(line_data, selected_line, f"for {selected_line}")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Distribution")
        fig_hist = px.histogram(
            line_data, x='OEE', nbins=15, title=f"OEE Distribution for {selected_line}",
            labels={'OEE': 'OEE Value', 'count': 'Number of Days'}
        )
        fig_hist.update_layout(xaxis_tickformat=',.0%')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Best vs Worst Days")
        best_day = line_data.loc[line_data['OEE'].idxmax()]
        worst_day = line_data.loc[line_data['OEE'].idxmin()]
        
        st.markdown("**🏆 Best Performance Day:**")
        st.markdown(f"Date: {best_day['Date'].strftime('%Y-%m-%d')}")
        st.markdown(f"OEE: {best_day['OEE']:.1%}")
        st.markdown(f"Output: {best_day['Total_Actual_Output']} units")
        
        st.markdown("**⚠️ Worst Performance Day:**")
        st.markdown(f"Date: {worst_day['Date'].strftime('%Y-%m-%d')}")
        st.markdown(f"OEE: {worst_day['OEE']:.1%}")
        st.markdown(f"Output: {worst_day['Total_Actual_Output']} units")
    
    st.subheader("📋 Daily Performance Data")
    display_data = line_data[['Date', 'OEE', 'Availability', 'Performance', 'Total_Actual_Output']].copy()
    display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
    for col in ['OEE', 'Availability', 'Performance']:
        display_data[col] = display_data[col].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)

def show_overall_analysis(overall_daily_oee):
    """Show overall daily analysis page"""
    st.header("📊 Overall Daily Analysis")
    
    date_range = st.date_input(
        "Select Date Range:",
        value=(overall_daily_oee['Date'].min().date(), overall_daily_oee['Date'].max().date()),
        min_value=overall_daily_oee['Date'].min().date(),
        max_value=overall_daily_oee['Date'].max().date()
    )
    
    filtered_data = overall_daily_oee.copy()
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = filtered_data[
            (filtered_data['Date'].dt.date >= start_date) & 
            (filtered_data['Date'].dt.date <= end_date)
        ]
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average OEE", f"{filtered_data['OEE'].mean():.1%}")
    with col2:
        st.metric("Best Day OEE", f"{filtered_data['OEE'].max():.1%}")
    with col3:
        st.metric("Worst Day OEE", f"{filtered_data['OEE'].min():.1%}")
    with col4:
        st.metric("Days Above 80%", f"{(filtered_data['OEE'] > 0.8).sum()}")
    
    st.subheader("📈 Overall Daily OEE Trend")
    fig_trend = create_oee_trend_chart(filtered_data, title_suffix="(All Lines Combined)")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.subheader("📊 OEE Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            filtered_data, x='OEE', nbins=20, title="OEE Distribution",
            labels={'OEE': 'OEE Value', 'count': 'Number of Days'}
        )
        fig_hist.update_layout(xaxis_tickformat=',.0%')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        performance_categories = pd.cut(
            filtered_data['OEE'], 
            bins=[0, 0.6, 0.8, 0.9, 1.0], 
            labels=['Poor (<60%)', 'Fair (60-80%)', 'Good (80-90%)', 'Excellent (>90%)']
        ).value_counts()
        
        fig_pie = px.pie(
            values=performance_categories.values, names=performance_categories.index,
            title="OEE Performance Categories"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("📋 Daily Performance Data")
    display_data = filtered_data.copy()
    display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
    for col in ['OEE', 'Availability', 'Performance', 'Quality']:
        display_data[col] = display_data[col].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()