Troubleshooting Guide
====================

This comprehensive troubleshooting guide helps diagnose and resolve common issues encountered when deploying and using the OEE Forecasting and Analytics system.

🔍 **Common Issues and Solutions**
=================================

**Installation and Setup Issues**
---------------------------------

**Issue: TensorFlow Installation Fails**

.. code-block::

   Error: Could not install packages due to an EnvironmentError
   ImportError: No module named 'tensorflow'

**Solution:**

.. code-block:: bash

   # Method 1: Install CPU-only version
   pip uninstall tensorflow
   pip install tensorflow-cpu

   # Method 2: Use conda for better dependency management
   conda create -n oee_env python=3.9
   conda activate oee_env
   conda install tensorflow

   # Method 3: For Apple Silicon Macs
   pip install tensorflow-macos tensorflow-metal

**Issue: Streamlit Won't Start**

.. code-block::

   Error: streamlit: command not found
   ModuleNotFoundError: No module named 'streamlit'

**Solution:**

.. code-block:: bash

   # Ensure virtual environment is activated
   source oee_env/bin/activate  # Linux/Mac
   # or
   oee_env\Scripts\activate     # Windows

   # Reinstall streamlit
   pip install --upgrade streamlit

   # Check installation
   streamlit --version

   # If port is occupied
   streamlit run app.py --server.port 8502

**Issue: GEMINI API Key Not Working**

.. code-block::

   Error: Invalid API key or quota exceeded
   APIKeyError: The provided API key is invalid

**Solution:**

.. code-block:: bash

   # Check API key format (should start with 'AI' for Gemini)
   echo $GEMINI_API_KEY

   # Set environment variable correctly
   export GEMINI_API_KEY="your_actual_api_key_here"

   # For Windows
   set GEMINI_API_KEY=your_actual_api_key_here

   # Verify in Python
   python -c "import os; print(os.getenv('GEMINI_API_KEY'))"

   # Check quota at Google AI Studio
   # https://makersuite.google.com/app/apikey

**Data Processing Issues**
-------------------------

**Issue: CSV File Not Found or Corrupted**

.. code-block::

   Error: FileNotFoundError: 'line_status_notcleaned.csv' not found
   pandas.errors.EmptyDataError: No columns to parse from file

**Solution:**

.. code-block:: python

   # Check file existence and format
   import os
   import pandas as pd

   # Verify file exists
   if not os.path.exists('line_status_notcleaned.csv'):
       print("File not found - check file path and name")

   # Check file encoding
   try:
       df = pd.read_csv('line_status_notcleaned.csv', encoding='utf-8')
   except UnicodeDecodeError:
       df = pd.read_csv('line_status_notcleaned.csv', encoding='latin1')

   # Check for empty file
   if df.empty:
       print("CSV file is empty")

   # Verify required columns
   required_columns = ['START_DATETIME', 'PRODUCTION_LINE', 'STATUS_NAME']
   missing_columns = [col for col in required_columns if col not in df.columns]
   if missing_columns:
       print(f"Missing columns: {missing_columns}")

**Issue: Date Parsing Errors**

.. code-block::

   Error: ValueError: time data '2024-13-01' does not match format
   pandas._libs.tslibs.parsing.DateParseError

**Solution:**

.. code-block:: python

   # Robust date parsing
   def safe_date_parsing(df, date_columns):
       """Safely parse date columns with multiple formats"""
       
       date_formats = [
           '%Y-%m-%d %H:%M:%S',
           '%m/%d/%Y %H:%M',
           '%d-%m-%Y %H:%M:%S',
           '%Y-%m-%d',
           '%m/%d/%Y'
       ]
       
       for col in date_columns:
           if col in df.columns:
               for fmt in date_formats:
                   try:
                       df[col] = pd.to_datetime(df[col], format=fmt)
                       print(f"Successfully parsed {col} with format {fmt}")
                       break
                   except:
                       continue
               else:
                   # Fallback to automatic parsing
                   df[col] = pd.to_datetime(df[col], errors='coerce')
                   print(f"Used automatic parsing for {col}")
       
       return df

   # Usage
   df = safe_date_parsing(df, ['START_DATETIME', 'FINISH_DATETIME'])

**Issue: Memory Errors with Large Datasets**

.. code-block::

   Error: MemoryError: Unable to allocate array
   killed (out of memory)

**Solution:**

.. code-block:: python

   # Process data in chunks
   def process_large_csv(file_path, chunk_size=10000):
       """Process large CSV files in chunks"""
       
       processed_chunks = []
       
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           # Process each chunk
           processed_chunk = process_chunk(chunk)
           processed_chunks.append(processed_chunk)
           
           # Optional: Save intermediate results
           chunk.to_pickle(f'temp_chunk_{len(processed_chunks)}.pkl')
       
       # Combine results
       final_result = pd.concat(processed_chunks, ignore_index=True)
       return final_result

   # Optimize memory usage
   def optimize_memory_usage(df):
       """Optimize DataFrame memory usage"""
       
       # Convert object columns to category where appropriate
       for col in df.select_dtypes(include=['object']):
           if df[col].nunique() / len(df) < 0.5:
               df[col] = df[col].astype('category')
       
       # Downcast numeric types
       for col in df.select_dtypes(include=['int64']):
           df[col] = pd.to_numeric(df[col], downcast='integer')
       
       for col in df.select_dtypes(include=['float64']):
           df[col] = pd.to_numeric(df[col], downcast='float')
       
       return df

**Model Training Issues**
------------------------

**Issue: Model Training Extremely Slow**

.. code-block::

   Issue: Model training takes hours or gets stuck
   Warning: Training speed is very slow

**Solution:**

.. code-block:: python

   # Check GPU availability
   import tensorflow as tf
   
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   
   # Enable GPU memory growth
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       try:
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           print(e)

   # Optimize training parameters
   def optimize_training_config():
       return {
           'batch_size': 32,  # Increase if you have enough memory
           'epochs': 50,      # Reduce for faster training
           'validation_split': 0.2,
           'callbacks': [
               tf.keras.callbacks.EarlyStopping(patience=10),
               tf.keras.callbacks.ReduceLROnPlateau(patience=5)
           ]
       }

   # Use mixed precision for faster training
   tf.keras.mixed_precision.set_global_policy('mixed_float16')

**Issue: Model Overfitting**

.. code-block::

   Issue: Training accuracy high, validation accuracy low
   Training loss decreases, validation loss increases

**Solution:**

.. code-block:: python

   # Implement regularization techniques
   def add_regularization(model):
       """Add regularization to prevent overfitting"""
       
       # Add dropout layers
       for i, layer in enumerate(model.layers):
           if hasattr(layer, 'units') and layer.units > 50:
               # Add dropout after large layers
               model.layers.insert(i+1, tf.keras.layers.Dropout(0.3))
       
       # Add L2 regularization
       for layer in model.layers:
           if hasattr(layer, 'kernel_regularizer'):
               layer.kernel_regularizer = tf.keras.regularizers.l2(0.001)
       
       return model

   # Data augmentation for time series
   def augment_training_data(X, y, noise_level=0.01):
       """Add noise to training data"""
       
       X_augmented = X + np.random.normal(0, noise_level, X.shape)
       y_augmented = y
       
       # Combine original and augmented data
       X_combined = np.concatenate([X, X_augmented])
       y_combined = np.concatenate([y, y_augmented])
       
       return X_combined, y_combined

**Issue: Poor Model Performance**

.. code-block::

   Issue: Model accuracy below expectations
   High prediction errors (MAE > 0.15, MAPE > 20%)

**Solution:**

.. code-block:: python

   # Diagnostic steps for poor performance
   def diagnose_model_performance(model, X_test, y_test):
       """Diagnose model performance issues"""
       
       predictions = model.predict(X_test)
       
       # Calculate detailed metrics
       mae = np.mean(np.abs(y_test - predictions.flatten()))
       rmse = np.sqrt(np.mean((y_test - predictions.flatten())**2))
       mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
       
       print(f"MAE: {mae:.4f}")
       print(f"RMSE: {rmse:.4f}")
       print(f"MAPE: {mape:.2f}%")
       
       # Check for patterns in errors
       errors = y_test - predictions.flatten()
       
       # Plot error distribution
       import matplotlib.pyplot as plt
       
       plt.figure(figsize=(12, 4))
       
       plt.subplot(1, 3, 1)
       plt.scatter(y_test, predictions.flatten())
       plt.xlabel('Actual')
       plt.ylabel('Predicted')
       plt.title('Actual vs Predicted')
       
       plt.subplot(1, 3, 2)
       plt.hist(errors, bins=50)
       plt.xlabel('Prediction Error')
       plt.title('Error Distribution')
       
       plt.subplot(1, 3, 3)
       plt.plot(errors)
       plt.xlabel('Sample')
       plt.ylabel('Error')
       plt.title('Error Over Time')
       
       plt.tight_layout()
       plt.show()
       
       # Recommendations based on error patterns
       if np.abs(np.mean(errors)) > 0.05:
           print("High bias detected - consider more complex model")
       
       if np.std(errors) > mae:
           print("High variance detected - consider regularization")

**RAG System Issues**
--------------------

**Issue: RAG System Not Initializing**

.. code-block::

   Error: Failed to load embedding model
   ModuleNotFoundError: No module named 'sentence_transformers'

**Solution:**

.. code-block:: bash

   # Install RAG dependencies
   pip install -r requirements_rag.txt

   # If sentence-transformers fails
   pip install sentence-transformers --no-cache-dir

   # For Apple Silicon Macs
   pip install sentence-transformers --no-deps
   pip install transformers torch torchvision torchaudio

   # Verify installation
   python -c "from sentence_transformers import SentenceTransformer; print('Success')"

**Issue: PDF Processing Fails**

.. code-block::

   Error: Failed to extract text from PDF
   PdfReadError: EOF marker not found

**Solution:**

.. code-block:: python

   # Robust PDF processing
   def robust_pdf_processing(pdf_path):
       """Process PDF with multiple fallback methods"""
       
       import PyPDF2
       import pdfplumber
       
       # Method 1: PyPDF2
       try:
           with open(pdf_path, 'rb') as file:
               pdf_reader = PyPDF2.PdfReader(file)
               text = ""
               for page in pdf_reader.pages:
                   text += page.extract_text()
               if text.strip():
                   return text
       except Exception as e:
           print(f"PyPDF2 failed: {e}")
       
       # Method 2: pdfplumber
       try:
           with pdfplumber.open(pdf_path) as pdf:
               text = ""
               for page in pdf.pages:
                   page_text = page.extract_text()
                   if page_text:
                       text += page_text
               if text.strip():
                   return text
       except Exception as e:
           print(f"pdfplumber failed: {e}")
       
       # Method 3: Manual text extraction
       print("All PDF methods failed - manual inspection required")
       return None

**Issue: Vector Database Performance Slow**

.. code-block::

   Issue: Slow similarity search
   FAISS index queries taking too long

**Solution:**

.. code-block:: python

   # Optimize FAISS index
   def optimize_faiss_index(embeddings, index_type='auto'):
       """Create optimized FAISS index"""
       
       import faiss
       
       dimension = embeddings.shape[1]
       n_vectors = embeddings.shape[0]
       
       if index_type == 'auto':
           if n_vectors < 10000:
               index_type = 'flat'
           elif n_vectors < 100000:
               index_type = 'ivf'
           else:
               index_type = 'pq'
       
       if index_type == 'flat':
           index = faiss.IndexFlatIP(dimension)
       elif index_type == 'ivf':
           nlist = min(int(np.sqrt(n_vectors)), 4096)
           quantizer = faiss.IndexFlatIP(dimension)
           index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
           index.train(embeddings)
           index.nprobe = min(nlist // 4, 100)
       elif index_type == 'pq':
           m = dimension // 8
           index = faiss.IndexPQ(dimension, m, 8)
           index.train(embeddings)
       
       index.add(embeddings)
       return index

**Streamlit Application Issues**
-------------------------------

**Issue: Streamlit App Crashes or Freezes**

.. code-block::

   Error: StreamlitAPIException
   App becomes unresponsive

**Solution:**

.. code-block:: python

   # Add error handling and debugging
   import streamlit as st
   import traceback

   def safe_streamlit_function(func):
       """Decorator for safe Streamlit function execution"""
       def wrapper(*args, **kwargs):
           try:
               return func(*args, **kwargs)
           except Exception as e:
               st.error(f"An error occurred: {str(e)}")
               st.error(f"Traceback: {traceback.format_exc()}")
               st.info("Please refresh the page and try again")
               return None
       return wrapper

   # Use session state properly
   def initialize_session_state():
       """Initialize session state variables"""
       
       if 'data_loaded' not in st.session_state:
           st.session_state.data_loaded = False
       
       if 'models_trained' not in st.session_state:
           st.session_state.models_trained = False
       
       if 'error_count' not in st.session_state:
           st.session_state.error_count = 0

   # Add caching for expensive operations
   @st.cache_data
   def load_and_process_data():
       """Cached data loading function"""
       try:
           return process_data_safely()
       except Exception as e:
           st.error(f"Data loading failed: {e}")
           return None

**Issue: Memory Usage Growing Over Time**

.. code-block::

   Issue: Streamlit app memory usage increases with each interaction
   Browser becomes slow or crashes

**Solution:**

.. code-block:: python

   # Clear cache and manage memory
   def manage_memory():
       """Memory management for Streamlit"""
       
       # Clear Streamlit cache periodically
       if st.session_state.get('interaction_count', 0) % 50 == 0:
           st.cache_data.clear()
           st.cache_resource.clear()
       
       # Increment interaction counter
       st.session_state.interaction_count = st.session_state.get('interaction_count', 0) + 1

   # Use efficient data structures
   def optimize_dataframes(df):
       """Optimize DataFrame memory usage"""
       
       # Convert to efficient types
       for col in df.select_dtypes(include=['object']):
           if df[col].nunique() / len(df) < 0.5:
               df[col] = df[col].astype('category')
       
       return df

**Performance Issues**
---------------------

**Issue: Application Response Time Slow**

.. code-block::

   Issue: Pages load slowly
   Model predictions take too long

**Solution:**

.. code-block:: python

   # Implement performance monitoring
   import time
   from functools import wraps

   def performance_monitor(func):
       """Monitor function performance"""
       @wraps(func)
       def wrapper(*args, **kwargs):
           start_time = time.time()
           result = func(*args, **kwargs)
           end_time = time.time()
           
           execution_time = end_time - start_time
           if execution_time > 5.0:  # Log slow operations
               print(f"Slow operation: {func.__name__} took {execution_time:.2f} seconds")
           
           return result
       return wrapper

   # Optimize model loading
   @st.cache_resource
   def load_models_efficiently():
       """Load models with caching and lazy loading"""
       
       models = {}
       
       # Load only when needed
       def lazy_load_model(model_name):
           if model_name not in models:
               models[model_name] = tf.keras.models.load_model(f'models/{model_name}.h5')
           return models[model_name]
       
       return lazy_load_model

   # Implement batch processing
   def batch_predictions(model, data_batches, batch_size=32):
       """Process predictions in batches"""
       
       all_predictions = []
       
       for i in range(0, len(data_batches), batch_size):
           batch = data_batches[i:i+batch_size]
           batch_predictions = model.predict(batch, verbose=0)
           all_predictions.extend(batch_predictions)
       
       return all_predictions

🔧 **Diagnostic Tools**
======================

**System Health Check Script**

.. code-block:: python

   # health_check.py - Comprehensive system diagnostics
   
   import sys
   import os
   import psutil
   import subprocess
   import importlib
   
   def run_health_check():
       """Run comprehensive system health check"""
       
       print("🏥 OEE Analytics System Health Check")
       print("=" * 50)
       
       # Check Python version
       print(f"Python Version: {sys.version}")
       if sys.version_info < (3, 8):
           print("❌ Python 3.8+ required")
       else:
           print("✅ Python version OK")
       
       # Check required packages
       required_packages = [
           'streamlit', 'pandas', 'numpy', 'tensorflow',
           'plotly', 'scikit-learn', 'matplotlib'
       ]
       
       print("\n📦 Package Verification:")
       for package in required_packages:
           try:
               importlib.import_module(package)
               print(f"✅ {package}")
           except ImportError:
               print(f"❌ {package} - Not installed")
       
       # Check optional RAG packages
       print("\n🤖 RAG System Packages:")
       rag_packages = [
           'sentence_transformers', 'faiss', 'google.generativeai',
           'PyPDF2', 'pdfplumber'
       ]
       
       for package in rag_packages:
           try:
               importlib.import_module(package)
               print(f"✅ {package}")
           except ImportError:
               print(f"⚠️ {package} - Optional, install for RAG features")
       
       # Check system resources
       print("\n💻 System Resources:")
       memory = psutil.virtual_memory()
       print(f"Memory: {memory.available / 1024**3:.1f}GB available / {memory.total / 1024**3:.1f}GB total")
       
       if memory.available < 2 * 1024**3:  # Less than 2GB
           print("⚠️ Low memory - may affect performance")
       else:
           print("✅ Memory OK")
       
       # Check GPU availability
       try:
           import tensorflow as tf
           gpus = tf.config.list_physical_devices('GPU')
           if gpus:
               print(f"✅ GPU available: {len(gpus)} device(s)")
           else:
               print("ℹ️ No GPU detected - using CPU")
       except:
           print("❌ Cannot check GPU status")
       
       # Check data files
       print("\n📁 Data Files:")
       data_files = ['line_status_notcleaned.csv', 'production_data.csv']
       for file in data_files:
           if os.path.exists(file):
               size = os.path.getsize(file) / 1024**2  # MB
               print(f"✅ {file} ({size:.1f}MB)")
           else:
               print(f"❌ {file} - Not found")
       
       # Check network connectivity
       print("\n🌐 Network Connectivity:")
       try:
           import requests
           response = requests.get('https://api.google.com', timeout=5)
           if response.status_code == 200:
               print("✅ Internet connectivity OK")
           else:
               print("⚠️ Internet connectivity issues")
       except:
           print("❌ No internet connection")
       
       print("\n🎉 Health check completed!")

   if __name__ == "__main__":
       run_health_check()

**Performance Profiler**

.. code-block:: python

   # profiler.py - Performance profiling tools
   
   import cProfile
   import pstats
   import io
   import time
   import psutil
   import threading
   
   class PerformanceProfiler:
       def __init__(self):
           self.profiler = cProfile.Profile()
           self.memory_usage = []
           self.monitoring = False
       
       def start_profiling(self):
           """Start performance profiling"""
           self.profiler.enable()
           self.monitoring = True
           self.monitor_memory()
       
       def stop_profiling(self):
           """Stop profiling and generate report"""
           self.profiler.disable()
           self.monitoring = False
           return self.generate_report()
       
       def monitor_memory(self):
           """Monitor memory usage in background"""
           def memory_monitor():
               while self.monitoring:
                   memory_percent = psutil.virtual_memory().percent
                   self.memory_usage.append({
                       'timestamp': time.time(),
                       'memory_percent': memory_percent
                   })
                   time.sleep(1)
           
           thread = threading.Thread(target=memory_monitor)
           thread.daemon = True
           thread.start()
       
       def generate_report(self):
           """Generate performance report"""
           s = io.StringIO()
           ps = pstats.Stats(self.profiler, stream=s)
           ps.sort_stats('cumulative')
           ps.print_stats(20)  # Top 20 functions
           
           report = {
               'profile_stats': s.getvalue(),
               'memory_usage': self.memory_usage,
               'peak_memory': max(self.memory_usage, key=lambda x: x['memory_percent'])['memory_percent'] if self.memory_usage else 0
           }
           
           return report

   # Usage example
   def profile_function(func, *args, **kwargs):
       """Profile a specific function"""
       profiler = PerformanceProfiler()
       profiler.start_profiling()
       
       result = func(*args, **kwargs)
       
       report = profiler.stop_profiling()
       print("Performance Report:")
       print(report['profile_stats'])
       print(f"Peak Memory Usage: {report['peak_memory']:.1f}%")
       
       return result

**Log Analysis Tool**

.. code-block:: python

   # log_analyzer.py - Analyze application logs for issues
   
   import re
   from collections import Counter, defaultdict
   from datetime import datetime
   
   class LogAnalyzer:
       def __init__(self, log_file_path):
           self.log_file_path = log_file_path
           self.error_patterns = {
               'memory_error': r'MemoryError|Out of memory',
               'connection_error': r'ConnectionError|Connection refused',
               'timeout_error': r'TimeoutError|Request timeout',
               'api_error': r'APIError|API key',
               'file_error': r'FileNotFoundError|No such file',
               'import_error': r'ImportError|ModuleNotFoundError'
           }
       
       def analyze_logs(self):
           """Analyze logs for common issues"""
           error_counts = Counter()
           error_details = defaultdict(list)
           
           try:
               with open(self.log_file_path, 'r') as f:
                   for line_num, line in enumerate(f, 1):
                       for error_type, pattern in self.error_patterns.items():
                           if re.search(pattern, line, re.IGNORECASE):
                               error_counts[error_type] += 1
                               error_details[error_type].append({
                                   'line_num': line_num,
                                   'content': line.strip()
                               })
           
           except FileNotFoundError:
               print(f"Log file not found: {self.log_file_path}")
               return None
           
           return {
               'error_counts': dict(error_counts),
               'error_details': dict(error_details),
               'total_errors': sum(error_counts.values())
           }
       
       def generate_recommendations(self, analysis):
           """Generate recommendations based on log analysis"""
           recommendations = []
           
           for error_type, count in analysis['error_counts'].items():
               if count > 0:
                   if error_type == 'memory_error':
                       recommendations.append(
                           "Memory issues detected. Consider: reducing batch size, "
                           "enabling data streaming, or increasing system memory."
                       )
                   elif error_type == 'connection_error':
                       recommendations.append(
                           "Connection issues detected. Check network connectivity "
                           "and firewall settings."
                       )
                   elif error_type == 'api_error':
                       recommendations.append(
                           "API key issues detected. Verify API key validity and quota."
                       )
                   elif error_type == 'file_error':
                       recommendations.append(
                           "File access issues detected. Check file paths and permissions."
                       )
           
           return recommendations

🛠️ **Quick Fix Scripts**
======================

**Environment Reset Script**

.. code-block:: bash

   #!/bin/bash
   # reset_environment.sh - Reset development environment
   
   echo "🔄 Resetting OEE Analytics Environment"
   
   # Deactivate current environment
   if [[ "$VIRTUAL_ENV" != "" ]]; then
       deactivate
   fi
   
   # Remove existing environment
   rm -rf oee_env
   
   # Create fresh environment
   python -m venv oee_env
   
   # Activate environment
   source oee_env/bin/activate
   
   # Upgrade pip
   pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   
   # Install optional RAG requirements
   read -p "Install RAG system dependencies? (y/n): " install_rag
   if [[ $install_rag == "y" ]]; then
       pip install -r requirements_rag.txt
   fi
   
   echo "✅ Environment reset completed"
   echo "Run 'source oee_env/bin/activate' to activate"

**Cache Clear Script**

.. code-block:: python

   # clear_cache.py - Clear all application caches
   
   import os
   import shutil
   import streamlit as st
   
   def clear_all_caches():
       """Clear all application caches"""
       
       print("🧹 Clearing application caches...")
       
       # Clear Streamlit cache
       try:
           st.cache_data.clear()
           st.cache_resource.clear()
           print("✅ Streamlit cache cleared")
       except:
           print("⚠️ Could not clear Streamlit cache")
       
       # Clear model cache directory
       model_cache_dir = "models/cache"
       if os.path.exists(model_cache_dir):
           shutil.rmtree(model_cache_dir)
           os.makedirs(model_cache_dir)
           print("✅ Model cache cleared")
       
       # Clear temporary files
       temp_dirs = ["temp", "tmp", "__pycache__"]
       for temp_dir in temp_dirs:
           if os.path.exists(temp_dir):
               shutil.rmtree(temp_dir)
               print(f"✅ {temp_dir} cleared")
       
       # Clear Python cache files
       for root, dirs, files in os.walk("."):
           for d in dirs:
               if d == "__pycache__":
                   shutil.rmtree(os.path.join(root, d))
       
       print("🎉 All caches cleared successfully")

   if __name__ == "__main__":
       clear_all_caches()

📞 **Getting Help**
==================

**When to Seek Additional Help**

.. code-block::

   Escalation Guidelines:
   
   🔍 Self-Diagnosis First:
   - Run the health check script
   - Check logs for error patterns
   - Try the quick fix scripts
   - Review this troubleshooting guide
   
   📞 Contact Support When:
   - Data corruption or loss occurs
   - Security vulnerabilities are discovered
   - Performance degradation persists after optimization
   - Integration with external systems fails
   - Critical production issues occur

**Information to Provide When Seeking Help**

.. code-block::

   Support Information Checklist:
   
   ✅ System Information:
   - Operating system and version
   - Python version
   - Package versions (pip freeze output)
   - Hardware specifications (RAM, CPU, GPU)
   
   ✅ Error Information:
   - Complete error messages
   - Steps to reproduce the issue
   - Log files (sanitized)
   - Screenshots if applicable
   
   ✅ Environment Details:
   - Deployment method (local, Docker, cloud)
   - Data volume and characteristics
   - Custom modifications made
   - Recent changes or updates

**Community Resources**

.. code-block::

   🌐 Online Resources:
   
   - GitHub Issues: https://github.com/HxRJILI/OEE-FORECAST/issues
   - Documentation: This comprehensive guide
   - Stack Overflow: Tag questions with 'oee-analytics'
   - Manufacturing Forums: Discuss domain-specific issues
   
   📚 Documentation Sections:
   
   - Installation Guide: Step-by-step setup instructions
   - API Reference: Detailed function documentation
   - Deployment Guide: Production deployment help
   - Model Optimization: Performance tuning tips

**Emergency Recovery Procedures**

.. code-block:: bash

   # emergency_recovery.sh - Emergency system recovery
   
   echo "🚨 Emergency Recovery Procedure"
   
   # Backup current state
   timestamp=$(date +%Y%m%d_%H%M%S)
   backup_dir="emergency_backup_$timestamp"
   mkdir -p $backup_dir
   
   # Copy important files
   cp -r models $backup_dir/ 2>/dev/null
   cp -r data $backup_dir/ 2>/dev/null
   cp *.csv $backup_dir/ 2>/dev/null
   cp .env* $backup_dir/ 2>/dev/null
   
   echo "✅ Emergency backup created in $backup_dir"
   
   # Reset to known good state
   echo "🔄 Resetting to clean state..."
   
   # Stop all running processes
   pkill -f streamlit
   pkill -f python
   
   # Clear caches
   python clear_cache.py
   
   # Reset environment
   ./reset_environment.sh
   
   echo "🎉 Emergency recovery completed"
   echo "📁 Backup available in: $backup_dir"

This troubleshooting guide covers the most common issues encountered with the OEE Forecasting and Analytics system. For additional help or to report new issues, please refer to the community resources or contact support with the information checklist provided.