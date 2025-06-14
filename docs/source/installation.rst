Installation Guide
==================

This guide will help you set up the OEE Forecasting and Analytics platform on your system.

System Requirements
-------------------

**Minimum Requirements:**
   - Python 3.8 or higher
   - 8 GB RAM (16 GB recommended for deep learning models)
   - 5 GB free disk space
   - Modern web browser (for Streamlit interface)

**Recommended Requirements:**
   - Python 3.9-3.11
   - 16 GB RAM or more
   - GPU with CUDA support (optional, for faster model training)
   - High-resolution display (for better visualization experience)

Basic Installation
------------------

1. **Clone the Repository**

.. code-block:: bash

   git clone https://github.com/HxRJILI/OEE-FORECAST.git
   cd OEE-FORECAST

2. **Create Virtual Environment** (Recommended)

.. code-block:: bash

   # Using venv
   python -m venv oee_env
   
   # Activate on Windows
   oee_env\Scripts\activate
   
   # Activate on macOS/Linux
   source oee_env/bin/activate

3. **Install Core Dependencies**

.. code-block:: bash

   pip install -r requirements.txt

Core Dependencies Installation
------------------------------

The project requires several key packages:

**Data Processing & Analysis:**

.. code-block:: bash

   pip install pandas>=1.5.0 numpy>=1.24.0 matplotlib>=3.6.0 seaborn>=0.12.0

**Machine Learning & Deep Learning:**

.. code-block:: bash

   pip install tensorflow>=2.10.0 scikit-learn>=1.3.0

**Web Interface:**

.. code-block:: bash

   pip install streamlit>=1.28.0 plotly>=5.15.0

**Time Series Analysis:**

.. code-block:: bash

   pip install statsmodels pmdarima

Optional: RAG Advisory System
-----------------------------

For the AI-powered advisory system, install additional dependencies:

.. code-block:: bash

   pip install -r requirements_rag.txt

Or manually install RAG components:

.. code-block:: bash

   # Core RAG dependencies
   pip install google-generativeai>=0.3.0
   pip install sentence-transformers>=2.2.0
   pip install faiss-cpu>=1.7.0
   
   # Document processing
   pip install PyPDF2>=3.0.0 pdfplumber>=0.9.0
   
   # NLP components
   pip install spacy>=3.7.0 nltk>=3.8.0
   
   # Download spaCy language model
   python -m spacy download en_core_web_sm

**Setup RAG System:**

Run the automated setup script:

.. code-block:: bash

   python setup_advisory.py

This will:
   - Install all required dependencies
   - Download NLP models
   - Create necessary directories
   - Test the system configuration

Platform-Specific Instructions
------------------------------

Windows
^^^^^^^

**Common Issues and Solutions:**

1. **TensorFlow DLL Loading Failed:**

.. code-block:: bash

   # Install Microsoft Visual C++ Redistributable
   # Download from: https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
   
   # Alternative: Use CPU-only TensorFlow
   pip uninstall tensorflow
   pip install tensorflow-cpu

2. **Long Path Names:**

.. code-block:: bash

   # Enable long path support or use shorter directory names
   git clone https://github.com/HxRJILI/OEE-FORECAST.git C:\oee

macOS
^^^^^

**M1/M2 Macs (Apple Silicon):**

.. code-block:: bash

   # Use conda for better ARM64 support
   conda create -n oee_env python=3.9
   conda activate oee_env
   conda install tensorflow
   pip install -r requirements.txt

**Intel Macs:**

.. code-block:: bash

   # Standard installation should work
   pip install -r requirements.txt

Linux
^^^^^

**Ubuntu/Debian:**

.. code-block:: bash

   # Install system dependencies
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip build-essential
   
   # Install project dependencies
   pip install -r requirements.txt

**CentOS/RHEL:**

.. code-block:: bash

   # Install system dependencies
   sudo yum install python3-devel gcc gcc-c++ make
   
   # Install project dependencies
   pip install -r requirements.txt

Development Installation
------------------------

For contributors and developers:

.. code-block:: bash

   # Clone with development branch
   git clone -b develop https://github.com/HxRJILI/OEE-FORECAST.git
   cd OEE-FORECAST
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt

Docker Installation
-------------------

Alternative installation using Docker:

.. code-block:: bash

   # Build Docker image
   docker build -t oee-forecast .
   
   # Run container
   docker run -p 8501:8501 oee-forecast

**Docker Compose** (with services):

.. code-block:: yaml

   version: '3.8'
   services:
     oee-app:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - ./data:/app/data
       environment:
         - GEMINI_API_KEY=${GEMINI_API_KEY}

Verification
------------

Test your installation:

.. code-block:: bash

   # Test basic imports
   python -c "import pandas, numpy, tensorflow, streamlit; print('All imports successful!')"
   
   # Test Streamlit app
   streamlit run app.py
   
   # Test RAG system (if installed)
   python -c "from advisory_integration import check_advisory_system_status; print(check_advisory_system_status())"

Data Setup
----------

1. **Required Data Files:**

Place these files in the project root directory:

.. code-block::

   line_status_notcleaned.csv    # Production line status data
   production_data.csv           # Manufacturing production data

2. **Optional Files:**

.. code-block::

   The Complete_Guide_to_Simple_OEE.pdf  # For RAG knowledge base

File format requirements are detailed in :doc:`data_requirements`.

Configuration
-------------

**Environment Variables:**

Create a `.env` file for sensitive configuration:

.. code-block:: bash

   # For RAG advisory system
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional: Custom model paths
   MODEL_CACHE_DIR=./models
   DOCUMENT_CACHE_DIR=./documents

Troubleshooting
---------------

**Common Installation Issues:**

1. **ImportError: No module named 'tensorflow'**

.. code-block:: bash

   # Ensure you're in the correct virtual environment
   pip install --upgrade tensorflow

2. **Streamlit not starting**

.. code-block:: bash

   # Check if port is available
   streamlit run app.py --server.port 8502

3. **RAG system initialization failed**

.. code-block:: bash

   # Run diagnostics
   python setup_advisory.py
   
   # Check API key
   echo $GEMINI_API_KEY

4. **Memory errors during model training**

.. code-block:: bash

   # Reduce batch size in configuration
   # Or increase system memory allocation

For more detailed troubleshooting, see :doc:`troubleshooting`.

Next Steps
----------

After successful installation:

1. Follow the :doc:`quickstart` guide
2. Review :doc:`data_requirements` for your data format
3. Explore the :doc:`streamlit/overview` for application features
4. Check out :doc:`notebooks/oee_insights_1` to understand the analysis process

**Need Help?**

- Check :doc:`troubleshooting` for common issues
- Create an issue on `GitHub <https://github.com/HxRJILI/OEE-FORECAST/issues>`_
- Review the FAQ in our documentation