OEE Forecasting and Analytics Documentation
==========================================

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :alt: Python Version

.. image:: https://img.shields.io/badge/TensorFlow-2.10%2B-orange
   :alt: TensorFlow Version

.. image:: https://img.shields.io/badge/Streamlit-1.28%2B-red
   :alt: Streamlit Version

**A comprehensive manufacturing analytics platform for Overall Equipment Effectiveness (OEE) analysis, forecasting, and optimization.**

This project provides a complete solution for manufacturing performance analysis through:

- **Data Processing Pipeline**: Automated preprocessing of production and line status data
- **Statistical Analysis**: ARIMA modeling and time series analysis for OEE patterns
- **Deep Learning Forecasting**: Advanced neural networks for multi-horizon OEE prediction
- **Interactive Dashboard**: Streamlit-based web application for real-time analytics
- **AI Advisory System**: RAG-powered intelligent recommendations for OEE improvement

 **Key Features**
==================

 **Comprehensive Analytics**
   - Real-time OEE calculation (Availability × Performance × Quality)
   - Multi-line production performance comparison
   - Historical trend analysis and pattern recognition

 **Advanced Forecasting**
   - Multiple deep learning architectures (RNN, CNN, WaveNet-style)
   - Multi-step ahead predictions with uncertainty quantification
   - Walk-forward validation for realistic performance assessment

 **AI-Powered Insights**
   - RAG-based advisory system for manufacturing optimization
   - Document-driven knowledge base for best practices
   - Intelligent recommendations based on performance patterns

 **Production Dashboard**
   - Line-specific performance monitoring
   - Interactive visualizations with Plotly
   - Real-time alerts and status indicators

 **Project Architecture**
==========================

The project is built around three core Jupyter notebooks that progressively build upon each other:

1. **OEE_Insights_1.ipynb**: Data preprocessing and OEE calculation foundation
2. **OEE_Insights_2.ipynb**: Statistical modeling and ARIMA-based forecasting
3. **OEE_Insights_3.ipynb**: Deep learning models for advanced time series prediction

These notebooks feed into a comprehensive Streamlit application that provides:

- Interactive dashboards for real-time monitoring
- Advanced forecasting capabilities with multiple model options
- AI-powered advisory system for optimization recommendations

 **Table of Contents**
========================

.. toctree::
   :maxdepth: 2
   :caption: 🚀 Getting Started
   :numbered:

   installation
   quickstart
   data_requirements

.. toctree::
   :maxdepth: 2
   :caption:  User Guide
   :numbered:

   streamlit/overview
   streamlit/dashboard
   streamlit/forecasting
   streamlit/advisory_system

.. toctree::
   :maxdepth: 2
   :caption:  Analysis Notebooks
   :numbered:

   notebooks/oee_insights_1
   notebooks/oee_insights_2
   notebooks/oee_insights_3

.. toctree::
   :maxdepth: 3
   :caption:  Models & Algorithms
   :numbered:

   models/statistical_models
   models/deep_learning_models
   models/evaluation_metrics

.. toctree::
   :maxdepth: 2
   :caption:  Developer Reference
   :numbered:

   api/data_processing
   api/forecasting
   api/advisory_system

.. toctree::
   :maxdepth: 2
   :caption:  Advanced Topics
   :numbered:

   advanced/rag_system
   advanced/model_optimization
   advanced/deployment

.. toctree::
   :maxdepth: 1
   :caption:  Resources

   troubleshooting
   changelog
   contributing
   license

 **Quick Start**
==================

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/HxRJILI/OEE-FORECAST.git
   cd OEE-FORECAST

   # Install dependencies
   pip install -r requirements.txt

   # Run the Streamlit application
   streamlit run app.py

For detailed installation instructions, see :doc:`installation`.

 **Results Overview**
=======================

Our comprehensive analysis across multiple production lines demonstrates:

**Statistical Models (ARIMA)**
   - LINE-01: ARIMA(0,1,2) - Best for non-stationary data with MA components
   - LINE-03: ARIMA(1,0,1) - Optimal for stationary data with AR and MA terms
   - LINE-04: ARIMA(2,0,0) - Pure autoregressive model for trend-following data
   - LINE-06: ARIMA(1,0,0) - Simple AR model for predictable patterns

**Deep Learning Models**
   - **Best Overall**: Multi-Kernel CNN (LB=30) - Superior pattern recognition
   - **Best for LINE-06**: Multi-Kernel CNN - MAE: 0.0591, MAPE: 8.63%
   - **Most Stable**: Stacked RNN with Masking - Consistent across all lines
   - **Fastest Training**: WaveNet-style CNN - Efficient dilated convolutions

**Forecasting Performance**
   - Average MAE across all models: 0.06-0.13 (6-13% error rate)
   - Best performing line: LINE-06 (Manufacturing optimization success)
   - Most challenging: LINE-01 (High variability, requires advanced models)

 **Contributing**
==================

We welcome contributions! Please see our :doc:`contributing` guide for details on:

- Code style and standards
- Testing requirements
- Documentation updates
- Feature requests and bug reports

 **License**
=============

This project is licensed under the MIT License - see the :doc:`license` file for details.

 **Links**
============

- **GitHub Repository**: `https://github.com/HxRJILI/OEE-FORECAST <https://github.com/HxRJILI/OEE-FORECAST>`_
- **Issue Tracker**: `GitHub Issues <https://github.com/HxRJILI/OEE-FORECAST/issues>`_
- **Documentation**: This site!

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`