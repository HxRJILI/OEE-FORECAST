# Installation Guide

This guide walks through the process of setting up your environment for the OEE-Forecasting project.

## System Requirements

- Python 3.11 recommended
- Git
- Sufficient disk space for datasets and model outputs
- At least 8GB RAM recommended for deep learning models

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/HxRJILI/OEE-FORECAST.git
cd OEE-FORECAST
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv oee-env
```

Activate the virtual environment:

- On Windows:
  ```bash
  oee-env\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source oee-env/bin/activate
  ```

### 3. Install Dependencies in the Correct Order

Due to some package dependencies, we need to install the requirements in a specific order:

#### Step 1: Install NumPy first
```bash
pip install numpy==1.24.3
```

#### Step 2: Install Cython
```bash
pip install Cython==0.29.36
```

#### Step 3: Install pmdarima
```bash
pip install pmdarima==2.0.3
```

#### Step 4: Install the remaining dependencies
```bash
pip install pandas matplotlib scikit-learn statsmodels tensorflow
```

### 4. Full Requirements List

Alternatively, you can create a `requirements.txt` file with the following content:

```
# Core data processing and analysis
numpy==1.24.3
pandas
matplotlib
scikit-learn

# Time series analysis
Cython==0.29.36
pmdarima==2.0.3
statsmodels

# Deep learning
tensorflow

# Additional utilities
glob2
```

And install using:
```bash
pip install -r requirements.txt
```

Note: The order in the requirements file is important, so if you encounter issues, try installing the packages manually in the order specified in the previous section.

## Package Versions

Here's a list of the exact versions used in the development of this project:

| Package        | Version |
|----------------|---------|
| numpy          | 1.24.3  |
| Cython         | 0.29.36 |
| pmdarima       | 2.0.3   |
| pandas         | latest  |
| matplotlib     | latest  |
| scikit-learn   | latest  |
| tensorflow     | latest  |
| statsmodels    | latest  |



## Jupyter Notebook Setup

This project uses Jupyter Notebooks extensively. To install and run Jupyter:

```bash
pip install jupyter
jupyter notebook
```

This will open a browser window where you can navigate to and open the project notebooks:
- OEE-Insight_1.ipynb
- OEE-Insight_2.ipynb
- OEE-Insight_3.ipynb

## Troubleshooting Common Installation Issues

### TensorFlow Installation Issues

If you encounter problems installing TensorFlow:

1. Ensure you have the latest pip version:
   ```bash
   pip install --upgrade pip
   ```

2. For GPU support, make sure you have the correct CUDA and cuDNN versions installed.

### pmdarima Installation Errors

If pmdarima installation fails:

1. Make sure you installed NumPy and Cython first
2. Try installing the development version:
   ```bash
   pip install git+https://github.com/alkaline-ml/pmdarima.git
   ```

### Memory Issues with Large Datasets

If you encounter memory errors when running the notebooks:

1. Try closing other applications to free up memory
2. Reduce batch sizes in the deep learning models
3. Consider sampling or partitioning the data for analysis

## Verifying Your Installation

To verify your installation is correct, run the following in a Python console:

```python
import numpy
import pandas
import pmdarima
import tensorflow
import matplotlib.pyplot
import statsmodels

print("NumPy version:", numpy.__version__)
print("pandas version:", pandas.__version__)
print("pmdarima version:", pmdarima.__version__)
print("TensorFlow version:", tensorflow.__version__)

print("All required packages are installed correctly!")
```

## Next Steps

Once your environment is set up, proceed to the [Data Overview](data_overview.html) section to learn about the datasets and how to prepare them for analysis.