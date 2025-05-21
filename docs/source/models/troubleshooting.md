# Troubleshooting

This section provides solutions to common issues you might encounter when working with the OEE-Forecasting project.

## Installation Issues

### Package Dependency Errors

**Issue**: Errors when installing required packages, particularly with conflicting dependencies.

**Solution**:
1. Create a fresh virtual environment
2. Install packages in the specific order mentioned in the installation guide
3. Make sure numpy and Cython are installed before pmdarima

### TensorFlow Installation Problems

**Issue**: TensorFlow fails to install or import properly.

**Solution**:
1. Ensure you have the compatible Python version (3.8-3.11 recommended for TensorFlow)
2. Try installing a specific version: `pip install tensorflow==2.10.0`
3. Check if your system requires a CPU-only version: `pip install tensorflow-cpu`

## Data Processing Issues

### Missing or Invalid Data

**Issue**: Errors when processing data due to missing values or unexpected formats.

**Solution**:
1. Check the structure of your input data against the expected format
2. Use the data cleaning functions provided in Notebook 1
3. Consider using the synthetic data generator if you don't have access to the real data

### OOE Calculation Errors

**Issue**: Invalid OOE values (e.g., values greater than 1 or negative values).

**Solution**:
1. Verify that the raw data is correct
2. Check the calculation formulas in Notebook 1
3. Implement additional data validation steps before calculation

## Model Training Issues

### Out of Memory Errors

**Issue**: Running out of memory when training deep learning models.

**Solution**:
1. Reduce batch size in model configuration
2. Simplify the model architecture
3. Reduce the sequence length for LSTM/GRU models
4. Use a machine with more RAM or GPU memory

### Poor Model Performance

**Issue**: Models show high error rates or fail to converge.

**Solution**:
1. Check for data normalization issues
2. Try different hyperparameter settings
3. Increase the training epochs
4. Experiment with different model architectures

### ARIMA Model Errors

**Issue**: pmdarima errors or warnings during model fitting.

**Solution**:
1. Check for stationarity in your time series
2. Try different differencing parameters
3. Increase the max_iter parameter in the model configuration
4. If the series is complex, consider using a seasonal model

## Visualization Issues

**Issue**: Plots don't display or look incorrect.

**Solution**:
1. Check matplotlib backend configuration
2. Make sure the data being plotted is valid
3. Adjust plot parameters for better visibility

## Additional Resources

If you encounter issues not covered here, you can:
1. Check the [GitHub repository](https://github.com/HxRJILI/OEE-FORECAST) for updates
2. Open an issue on GitHub
3. Refer to the documentation of individual libraries (TensorFlow, pmdarima, etc.)