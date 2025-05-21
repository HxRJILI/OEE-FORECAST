# Deep Learning Models

This section describes the deep learning models implemented in the OEE-Forecasting project, corresponding to the analysis in Notebook 3 (OEE-Insight_3).

## Overview

Deep learning models can capture complex temporal patterns in time series data that might be difficult for statistical models to identify. For the OOE forecasting task, we experimented with several neural network architectures to find the most effective approach.

## Data Preparation for Deep Learning

Before training our models, we performed specific preprocessing steps for deep learning:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Scale the data
scaler = RobustScaler()
scaled_data = scaler.fit_transform(ooe_data[['OOE']])

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Sequence length (lookback period)
seq_length = 30

# Create training sequences
X, y = create_sequences(scaled_data, seq_length)

# Reshape for LSTM input [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```

## Recurrent Neural Networks (RNNs)

### Long Short-Term Memory (LSTM)

LSTMs are specifically designed to handle the vanishing gradient problem in traditional RNNs, making them well-suited for capturing long-term dependencies in time series data.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Build LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

# Compile model
lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001
)

# Train model
history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

### Gated Recurrent Unit (GRU)

GRUs are similar to LSTMs but with a simplified architecture that can be faster to train while still capturing complex temporal dependencies.

```python
from tensorflow.keras.layers import GRU

# Build GRU model
gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    GRU(32),
    Dense(1)
])

# Compile and train (similar to LSTM)
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
gru_history = gru_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

### Simple RNN

For comparison, we also implemented a simple RNN model:

```python
from tensorflow.keras.layers import SimpleRNN

# Build Simple RNN model
rnn_model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    SimpleRNN(32),
    Dense(1)
])

# Compile and train
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
```

## Hybrid Models

### CNN + LSTM

We also experimented with hybrid models combining convolutional layers for feature extraction with recurrent layers for temporal modeling:

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Build CNN + LSTM model
cnn_lstm_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, 1)),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=False),
    Dense(1)
])

# Compile and train
cnn_lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
```

### LSTM Autoencoder + Dense

Another hybrid approach used an LSTM autoencoder for feature extraction followed by dense layers for prediction:

```python
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

# Encoder
encoder_inputs = Input(shape=(seq_length, 1))
encoder = LSTM(64, return_sequences=False)(encoder_inputs)

# Decoder for reconstruction
decoder = RepeatVector(seq_length)(encoder)
decoder = LSTM(64, return_sequences=True)(decoder)
decoder_outputs = TimeDistributed(Dense(1))(decoder)

# Autoencoder model
autoencoder = Model(encoder_inputs, decoder_outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)

# Extract encoder for feature extraction
encoder_model = Model(encoder_inputs, encoder)
encoded_features = encoder_model.predict(X_train)

# Build prediction model using encoded features
prediction_model = Sequential([
    Dense(32, activation='relu', input_shape=(64,)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

prediction_model.compile(optimizer='adam', loss='mse')
prediction_model.fit(encoded_features, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

## Advanced Architectures: TCN

Temporal Convolutional Networks (TCNs) use dilated convolutions to capture long-range dependencies with fewer parameters than RNNs:

```python
from tensorflow.keras.layers import Conv1D, Activation, Add, Lambda

# TCN Residual Block
def residual_block(x, dilation_rate, filters):
    # Dilated Causal Convolution
    skip = x
    x = Conv1D(filters, kernel_size=2, dilation_rate=dilation_rate, padding='causal', activation='relu')(x)
    x = Conv1D(filters, kernel_size=2, dilation_rate=dilation_rate, padding='causal', activation='relu')(x)
    
    # Residual connection
    if skip.shape[-1] != filters:
        skip = Conv1D(filters, kernel_size=1)(skip)
    x = Add()([x, skip])
    return x

# TCN Model
tcn_input = Input(shape=(seq_length, 1))
x = tcn_input

# Multiple TCN blocks with increasing dilation rates
for dilation_rate in [1, 2, 4, 8]:
    x = residual_block(x, dilation_rate, filters=64)

# Global pooling and prediction
x = GlobalAveragePooling1D()(x)
tcn_output = Dense(1)(x)

# Create and compile model
tcn_model = Model(tcn_input, tcn_output)
tcn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
```

## Model Evaluation

We evaluated all models using the same metrics:

```python
# Function to evaluate models
def evaluate_model(model, X_test, y_test, scaler, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform if data was scaled
    if scaler is not None:
        y_test_transform = np.zeros((len(y_test), 1))
        y_test_transform[:, 0] = y_test.flatten()
        y_pred_transform = np.zeros((len(y_pred), 1))
        y_pred_transform[:, 0] = y_pred.flatten()
        
        y_test = scaler.inverse_transform(y_test_transform)[:, 0]
        y_pred = scaler.inverse_transform(y_pred_transform)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mse, rmse, mae, r2, y_pred
```

## Results and Visualization

We compared the performance of all models and visualized their forecasts:

```python
# Plot all model predictions
plt.figure(figsize=(15, 8))
plt.plot(y_test, label='Actual')
plt.plot(lstm_pred, label='LSTM')
plt.plot(gru_pred, label='GRU')
plt.plot(cnn_lstm_pred, label='CNN+LSTM')
plt.plot(tcn_pred, label='TCN')
plt.title('OOE Forecasting: Model Comparison')
plt.xlabel('Time')
plt.ylabel('OOE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Conclusion

In our experiments, the hybrid CNN+LSTM model and the TCN architecture generally outperformed the traditional RNN approaches for OOE forecasting. These models better captured both the local patterns and long-term dependencies in the OOE time series data.

The TCN model, in particular, offered a good balance between forecasting accuracy and computational efficiency, making it suitable for production deployment.

For detailed implementation and results, refer to Notebook 3 (OEE-Insight_3) in the project repository.