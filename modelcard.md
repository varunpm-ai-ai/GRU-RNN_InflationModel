# Model Card: GRU-RNN Inflation Forecasting Model for [gru_model.h5 and gru_model2.h5]

## Model description
This model is a Gated Recurrent Unit (GRU) Neural Network designed for time-series forecasting of the Consumer Price Index (CPI) in [DATASET/REGION]. It leverages 12 macroeconomic indicators to capture the complex, non-linear relationships driving inflation trends over time.

## Key Factors Used
The model uses the following 12 factors to predict CPI, incorporating a 4-year sequence for temporal learning:
CPI (Consumer Price Index) (Lagged)
GDP Growth
Unemployment Rate
Crude Oil Prices
Gold Prices
Real Estate Index
Stock Index
Money Supply M1
Money Supply M2
Industrial Production
Retail Sales
Exchange Rate

## Architecture
The architecture is a deep recurrent network suitable for short-sequence time series (N=4 time steps):
Input Layer: (4 time steps, 12 features)
GRU Layers: Two stacked GRU layers (16 units and 8 units) with a 20% Dropout layer between them to prevent overfitting.
Dense Layers: A ReLU Dense layer with L2 regularization, followed by a final linear output layer for the predicted CPI value.
Intended Use
Primary Use: Multi-year forecasting of CPI for [DATASET/REGION].
Out-of-Scope: Predicting short-term (e.g., monthly) volatility or forecasting regions/countries not represented in the training data.

## Training Details
The model was trained on historical annual data (aggregated from monthly reports) and validated on a test set (e.g., the last 3 years of available data).
Training Time Steps: $N=4$ years (meaning the last 4 years of data are used to predict the next year).
Loss Function: Mean Squared Error (MSE).
Optimizer: Adam.
Key Performance Metric (Test Set): R-squared ($R^2$) and Root Mean Squared Error (RMSE).

### How to Use
The model files (GRU model and scalers) can be loaded into the provided Streamlit application (streamlit_inflation_app.py) for interactive multi-year forecasting.
bash
```
import joblib
from tensorflow.keras.models import load_model

# Load model and scalers for feature transformation and inverse-scaling of the output
gru_model = load_model('gru_model.h5')
scaler_X = joblib.load('scaler_X.joblib')
scaler_y = joblib.load('scaler_y.joblib')

# Example prediction setup:
# 1. Create a 4-step sequence (4x12 NumPy array) of your latest known data.
# 2. Scale the input data using scaler_X.
# 3. Predict the future scaled CPI using gru_model.predict(X_scaled).
# 4. Inverse transform the prediction using scaler_y to get the actual CPI.
```