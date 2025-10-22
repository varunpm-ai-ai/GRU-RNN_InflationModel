# 🇮🇳 India & Bengaluru Inflation Forecasting Models (GRU-RNN Model)
Predicting inflation rates of indian economy and benglore economy using an upgraded method of RNN, GRU-RNN method to get the best results 

## Project Overview

This project utilizes a GRU-RNN (Gated Recurrent Unit - Recurrent Neural Network) to predict inflation rates for both the Indian economy and the Bangalore region. The model leverages multiple economic indicators to achieve high accuracy in forecasting.

bash
```
Model                     Focus           Key Feature

Model 1: India National  National-level  Multi-year, iterative forecasting using user-defined
CPI Forecast             inflation.      baseline inputs.

Model 2: Bengaluru       Local/Regional  Demonstrates adaptability of the GRU architecture for
CPI Forecast             inflation.      specific regional data.
```

## Key Factors Used

- **CPI (Consumer Price Index)**
- **GDP Growth**
- **Unemployment Rate**
- **Crude Oil Prices**
- **Gold Prices**
- **Real Estate Index**
- **Stock Index**
- **Money Supply M1**
- **Money Supply M2**
- **Industrial Production**
- **Retail Sales**
- **Exchange Rate**

These factors are consistently used for both the Indian and Bangalore inflation predictions.

## Model Details

- **Model Type:** GRU-RNN
- **Accuracy:**
    - Bangalore: **82%**
    - India: **80%**

## Usage

1. Clone the repository.
2. Prepare your dataset with the above factors.
3. Train the GRU-RNN model using the provided scripts.
4. Evaluate the model performance for your region of interest.

## License

This project is released under the MIT License.
