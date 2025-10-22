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

##  Key Economic Indicators Used

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

## Technology Stack
Primary Language: Python 3.x
Modeling Framework: TensorFlow / Keras (GRU RNN)
Data Handling: Pandas, NumPy
Web Application: Streamlit (for interactive demonstration)
Model Persistence: joblib (for saving and loading scalers)
Visualization: Matplotlib / Streamlit native charting

##  Getting Started (Local Setup)
To run the interactive Streamlit application and recreate the models, follow these steps.

# Prerequisites
Python: Ensure you have Python 3.8+ installed.
Required Libraries: Install all necessary dependencies in requiremets.txt using pip.

bash
```
pip install requirements.txt
```

## File Structure
Your project directory must be structured as follows. You need to train and save all necessary files into the /models folder.

bash
```
/root_foulder
├──/datasets
│    ├──/Bengaloredataset
│    │   ├──bangalore_inflation_dataset.csv
│    ├──/IndiaDataset
│        ├──india_inflation_dataset.csv
│
├──/models
│   ├──gru_model.h5(indian model)
│   ├──gru_model2.h5(bengalore model)
│   ├──scaler.save(indian scaler model)
│   ├──scaler2.save(bengalore scaler model)
│
├──/NoteBooks
│   ├──/bengaluru
│   │   ├──bengaluru-inflation-model.ipynb
│   ├──/india
│       ├──india-inflation-model.ipynb
│
├──app.py
├──app2.py
├──README.md
```

## Usage

1. Clone the repository.
2. Prepare your dataset with the above factors.
3. Train the GRU-RNN model using the provided scripts.
4. Evaluate the model performance for your region of interest.

## License

This project is released under the MIT License.