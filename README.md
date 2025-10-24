# 🇮🇳 India & Bengaluru Inflation Forecasting Models (GRU-RNN Model)
Predicting inflation rates of indian economy and benglore economy using an upgraded method of RNN, GRU-RNN method to get the best results 

## Deployed app
https://varunpm-ai-ai-gru-rnn-inflationmodel-app-dr4zjj.streamlit.app/

# Screen shots 
![image alt](https://github.com/varunpm-ai-ai/GRU-RNN_InflationModel/blob/main/Screenshot%202025-10-22%20144236.png?raw=true)
<div align="center">
 <img src="https://github.com/varunpm-ai-ai/GRU-RNN_InflationModel/blob/main/Screenshot%202025-10-22%20144737.png?raw=true" alt="img1" width="100"  />
 <img src="https://github.com/varunpm-ai-ai/GRU-RNN_InflationModel/blob/main/Screenshot%202025-10-22%20144759.png?raw=true" alt="img2" width="100"  />
 <img src="https://github.com/varunpm-ai-ai/GRU-RNN_InflationModel/blob/main/Screenshot%202025-10-22%20193512.png?raw=true" alt="img3" width="100"  />
</div>

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
├──notebookscript.py
├──README.md
├──modelcard.md
├──CODE_OF_CONDUCT.md
├──CONTRIBUTING.md
├──pull_request_template.md
├──SECURITY.md
```

## Usage

1. Clone the repository.
2. Prepare your dataset with the above factors.
3. Train the GRU-RNN model using the provided scripts.
4. Evaluate the model performance for your region of interest.
5. Add the mode to the streamlit app 
6. bash
```
streamlit run app.py  
```
The application will open in your browser, allowing for multi-year forecasting based on user inputs.

## Model Architecture & Methodology
The core methodology for both the national and regional models is a standard practice for time series forecasting, allowing the model to learn relationships between historical economic data and future inflation.

# 1. Data Preparation (Crucial Steps)

Time Step Aggregation: Input data is processed using annual averages to focus on macro-economic shifts rather than monthly volatility.

Lagged Features: The model employs a time window of $N=4$ preceding years ($T-4$ to $T-1$) to predict the CPI in the target year $T$. This captures the historical inertia of inflation and economic policy impacts.

Feature Scaling: All 12 input features (including lagged CPI) are crucial for the GRU model. They are standardized using StandardScaler ($\mu=0, \sigma=1$) to prevent features with larger numerical scales (like Stock_Index) from dominating the learning process.

# 2. GRU Network Structure

Both Model 1 and Model 2 share this identical architecture, demonstrating model transferability across datasets:
bash
```
Layer   Type      Configuration                  Purpose

L1      GRU       16 units, tanh activation,     Extracts sequential dependencies and prepares
                  return_sequences=True          output for next GRU layer
 
L2      Dropout   20%                            Reduces complex co-adaptations between neurons
                                                 (overfitting)

L3      GRU       8 units, tanh activation,      Consolidates the 4-year sequence data into a final,
                  return_sequences=False         comprehensive vector for prediction.

L4      Dropout   20%                            Prevents overfitting.


L5      Dense     16 units, ReLU activation, L2  Performs non-linear transformation and feature
                  Regularization                 weighting prior to output.

output  Dense     1 unit (Linear activation)     Outputs the single predicted CPI value (scaled).

```

# 3. Forecasting Logic

The Streamlit app utilizes Iterative Multi-Step Prediction:

Prediction (Year $T$): The model takes the last 4 known years ($T-4$ to $T-1$) and predicts CPI for year $T$.

Iteration (Year $T+1$): To forecast the subsequent year ($T+1$), the oldest data point ($T-4$) is dropped. The new sequence is formed by $T-3, T-2, T-1,$ and the predicted CPI from year $T$. The other economic indicators are held constant at the user-provided baseline.

Output: This process repeats until the selected target year is reached, generating a robust, long-term forecast series.


## License

This project is released under the MIT License.
